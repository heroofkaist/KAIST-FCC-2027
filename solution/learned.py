"""Versioned local learned-model inference; optional libraries load only on use."""
from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np
from .features import swap_features
from .evidence import validate_blocks
from .pipeline import sha256


@dataclass
class LoadedLearner:
    manifest: dict
    backend: object
    root: Path


def load_learner(path, expected_features=None, *, _native=False):
    root = Path(path).resolve(); manifest = json.loads((root / 'model.json').read_text())
    kind = manifest.get('kind'); names = manifest.get('feature_names')
    if manifest.get('schema_version') != 1 or manifest.get('model_fps') != 5:
        raise ValueError('unsupported learned artifact schema/cadence')
    if kind not in ('lightgbm', 'catboost', 'xgboost', 'tcn', 'bilstm', 'transformer'):
        raise ValueError('unknown learned backend')
    if not isinstance(names, list) or not names or not all(isinstance(name, str) for name in names):
        raise ValueError('invalid feature schema')
    swap_features(np.empty((0, len(names))), names)
    if expected_features is not None and list(expected_features) != names:
        raise ValueError('artifact feature schema does not match inference')
    filename = manifest.get('weights_file')
    if not isinstance(filename, str) or Path(filename).name != filename:
        raise ValueError('weights path must be a filename inside the artifact')
    weights = (root / filename).resolve()
    if weights.parent != root or sha256(weights) != manifest.get('weights_sha256'):
        raise ValueError('model weights hash/location mismatch')
    if not _native:
        return LoadedLearner(manifest, None, root)
    if kind == 'lightgbm':
        import lightgbm as lgb
        backend = lgb.Booster(model_file=str(weights))
    elif kind == 'catboost':
        from catboost import CatBoostClassifier
        backend = CatBoostClassifier(); backend.load_model(str(weights))
    elif kind == 'xgboost':
        from xgboost import XGBClassifier
        backend = XGBClassifier(); backend.load_model(weights)
    else:
        import torch
        from experiments.networks import build_network
        backend = build_network(kind, len(names), manifest['configuration'])
        backend.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True), strict=True)
        backend.eval()
    return LoadedLearner(manifest, backend, root)


def predict_probabilities(learner, features, blocks=None):
    if not isinstance(learner, LoadedLearner):
        learner = load_learner(learner)
    x = np.asarray(features, dtype=np.float32); names = learner.manifest['feature_names']
    if x.ndim != 2 or x.shape[1] != len(names) or not np.isfinite(x).all():
        raise ValueError('invalid learned inference features')
    blocks = blocks if blocks is not None else [(0, len(x))]
    validate_blocks(blocks, len(x))
    if learner.backend is None:
        from experiments.native_worker import predict_isolated
        return predict_isolated(learner.root, x, blocks)
    def raw(values):
        kind = learner.manifest['kind']
        if kind in ('tcn', 'bilstm', 'transformer'):
            from experiments.networks import network_probabilities
            return network_probabilities(learner.backend, values, blocks)
        if kind == 'lightgbm':
            return learner.backend.predict(values, num_threads=4)
        return learner.backend.predict_proba(values)
    p = np.asarray(raw(x), dtype=np.float64)
    q = np.asarray(raw(swap_features(x, names)), dtype=np.float64)[:, [1, 0, 2]]
    result = (p + q) * .5
    if result.shape != (len(x), 3) or not np.isfinite(result).all() or (result < 0).any():
        raise ValueError('invalid learned probabilities')
    result = np.maximum(result, 1e-9)
    return result / result.sum(axis=1, keepdims=True)
