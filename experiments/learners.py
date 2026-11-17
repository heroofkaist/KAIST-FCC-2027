"""Train permitted external observations; export safe, hash-checked local weights."""
import json
from pathlib import Path
import time
import numpy as np
from solution.evidence import validate_blocks
from solution.features import swap_features
from solution.pipeline import sha256


def fit_learner(kind, config, training, out, seed=20261109, *, _native=False):
    out = Path(out)
    if out.exists():
        raise FileExistsError('learner output already exists')
    if not training:
        raise ValueError('external training sequences are required')
    if not _native:
        from .native_worker import fit_isolated
        return fit_isolated(kind, config, training, out, seed)
    names = tuple(training[0]['feature_names']); width = len(names)
    swap = swap_features(np.arange(width)[None], names)[0]
    xs, ys, weights = [], [], []
    for row in training:
        x = np.asarray(row['x'], dtype=np.float32); y = np.asarray(row['y'])
        if tuple(row['feature_names']) != names or x.ndim != 2 or x.shape[1] != width or y.shape != (len(x),):
            raise ValueError('inconsistent training schema')
        if not np.isfinite(x).all() or not np.isin(y, [0, 1, 2, 3]).all():
            raise ValueError('invalid training observations/labels')
        validate_blocks(row['blocks'], len(x)); valid = y < 3
        weight = float(row.get('weight', 1))
        if not valid.any() or not np.isfinite(weight) or weight <= 0:
            raise ValueError('training sequence needs valid labels and a positive weight')
        xs.append(x[valid]); ys.append(y[valid].astype(np.int64))
        weights.append(np.full(valid.sum(), weight / valid.sum()))
    x = np.concatenate(xs); y = np.concatenate(ys); w = np.concatenate(weights)
    if set(y.tolist()) != {0, 1, 2}:
        raise ValueError('training corpus must cover H/A/D')
    w /= w.mean(); out.mkdir(parents=True, exist_ok=False); started = time.monotonic()
    if kind in ('lightgbm', 'catboost', 'xgboost'):
        x = np.concatenate([x, x[:, swap]]); y = np.r_[y, np.where(y < 2, 1 - y, y)]; w = np.r_[w, w]
    history = None
    if kind == 'lightgbm':
        import lightgbm as lgb
        params = {'objective': 'multiclass', 'num_class': 3, 'learning_rate': config.get('learning_rate', .05),
                  'num_leaves': config.get('leaves', 31), 'min_data_in_leaf': config.get('min_leaf', 200),
                  'lambda_l2': config.get('l2', 3.), 'verbosity': -1, 'seed': seed,
                  'num_threads': 4, 'deterministic': True, 'force_col_wise': True}
        model = lgb.train(params, lgb.Dataset(x, label=y, weight=w, feature_name=list(names)),
                          num_boost_round=config.get('rounds', 300))
        filename = 'weights.txt'; model.save_model(str(out / filename)); version = lgb.__version__
    elif kind == 'catboost':
        import catboost
        model = catboost.CatBoostClassifier(iterations=config.get('rounds', 350), depth=config.get('depth', 6),
                                            learning_rate=config.get('learning_rate', .07),
                                            l2_leaf_reg=config.get('l2', 5.), loss_function='MultiClass',
                                            thread_count=4, random_seed=seed, verbose=False, allow_writing_files=False)
        model.fit(x, y, sample_weight=w)
        filename = 'weights.cbm'; model.save_model(str(out / filename)); version = catboost.__version__
    elif kind == 'xgboost':
        import xgboost
        model = xgboost.XGBClassifier(n_estimators=config.get('rounds', 300), max_depth=config.get('depth', 4),
                                      learning_rate=config.get('learning_rate', .05), reg_lambda=config.get('l2', 5.),
                                      min_child_weight=config.get('min_child_weight', 30), subsample=.9,
                                      colsample_bytree=.9, objective='multi:softprob', num_class=3,
                                      tree_method='hist', n_jobs=4, random_state=seed, verbosity=0)
        model.fit(x, y, sample_weight=w)
        filename = 'weights.ubj'; model.save_model(out / filename); version = xgboost.__version__
    elif kind in ('tcn', 'bilstm', 'transformer'):
        import torch
        from .networks import train_network
        model, history = train_network(kind, config, training, swap, seed)
        filename = 'weights.pt'; torch.save(model.state_dict(), out / filename); version = torch.__version__
    else:
        raise ValueError('unknown learner')
    manifest = {'schema_version': 1, 'kind': kind, 'backend_version': version,
                'feature_names': list(names), 'model_fps': 5, 'configuration': config, 'seed': seed,
                'weights_file': filename, 'weights_sha256': sha256(out / filename),
                'training_matches': sorted({row['match_id'] for row in training}),
                'training_rows': int(sum(len(values) for values in ys)),
                'seconds': time.monotonic() - started, 'loss_history': history,
                'team_symmetry': 'average native and exchanged-team predictions'}
    (out / 'model.json').write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + '\n')
    return manifest
