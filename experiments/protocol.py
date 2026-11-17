"""Frozen group splits and observation-only missing-ball experiments."""
import hashlib
import json
from pathlib import Path
import platform

import numpy as np

from solution.pipeline import ROOT, _check_output, sha256
from .corpus import load_corpus_match, MODEL_FPS

PROTOCOL_PATH = Path(__file__).resolve().parents[1] / 'configs/research-protocol.json'
PROTOCOL = json.loads(PROTOCOL_PATH.read_text())


def development_folds():
    games = PROTOCOL['sportec_development']
    return [(tuple(game for game in games if game not in valid), tuple(valid))
            for valid in PROTOCOL['sportec_validation_folds']]


def validate_split(train, validation):
    train, validation = tuple(train), tuple(validation)
    forbidden = set(PROTOCOL['sportec_test'] + PROTOCOL['skillcorner_test'])
    if not train or not validation or len(set(train)) != len(train) or len(set(validation)) != len(validation):
        raise ValueError('empty or duplicate match split')
    if set(train) & set(validation):
        raise ValueError('training/validation match overlap')
    if (set(train) | set(validation)) & forbidden:
        raise ValueError('held-out test cannot enter development splits')


def load_match(root, source, match_id, *, allow_test=False):
    match_id = str(match_id)
    if source not in ('sportec', 'skillcorner') or not match_id.isalnum():
        raise ValueError('invalid external source/match ID')
    test = PROTOCOL[source + '_test']
    if match_id in test and not allow_test:
        raise ValueError('held-out match access must be explicit')
    if source == 'sportec' and match_id not in PROTOCOL['sportec_development'] + test:
        raise ValueError('match is not in the frozen Sportec protocol')
    if source == 'skillcorner' and not match_id.isdigit():
        raise ValueError('invalid SkillCorner match ID')
    path = Path(root) / source / (match_id + '.npz')
    data = load_corpus_match(path)
    if data['provenance']['source'] != source or data['provenance']['match_id'] != match_id:
        raise ValueError('corpus identity does not match requested group')
    return data


def observe(data, scenario, seed=20261109, dropout_pattern=None):
    if scenario not in ('clean', 'masked', 'long', 'native', 'skill_dropout'):
        raise ValueError('unknown observation scenario')
    indices = data['indices']; n = len(indices)
    blocks = [tuple(map(int, b)) for b in data['sample_blocks']]
    codes = data['nearest'][indices].copy()
    ball = data['ball'].copy(); players = data['players'].copy(); height = data['height'].copy()
    identity = data['provenance']
    material = f"{seed}|{identity['source']}|{identity['match_id']}|{scenario}"
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(material.encode()).digest()[:8], 'little'))
    missing = np.zeros(n, bool)
    if scenario == 'native':
        codes = data['observed'][indices].copy()
        missing = ~data['ball_detected'][indices]
        players[~data['player_detected']] = np.nan
    elif scenario != 'clean':
        pitch = np.asarray(identity['pitch'])
        missing |= (ball < 0).any(axis=1) | (ball > pitch).any(axis=1) | (height > 2)
        for start, stop in blocks:
            if scenario == 'skill_dropout':
                pattern = np.asarray(dropout_pattern, dtype=bool)
                if pattern.ndim != 1 or not len(pattern):
                    raise ValueError('a nonempty external dropout pattern is required')
                offset = rng.integers(len(pattern))
                missing[start:stop] |= pattern[(np.arange(stop - start) + offset) % len(pattern)]
                continue
            cursor = start
            while cursor < stop:
                length = MODEL_FPS if scenario == 'masked' else int(rng.integers(2 * MODEL_FPS, 8 * MODEL_FPS + 1))
                end = min(stop, cursor + length)
                if rng.random() < (.35 if scenario == 'masked' else .45):
                    missing[cursor:end] = True
                cursor = end
            if scenario == 'long':
                # Brief nearest-player attribution mistakes, independent of human truth.
                for cursor in np.flatnonzero(rng.random(stop - start) < .015) + start:
                    end = min(cursor + 2, stop)
                    region = codes[cursor:end]
                    region[region < 2] = 1 - region[region < 2]
    codes[missing] = 2; ball[missing] = np.nan; height[missing] = np.nan
    return {'codes': codes, 'ball': ball, 'players': players, 'height': height,
            'team': data['team'].copy(), 'blocks': blocks, 'fps': MODEL_FPS,
            'pitch': tuple(identity['pitch'])}


def create_run(out, config):
    out = _check_output(out, ROOT / 'data', [])
    out.mkdir(parents=True, exist_ok=False)
    source_paths = sorted((ROOT / 'solution').glob('*.py')) + sorted((ROOT / 'experiments').glob('*.py'))
    manifest = {'configuration': config, 'protocol': PROTOCOL,
                'protocol_sha256': sha256(PROTOCOL_PATH),
                'source_sha256': {str(p.relative_to(ROOT)): sha256(p) for p in source_paths},
                'runtime': {'python': platform.python_version(), 'numpy': np.__version__}}
    (out / 'run.json').write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + '\n')
    return out
