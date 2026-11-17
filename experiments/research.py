"""Grouped external-only model comparisons, with immutable run directories.

python -m experiments.research cv --out artifacts/research-cv
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import time

import numpy as np

from solution.config import Config
from solution.evidence import Evidence
from solution.features import categorical_features
from solution.spatial import spatial_features
from solution.io import Tracking
from solution.fast_decode import fast_decode
from solution.learned import load_learner, predict_probabilities
from solution.pipeline import sha256
from solution import temporal
from starter.baseline import smooth
from .corpus import upsample_probabilities
from .protocol import PROTOCOL, development_folds, validate_split, load_match, observe, create_run
from .diagnostics import score_match
from .learners import fit_learner

REGISTRY_PATH = Path(__file__).resolve().parents[1] / 'configs/research-candidates.json'
CANDIDATES = json.loads(REGISTRY_PATH.read_text())
DEFAULT_DECODER = {'transition_seconds': .4, 'short_segment_cost': 2.}


def build_row(data, scenario, candidate, seed=20261109, dropout_pattern=None):
    observations = observe(data, scenario, seed, dropout_pattern)
    if candidate['kind'] == 'linear':
        letters = np.array(['H', 'A', 'D', 'X'])[observations['codes']]
        x = temporal.temporal_features(letters, observations['blocks'], 5).astype(np.float32)
        names = temporal.FEATURES
    else:
        x, names = categorical_features(observations['codes'], observations['blocks'], 5)
        if candidate.get('features') == 'spatial':
            tracking = Tracking(observations['ball'], observations['players'], observations['team'],
                                np.full(observations['players'].shape[:2], np.nan),
                                ('Home', 'Away'), observations['pitch'])
            geometry, spatial_names = spatial_features(tracking, observations['blocks'], 5, observations['height'])
            x = np.column_stack([x, geometry]); names += spatial_names
    return {'x': x, 'y': data['labels'][data['indices']].copy(),
            'codes': observations['codes'], 'blocks': observations['blocks'],
            'feature_names': names, 'match_id': data['provenance']['match_id'], 'weight': 1.}


def decode_probabilities(probability, indices, blocks, n_frames, native_fps, decoder):
    p = upsample_probabilities(probability, indices, blocks, n_frames)
    config = replace(Config(), fps=float(native_fps), mature_frames=round(native_fps),
                     transition_cost=decoder['transition_seconds'] * native_fps,
                     short_segment_cost=decoder['short_segment_cost'] * native_fps / 5)
    labels = fast_decode(Evidence(np.log(np.maximum(p, 1e-9)), np.ones(n_frames, bool)), blocks, config)
    codes = np.array([{'H': 0, 'A': 1, 'D': 2, 'X': 3}[x] for x in labels], dtype=np.int8)
    return codes, p


def summarize(rows):
    expected = {(game, scenario) for game in PROTOCOL['sportec_development']
                for scenario in PROTOCOL['selection_scenarios']}
    grouped = {}
    for row in rows:
        grouped.setdefault(row['candidate'], []).append(row)
    results = []
    for name in sorted(grouped):
        values = grouped[name]
        keys = [(row['match_id'], row['scenario']) for row in values]
        if len(keys) != len(set(keys)) or set(keys) != expected:
            raise ValueError('selection requires exactly one score per development match/scenario')
        values = sorted(values, key=lambda row: (row['match_id'], row['scenario']))
        by_match = {game: float(np.mean([row['scores']['spellF1'] for row in values if row['match_id'] == game]))
                    for game in PROTOCOL['sportec_development']}
        results.append({'candidate': name, 'mean_spellF1': float(np.mean(list(by_match.values()))),
                        'mean_macroF1': float(np.mean([row['scores']['macroF1'] for row in values])),
                        'mean_dead_f1': float(np.mean([row['scores']['dead_f1'] for row in values])),
                        'by_match': by_match,
                        'by_scenario': {scenario: float(np.mean([row['scores']['spellF1'] for row in values if row['scenario'] == scenario]))
                                        for scenario in PROTOCOL['selection_scenarios']}})
    return sorted(results, key=lambda row: (-row['mean_spellF1'], row['candidate']))


def run_cv(root, out, candidates, folds=None, seed=20261109):
    root = Path(root)
    folds = folds if folds is not None else development_folds()
    for train, valid in folds:
        validate_split(train, valid)
    out = create_run(out, {'stage': 'cv', 'candidates': candidates, 'folds': folds,
                           'seed': seed, 'initial_decoder': DEFAULT_DECODER,
                           'candidate_registry_sha256': sha256(REGISTRY_PATH)})
    games = sorted({game for train, valid in folds for game in tuple(train) + tuple(valid)})
    data = {game: load_match(root, 'sportec', game) for game in games}
    paths = {game: root / 'sportec' / (game + '.npz') for game in games}
    weak_games = []
    if any(c.get('weak_weight') or c.get('skill_dropout') for c in candidates.values()):
        weak_games = [p.stem for p in sorted((root / 'skillcorner').glob('*.npz'))
                      if p.stem not in PROTOCOL['skillcorner_test']]
        for game in weak_games:
            data[game] = load_match(root, 'skillcorner', game)
            paths[game] = root / 'skillcorner' / (game + '.npz')
    (out / 'corpus.json').write_text(json.dumps({game: sha256(path) for game, path in paths.items()}, indent=2, sort_keys=True) + '\n')
    pattern = None
    if any(c.get('skill_dropout') for c in candidates.values()):
        pattern = np.concatenate([~data[g]['ball_detected'][data[g]['indices']] for g in weak_games])
    row_cache = {}; rows = []; predictions = out / 'predictions'; predictions.mkdir()
    def row_for(game, scenario, candidate):
        key = (game, scenario, candidate['kind'] == 'linear', candidate.get('features', 'categorical'))
        if key not in row_cache:
            row_cache[key] = build_row(data[game], scenario, candidate, seed, pattern)
        return dict(row_cache[key])
    for name, candidate in candidates.items():
        for fold_id, (train, valid) in enumerate(folds):
            started = time.monotonic(); kind = candidate['kind']
            print('START', name, 'fold', fold_id, 'train', ','.join(train), 'validate', ','.join(valid), flush=True)
            model_root = out / 'models' / name / f'fold-{fold_id}'
            scenarios = list(PROTOCOL['selection_scenarios'])
            if candidate.get('skill_dropout'):
                scenarios.append('skill_dropout')
            training = []
            if kind != 'smooth':
                for game in train:
                    for scenario in scenarios:
                        row = row_for(game, scenario, candidate); row['weight'] = 1 / len(scenarios)
                        training.append(row)
                if candidate.get('weak_weight'):
                    for game in weak_games:
                        row = row_for(game, 'native', candidate); row['weight'] = candidate['weak_weight']; training.append(row)
                if kind == 'linear':
                    xs, ys = [], []
                    for row in training:
                        take = np.arange(0, len(row['y']), 2); take = take[row['y'][take] < 3]
                        xs.append(row['x'][take]); ys.append(row['y'][take])
                    model = temporal.fit(np.concatenate(xs), np.concatenate(ys), **candidate.get('configuration', {}))
                    model_root.mkdir(parents=True); temporal.save_model(model_root / 'model.json', model)
                else:
                    fit_learner(kind, candidate.get('configuration', {}), training, model_root, seed)
                    model = load_learner(model_root)
            for game in valid:
                match = data[game]
                blocks = [tuple(map(int, x)) for x in match['blocks']]
                for scenario in PROTOCOL['selection_scenarios']:
                    row = row_for(game, scenario, candidate)
                    if kind == 'smooth':
                        raw = np.where(row['codes'] < 2, row['codes'], -1)
                        coded = np.full(len(raw), 3, dtype=np.int8)
                        for a, b in row['blocks']:
                            labels = smooth(raw[a:b], round(candidate['seconds'] * 5))
                            coded[a:b] = np.where(labels == 'H', 0, np.where(labels == 'A', 1, 2))
                        probability = np.eye(3)[np.minimum(coded, 2)]
                        full_p = upsample_probabilities(probability, match['indices'], blocks, len(match['labels']))
                        prediction = full_p.argmax(axis=1).astype(np.int8)
                    else:
                        probability = np.exp(temporal.predict_scores(row['x'], model)) if kind == 'linear' else predict_probabilities(model, row['x'], row['blocks'])
                        prediction, full_p = decode_probabilities(probability, match['indices'], blocks, len(match['labels']), match['provenance']['native_fps'], DEFAULT_DECODER)
                    scores = score_match(match['labels'], prediction, blocks, match['provenance']['native_fps'], full_p)
                    filename = f'{name}__{game}__{scenario}.npz'
                    np.savez_compressed(predictions / filename, probability=probability, prediction=prediction)
                    result = {'candidate': name, 'definition': candidate, 'match_id': game,
                              'scenario': scenario, 'fold': fold_id, 'seed': seed,
                              'prediction_file': 'predictions/' + filename,
                              'prediction_sha256': sha256(predictions / filename), 'scores': scores}
                    rows.append(result)
                    (out / 'scores.json').write_text(json.dumps(rows, indent=2, sort_keys=True, allow_nan=False) + '\n')
                    print(name, game, scenario, 'spellF1', round(scores['spellF1'], 4), flush=True)
            print('DONE', name, 'fold', fold_id, 'seconds', round(time.monotonic() - started, 1), flush=True)
        if len(folds) == 3:
            complete = [row for row in rows if row['candidate'] == name]
            summary = summarize(complete)[0]
            print('SUMMARY', json.dumps(summary, sort_keys=True), flush=True)
    if len(folds) == 3:
        (out / 'summary.json').write_text(json.dumps(summarize(rows), indent=2, sort_keys=True) + '\n')
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('cv',))
    parser.add_argument('--data-dir', type=Path, default=Path('data/external/prepared-v2'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--candidates', nargs='+')
    parser.add_argument('--seed', type=int, default=PROTOCOL['seed'])
    args = parser.parse_args()
    names = args.candidates or [name for name in CANDIDATES if name not in ('spatial_lgb31', 'weak_lgb31', 'dropout_lgb31')]
    unknown = set(names) - set(CANDIDATES)
    if unknown:
        parser.error('unknown candidates: ' + ', '.join(sorted(unknown)))
    candidates = {name: CANDIDATES[name] for name in names}
    run_cv(args.data_dir, args.out, candidates, seed=args.seed)


if __name__ == '__main__':
    main()
