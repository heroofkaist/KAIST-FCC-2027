"""Local, label-separated external trajectories and native/model time scales.

This module never reads competition files. Labels remain native-resolution;
positions are sampled at 5 Hz without crossing period or missing-frame breaks.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from solution.evidence import validate_blocks
from solution.pipeline import sha256
from .sportec import read_sportec, nearest_labels

MODEL_FPS = 5
SPORT = ('J03WMX', 'J03WN1', 'J03WOH', 'J03WOY', 'J03WPY', 'J03WQQ', 'J03WR9')


def sample_indices(blocks, n_frames, native_fps, model_fps=MODEL_FPS):
    validate_blocks(blocks, n_frames)
    if not np.isfinite(native_fps) or not np.isfinite(model_fps) or model_fps <= 0:
        raise ValueError('invalid cadence')
    ratio = native_fps / model_fps
    if ratio < 1 or not np.isclose(ratio, round(ratio)):
        raise ValueError('native cadence must be an integer multiple of model cadence')
    indices, result = [], []
    for start, stop in blocks:
        offset = len(indices)
        indices.extend(range(start, stop, round(ratio)))
        result.append((offset, len(indices)))
    return np.asarray(indices, dtype=np.int64), result


def upsample_probabilities(probability, indices, native_blocks, n_frames):
    validate_blocks(native_blocks, n_frames)
    p = np.asarray(probability, dtype=float); indices = np.asarray(indices)
    if p.shape != (len(indices), 3) or not np.isfinite(p).all() or (p < 0).any():
        raise ValueError('invalid model probabilities')
    if not np.allclose(p.sum(axis=1), 1) or indices.ndim != 1 or indices.dtype.kind not in 'iu':
        raise ValueError('invalid probability sums or sample indices')
    if len(indices) and ((np.diff(indices) <= 0).any() or indices[0] < 0 or indices[-1] >= n_frames):
        raise ValueError('sample indices must increase inside the native timeline')
    result = np.full((n_frames, 3), 1 / 3, dtype=float)
    covered = np.zeros(len(indices), bool)
    for start, stop in native_blocks:
        chosen = (indices >= start) & (indices < stop)
        xs = indices[chosen]
        if not len(xs) or xs[0] != start:
            raise ValueError('every native block must have a sample at its start')
        for k in range(3):
            result[start:stop, k] = np.interp(np.arange(start, stop), xs, p[chosen, k])
        covered |= chosen
    if not covered.all():
        raise ValueError('model samples outside valid blocks')
    return result


def _codes(labels):
    return np.asarray([{'H': 0, 'A': 1, 'D': 2, 'X': 3}[x] for x in labels], dtype=np.int8)


def _write(out, arrays, provenance, input_hashes):
    out = Path(out)
    if out.exists():
        raise FileExistsError('prepared corpus match already exists')
    for path, expected in input_hashes.items():
        if sha256(path) != expected:
            raise ValueError('source changed during preparation')
    provenance.update({'schema_version': 2, 'model_fps': MODEL_FPS,
                       'inputs': {Path(p).name: value for p, value in input_hashes.items()},
                       'adapter_sha256': sha256(__file__)})
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('xb') as handle:
        np.savez_compressed(handle, **arrays, provenance=np.array(json.dumps(provenance, sort_keys=True)))
    return provenance


def prepare_sportec(meta, raw, out):
    hashes = {str(p): sha256(p) for p in (meta, raw)}
    match = read_sportec(meta, raw)
    n = len(match.labels); indices, sample_blocks = sample_indices(match.blocks, n, 25)
    nearest = _codes(nearest_labels(match.tracking))
    arrays = {'labels': _codes(match.labels), 'nearest': nearest, 'observed': nearest.copy(),
              'blocks': np.asarray(match.blocks, dtype=np.int64), 'indices': indices,
              'sample_blocks': np.asarray(sample_blocks, dtype=np.int64),
              'ball': match.tracking.ball[indices].astype(np.float32),
              'players': match.tracking.players[indices].astype(np.float32),
              'player_detected': np.isfinite(match.tracking.players[indices]).all(axis=2),
              'team': match.tracking.team, 'height': match.height[indices].astype(np.float32),
              'ball_detected': np.isfinite(match.tracking.ball).all(axis=1)}
    return _write(out, arrays, {'match_id': Path(raw).stem, 'source': 'sportec',
                               'native_fps': 25, 'weak_labels': False,
                               'pitch': list(match.tracking.pitch), 'license': 'CC BY 4.0',
                               'dataset': 'https://doi.org/10.6084/m9.figshare.28196177'}, hashes)


def _point(data, centre):
    if not data or data.get('x') is None or data.get('y') is None:
        return np.array([np.nan, np.nan], dtype=np.float32)
    xy = np.array([data['x'], data['y']], dtype=np.float32)
    if not np.isfinite(xy).all():
        raise ValueError('non-finite external coordinates')
    return xy + centre


def prepare_skillcorner(meta, raw, phases, out):
    if Path(out).exists():
        raise FileExistsError('prepared corpus match already exists')
    hashes = {str(p): sha256(p) for p in (meta, raw, phases)}
    metadata = json.loads(Path(meta).read_text())
    pitch = [float(metadata['pitch_length']), float(metadata['pitch_width'])]
    if not np.isfinite(pitch).all() or min(pitch) <= 0:
        raise ValueError('invalid pitch')
    centre = np.asarray(pitch, dtype=np.float32) / 2
    team_ids = [metadata['home_team']['id'], metadata['away_team']['id']]
    if team_ids[0] == team_ids[1]:
        raise ValueError('teams must differ')
    roster = [p for p in metadata['players'] if p['team_id'] in team_ids]
    slots = {p['id']: i for i, p in enumerate(roster)}
    if len(slots) != len(roster) or not roster:
        raise ValueError('invalid roster')
    team = np.asarray([team_ids.index(p['team_id']) for p in roster], dtype=np.int8)
    if set(team.tolist()) != {0, 1}:
        raise ValueError('both teams require players')
    with Path(raw).open() as handle:
        n = sum(bool(line.strip()) for line in handle)
    labels = np.full(n, 3, dtype=np.int8); nearest = np.full(n, 2, dtype=np.int8)
    observed = nearest.copy(); detected = np.zeros(n, bool)
    blocks = []; active_start = None; previous_period = None; previous_frame = None
    indices = []; balls = []; players = []; player_flags = []; heights = []
    i = -1
    with Path(raw).open() as handle:
        for line in handle:
            if not line.strip():
                continue
            i += 1; row = json.loads(line); frame = row['frame']; period = row.get('period')
            if not isinstance(frame, int) or isinstance(frame, bool) or (previous_frame is not None and frame <= previous_frame):
                raise ValueError('tracking frame IDs must strictly increase')
            valid = period in (1, 2, 3, 4)
            boundary = (not valid or period != previous_period or previous_frame is None or frame != previous_frame + 1)
            if boundary and active_start is not None:
                blocks.append((active_start, i)); active_start = None
            if valid and active_start is None:
                active_start = i
            previous_frame = frame; previous_period = period
            if not valid:
                continue
            group = (row.get('possession') or {}).get('group')
            if group not in (None, 'home team', 'away team'):
                raise ValueError('unknown SkillCorner possession group')
            labels[i] = {'home team': 0, 'away team': 1, None: 3}[group]
            ball_data = row.get('ball_data') or {}; ball = _point(ball_data, centre)
            current = np.full((len(slots), 2), np.nan, dtype=np.float32)
            flags = np.zeros(len(slots), bool); seen = set()
            for player in row.get('player_data') or []:
                person = player['player_id']
                if person not in slots or person in seen:
                    raise ValueError('unknown or duplicate player ID')
                seen.add(person); slot = slots[person]
                current[slot] = _point(player, centre)
                flags[slot] = player.get('is_detected') is True
            distance = np.linalg.norm(current - ball, axis=1)
            distance[~np.isfinite(distance)] = np.inf
            if np.isfinite(distance).any():
                nearest[i] = team[distance.argmin()]
            detected[i] = ball_data.get('is_detected') is True and np.isfinite(ball).all()
            observed_distance = np.where(flags, distance, np.inf)
            if detected[i] and np.isfinite(observed_distance).any():
                observed[i] = team[observed_distance.argmin()]
            if (i - active_start) % 2 == 0:
                indices.append(i); balls.append(ball); players.append(current); player_flags.append(flags)
                height = ball_data.get('z')
                heights.append(float(height) if height is not None else np.nan)
    if active_start is not None:
        blocks.append((active_start, n))
    expected, sample_blocks = sample_indices(blocks, n, 10)
    if not np.array_equal(indices, expected) or not len(expected):
        raise ValueError('invalid or empty sampled timeline')
    arrays = {'labels': labels, 'nearest': nearest, 'observed': observed, 'ball_detected': detected,
              'blocks': np.asarray(blocks, dtype=np.int64), 'indices': expected,
              'sample_blocks': np.asarray(sample_blocks, dtype=np.int64),
              'ball': np.asarray(balls, dtype=np.float32), 'players': np.asarray(players, dtype=np.float32),
              'player_detected': np.asarray(player_flags, dtype=bool), 'team': team,
              'height': np.asarray(heights, dtype=np.float32)}
    return _write(out, arrays, {'match_id': str(metadata['id']), 'source': 'skillcorner',
                               'native_fps': 10, 'weak_labels': True, 'pitch': pitch,
                               'license': 'MIT; retain SkillCorner LICENSE and attribution',
                               'dataset': 'https://github.com/SkillCorner/opendata'}, hashes)


def load_corpus_match(path):
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key].copy() for key in archive.files if key != 'provenance'}
        data['provenance'] = json.loads(str(archive['provenance']))
    if data['provenance'].get('schema_version') != 2:
        raise ValueError('unsupported corpus schema')
    n = len(data['labels']); m = len(data['indices'])
    blocks = [tuple(map(int, b)) for b in data['blocks']]
    expected, samples = sample_indices(blocks, n, data['provenance']['native_fps'])
    if not np.array_equal(expected, data['indices']) or not np.array_equal(samples, data['sample_blocks']):
        raise ValueError('corrupt sample timeline')
    for key in ('labels', 'nearest', 'observed'):
        if data[key].shape != (n,) or not np.isin(data[key], [0, 1, 2, 3]).all():
            raise ValueError('invalid corpus label channel')
    if data['ball'].shape != (m, 2) or data['players'].shape != (m, len(data['team']), 2):
        raise ValueError('corrupt coordinate shape')
    if np.isinf(data['ball']).any() or np.isinf(data['players']).any():
        raise ValueError('infinite corpus coordinate')
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', choices=('sportec', 'skillcorner', 'all'), default='all')
    parser.add_argument('--data-dir', type=Path, default=Path('data/external'))
    parser.add_argument('--out', type=Path, default=Path('data/external/prepared-v2'))
    args = parser.parse_args()
    jobs = []
    if args.source in ('sportec', 'all'):
        for game in SPORT:
            folder = args.data_dir / 'sportec'
            jobs.append((prepare_sportec, (folder / (game + '-meta.xml'), folder / (game + '.xml')),
                         args.out / 'sportec' / (game + '.npz')))
    if args.source in ('skillcorner', 'all'):
        for folder in sorted((args.data_dir / 'skillcorner/data/matches').iterdir()):
            game = folder.name
            jobs.append((prepare_skillcorner, tuple(folder / (game + suffix) for suffix in
                         ('_match.json', '_tracking_extrapolated.jsonl', '_phases_of_play.csv')),
                         args.out / 'skillcorner' / (game + '.npz')))
    for function, inputs, out in jobs:
        if out.exists():
            load_corpus_match(out)
            print(out.stem, 'existing verified cache', flush=True)
            continue
        provenance = function(*inputs, out)
        print(provenance['source'], provenance['match_id'], 'prepared', flush=True)


if __name__ == '__main__':
    main()
