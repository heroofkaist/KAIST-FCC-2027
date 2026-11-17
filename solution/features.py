"""Label-independent multiscale features of automatic owner observations."""
import numpy as np
from .evidence import validate_blocks

WINDOWS = (.2, .6, 1.5, 3., 6., 12., 24.)


def swap_features(features, names):
    names = tuple(names)
    lookup = {name: i for i, name in enumerate(names)}
    if len(lookup) != len(names):
        raise ValueError('feature names must be unique')
    swapped = [name.replace('home', 'TEMP').replace('away', 'home').replace('TEMP', 'away') for name in names]
    if any(name not in lookup for name in swapped):
        raise ValueError('incomplete home/away feature pairs')
    x = np.asarray(features)
    if x.ndim != 2 or x.shape[1] != len(names):
        raise ValueError('feature shape does not match schema')
    return x[:, [lookup[name] for name in swapped]]


def _duration(frames, fps):
    return np.log1p(np.minimum(np.asarray(frames) / fps, 120)) / np.log(121)


def categorical_features(labels, blocks, fps):
    labels = np.asarray(labels)
    if labels.ndim != 1 or not np.isin(labels, [0, 1, 2, 3]).all():
        raise ValueError('observations must be H/A/D/X codes')
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError('fps must be positive')
    validate_blocks(blocks, len(labels))
    names = [f'{side}_now' for side in ('home', 'away', 'missing')]
    names += [f'{side}_{direction}_{seconds:g}s' for seconds in WINDOWS
              for direction in ('past', 'future') for side in ('home', 'away', 'missing')]
    names += ['run_before', 'run_after', 'run_length', 'since_home', 'until_home',
              'since_away', 'until_away', 'since_known', 'until_known',
              'previous_home', 'previous_away', 'next_home', 'next_away', 'same_owner_gap']
    names = tuple(names); lookup = {name: i for i, name in enumerate(names)}
    features = np.zeros((len(labels), len(names)), dtype=np.float32)
    for start, stop in blocks:
        sequence = labels[start:stop]; n = len(sequence); index = np.arange(n)
        indicators = np.column_stack((sequence == 0, sequence == 1, sequence >= 2))
        features[start:stop, :3] = indicators
        cumulative = np.vstack((np.zeros((1, 3)), np.cumsum(indicators, axis=0)))
        col = 3
        for seconds in WINDOWS:
            radius = max(1, round(seconds * fps))
            for low, high in ((np.maximum(0, index - radius), index + 1),
                              (index, np.minimum(n, index + radius + 1))):
                features[start:stop, col:col + 3] = (cumulative[high] - cumulative[low]) / (high - low)[:, None]
                col += 3
        canonical = np.minimum(sequence, 2)
        bounds = np.r_[0, np.flatnonzero(canonical[1:] != canonical[:-1]) + 1, n]
        for a, b in zip(bounds[:-1], bounds[1:]):
            features[start + a:start + b, lookup['run_before']] = _duration(np.arange(1, b - a + 1), fps)
            features[start + a:start + b, lookup['run_after']] = _duration(np.arange(b - a, 0, -1), fps)
            features[start + a:start + b, lookup['run_length']] = _duration(b - a, fps)
        for name, mask in [('home', sequence == 0), ('away', sequence == 1), ('known', sequence < 2)]:
            previous = np.maximum.accumulate(np.where(mask, index, -10**9))
            following = np.minimum.accumulate(np.where(mask, index, 10**9)[::-1])[::-1]
            features[start:stop, lookup['since_' + name]] = _duration(index - previous, fps)
            features[start:stop, lookup['until_' + name]] = _duration(following - index, fps)
        known = sequence < 2
        previous = np.maximum.accumulate(np.where(known, index, -1))
        following = np.minimum.accumulate(np.where(known, index, n)[::-1])[::-1]
        prev_owner = np.where(previous >= 0, sequence[np.maximum(previous, 0)], 3)
        next_owner = np.where(following < n, sequence[np.minimum(following, n - 1)], 3)
        for side, code in [('home', 0), ('away', 1)]:
            features[start:stop, lookup['previous_' + side]] = prev_owner == code
            features[start:stop, lookup['next_' + side]] = next_owner == code
        features[start:stop, lookup['same_owner_gap']] = (prev_owner == next_owner) & (prev_owner < 2) & ~known
    return features, names
