"""Compiled equivalent of the exact age-expanded decoder for bounded searches."""
import numpy as np
from numba import njit
from .evidence import validate_blocks


@njit(cache=True)
def _path(scores, k, transition, short_cost):
    n = len(scores); penalty = short_cost * (k - np.arange(1, k + 1)) / k
    forward = np.full((n, 3, k), -np.inf); back = np.zeros((n, 3, k), np.int16)
    for state in range(3):
        forward[0, state, 0] = scores[0, state]
    for t in range(1, n):
        for state in range(3):
            best = -np.inf; predecessor = 0
            for other in range(3):
                if other == state:
                    continue
                for age in range(k):
                    value = forward[t - 1, other, age] - penalty[age]
                    if value > best:
                        best = value; predecessor = other * k + age
            forward[t, state, 0] = best - transition
            back[t, state, 0] = predecessor
            for age in range(1, k):
                forward[t, state, age] = forward[t - 1, state, age - 1]
                back[t, state, age] = state * k + age - 1
            if forward[t - 1, state, k - 1] >= forward[t, state, k - 1]:
                forward[t, state, k - 1] = forward[t - 1, state, k - 1]
                back[t, state, k - 1] = state * k + k - 1
            for age in range(k):
                forward[t, state, age] += scores[t, state]
    optimum = -np.inf; state = 0
    for other in range(3):
        for age in range(k):
            value = forward[n - 1, other, age] - penalty[age]
            if value > optimum:
                optimum = value; state = other * k + age
    labels = np.empty(n, np.int8)
    for t in range(n - 1, -1, -1):
        labels[t] = state // k
        state = int(back[t, state // k, state % k])
    suffix = np.empty((3, k))
    for other in range(3):
        suffix[other] = -penalty
    for t in range(n - 1, -1, -1):
        home = -np.inf; away = -np.inf
        for age in range(k):
            home = max(home, forward[t, 0, age] + suffix[0, age])
            away = max(away, forward[t, 1, age] + suffix[1, age])
        if labels[t] != 2 and abs(home - optimum) < 1e-8 and abs(away - optimum) < 1e-8:
            labels[t] = 3
        if t:
            previous = np.empty((3, k))
            for current in range(3):
                switch = -np.inf
                for other in range(3):
                    if other != current:
                        switch = max(switch, scores[t, other] + suffix[other, 0])
                switch -= transition
                for age in range(k):
                    stay = scores[t, current] + suffix[current, min(age + 1, k - 1)]
                    previous[current, age] = max(stay, switch - penalty[age])
            suffix = previous
    return labels


def fast_decode(evidence, blocks, config):
    scores = np.asarray(evidence.scores, dtype=np.float64)
    informative = np.asarray(evidence.informative)
    if scores.ndim != 2 or scores.shape[1] != 3 or informative.shape != (len(scores),) or not np.isfinite(scores).all():
        raise ValueError('invalid decoder evidence')
    validate_blocks(blocks, len(scores)); result = np.full(len(scores), 'X', dtype='<U1')
    for start, stop in blocks:
        if informative[start:stop].any():
            codes = _path(scores[start:stop], config.mature_frames, config.transition_cost, config.short_segment_cost)
            result[start:stop] = np.array(['H', 'A', 'D', 'X'])[codes]
    return result
