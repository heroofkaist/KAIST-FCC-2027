"""Official possession metrics with native-cadence error and confidence reports."""
import numpy as np
from evaluation import possession_metrics as pm
from solution.evidence import validate_blocks


def score_match(truth, prediction, blocks, fps, probability=None):
    gt = np.asarray(truth); pred = np.asarray(prediction)
    if gt.ndim != 1 or pred.shape != gt.shape or not np.isin(gt, [0, 1, 2, 3]).all() or not np.isin(pred, [0, 1, 2, 3]).all():
        raise ValueError('invalid scored labels')
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError('invalid score cadence')
    validate_blocks(blocks, len(gt))
    runs = []
    for start, stop in blocks:
        runs.extend((start + a, start + b) for a, b in pm.valid_runs(gt[start:stop]))
    valid = np.zeros(len(gt), bool)
    for start, stop in runs:
        valid[start:stop] = True
    cm = pm.scored_confusion(pm.confusion(gt[valid], pred[valid]))
    filled = pm.fill_abstentions(pred, runs)
    spells = pm.spell_scores(gt, filled, runs)
    result = {'spellF1': spells['mean_f1'], 'macroF1': pm.macro_f1(cm),
              'dead_f1': pm.per_class_prf(cm, 2)[2], 'alive_f1': pm.alive_dead_f1(cm),
              'per_class_f1': {name: pm.per_class_prf(cm, k)[2] for k, name in enumerate(('H', 'A', 'D'))},
              'frames': int(valid.sum()), 'n_gt_spells': spells['n_gt'], 'n_pred_spells': spells['n_pred'],
              'spell_by_iou': {str(k): v for k, v in spells['per_iou'].items()},
              'confusion': cm.tolist()}
    gt_trans = pm.transitions(gt, runs); pred_trans = pm.transitions(filled, runs)
    for seconds in (.5, 1.):
        report = pm.transition_scores(gt_trans, pred_trans, round(seconds * fps))
        result[f'transitions_{seconds:g}s'] = {key: value for key, value in report.items() if key not in ('pairs', 'missed', 'spurious')}
        latency = report['mean_abs_latency']
        result[f'transitions_{seconds:g}s']['mean_abs_latency_seconds'] = latency / fps if latency is not None else None
    if probability is not None:
        p = np.asarray(probability)
        if p.shape != (len(gt), 3) or not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(axis=1), 1):
            raise ValueError('invalid confidence probabilities')
        if valid.any():
            p = p[valid]; y = gt[valid].astype(int)
            target = np.eye(3)[y]
            confidence = p.max(axis=1); correct = p.argmax(axis=1) == y
            ece = 0.
            for low in np.arange(10) / 10:
                mask = (confidence > low) & (confidence <= low + .1)
                if mask.any():
                    ece += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
            result['confidence'] = {'brier': float(np.mean(np.sum((p - target)**2, axis=1))),
                                    'ece_10_bins': float(ece), 'mean_max_probability': float(confidence.mean())}
    return result


def paired_match_interval(candidate, reference, seed=20261109, draws=10000):
    a, b = np.asarray(candidate, dtype=float), np.asarray(reference, dtype=float)
    if a.ndim != 1 or a.shape != b.shape or not len(a) or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('paired finite match scores are required')
    if not isinstance(draws, int) or draws < 1:
        raise ValueError('bootstrap draws must be positive')
    difference = a - b
    rng = np.random.default_rng(seed)
    means = difference[rng.integers(0, len(a), (draws, len(a)))].mean(axis=1)
    low, high = np.percentile(means, [5, 95])
    return {'n_matches': len(a), 'mean_difference': float(difference.mean()),
            'low': float(low), 'high': float(high), 'percentiles': [5, 95],
            'seed': seed, 'draws': draws,
            'limitation': 'Match bootstrap is descriptive with few matches; a one-match interval is not independent uncertainty evidence.'}
