"""Fresh-process numerical backends, avoiding incompatible OpenMP runtimes.

Only NumPy/JSON cross the process boundary. No pickle, remote service, modified
system libraries, or duplicate-OpenMP suppression is used.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _invoke(request, directory):
    request_path = directory / 'request.json'
    request_path.write_text(json.dumps(request, allow_nan=False))
    result = subprocess.run([sys.executable, '-m', 'experiments.native_worker', str(request_path)], cwd=ROOT)
    response_path = directory / 'response.json'
    if not response_path.exists():
        raise RuntimeError(f'native backend worker exited {result.returncode} without a response')
    response = json.loads(response_path.read_text())
    if result.returncode:
        errors = {'ValueError': ValueError, 'FileExistsError': FileExistsError, 'FileNotFoundError': FileNotFoundError}
        raise errors.get(response.get('type'), RuntimeError)(response.get('error', 'native backend failed'))
    return response


def fit_isolated(kind, config, training, out, seed):
    with tempfile.TemporaryDirectory(prefix='fcc-native-fit-') as tmp:
        directory = Path(tmp); arrays = {}; rows = []
        for i, row in enumerate(training):
            arrays[f'x_{i}'] = np.asarray(row['x']); arrays[f'y_{i}'] = np.asarray(row['y'])
            rows.append({key: row[key] for key in ('blocks', 'feature_names', 'match_id')})
            rows[-1]['weight'] = float(row.get('weight', 1))
        np.savez(directory / 'inputs.npz', **arrays)
        return _invoke({'action': 'fit', 'kind': kind, 'configuration': config, 'training': rows,
                        'out': str(Path(out).resolve()), 'seed': seed}, directory)['manifest']


def predict_isolated(path, x, blocks):
    with tempfile.TemporaryDirectory(prefix='fcc-native-predict-') as tmp:
        directory = Path(tmp); np.save(directory / 'inputs.npy', x, allow_pickle=False)
        _invoke({'action': 'predict', 'model': str(Path(path).resolve()), 'blocks': blocks}, directory)
        return np.load(directory / 'probability.npy', allow_pickle=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('request', type=Path)
    args = parser.parse_args(); directory = args.request.parent
    try:
        request = json.loads(args.request.read_text())
        if request['action'] == 'fit':
            from .learners import fit_learner
            with np.load(directory / 'inputs.npz', allow_pickle=False) as archive:
                training = [{**row, 'x': archive[f'x_{i}'].copy(), 'y': archive[f'y_{i}'].copy()}
                            for i, row in enumerate(request['training'])]
            manifest = fit_learner(request['kind'], request['configuration'], training,
                                   request['out'], request['seed'], _native=True)
            response = {'manifest': manifest}
        elif request['action'] == 'predict':
            from solution.learned import load_learner, predict_probabilities
            model = load_learner(request['model'], _native=True)
            x = np.load(directory / 'inputs.npy', allow_pickle=False)
            p = predict_probabilities(model, x, request['blocks'])
            np.save(directory / 'probability.npy', p, allow_pickle=False)
            response = {'rows': len(p)}
        else:
            raise ValueError('unknown native worker action')
    except Exception as error:
        (directory / 'response.json').write_text(json.dumps({'type': type(error).__name__, 'error': str(error)}))
        raise SystemExit(1)
    (directory / 'response.json').write_text(json.dumps(response, allow_nan=False))


if __name__ == '__main__':
    main()
