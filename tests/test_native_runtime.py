import subprocess
import sys
import textwrap
import unittest


class NativeRuntimeTests(unittest.TestCase):
    def test_torch_and_tree_families_coexist_without_shared_native_runtime(self):
        script = textwrap.dedent('''
            from pathlib import Path
            import tempfile
            import numpy as np
            import torch
            from experiments.learners import fit_learner
            from solution.learned import load_learner, predict_probabilities
            y = np.tile(np.arange(3), 10)
            x = np.eye(3, dtype=np.float32)[y]
            row = {'x': x, 'y': y, 'blocks': [(0, len(y))],
                   'feature_names': ('home_now', 'away_now', 'missing_now'),
                   'weight': 1., 'match_id': 'synthetic'}
            with tempfile.TemporaryDirectory() as tmp:
                for kind in ['lightgbm', 'catboost', 'xgboost']:
                    out = Path(tmp) / kind
                    fit_learner(kind, {'rounds': 5, 'min_leaf': 1}, [row], out)
                    probability = predict_probabilities(load_learner(out), x, row['blocks'])
                    assert np.isfinite(probability).all()
                assert torch.ones(3).sum().item() == 3
            print('isolated-native-families-ok')
        ''')
        result = subprocess.run([sys.executable, '-X', 'faulthandler', '-c', script],
                                text=True, capture_output=True, timeout=90)
        self.assertEqual(result.returncode, 0, result.stderr[-2000:])
        self.assertIn('isolated-native-families-ok', result.stdout)
