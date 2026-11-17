import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

try:
    from experiments.learners import fit_learner
    from solution.learned import load_learner, predict_probabilities
    AVAILABLE = True
except ImportError:
    AVAILABLE = False


class LearnedTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(AVAILABLE, 'interchangeable external learners are missing')

    def fixture(self):
        y = np.tile(np.arange(3), 40)
        x = np.eye(3, dtype=np.float32)[y]
        return {'x': x, 'y': y, 'blocks': [(0, len(y))],
                'feature_names': ('home_now', 'away_now', 'missing_now'),
                'weight': 1., 'match_id': 'synthetic'}

    def test_real_boosters_fit_round_trip_and_exact_team_symmetry(self):
        data = self.fixture()
        settings = {'lightgbm': {'rounds': 25, 'leaves': 7, 'min_leaf': 1},
                    'catboost': {'rounds': 25, 'depth': 3},
                    'xgboost': {'rounds': 25, 'depth': 3, 'min_child_weight': 1}}
        with tempfile.TemporaryDirectory() as tmp:
            for kind, config in settings.items():
                with self.subTest(kind=kind):
                    out = Path(tmp) / kind
                    fit_learner(kind, config, [data], out, seed=1)
                    loaded = load_learner(out)
                    a = predict_probabilities(loaded, data['x'], data['blocks'])
                    b = predict_probabilities(load_learner(out), data['x'], data['blocks'])
                    np.testing.assert_array_equal(a, b)
                    self.assertGreater(np.mean(a.argmax(axis=1) == data['y']), .95)
                    swapped = predict_probabilities(loaded, data['x'][:, [1, 0, 2]], data['blocks'])
                    np.testing.assert_array_equal(swapped, a[:, [1, 0, 2]])
                    with self.assertRaises(FileExistsError):
                        fit_learner(kind, config, [data], out)

    def test_invalid_artifacts_and_nonfinite_inputs_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'model'; data = self.fixture()
            fit_learner('lightgbm', {'rounds': 5, 'min_leaf': 1}, [data], out)
            model = load_learner(out)
            bad = data['x'].copy(); bad[0, 0] = np.nan
            with self.assertRaises(ValueError):
                predict_probabilities(model, bad, data['blocks'])
            path = out / 'model.json'; manifest = json.loads(path.read_text())
            manifest['weights_file'] = '../escape.txt'; path.write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                load_learner(out)
