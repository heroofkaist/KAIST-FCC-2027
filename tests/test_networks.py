from pathlib import Path
import tempfile
import unittest
import numpy as np

try:
    import torch
    from experiments.networks import build_network, network_probabilities
    from experiments.learners import fit_learner
    from solution.learned import load_learner, predict_probabilities
    AVAILABLE = True
except ImportError:
    AVAILABLE = False


class NetworkTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(AVAILABLE, 'compact temporal networks are missing')

    def test_tcn_chunking_and_period_isolation(self):
        torch.manual_seed(3)
        model = build_network('tcn', 3, {'channels': 8, 'layers': 4})
        x = np.random.default_rng(1).random((317, 3)).astype(np.float32)
        blocks = [(0, 171), (171, 317)]
        a = network_probabilities(model, x, blocks, chunk_size=0)
        b = network_probabilities(model, x, blocks, chunk_size=53)
        np.testing.assert_allclose(a, b, atol=1e-7)
        separate = network_probabilities(model, x[:171], [(0, 171)], chunk_size=53)
        np.testing.assert_array_equal(b[:171], separate)

    def test_recurrent_and_attention_networks_are_finite(self):
        x = np.random.default_rng(4).random((53, 3)).astype(np.float32)
        for kind in ['bilstm', 'transformer']:
            model = build_network(kind, 3, {'channels': 8, 'layers': 1, 'heads': 2})
            p = network_probabilities(model, x, [(0, 53)])
            self.assertEqual(p.shape, (53, 3))
            self.assertTrue(np.isfinite(p).all())
            np.testing.assert_allclose(p.sum(axis=1), 1, atol=1e-6)

    def test_tcn_training_and_safe_weight_round_trip(self):
        y = np.tile(np.repeat(np.arange(3), 20), 3)
        x = np.eye(3, dtype=np.float32)[y]
        data = {'x': x, 'y': y, 'blocks': [(0, len(y))],
                'feature_names': ('home_now', 'away_now', 'missing_now'),
                'weight': 1., 'match_id': 'synthetic'}
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'tcn'
            fit_learner('tcn', {'channels': 8, 'layers': 2, 'epochs': 5,
                              'steps_per_epoch': 8, 'batch_size': 4, 'window': 32,
                              'context': 8, 'learning_rate': .01}, [data], out, seed=4)
            a = predict_probabilities(load_learner(out), x, data['blocks'])
            b = predict_probabilities(load_learner(out), x, data['blocks'])
            np.testing.assert_array_equal(a, b)
            self.assertGreater(np.mean(a.argmax(axis=1) == y), .9)
