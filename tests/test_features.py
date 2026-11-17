import unittest
import numpy as np

try:
    from solution.features import categorical_features, swap_features
    AVAILABLE = True
except ImportError:
    AVAILABLE = False


class FeatureTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(AVAILABLE, 'multiscale feature layer is missing')

    def test_short_blocks_and_gaps_are_isolated(self):
        codes = np.array([0, 2, 0, 3, 1, 1, 2])
        full, names = categorical_features(codes, [(0, 3), (4, 7)], 5)
        a, _ = categorical_features(codes[:3], [(0, 3)], 5)
        b, _ = categorical_features(codes[4:], [(0, 3)], 5)
        np.testing.assert_array_equal(full[:3], a)
        np.testing.assert_array_equal(full[4:], b)
        np.testing.assert_array_equal(full[3], 0)
        self.assertEqual(full.dtype, np.float32)
        self.assertTrue(np.isfinite(full).all())
        self.assertEqual(len(names), full.shape[1])

    def test_team_swap_and_context_duration(self):
        codes = np.array([0, 0, 2, 2, 2, 1, 0, 3])
        a, names = categorical_features(codes, [(0, 8)], 5)
        b, names_b = categorical_features(np.array([1, 1, 2, 2, 2, 0, 1, 3]), [(0, 8)], 5)
        self.assertEqual(names, names_b)
        np.testing.assert_array_equal(swap_features(a, names), b)
        self.assertLess(a[2, names.index('since_home')], a[4, names.index('since_home')])
        self.assertGreater(a[3, names.index('run_length')], 0)
        self.assertEqual(a[2, names.index('home_past_0.2s')], .5)
        self.assertEqual(a[2, names.index('away_future_0.2s')], 0)

    def test_empty_unknown_and_invalid_inputs(self):
        a, _ = categorical_features(np.full(20, 3), [(0, 20)], 5)
        self.assertTrue(np.isfinite(a).all())
        with self.assertRaises(ValueError):
            categorical_features(np.array([4]), [(0, 1)], 5)
        with self.assertRaises(ValueError):
            categorical_features(np.array([0]), [(0, 1)], 0)
