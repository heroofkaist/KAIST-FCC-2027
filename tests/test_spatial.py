import unittest
import numpy as np
from solution.io import Tracking

try:
    from solution.spatial import spatial_features
    from solution.features import swap_features
    AVAILABLE = True
except ImportError:
    AVAILABLE = False


class SpatialTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(AVAILABLE, 'invariant spatial features are missing')

    def fixture(self):
        ball = np.array([[50., 30], [51, 30], [52, 30], [10, 10]])
        players = np.array([[[49., 30], [55, 30], [70, 20]],
                            [[50., 30], [56, 30], [71, 20]],
                            [[51., 30], [57, 30], [72, 20]],
                            [[9., 10], [15, 10], [30, 20]]])
        return Tracking(ball, players, np.array([0, 0, 1]),
                        np.full((4, 3), np.nan), ('Home', 'Away'), (100., 60.))

    def test_slot_permutation_and_team_exchange(self):
        t = self.fixture(); blocks = [(0, 3), (3, 4)]
        a, names = spatial_features(t, blocks, 5)
        order = [2, 0, 1]
        permuted = Tracking(t.ball, t.players[:, order], t.team[order], t.speed[:, order], t.team_names, t.pitch)
        b, _ = spatial_features(permuted, blocks, 5)
        np.testing.assert_array_equal(a, b)
        exchanged = Tracking(t.ball, t.players, 1 - t.team, t.speed, t.team_names[::-1], t.pitch)
        c, _ = spatial_features(exchanged, blocks, 5)
        np.testing.assert_array_equal(swap_features(a, names), c)
        self.assertLess(a[0, names.index('home_nearest_1')], a[0, names.index('away_nearest_1')])

    def test_missing_coordinates_and_block_velocity(self):
        t = self.fixture(); a, names = spatial_features(t, [(0, 3), (3, 4)], 5)
        self.assertGreater(a[1, names.index('ball_speed')], 0)
        self.assertEqual(a[3, names.index('ball_speed')], 0)
        t.ball[:] = np.nan; t.players[:] = np.nan
        b, names = spatial_features(t, [(0, 4)], 5)
        self.assertTrue(np.isfinite(b).all())
        self.assertEqual(b[:, names.index('ball_valid')].sum(), 0)
        self.assertEqual(b[:, names.index('home_visible_fraction')].sum(), 0)
        self.assertEqual(b[:, names.index('away_visible_fraction')].sum(), 0)
