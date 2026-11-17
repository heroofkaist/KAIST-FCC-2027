from pathlib import Path
import tempfile
import unittest
import numpy as np
from evaluation import possession_metrics as pm

try:
    from experiments.protocol import development_folds, validate_split, load_match, observe, create_run
    from experiments.diagnostics import score_match, paired_match_interval
    AVAILABLE = True
except ImportError:
    AVAILABLE = False


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(AVAILABLE, 'guarded research protocol is missing')

    def test_group_folds_and_holdout_guards(self):
        folds = development_folds()
        validation = []
        for train, valid in folds:
            validate_split(train, valid)
            validation.extend(valid)
            self.assertFalse(set(train) & set(valid))
        self.assertEqual(len(set(validation)), 6)
        with self.assertRaises(ValueError):
            validate_split(['J03WOH'], ['J03WOH'])
        with self.assertRaises(ValueError):
            validate_split(['J03WR9'], ['J03WOH'])
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                load_match(Path(tmp), 'sportec', 'J03WR9')
            with self.assertRaises(ValueError):
                load_match(Path(tmp), 'sportec', '../../evaluation/gt/target')

    def fixture(self):
        n = 120
        return {'indices': np.arange(n), 'sample_blocks': np.array([[0, 60], [60, 120]]),
                'labels': np.r_[np.zeros(60, int), np.ones(60, int)],
                'nearest': np.r_[np.zeros(60, int), np.ones(60, int)],
                'observed': np.zeros(n, int), 'ball_detected': np.ones(n, bool),
                'ball': np.full((n, 2), 20.), 'height': np.zeros(n),
                'players': np.full((n, 2, 2), 21.), 'player_detected': np.ones((n, 2), bool),
                'team': np.array([0, 1]), 'provenance': {'source': 'sportec', 'match_id': 'J03WOH', 'pitch': [105, 68]}}

    def test_corruption_depends_only_on_observations_and_seed(self):
        a = self.fixture(); b = dict(a); b['labels'] = 2 - a['labels']
        for scenario in ['clean', 'masked', 'long', 'native']:
            x = observe(a, scenario, seed=9); y = observe(b, scenario, seed=9)
            for key in ['codes', 'ball', 'players', 'height']:
                np.testing.assert_array_equal(x[key], y[key])
        x = observe(a, 'long', seed=9)
        self.assertTrue((x['codes'] == 2).any())
        self.assertTrue(np.isnan(x['ball']).any())
        self.assertEqual(x['blocks'], [(0, 60), (60, 120)])

    def test_official_spell_metric_and_error_diagnostics(self):
        gt = np.r_[np.zeros(100, int), np.full(25, 2), np.ones(100, int)]
        pred = gt.copy(); pred[40:50] = 1
        blocks = [(0, len(gt))]
        expected = pm.spell_scores(gt, pred, blocks)
        result = score_match(gt, pred, blocks, 25)
        self.assertEqual(result['spellF1'], expected['mean_f1'])
        self.assertEqual(tuple(float(k) for k in result['spell_by_iou']), pm.SPELL_IOUS)
        self.assertEqual(score_match(gt, gt, blocks, 25)['spellF1'], 1)
        self.assertGreater(result['n_pred_spells'], result['n_gt_spells'])
        self.assertGreater(result['transitions_0.5s']['n_spurious'], 0)
        self.assertEqual(result['dead_f1'], 1)

    def test_paired_match_bootstrap_and_output_reuse(self):
        a = paired_match_interval([.5, .6, .7], [.4, .5, .6], seed=11)
        b = paired_match_interval([.5, .6, .7], [.4, .5, .6], seed=11)
        self.assertEqual(a, b)
        self.assertAlmostEqual(a['mean_difference'], .1)
        self.assertGreater(a['low'], 0)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'run'
            create_run(out, {'seed': 1})
            with self.assertRaises(FileExistsError):
                create_run(out, {'seed': 2})
