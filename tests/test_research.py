import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from experiments.protocol import development_folds

try:
    from experiments.research import run_cv, summarize, build_row
    AVAILABLE = True
except ImportError:
    AVAILABLE = False


class ResearchTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(AVAILABLE, 'grouped research runner is missing')

    def write_match(self, root, game):
        n = 125; idx = np.arange(0, n, 5); m = len(idx)
        labels = np.r_[np.zeros(40, int), np.ones(40, int), np.full(45, 2)]
        provenance = {'schema_version': 2, 'match_id': game, 'source': 'sportec',
                      'native_fps': 25, 'model_fps': 5, 'pitch': [105, 68], 'weak_labels': False}
        folder = root / 'sportec'; folder.mkdir(exist_ok=True)
        np.savez(folder / (game + '.npz'), labels=labels, nearest=labels, observed=labels,
                 ball_detected=labels < 2, indices=idx, blocks=np.array([[0, n]]),
                 sample_blocks=np.array([[0, m]]), ball=np.full((m, 2), 20.),
                 players=np.full((m, 2, 2), 21.), player_detected=np.ones((m, 2), bool),
                 team=np.array([0, 1]), height=np.zeros(m), provenance=np.array(json.dumps(provenance)))

    def test_cv_fits_only_training_groups_and_partial_selection_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); corpus = root / 'corpus'; corpus.mkdir()
            train, valid = development_folds()[0]
            for game in train + valid:
                self.write_match(corpus, game)
            out = root / 'cv'
            candidate = {'tiny': {'kind': 'lightgbm', 'configuration': {'rounds': 3, 'min_leaf': 1}}}
            rows = run_cv(corpus, out, candidate, folds=[(train, valid)])
            self.assertEqual(len(rows), 6)
            manifest = json.loads((out / 'models/tiny/fold-0/model.json').read_text())
            self.assertEqual(set(manifest['training_matches']), set(train))
            self.assertFalse(set(manifest['training_matches']) & set(valid))
            with self.assertRaises(ValueError):
                summarize(rows)
            with self.assertRaises(FileExistsError):
                run_cv(corpus, out, candidate, folds=[(train, valid)])

    def test_summary_balances_matches_and_scenarios_and_is_deterministic(self):
        rows = []
        for game in ['J03WMX', 'J03WN1', 'J03WOH', 'J03WOY', 'J03WPY', 'J03WQQ']:
            for scenario in ['clean', 'masked', 'long']:
                rows.append({'candidate': 'a', 'match_id': game, 'scenario': scenario,
                             'scores': {'spellF1': .5, 'macroF1': .6, 'dead_f1': .7}})
        a = summarize(rows); b = summarize(list(reversed(rows)))
        self.assertEqual(a, b)
        self.assertAlmostEqual(a[0]['mean_spellF1'], .5)
        with self.assertRaises(ValueError):
            summarize(rows + [rows[0]])
