import tempfile
from pathlib import Path
import unittest
import numpy as np
import test_sportec
try:
    from experiments.prepare import prepare_files
    AVAILABLE=True
except ImportError:AVAILABLE=False

class PreparationTests(unittest.TestCase):
    def test_cache_records_provenance_and_separates_observations_from_truth(self):
        self.assertTrue(AVAILABLE,'reproducible external preparation missing')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);meta,raw=test_sportec.SportecTests().fixture(root)
            report=prepare_files(meta,raw,root/'cache.npz',seed=11)
            self.assertEqual(len(report['tracking_sha256']),64)
            with np.load(root/'cache.npz',allow_pickle=False) as cache:
                self.assertEqual(cache['labels'].tolist(),['H','D','A'])
                self.assertEqual(cache['clean'].tolist(),['H','A','D'])
            with self.assertRaises(FileExistsError):prepare_files(meta,raw,root/'cache.npz',seed=11)
