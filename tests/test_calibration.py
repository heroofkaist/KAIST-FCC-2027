import tempfile
from pathlib import Path
import unittest
import numpy as np
from solution.temporal import fit,temporal_features,save_model
from solution.evidence import Evidence
from solution.config import Config
try:
    from solution.temporal import calibrated_evidence
    from experiments.calibrate import choose_candidate
    AVAILABLE=True
except ImportError:AVAILABLE=False

class CalibrationTests(unittest.TestCase):
    def test_external_selection_uses_fixed_equal_scenario_metric(self):
        self.assertTrue(AVAILABLE,'calibration selection missing')
        candidates=[{'name':'a','scores':{'clean':{'spellF1':0.8},'masked':{'spellF1':0.2}}},
                    {'name':'b','scores':{'clean':{'spellF1':0.6},'masked':{'spellF1':0.6}}}]
        self.assertEqual(choose_candidate(candidates)['name'],'b')

    def test_neutral_observations_remain_uninformative(self):
        self.assertTrue(AVAILABLE,'calibrated evidence missing')
        labels=np.array(list('H'*20+'A'*20+'D'*20))
        x=temporal_features(labels,[(0,60)],25)
        model=fit(x,np.repeat([0,1,2],20),steps=10)
        e=calibrated_evidence(np.full(60,'X'),[(0,60)],Config(),model)
        self.assertFalse(e.informative.any())
        self.assertTrue(np.isfinite(e.scores).all())
