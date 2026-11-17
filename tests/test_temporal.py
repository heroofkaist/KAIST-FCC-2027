import tempfile
from pathlib import Path
import unittest
import numpy as np
try:
    from solution.temporal import temporal_features, fit, predict_scores, save_model, load_model, swap_features
    AVAILABLE=True
except ImportError:AVAILABLE=False

class TemporalTests(unittest.TestCase):
    def setUp(self):self.assertTrue(AVAILABLE,'external temporal learner is missing')

    def test_features_do_not_cross_halves(self):
        x=temporal_features(np.array(list('HHHHHAAAAA')),[(0,5),(5,10)],25)
        first=temporal_features(np.array(list('HHHHH')),[(0,5)],25)
        np.testing.assert_array_equal(x[:5],first)
        self.assertTrue(np.isfinite(x).all())

    def test_swap_equivariance(self):
        labels=np.array(list('HHHDDDAAXAAHH'))
        swapped=np.array([{'H':'A','A':'H','D':'D','X':'X'}[s] for s in labels])
        x=temporal_features(labels,[(0,len(labels))],25)
        y=temporal_features(swapped,[(0,len(labels))],25)
        np.testing.assert_allclose(swap_features(x),y)
        target=np.array([0]*6+[1]*7)
        model=fit(x,target,steps=30)
        np.testing.assert_allclose(predict_scores(y,model),predict_scores(x,model)[:,[1,0,2]],atol=1e-10)

    def test_external_supervision_can_learn_and_round_trip(self):
        labels=np.array(list('H'*80+'D'*80+'A'*80))
        x=temporal_features(labels,[(0,240)],25)
        truth=np.repeat([0,2,1],80)
        model=fit(x,truth,steps=150)
        scores=predict_scores(x,model)
        self.assertGreater(np.mean(scores.argmax(axis=1)==truth),0.9)
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'model.json';save_model(p,model)
            np.testing.assert_array_equal(scores,predict_scores(x,load_model(p)))

    def test_rejects_invalid_labels_and_model(self):
        with self.assertRaises(ValueError):temporal_features(np.array(['bad']),[(0,1)],25)
        x=temporal_features(np.array(['H','A']),[(0,2)],25)
        with self.assertRaises(ValueError):fit(x,np.array([0,3]))
        with self.assertRaises(ValueError):predict_scores(x,{'weights':[[float('nan')]]})
