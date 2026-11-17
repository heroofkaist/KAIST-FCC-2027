import unittest
import numpy as np
try:
    from solution.io import Tracking, Tag
    from solution.config import Config
    from solution.evidence import tracking_evidence, bootstrap_evidence, add_tags, Evidence
    AVAILABLE = True
except ImportError:
    AVAILABLE = False

class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(AVAILABLE, 'solution evidence implementation is missing')
        self.c=Config()

    def fixture(self,n=100):
        players=np.tile(np.array([[20.,20.],[80.,20.]]),(n,1,1))
        return Tracking(np.tile([20.5,20.],(n,1)), players, np.array([0,1]), np.zeros((n,2)), ('Home','Away'), (105.3,68.))

    def test_control_and_stationary_live_ball(self):
        e=tracking_evidence(self.fixture(),[(0,100)],self.c)
        self.assertTrue(np.all(e.scores[:,0]>e.scores[:,1]))
        self.assertTrue(np.all(e.scores[:,0]>e.scores[:,2]))

    def test_missing_ball_is_neutral_not_dead(self):
        t=self.fixture();t.ball[:]=np.nan
        e=tracking_evidence(t,[(0,100)],self.c)
        self.assertTrue(np.all(e.scores==0))
        self.assertFalse(e.informative.any())

    def test_outside_field_gives_dead_evidence(self):
        t=self.fixture();t.ball[:]=[110,20]
        e=tracking_evidence(t,[(0,100)],self.c)
        self.assertTrue(np.all(e.scores[:,2]>e.scores[:,:2].max(axis=1)))

    def test_slot_permutation_is_invariant(self):
        t=self.fixture();a=tracking_evidence(t,[(0,100)],self.c)
        t.players=t.players[:,::-1];t.team=t.team[::-1];t.speed=t.speed[:,::-1]
        b=tracking_evidence(t,[(0,100)],self.c)
        np.testing.assert_array_equal(a.scores,b.scores)

    def test_short_occlusion_filled_but_never_across_blocks(self):
        t=self.fixture();t.ball[30:35]=np.nan
        e=tracking_evidence(t,[(0,100)],self.c)
        self.assertTrue(e.informative[30:35].all())
        t=self.fixture();t.ball[40:60]=np.nan
        e=tracking_evidence(t,[(0,50),(50,100)],self.c)
        self.assertFalse(e.informative[40:60].any())

    def test_tags_defense_na_conflict_and_duplicate_cap(self):
        base=Evidence(np.zeros((100,3)),np.zeros(100,dtype=bool))
        h=Tag(10,90,0,'attack');a=Tag(10,90,1,'attack')
        one=add_tags(base,[h],self.c)
        repeated=add_tags(base,[h]*30,self.c)
        np.testing.assert_allclose(one.scores,repeated.scores)
        conflict=add_tags(base,[h,a],self.c)
        np.testing.assert_allclose(conflict.scores[:,0],conflict.scores[:,1])
        defense=add_tags(base,[Tag(10,90,0,'defense')],self.c)
        self.assertGreater(defense.scores[50,1],defense.scores[50,0])
        neutral=add_tags(base,[Tag(10,90,-1,'restart')],self.c)
        self.assertTrue(np.all(neutral.scores==0))
        self.assertFalse(neutral.informative.any())
        np.testing.assert_array_equal(base.scores,0)

    def test_bootstrap_short_missing_gap_has_no_dead_reward(self):
        labels=np.array(list('H'*40+'D'*10+'H'*40))
        e=bootstrap_evidence(labels,[(0,90)],self.c)
        self.assertTrue(np.all(e.scores[40:50,2]==0))
        self.assertGreater(e.scores[10,0],e.scores[10,1])

    def test_all_x_bootstrap_neutral(self):
        e=bootstrap_evidence(np.full(100,'X'),[(0,100)],self.c)
        self.assertFalse(e.informative.any())
