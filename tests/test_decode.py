import itertools
import unittest
import numpy as np
from solution.config import Config
from solution.evidence import Evidence, tracking_evidence, bootstrap_evidence
try:
    from solution.decode import decode
    AVAILABLE=True
except ImportError:
    AVAILABLE=False

class DecodeTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(AVAILABLE,'duration-aware decoder is missing')
        self.c=Config()

    def test_missing_observations_and_isolated_touch_do_not_split_possession(self):
        x=np.zeros((200,3));x[:,0]=1
        x[50:100]=0
        x[130]=[0,2,0]
        labels=decode(Evidence(x,np.max(x,axis=1)>0),[(0,200)],self.c)
        self.assertTrue(np.all(labels=='H'))

    def test_strong_short_possession_not_forbidden(self):
        x=np.zeros((200,3));x[:,0]=1
        x[80:90]=[0,20,0]
        labels=decode(Evidence(x,np.ones(200,bool)),[(0,200)],self.c)
        self.assertTrue(np.all(labels[80:90]=='A'))
        self.assertEqual(labels[79],'H');self.assertEqual(labels[90],'H')

    def test_dead_interval_and_independent_halves(self):
        x=np.zeros((150,3));x[:50,0]=2;x[20:35]=[0,0,4];x[100:,1]=2
        labels=decode(Evidence(x,np.max(x,axis=1)>0),[(0,50),(100,150)],self.c)
        self.assertEqual(labels[25],'D');self.assertEqual(labels[40],'H')
        self.assertTrue(np.all(labels[50:100]=='X'))
        self.assertTrue(np.all(labels[100:]=='A'))

    def test_no_observations_and_no_team_evidence_abstain(self):
        x=np.zeros((100,3))
        labels=decode(Evidence(x,np.zeros(100,bool)),[(0,100)],self.c)
        self.assertTrue(np.all(labels=='X'))
        x[:,:2]=1
        labels=decode(Evidence(x,np.ones(100,bool)),[(0,100)],self.c)
        self.assertTrue(np.all(labels=='X'))

    def test_rejects_invalid_shapes_nonfinite_and_overlapping_blocks(self):
        for e,blocks in [(Evidence(np.zeros((5,2)),np.zeros(5,bool)),[(0,5)]),
                         (Evidence(np.full((5,3),np.nan),np.ones(5,bool)),[(0,5)]),
                         (Evidence(np.zeros((5,3)),np.ones(5,bool)),[(0,4),(3,5)])]:
            with self.assertRaises(ValueError):decode(e,blocks,self.c)

    def test_matches_exhaustive_path_objective(self):
        c=Config(transition_cost=0.7,short_segment_cost=0.8,mature_frames=3)
        x=np.array([[2.,0,0],[0,1,0],[0,3,0],[0,0,2],[0,0,1],[0,0,4]])
        def objective(path):
            value=sum(x[i,s] for i,s in enumerate(path))
            segments=[];length=1
            for previous,current in zip(path,path[1:]):
                if previous==current:length+=1
                else:segments.append(length);length=1
            segments.append(length)
            return value-0.7*(len(segments)-1)-sum(0.8*max(0,3-n)/3 for n in segments)
        expected=max(objective(p) for p in itertools.product(range(3),repeat=len(x)))
        labels=decode(Evidence(x,np.ones(len(x),bool)),[(0,len(x))],c)
        actual=objective([{'H':0,'A':1,'D':2}[s] for s in labels])
        self.assertAlmostEqual(actual,expected)

    def test_age_one_reduces_to_standard_switch_penalty(self):
        x=np.array([[5.,0,0],[0,5,0],[0,0,5]])
        labels=decode(Evidence(x,np.ones(3,bool)),[(0,3)],Config(mature_frames=1,transition_cost=1))
        self.assertEqual(labels.tolist(),['H','A','D'])

    def test_tied_live_segment_after_stoppage_abstains_without_home_bias(self):
        x=np.zeros((120,3));x[:40,1]=5;x[40:80,2]=5;x[80:,:2]=1
        original=decode(Evidence(x,np.ones(120,bool)),[(0,120)],self.c)
        swapped=decode(Evidence(x[:,[1,0,2]],np.ones(120,bool)),[(0,120)],self.c)
        self.assertTrue(np.all(original[:40]=='A'))
        self.assertTrue(np.all(original[40:80]=='D'))
        self.assertTrue(np.all(original[80:]=='X'))
        translated=np.array([{'H':'A','A':'H','D':'D','X':'X'}[s] for s in original])
        np.testing.assert_array_equal(swapped,translated)

    def test_equal_alternating_team_votes_abstain(self):
        for order in ('HA','AH'):
            labels=np.array(list(order*50))
            e=bootstrap_evidence(labels,[(0,100)],self.c)
            actual=decode(e,[(0,100)],self.c)
            self.assertTrue(np.all(actual=='X'))

    def test_ownership_abstentions_match_all_optimal_paths(self):
        rng=np.random.default_rng(9)
        config=Config(mature_frames=2,transition_cost=1,short_segment_cost=1)
        for _ in range(20):
            x=rng.integers(-1,3,size=(5,3)).astype(float)
            paths=list(itertools.product(range(3),repeat=5))
            values=[]
            for path in paths:
                lengths=[];length=1
                for a,b in zip(path,path[1:]):
                    if a==b:length+=1
                    else:lengths.append(length);length=1
                lengths.append(length)
                values.append(sum(x[t,s] for t,s in enumerate(path))-(len(lengths)-1)-sum(max(0,2-n)/2 for n in lengths))
            best=max(values);optimal=[p for p,v in zip(paths,values) if v==best]
            result=decode(Evidence(x,np.ones(5,bool)),[(0,5)],config)
            for t,label in enumerate(result):
                possible={p[t] for p in optimal}
                if {0,1}.issubset(possible) and label!='D':self.assertEqual(label,'X')
                elif label!='X':self.assertIn({'H':0,'A':1,'D':2}[label],possible)
