from dataclasses import replace
import unittest
import numpy as np
from solution.config import Config
from solution.evidence import Evidence
from solution.decode import decode

try:
    from solution.fast_decode import fast_decode
    AVAILABLE = True
except ImportError:
    AVAILABLE = False


class FastDecodeTests(unittest.TestCase):
    def test_compiled_decoder_agrees_with_exact_reference_and_ties(self):
        self.assertTrue(AVAILABLE, 'compiled exact decoder is missing')
        rng = np.random.default_rng(7)
        for k in [1, 3, 5]:
            config = replace(Config(), mature_frames=k, transition_cost=1., short_segment_cost=2.)
            for scores in [np.zeros((30, 3)), rng.normal(size=(30, 3)), rng.integers(-2, 3, (30, 3))]:
                evidence = Evidence(scores, np.ones(30, bool)); blocks = [(0, 13), (16, 30)]
                np.testing.assert_array_equal(fast_decode(evidence, blocks, config), decode(evidence, blocks, config))
        e = Evidence(np.zeros((30, 3)), np.zeros(30, bool))
        self.assertTrue((fast_decode(e, [(0, 30)], Config()) == 'X').all())
