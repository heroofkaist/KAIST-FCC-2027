from pathlib import Path
import tempfile
import unittest
import numpy as np
try:
    from experiments.sportec import read_sportec, nearest_labels, mask_observations
    AVAILABLE=True
except ImportError:AVAILABLE=False

class SportecTests(unittest.TestCase):
    def setUp(self):self.assertTrue(AVAILABLE,'external Sportec adapter is missing')

    def fixture(self,root):
        meta=root/'meta.xml'
        meta.write_text('<root><General HomeTeamId="h" GuestTeamId="a" HomeTeamName="Home" GuestTeamName="Away"/><Environment PitchX="100" PitchY="60"/></root>')
        raw=root/'positions.xml'
        raw.write_text('''<root>
<FrameSet GameSection="firstHalf" TeamId="h" PersonId="p1"><Frame N="10" X="-10" Y="0" S="3.6"/><Frame N="11" X="-10" Y="0" S="0"/></FrameSet>
<FrameSet GameSection="firstHalf" TeamId="a" PersonId="p2"><Frame N="10" X="20" Y="0" S="0"/><Frame N="11" X="20" Y="0" S="0"/></FrameSet>
<FrameSet GameSection="firstHalf" TeamId="BALL" PersonId="ball"><Frame N="10" X="-9" Y="0" Z="0" BallStatus="1" BallPossession="1"/><Frame N="11" X="20" Y="0" Z="0" BallStatus="0" BallPossession="2"/></FrameSet>
<FrameSet GameSection="secondHalf" TeamId="BALL" PersonId="ball"><Frame N="100" X="1" Y="0" Z="0" BallStatus="1" BallPossession="2"/></FrameSet>
</root>''')
        return meta,raw

    def test_side_labels_coordinates_speed_and_half_boundaries(self):
        with tempfile.TemporaryDirectory() as tmp:
            meta,raw=self.fixture(Path(tmp));match=read_sportec(meta,raw)
            self.assertEqual(match.labels.tolist(),['H','D','A'])
            self.assertEqual(match.blocks,[(0,2),(2,3)])
            np.testing.assert_allclose(match.tracking.ball[0],[41,30])
            self.assertEqual(match.tracking.speed[0,0],1)
            self.assertEqual(nearest_labels(match.tracking).tolist(),['H','A','D'])

    def test_human_labels_cannot_change_observation_features(self):
        with tempfile.TemporaryDirectory() as tmp:
            meta,raw=self.fixture(Path(tmp));a=read_sportec(meta,raw)
            raw.write_text(raw.read_text().replace('BallPossession="1"','BallPossession="2"').replace('BallStatus="0"','BallStatus="1"'))
            b=read_sportec(meta,raw)
            self.assertFalse(np.array_equal(a.labels,b.labels))
            np.testing.assert_array_equal(nearest_labels(a.tracking),nearest_labels(b.tracking))
            np.testing.assert_array_equal(mask_observations(a,seed=7),mask_observations(b,seed=7))

    def test_duplicate_ball_frames_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            meta,raw=self.fixture(Path(tmp));raw.write_text(raw.read_text().replace('N="11" X="20" Y="0" Z="0"','N="10" X="20" Y="0" Z="0"'))
            with self.assertRaises(ValueError):read_sportec(meta,raw)
