import csv
import tempfile
import unittest
from pathlib import Path
import numpy as np

try:
    from solution.io import read_tracking, read_baseline, read_tags
    from solution.config import Config, load_config
    AVAILABLE = True
except ImportError:
    AVAILABLE = False

META = '<root><Team id="h"><Name>Home</Name></Team><Team id="a"><Name>Away</Name></Team><Player teamId="h"/><Player teamId="a"/></root>'

class InputTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(AVAILABLE, 'solution input implementation is missing')
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.meta = self.root/'meta.xml'
        self.meta.write_text(META)
        self.raw = self.root/'raw.txt'

    def test_tracking_scales_coordinates_and_keeps_missing(self):
        self.raw.write_text('1:0.1,0.2,3;NaN,NaN,NaN:0.2,0.3,4\n2:0.1,0.2,3;0.9,0.8,2:NaN,NaN,NaN\n')
        t = read_tracking(self.meta, self.raw, 2, (100., 50.))
        np.testing.assert_allclose(t.ball[0], [20,15])
        self.assertTrue(np.isnan(t.ball[1]).all())
        np.testing.assert_equal(t.team, [0,1])

    def test_tracking_rejects_internal_duplicate_even_with_correct_endpoints(self):
        self.raw.write_text(''.join(f'{i}:0,0,0;1,1,0:0,0,0\n' for i in (1,2,2,4)))
        with self.assertRaises(ValueError): read_tracking(self.meta,self.raw,4,(100,50))

    def test_tracking_rejects_truncation_extra_rows_slots_inf_and_partial_xy(self):
        for content in ('1:0,0,0;1,1,0:0,0,0\n',
                        '1:0,0,0:0,0,0\n2:0,0,0:0,0,0\n',
                        '1:0,0,0;1,1,0:inf,0,0\n2:0,0,0;1,1,0:0,0,0\n',
                        '1:0,0,0;1,1,0:NaN,0,0\n2:0,0,0;1,1,0:0,0,0\n'):
            with self.subTest(content=content):
                self.raw.write_text(content)
                with self.assertRaises(ValueError): read_tracking(self.meta,self.raw,2,(100,50))

    def test_unknown_team_rejected(self):
        self.meta.write_text(META.replace('teamId="a"','teamId="unknown"'))
        self.raw.write_text('1:0,0,0;1,1,0:0,0,0\n')
        with self.assertRaises(ValueError): read_tracking(self.meta,self.raw,1,(100,50))

    def test_baseline_contract_and_gt_path_guard(self):
        p=self.root/'baseline_raw.csv'
        p.write_text('frame,label\n0,H\n1,D\n2,A\n')
        self.assertEqual(read_baseline(p,3).tolist(), ['H','D','A'])
        p.write_text('frame,label\n0,H\n0,A\n2,D\n')
        with self.assertRaises(ValueError): read_baseline(p,3)
        p=self.root/'possession_gt_public.csv'
        p.write_text('frame,label\n0,H\n')
        with self.assertRaises(ValueError): read_baseline(p,1)

    def test_tag_time_and_side_mapping(self):
        p=self.root/'tags.xml'
        p.write_text('<root><instance><start>0.04</start><end>0.12</end><code>Defending in Middle Third</code><label><group>Team</group><text>Home</text></label></instance></root>')
        tags=read_tags(p,10,25,('Home','Away'))
        self.assertEqual((tags[0].start,tags[0].stop,tags[0].team,tags[0].kind),(1,3,0,'defense'))
        p.write_text(p.read_text().replace('0.12','nan'))
        with self.assertRaises(ValueError): read_tags(p,10,25,('Home','Away'))

    def test_config_rejects_unknown_nonfinite_and_invalid_values(self):
        for text in ('{"fps": 0}', '{"transition_cost": -1}', '{"tag_weight": NaN}', '{"typo": 1}', '{"mature_frames": 2.5}'):
            p=self.root/'config.json';p.write_text(text)
            with self.assertRaises(ValueError): load_config(p)

class HomographyTests(unittest.TestCase):
    def test_frame_order_and_valid_matrix_are_checked(self):
        self.assertTrue(AVAILABLE)
        from solution.io import validate_homography
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'h.csv'
            header='frame,h00,h01,h02,h10,h11,h12,h20,h21,h22,reprojection_error,valid\n'
            p.write_text(header+'0,1,0,0,0,1,0,0,0,1,0.2,1\n')
            validate_homography(p,1)
            for row in ('1,1,0,0,0,1,0,0,0,1,0.2,1\n','0,nan,0,0,0,1,0,0,0,1,0.2,1\n'):
                p.write_text(header+row)
                with self.assertRaises(ValueError):validate_homography(p,1)
