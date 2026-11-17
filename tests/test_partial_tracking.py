import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from solution.pipeline import sha256

try:
    from solution.partial_tracking import read_tracking_prefix, reproduce_prefix
    AVAILABLE = True
except ImportError:
    AVAILABLE = False


class PartialTrackingTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(AVAILABLE, 'explicit partial tracking support is missing')

    def fixture(self, root):
        meta = root / 'meta.xml'
        meta.write_text('<root><Team id="h"><Name>Home</Name></Team><Team id="a"><Name>Away</Name></Team><Player teamId="h"/><Player teamId="a"/></root>')
        raw = root / 'prefix.txt'; raw.write_text('1:0,0,0;1,1,0:0.5,0.5,0\n2:0,0,0;1,1,0:NaN,NaN,NaN\n')
        manifest = {'rows': 2, 'first_frame': 1, 'last_frame': 2, 'n_slots': 2,
                    'prefix_sha256': sha256(raw), 'complete_member': False}
        path = root / 'manifest.json'; path.write_text(json.dumps(manifest))
        return meta, raw, path

    def test_explicit_coverage_and_hash_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); meta, raw, manifest = self.fixture(root)
            tracking, coverage = read_tracking_prefix(meta, raw, manifest, 5, (100., 60.))
            np.testing.assert_array_equal(coverage, [True, True, False, False, False])
            np.testing.assert_allclose(tracking.ball[0], [50, 30])
            self.assertTrue(np.isnan(tracking.players[2:]).all())
            self.assertTrue(np.isnan(tracking.ball[2:]).all())
            raw.write_text(raw.read_text().replace('0.5', '0.4'))
            with self.assertRaises(ValueError):
                read_tracking_prefix(meta, raw, manifest, 5, (100., 60.))

    def test_full_source_can_regenerate_exact_prefix_and_never_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); meta, raw, manifest = self.fixture(root)
            full = root / 'full.txt'; full.write_bytes(raw.read_bytes() + b'3:0,0,0;1,1,0:0,0,0\n')
            out = root / 'reproduced.txt'
            reproduce_prefix(full, manifest, out)
            self.assertEqual(out.read_bytes(), raw.read_bytes())
            with self.assertRaises(FileExistsError):
                reproduce_prefix(full, manifest, out)
