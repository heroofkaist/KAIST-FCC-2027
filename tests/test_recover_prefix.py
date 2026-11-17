import io
from pathlib import Path
import struct
import tempfile
import unittest
import zipfile

try:
    from experiments.recover_prefix import recover_atd_prefix
    AVAILABLE = True
except ImportError:
    AVAILABLE = False


class RecoveryTests(unittest.TestCase):
    def test_truncated_deflate_recovers_only_complete_validated_records(self):
        self.assertTrue(AVAILABLE, 'partial ATD recovery is missing')
        original = ''.join(f'{i}:0.1,0.2,0;0.3,0.4,0:0.2,0.3,{i % 7}\n' for i in range(1, 1001)).encode()
        memory = io.BytesIO()
        with zipfile.ZipFile(memory, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr('tactical/test_atd.txt', original)
        raw = memory.getvalue(); values = struct.unpack('<5H3I2H', raw[4:30])
        start = 30 + values[-2] + values[-1]; size = values[6]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); partial = root / 'incomplete.zip'
            partial.write_bytes(raw[:start + size - 20])
            report = recover_atd_prefix(partial, root / 'recovered', n_slots=2)
            recovered = (root / 'recovered/tracking-prefix.txt').read_bytes()
            self.assertGreater(report['rows'], 0); self.assertLess(report['rows'], 1000)
            self.assertEqual(recovered, b''.join(original.splitlines(keepends=True)[:report['rows']]))
            self.assertFalse(report['complete_member'])
            self.assertEqual(report['first_frame'], 1)
            self.assertEqual(report['last_frame'], report['rows'])
            with self.assertRaises(FileExistsError):
                recover_atd_prefix(partial, root / 'recovered', n_slots=2)

    def test_invalid_structure_or_wrong_member_is_rejected(self):
        self.assertTrue(AVAILABLE, 'partial ATD recovery is missing')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); path = root / 'bad.zip'
            with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
                archive.writestr('wrong.mp4', b'1:bad\n')
            with self.assertRaises(ValueError):
                recover_atd_prefix(path, root / 'output', n_slots=2)
