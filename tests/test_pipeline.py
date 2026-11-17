import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from test_inputs import META
try:
    from solution.pipeline import run, Match
    AVAILABLE=True
except ImportError:
    AVAILABLE=False

class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(AVAILABLE,'pipeline implementation is missing')
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.data=self.root/'data';self.folder=self.data/'tactical';self.folder.mkdir(parents=True)
        (self.folder/'BAR-ATH_tactical_atd.xml').write_text(META)
        self.baseline=self.root/'baseline.csv'
        self.baseline.write_text('frame,label\n'+''.join(f'{i},H\n' for i in range(100)))
        self.config=self.root/'config.json';self.config.write_text('{}')
        self.match=Match(n_frames=100,blocks=((0,100),))

    def generate(self,name='out',mode='bootstrap'):
        return run(mode,self.data,self.baseline,self.config,self.root/name,match=self.match)

    def test_two_runs_identical_and_manifest_matches_bytes(self):
        report=self.generate('a');self.generate('b')
        a=(self.root/'a/submission.csv').read_bytes()
        self.assertEqual(a,(self.root/'b/submission.csv').read_bytes())
        self.assertEqual(report['output_sha256'],hashlib.sha256(a).hexdigest())
        self.assertEqual(report['mode'],'bootstrap')
        self.assertIn('smart_tagging',report['missing_optional'])
        self.assertEqual(len(a.decode().splitlines()),101)
        self.assertEqual(json.loads((self.root/'a/manifest.json').read_text())['output_sha256'],report['output_sha256'])

    def test_missing_tracking_does_not_create_output(self):
        with self.assertRaises(FileNotFoundError):self.generate(mode='tracking')
        self.assertFalse((self.root/'out').exists())

    def test_tracking_cli_core_generates_valid_fixture(self):
        (self.folder/'BAR-ATH_tactical_atd.txt').write_text(''.join(f'{i}:0.1,0.2,0;0.9,0.8,0:0.105,0.2,0\n' for i in range(1,101)))
        report=self.generate(mode='tracking')
        self.assertEqual(report['label_counts']['H'],100)
        self.assertFalse(report['provisional'])

    def test_invalid_optional_tags_stop_before_output(self):
        (self.folder/'BAR-ATH_tactical_smart_tagging.xml').write_text('<root><instance><start>4</start><end>2</end></instance></root>')
        with self.assertRaises(ValueError):self.generate()
        self.assertFalse((self.root/'out').exists())

    def test_no_overwrite_of_existing_runs_or_input_directories(self):
        self.generate()
        with self.assertRaises(FileExistsError):self.generate()
        with self.assertRaises(ValueError):
            run('bootstrap',self.data,self.baseline,self.config,self.folder/'result',match=self.match)

    def test_repository_protected_output_and_wrong_fps(self):
        repo=Path(__file__).resolve().parents[1]
        with self.assertRaises(ValueError):
            run('bootstrap',self.data,self.baseline,self.config,repo/'evaluation/results/new',match=self.match)
        self.config.write_text('{"fps": 30}')
        with self.assertRaises(ValueError):self.generate()

    def test_cli_missing_inputs_fails_cleanly(self):
        result=subprocess.run([sys.executable,'-m','solution','--data-dir',str(self.data),'--out-dir',str(self.root/'cli')],text=True,capture_output=True)
        self.assertEqual(result.returncode,2)
        self.assertIn('error:',result.stderr)
        self.assertNotIn('Traceback',result.stderr)
        self.assertFalse((self.root/'cli').exists())

    def test_external_calibration_is_explicit_and_requires_matching_configuration(self):
        from dataclasses import asdict
        from solution.config import Config
        calibration=self.root/'external.json'
        calibration.write_text(json.dumps({'method':'baseline_raw','configuration':asdict(Config()),'training_match':'J03WOH','validation_match':'J03WOY','held_out_match':'J03WPY'}))
        report=run('bootstrap',self.data,self.baseline,self.config,self.root/'calibrated',match=self.match,calibration_path=calibration)
        self.assertEqual(report['calibration_method'],'baseline_raw')
        self.assertIn('calibration',report['inputs'])
        selection=json.loads(calibration.read_text());selection['configuration']['transition_cost']=999
        calibration.write_text(json.dumps(selection))
        with self.assertRaises(ValueError):
            run('bootstrap',self.data,self.baseline,self.config,self.root/'bad-calibration',match=self.match,calibration_path=calibration)

    def test_learned_calibration_in_pipeline(self):
        from solution.pipeline import ROOT
        config=ROOT/'configs/learned-selected.json'
        calibration=ROOT/'configs/learned-calibration.json'
        report=run('bootstrap',self.data,self.baseline,config,self.root/'learned_out',match=self.match,calibration_path=calibration)
        self.assertEqual(report['calibration_method'],'learned')
        self.assertIn('model_manifest_0',report['inputs'])
        self.assertIn('model_weights_0',report['inputs'])
        self.assertIn('model_manifest_1',report['inputs'])
        self.assertIn('model_weights_1',report['inputs'])
        self.assertEqual(sum(report['label_counts'].values()),100)
        out_sub=self.root/'learned_out/submission.csv'
        self.assertTrue(out_sub.is_file())
        self.assertEqual(len(out_sub.read_bytes().decode().splitlines()),101)

    def test_spatial_learned_tracking_in_pipeline(self):
        from solution.pipeline import ROOT
        config=ROOT/'configs/learned-selected.json'
        calibration=ROOT/'configs/learned-spatial-ensemble.json'
        (self.folder/'BAR-ATH_tactical_atd.txt').write_text(''.join(f'{i}:0.1,0.2,0;0.9,0.8,0:0.105,0.2,0\n' for i in range(1,101)))
        report=run('tracking',self.data,self.baseline,config,self.root/'spatial_out',match=self.match,calibration_path=calibration)
        self.assertEqual(report['mode'],'tracking')
        self.assertFalse(report['provisional'])
        self.assertEqual(report['calibration_method'],'learned')
        self.assertIn('tracking',report['inputs'])
        self.assertIn('model_manifest_0',report['inputs'])
        self.assertIn('model_weights_0',report['inputs'])
        self.assertEqual(sum(report['label_counts'].values()),100)
        out_sub=self.root/'spatial_out/submission.csv'
        self.assertTrue(out_sub.is_file())
        self.assertEqual(len(out_sub.read_bytes().decode().splitlines()),101)
