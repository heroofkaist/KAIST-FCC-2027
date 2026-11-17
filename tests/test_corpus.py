import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

try:
    from experiments.corpus import (sample_indices, upsample_probabilities,
                                    prepare_skillcorner, prepare_sportec,
                                    load_corpus_match)
    AVAILABLE = True
except ImportError:
    AVAILABLE = False


class CorpusTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(AVAILABLE, 'normalized external corpus is missing')

    def skill_fixture(self, root):
        meta = root / '42_match.json'
        meta.write_text(json.dumps({'id': 42, 'pitch_length': 100, 'pitch_width': 60,
                                    'home_team': {'id': 1}, 'away_team': {'id': 2},
                                    'players': [{'id': 11, 'team_id': 1},
                                                {'id': 12, 'team_id': 2}]}))
        rows = []
        for i, frame in enumerate([0, 1, 2, 100, 101, 102]):
            rows.append({'frame': frame, 'period': 1 if i < 3 else 2,
                         'ball_data': {'x': -9, 'y': 0, 'z': 0, 'is_detected': i != 1},
                         'player_data': [{'player_id': 11, 'x': -10, 'y': 0, 'is_detected': True},
                                         {'player_id': 12, 'x': 20, 'y': 0, 'is_detected': True}],
                         'possession': {'group': ['home team', 'away team', None][i % 3]}})
        raw = root / '42_tracking_extrapolated.jsonl'
        raw.write_text(''.join(json.dumps(x) + '\n' for x in rows))
        phases = root / '42_phases_of_play.csv'
        phases.write_text('frame_start,frame_end,team_in_possession_id\n0,2,1\n')
        return meta, raw, phases

    def test_cadence_and_interpolation_never_cross_blocks(self):
        indices, blocks = sample_indices([(0, 3), (5, 6), (7, 11)], 11, 10, 5)
        np.testing.assert_array_equal(indices, [0, 2, 5, 7, 9])
        self.assertEqual(blocks, [(0, 2), (2, 3), (3, 5)])
        p = np.array([[1., 0, 0], [0, 1., 0], [0, 0, 1.],
                      [.6, .3, .1], [.2, .3, .5]])
        full = upsample_probabilities(p, indices, [(0, 3), (5, 6), (7, 11)], 11)
        np.testing.assert_allclose(full[1], [.5, .5, 0])
        np.testing.assert_allclose(full[5], [0, 0, 1])
        np.testing.assert_allclose(full[10], [.2, .3, .5])
        np.testing.assert_allclose(full[3:5], 1 / 3)
        with self.assertRaises(ValueError):
            sample_indices([(0, 3)], 3, 12, 5)
        with self.assertRaises(ValueError):
            upsample_probabilities(p[:-1], indices, [(0, 3)], 11)

    def test_skillcorner_weak_labels_nulls_visibility_and_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            meta, raw, phases = self.skill_fixture(root)
            out = root / '42.npz'
            prepare_skillcorner(meta, raw, phases, out)
            match = load_corpus_match(out)
            self.assertEqual(match['labels'].tolist(), [0, 1, 3, 0, 1, 3])
            self.assertEqual(match['nearest'].tolist(), [0] * 6)
            self.assertEqual(match['observed'].tolist(), [0, 2, 0, 0, 0, 0])
            self.assertEqual(match['blocks'].tolist(), [[0, 3], [3, 6]])
            self.assertEqual(match['indices'].tolist(), [0, 2, 3, 5])
            np.testing.assert_allclose(match['ball'][0], [41, 30])
            self.assertTrue(match['provenance']['weak_labels'])
            with self.assertRaises(FileExistsError):
                prepare_skillcorner(meta, raw, phases, out)

    def test_skillcorner_labels_and_phases_cannot_change_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); meta, raw, phases = self.skill_fixture(root)
            prepare_skillcorner(meta, raw, phases, root / 'before.npz')
            raw.write_text(raw.read_text().replace('home team', 'SWAP').replace('away team', 'home team').replace('SWAP', 'away team'))
            phases.write_text('frame_start,frame_end,team_in_possession_id\n0,102,2\n')
            prepare_skillcorner(meta, raw, phases, root / 'after.npz')
            a, b = [load_corpus_match(root / name) for name in ['before.npz', 'after.npz']]
            self.assertFalse(np.array_equal(a['labels'], b['labels']))
            for key in ['nearest', 'observed', 'indices', 'ball', 'players', 'team', 'blocks', 'height']:
                np.testing.assert_array_equal(a[key], b[key])

    def test_skillcorner_unknown_ids_and_duplicate_frames_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); meta, raw, phases = self.skill_fixture(root)
            original = raw.read_text()
            raw.write_text(original.replace('"player_id": 11', '"player_id": 999'))
            with self.assertRaises(ValueError):
                prepare_skillcorner(meta, raw, phases, root / 'bad.npz')
            raw.write_text(original.replace('"frame": 1,', '"frame": 0,'))
            with self.assertRaises(ValueError):
                prepare_skillcorner(meta, raw, phases, root / 'bad.npz')

    def test_sportec_supervision_is_separate_and_partial_blocks_sample(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); meta = root / 'meta.xml'; raw = root / 'JTEST.xml'
            meta.write_text('<r><General HomeTeamId="h" GuestTeamId="a" HomeTeamName="Home" GuestTeamName="Away"/><Environment PitchX="100" PitchY="60"/></r>')
            raw.write_text('<r><FrameSet GameSection="firstHalf" TeamId="h" PersonId="1"><Frame N="1" X="-10" Y="0" S="0"/></FrameSet><FrameSet GameSection="firstHalf" TeamId="a" PersonId="2"><Frame N="1" X="20" Y="0" S="0"/></FrameSet><FrameSet GameSection="firstHalf" TeamId="BALL"><Frame N="1" X="-9" Y="0" Z="0" BallStatus="1" BallPossession="1"/></FrameSet></r>')
            prepare_sportec(meta, raw, root / 'a.npz')
            raw.write_text(raw.read_text().replace('BallStatus="1"', 'BallStatus="0"'))
            prepare_sportec(meta, raw, root / 'b.npz')
            a, b = [load_corpus_match(root / name) for name in ['a.npz', 'b.npz']]
            self.assertEqual(a['labels'].tolist(), [0]); self.assertEqual(b['labels'].tolist(), [2])
            self.assertEqual(a['sample_blocks'].tolist(), [[0, 1]])
            for key in ['nearest', 'observed', 'ball', 'players', 'indices', 'height']:
                np.testing.assert_array_equal(a[key], b[key])
