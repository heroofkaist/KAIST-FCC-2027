"""Explicit observed-prefix coverage, including reproduction from the full bundle."""
import argparse
import json
from pathlib import Path
import numpy as np
from .io import Tracking, read_tracking
from .pipeline import sha256


def _manifest(path):
    report = json.loads(Path(path).read_text())
    rows = report.get('rows')
    if not isinstance(rows, int) or rows < 1 or report.get('first_frame') != 1 or report.get('last_frame') != rows:
        raise ValueError('invalid partial-tracking coverage')
    checksum = report.get('prefix_sha256')
    if not isinstance(checksum, str) or len(checksum) != 64 or any(c not in '0123456789abcdef' for c in checksum):
        raise ValueError('invalid partial-tracking checksum')
    return report


def read_tracking_prefix(meta, raw, manifest, n_frames, pitch):
    report = _manifest(manifest)
    if not isinstance(n_frames, int) or n_frames < report['rows']:
        raise ValueError('prefix does not fit the full timeline')
    if sha256(raw) != report['prefix_sha256']:
        raise ValueError('partial-tracking checksum mismatch')
    prefix = read_tracking(meta, raw, report['rows'], pitch)
    if len(prefix.team) != report.get('n_slots'):
        raise ValueError('partial-tracking metadata slot mismatch')
    ball = np.full((n_frames, 2), np.nan, np.float32)
    players = np.full((n_frames, len(prefix.team), 2), np.nan, np.float32)
    speed = np.full((n_frames, len(prefix.team)), np.nan, np.float32)
    count = report['rows']; ball[:count] = prefix.ball; players[:count] = prefix.players; speed[:count] = prefix.speed
    coverage = np.arange(n_frames) < count
    return Tracking(ball, players, prefix.team, speed, prefix.team_names, tuple(pitch)), coverage


def reproduce_prefix(full_tracking, manifest, out):
    report = _manifest(manifest); out = Path(out)
    if out.exists():
        raise FileExistsError('prefix output already exists')
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        with Path(full_tracking).open('rb') as source, out.open('xb') as target:
            for _ in range(report['rows']):
                line = source.readline()
                if not line:
                    raise ValueError('full source is shorter than the recorded prefix')
                target.write(line)
        if sha256(out) != report['prefix_sha256']:
            raise ValueError('full bundle does not reproduce the recorded prefix')
    except Exception:
        out.unlink(missing_ok=True)
        raise
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(); reproduce_prefix(args.source, args.manifest, args.out)
    print('Prefix reproduced and SHA-256 matched; protected coordinates remain local.')


if __name__ == '__main__':
    main()
