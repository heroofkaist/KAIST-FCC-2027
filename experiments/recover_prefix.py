"""Recover structurally valid ATD records from an interrupted ZIP download.

The original is never modified. Only complete records are written, with a
manifest explicitly distinguishing structural validation from unavailable CRC.
"""
import argparse
import json
import math
from pathlib import Path
import shutil
import struct
import tempfile
import zlib
from solution.pipeline import sha256


def _validate_record(line, expected, n_slots):
    parts = line.decode('utf-8').strip().split(':')
    if len(parts) != 3 or parts[0] != str(expected):
        raise ValueError('recovered tracking frame IDs must be exactly 1..N')
    groups = parts[1].split(';') + [parts[2]]
    if len(groups) != n_slots + 1:
        raise ValueError('recovered tracking slot count mismatch')
    for group in groups:
        values = [float(x) for x in group.split(',')]
        if len(values) != 3 or any(math.isinf(x) for x in values):
            raise ValueError('invalid recovered numeric shape/value')
        if math.isnan(values[0]) != math.isnan(values[1]) or (math.isfinite(values[2]) and values[2] < 0):
            raise ValueError('partial coordinates or negative speed in recovered record')


def recover_atd_prefix(archive, out, n_slots):
    archive, out = Path(archive), Path(out)
    if out.exists():
        raise FileExistsError('recovery output already exists')
    if not isinstance(n_slots, int) or n_slots < 1:
        raise ValueError('positive track slot count is required')
    before = sha256(archive)
    with archive.open('rb') as handle:
        header = handle.read(30)
        if len(header) != 30 or header[:4] != b'PK\x03\x04':
            raise ValueError('archive has no first ZIP local header')
        version, flags, method, _, _, crc, size, expected_size, name_length, extra_length = struct.unpack('<5H3I2H', header[4:])
        name = handle.read(name_length).decode('utf-8')
        if method != 8 or flags & 1 or not name.endswith('_atd.txt'):
            raise ValueError('expected an unencrypted Deflate ATD member')
        handle.read(extra_length)
        out.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix='.recover-atd-', dir=out.parent))
        try:
            target = staging / 'tracking-prefix.txt'; decoder = zlib.decompressobj(-15)
            carry = b''; rows = 0; decoded_bytes = 0; running_crc = 0
            with target.open('wb') as output:
                while chunk := handle.read(1024 * 1024):
                    decoded = decoder.decompress(chunk)
                    decoded_bytes += len(decoded); running_crc = zlib.crc32(decoded, running_crc)
                    carry += decoded
                    end = carry.rfind(b'\n')
                    if end >= 0:
                        complete, carry = carry[:end + 1], carry[end + 1:]
                        for line in complete.splitlines(keepends=True):
                            _validate_record(line, rows + 1, n_slots); rows += 1
                            output.write(line)
                    if decoder.eof:
                        if carry:
                            _validate_record(carry, rows + 1, n_slots); rows += 1
                            output.write(carry); carry = b''
                        break
            if not rows:
                raise ValueError('no complete tracking records could be recovered')
            crc_verified = decoder.eof and not (flags & 8)
            if crc_verified and (running_crc != crc or decoded_bytes != expected_size):
                raise ValueError('complete member CRC/size mismatch')
            if sha256(archive) != before:
                raise ValueError('source archive changed during recovery')
            report = {'source_archive_sha256': before, 'member': name, 'rows': rows,
                      'first_frame': 1, 'last_frame': rows, 'complete_member': decoder.eof,
                      'member_crc_verified': bool(crc_verified),
                      'integrity': 'Complete records structurally validated; an interrupted member has no verifiable final CRC.',
                      'discarded_partial_record_bytes': len(carry), 'n_slots': n_slots,
                      'prefix_sha256': sha256(target), 'protected_local_data': True}
            (staging / 'manifest.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
            if out.exists():
                raise FileExistsError('recovery output was created during processing')
            staging.rename(out)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--slots', type=int, required=True)
    args = parser.parse_args(); report = recover_atd_prefix(args.archive, args.out, args.slots)
    print('Recovered', report['rows'], 'complete records; member complete:', report['complete_member'])
    print('No coordinates or labels are printed; local source/prefix hashes recorded in manifest.')


if __name__ == '__main__':
    main()
