"""Run local inference. Ground truth and video are never loaded."""
import argparse
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
from games import DATA_DIR
from .pipeline import ROOT, run


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('tracking','bootstrap'),default='tracking')
    parser.add_argument('--data-dir',type=Path,default=Path(DATA_DIR))
    parser.add_argument('--baseline',type=Path,default=ROOT/'evaluation/results/baseline_raw.csv',help='Algorithmic predictions only; human labels are forbidden')
    parser.add_argument('--config',type=Path,default=ROOT/'configs/possession.json')
    parser.add_argument('--calibration',type=Path,help='Frozen external selection JSON; use its matching --config')
    parser.add_argument('--out-dir',type=Path,required=True,help='New local artifacts/<run> directory')
    args=parser.parse_args()
    try:
        report=run(args.mode,args.data_dir,args.baseline,args.config,args.out_dir,calibration_path=args.calibration)
    except (OSError,ValueError,ET.ParseError) as exc:
        print(f'error: {exc}',file=sys.stderr)
        return 2
    print(f'Wrote {args.out_dir / "submission.csv"} ({sum(report["label_counts"].values())} frames)')
    print(f'Mode: {report["mode"]}; SHA256: {report["output_sha256"]}')
    if report['provisional']:print('Provisional bootstrap candidate: full tactical tracking was not used.')
    print('No ground truth was read. Hidden-test quality is unknown.')
    return 0

if __name__=='__main__':raise SystemExit(main())
