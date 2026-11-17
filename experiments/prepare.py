"""Download/prepare three permitted external Sportec matches, never BAR-ATH."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from urllib.request import urlopen
import numpy as np
from .sportec import read_sportec,nearest_labels,mask_observations
from solution.pipeline import sha256

GAMES={'J03WOH':(11,348945715),'J03WOY':(22,362567859),'J03WPY':(33,371688499)}
BASE='https://huggingface.co/datasets/pysport/idsse-data/resolve/main/'


def prepare_files(meta_path,raw_path,out_path,seed):
    out_path=Path(out_path)
    if out_path.exists():raise FileExistsError('prepared cache already exists')
    before={str(p):sha256(p) for p in (meta_path,raw_path)}
    match=read_sportec(meta_path,raw_path)
    report={'tracking_sha256':before[str(raw_path)],'metadata_sha256':before[str(meta_path)],
            'frames':len(match.labels),'blocks':match.blocks,'seed':seed,
            'adapter_sha256':sha256(Path(__file__).with_name('sportec.py')),
            'dataset':'https://doi.org/10.6084/m9.figshare.28196177','license':'CC BY 4.0'}
    for p in (meta_path,raw_path):
        if sha256(p)!=before[str(p)]:raise ValueError('external source changed during preparation')
    out_path.parent.mkdir(parents=True,exist_ok=True)
    with out_path.open('xb') as handle:
        np.savez_compressed(handle,labels=match.labels,clean=nearest_labels(match.tracking),masked=mask_observations(match,seed=seed),blocks=np.array(match.blocks))
    out_path.with_suffix('.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    return report


def download(root,game):
    for kind,template in (('','DFL_04_03_positions_raw_observed_DFL-COM-000002_DFL-MAT-{}.xml'),('-meta','DFL_02_01_matchinformation_DFL-COM-000002_DFL-MAT-{}.xml')):
        target=root/(game+kind+'.xml')
        if target.exists():continue
        part=target.with_suffix('.download.part')
        with urlopen(BASE+template.format(game),timeout=120) as response,part.open('wb') as handle:
            shutil.copyfileobj(response,handle,1024*1024)
        if not kind and part.stat().st_size!=GAMES[game][1]:raise ValueError('external download has unexpected size')
        part.rename(target)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,default=Path('data/external/sportec'))
    parser.add_argument('--download',action='store_true')
    args=parser.parse_args();args.data_dir.mkdir(parents=True,exist_ok=True)
    for game,(seed,size) in GAMES.items():
        if args.download:download(args.data_dir,game)
        raw=args.data_dir/(game+'.xml');meta=args.data_dir/(game+'-meta.xml')
        if raw.stat().st_size!=size:raise ValueError('external source size does not match the declared match')
        cache=args.data_dir/(game+'-benchmark.npz')
        if cache.exists():
            print(game,'cache already exists; use a fresh directory to regenerate')
            continue
        report=prepare_files(meta,raw,cache,seed)
        print(game,report['frames'],'frames prepared',flush=True)

if __name__=='__main__':main()
