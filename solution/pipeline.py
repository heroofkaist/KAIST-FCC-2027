"""File-to-file inference with provenance and protected output locations."""
from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib
import json
import os
import platform
import shutil
import subprocess
import tempfile
import numpy as np
from games import GAMES, GAME, asset_paths, half_video_blocks
from .config import load_config
from .io import Tracking, read_metadata, read_tracking, read_baseline, read_tags, validate_homography
from .evidence import Evidence, tracking_evidence, bootstrap_evidence, add_tags, validate_blocks
from .decode import decode
from .temporal import calibrated_evidence

ROOT=Path(__file__).resolve().parents[1]

@dataclass(frozen=True)
class Match:
    n_frames: int = GAMES[GAME]['n_video_frames']
    blocks: tuple = tuple((a,b+1) for _,a,b in half_video_blocks(GAME))
    pitch: tuple = GAMES[GAME]['pitch']
    fps: float = GAMES[GAME]['fps']


def sha256(path):
    digest=hashlib.sha256()
    with open(path,'rb') as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


def _check_output(out_dir,data_dir,input_paths):
    out=Path(out_dir).resolve()
    data=Path(data_dir).resolve()
    if out==data or data in out.parents or out in data.parents:
        raise ValueError('output must be separate from input data directory')
    for path in input_paths:
        source=Path(path).resolve()
        if out==source or out in source.parents:
            raise ValueError('output directory cannot contain an input file')
    if out==ROOT or ROOT in out.parents:
        artifacts=ROOT/'artifacts'
        if artifacts not in out.parents:
            raise ValueError('repository outputs must be in a new artifacts/<run> directory')
    if out.exists():raise FileExistsError('output directory already exists; choose a new run name')
    return out


def run(mode,data_dir,baseline_path,config_path,out_dir,*,match=None,calibration_path=None):
    match=match or Match()
    if mode not in ('tracking','bootstrap'):raise ValueError('mode must be tracking or bootstrap')
    config=load_config(config_path)
    if config.fps!=match.fps:raise ValueError('config fps must equal the match frame rate')
    validate_blocks(match.blocks,match.n_frames)
    paths={key:Path(value) for key,value in asset_paths(data_dir=str(data_dir)).items()}
    required={'metadata':paths['atd_meta'],'configuration':Path(config_path)}
    calibration=None
    if calibration_path is not None:
        if mode not in ('bootstrap','tracking'):raise ValueError('external calibration requires bootstrap or tracking mode')
        calibration=json.loads(Path(calibration_path).read_text(encoding='utf-8'))
        if not isinstance(calibration,dict) or calibration.get('method') not in ('baseline_raw','baseline_smooth50','heuristic','temporal','learned'):
            raise ValueError('invalid external calibration method')
        if calibration.get('configuration')!=asdict(config):
            raise ValueError('configuration must match the frozen external calibration')
        required['calibration']=Path(calibration_path)
        if calibration.get('method')=='learned':
            entries=[calibration] if 'model_dir' in calibration else calibration.get('models',[])
            for idx,entry in enumerate(entries):
                model_dir=ROOT/entry['model_dir']
                manifest_path=model_dir/'model.json'
                required[f'model_manifest_{idx}']=manifest_path
                w_name='weights.txt'
                if manifest_path.is_file():
                    try:w_name=json.loads(manifest_path.read_text(encoding='utf-8')).get('weights_file','weights.txt')
                    except Exception:pass
                required[f'model_weights_{idx}']=model_dir/w_name
    if mode=='tracking':required['tracking']=paths['atd_raw']
    else:required['algorithmic_baseline']=Path(baseline_path)
    for key,path in required.items():
        if not path.is_file():raise FileNotFoundError(f'missing {key}: {path.name}')
    missing=[]
    for key in ('smart_tagging','homography'):
        if paths[key].is_file():required[key]=paths[key]
        else:missing.append(key)
    out=_check_output(out_dir,data_dir,required.values())
    hashes={key:sha256(path) for key,path in required.items()}
    _,names=read_metadata(paths['atd_meta'])
    tags=read_tags(paths['smart_tagging'],match.n_frames,config.fps,names) if 'smart_tagging' in required else []
    if 'homography' in required:validate_homography(paths['homography'],match.n_frames)
    if mode=='tracking':
        tracking=read_tracking(paths['atd_meta'],paths['atd_raw'],match.n_frames,match.pitch)
        barriers=[(tag.start,tag.stop) for tag in tags if tag.kind=='restart']
        if calibration and calibration.get('method')=='learned':
            from .features import categorical_features
            from .spatial import spatial_features
            from .learned import load_learner, predict_probabilities
            from experiments.sportec import nearest_labels
            from experiments.corpus import upsample_probabilities
            indices=np.concatenate([np.arange(a,b,5) for a,b in match.blocks])
            sample_blocks=[]
            cursor=0
            for a,b in match.blocks:
                count=len(np.arange(a,b,5))
                sample_blocks.append((cursor,cursor+count))
                cursor+=count
            nearest=nearest_labels(tracking)
            coded=np.array([{'H':0,'A':1,'D':2,'X':3}[x] for x in nearest[indices]],dtype=np.int8)
            cat_features,_=categorical_features(coded,sample_blocks,5)
            first_dir=calibration.get('model_dir') or calibration['models'][0]['model_dir']
            manifest=json.loads((ROOT/first_dir/'model.json').read_text(encoding='utf-8'))
            n_feat=len(manifest.get('feature_names',[]))
            if n_feat==92:
                sub_tracking=Tracking(tracking.ball[indices],tracking.players[indices],tracking.team,tracking.speed[indices],tracking.team_names,tracking.pitch)
                spat_features,_=spatial_features(sub_tracking,sample_blocks,5)
                features=np.column_stack([cat_features,spat_features])
            else:
                features=cat_features
            if 'model_dir' in calibration:
                learner=load_learner(ROOT/calibration['model_dir'])
                prob=predict_probabilities(learner,features,sample_blocks)
            elif 'models' in calibration:
                probs=[]
                for entry in calibration['models']:
                    learner=load_learner(ROOT/entry['model_dir'])
                    p=predict_probabilities(learner,features,sample_blocks)
                    probs.append(p*entry.get('weight',1.0))
                prob=sum(probs)/sum(e.get('weight',1.0) for e in calibration['models'])
            else:
                raise ValueError('learned calibration must specify model_dir or models')
            full_p=upsample_probabilities(prob,indices,match.blocks,match.n_frames)
            scores=np.log(np.maximum(full_p,1e-9))
            dead_bias=float(calibration.get('dead_bias',0.0))
            if dead_bias!=0.0:scores[:,2]+=dead_bias
            evidence=Evidence(scores,np.ones(match.n_frames,bool))
        else:
            evidence=tracking_evidence(tracking,match.blocks,config,barriers=barriers)
    else:
        labels=read_baseline(baseline_path,match.n_frames)
        if calibration and calibration['method']=='temporal':
            evidence=calibrated_evidence(labels,match.blocks,config,calibration['model'])
        elif calibration and calibration['method']=='learned':
            from .features import categorical_features
            from .learned import load_learner, predict_probabilities
            from experiments.corpus import upsample_probabilities
            indices=np.concatenate([np.arange(a,b,5) for a,b in match.blocks])
            sample_blocks=[]
            cursor=0
            for a,b in match.blocks:
                count=len(np.arange(a,b,5))
                sample_blocks.append((cursor,cursor+count))
                cursor+=count
            coded=np.array([{'H':0,'A':1,'D':2,'X':3}[x] for x in labels[indices]],dtype=np.int8)
            features,_=categorical_features(coded,sample_blocks,5)
            if 'model_dir' in calibration:
                learner=load_learner(ROOT/calibration['model_dir'])
                prob=predict_probabilities(learner,features,sample_blocks)
            elif 'models' in calibration:
                probs=[]
                for entry in calibration['models']:
                    learner=load_learner(ROOT/entry['model_dir'])
                    p=predict_probabilities(learner,features,sample_blocks)
                    probs.append(p*entry.get('weight',1.0))
                prob=sum(probs)/sum(e.get('weight',1.0) for e in calibration['models'])
            else:
                raise ValueError('learned calibration must specify model_dir or models')
            full_p=upsample_probabilities(prob,indices,match.blocks,match.n_frames)
            scores=np.log(np.maximum(full_p,1e-9))
            dead_bias=float(calibration.get('dead_bias',0.0))
            if dead_bias!=0.0:scores[:,2]+=dead_bias
            evidence=Evidence(scores,np.ones(match.n_frames,bool))
        else:evidence=bootstrap_evidence(labels,match.blocks,config)
    baseline_only=calibration and calibration['method'].startswith('baseline_')
    if baseline_only:
        from starter.baseline import smooth
        codes=np.where(labels=='H',0,np.where(labels=='A',1,-1))
        labels=np.full(match.n_frames,'X',dtype='<U1')
        window=50 if calibration['method']=='baseline_smooth50' else 0
        for start,stop in match.blocks:labels[start:stop]=smooth(codes[start:stop],window)
    else:
        evidence=add_tags(evidence,tags,config)
        labels=decode(evidence,match.blocks,config)
    for key,path in required.items():
        if sha256(path)!=hashes[key]:raise ValueError('input changed during inference; rerun with stable inputs')
    if len(labels)!=match.n_frames or not np.isin(labels,['H','A','D','X']).all():
        raise ValueError('internal output contract violation')
    code_paths=sorted((ROOT/'solution').glob('*.py'))+[ROOT/'games.py']
    code_hashes={str(p.relative_to(ROOT)):sha256(p) for p in code_paths}
    git=subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,text=True,capture_output=True)
    report={
        'schema_version':1,'mode':mode,'provisional':mode=='bootstrap',
        'calibration_method':calibration['method'] if calibration else None,
        'tag_use':'disabled by baseline-only selection' if baseline_only else 'capped soft evidence',
        'quality_claim':'Unvalidated on hidden competition ground truth',
        'configuration':asdict(config),'match':asdict(match),
        'inputs':{key:{'filename':path.name,'sha256':hashes[key]} for key,path in required.items()},
        'missing_optional':missing,'homography_use':'validated only; no possession evidence',
        'source_sha256':code_hashes,'git_commit':git.stdout.strip() if git.returncode==0 else None,
        'runtime':{'python':platform.python_version(),'numpy':np.__version__},
        'label_counts':{label:int(np.sum(labels==label)) for label in ('H','A','D','X')},
    }
    out.parent.mkdir(parents=True,exist_ok=True)
    staging=Path(tempfile.mkdtemp(prefix='.possession-',dir=out.parent))
    try:
        submission=staging/'submission.csv'
        with submission.open('w',encoding='utf-8',newline='\n') as handle:
            handle.write('frame,label\n')
            handle.writelines(f'{i},{label}\n' for i,label in enumerate(labels))
        report['output_sha256']=sha256(submission)
        (staging/'manifest.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n',encoding='utf-8')
        # Recheck after long computation; never replace a preexisting run.
        if out.exists():raise FileExistsError('output directory was created during inference')
        os.rename(staging,out)
    finally:
        if staging.exists():shutil.rmtree(staging)
    return report
