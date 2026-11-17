"""Fixed match-level external model selection; never loads competition data.

python -m experiments.calibrate fit --data-dir data/external/sportec --out artifacts/external-fit
python -m experiments.calibrate test --data-dir data/external/sportec --out artifacts/external-test --selection artifacts/external-fit/selection.json
"""
import argparse
from dataclasses import asdict,replace
import json
from pathlib import Path
import numpy as np
from evaluation import possession_metrics as pm
from starter.baseline import smooth
from solution.config import Config
from solution.evidence import bootstrap_evidence
from solution.decode import decode
from solution.temporal import temporal_features,fit,calibrated_evidence
from solution.pipeline import sha256

TRAIN='J03WOH'
VALIDATION='J03WOY'
TEST='J03WPY'
SCENARIOS=('clean','masked')


def choose_candidate(candidates):
    return max(candidates,key=lambda row:sum(row['scores'][s]['spellF1'] for s in SCENARIOS)/len(SCENARIOS))


def score(truth,prediction,blocks):
    conversion={'H':0,'A':1,'D':2,'X':3}
    gt=np.array([conversion[x] for x in truth],dtype=np.int8)
    pred=np.array([conversion[x] for x in prediction],dtype=np.int8)
    runs=[]
    for start,stop in blocks:
        runs.extend((a+start,b+start) for a,b in pm.valid_runs(gt[start:stop]))
    filled=pm.fill_abstentions(pred,runs)
    spell=pm.spell_scores(gt,filled,runs)
    valid=gt!=3
    f1=[]
    for side in (0,1,2):
        tp=np.sum((gt==side)&(pred==side)&valid)
        fp=np.sum((gt!=side)&(pred==side)&valid)
        fn=np.sum((gt==side)&(pred!=side)&valid)
        f1.append(float(2*tp/(2*tp+fp+fn)) if 2*tp+fp+fn else 0.)
    return {'spellF1':spell['mean_f1'],'macroF1':float(np.mean(f1)),'frames':int(valid.sum())}


def load_match(root,game):
    path=Path(root)/(game+'-benchmark.npz')
    with np.load(path,allow_pickle=False) as data:
        return {key:data[key].copy() for key in data.files}


def predict(sequence,blocks,selection):
    method=selection['method'];config=Config(**selection['configuration'])
    if method in ('baseline_raw','baseline_smooth50'):
        result=np.full(len(sequence),'X',dtype='<U1')
        codes=np.where(sequence=='H',0,np.where(sequence=='A',1,-1))
        for a,b in blocks:result[a:b]=smooth(codes[a:b],50 if method=='baseline_smooth50' else 0)
        return result
    if method=='heuristic':e=bootstrap_evidence(sequence,blocks,config)
    elif method=='temporal':e=calibrated_evidence(sequence,blocks,config,selection['model'])
    else:raise ValueError('unknown candidate method')
    return decode(e,blocks,config)


def benchmark(data,selection):
    blocks=[tuple(map(int,x)) for x in data['blocks']]
    return {scenario:score(data['labels'],predict(data[scenario],blocks,selection),blocks) for scenario in SCENARIOS}


def train_select(root,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    train=load_match(root,TRAIN);validation=load_match(root,VALIDATION)
    blocks=[tuple(map(int,x)) for x in train['blocks']]
    conversion={'H':0,'A':1,'D':2,'X':3}
    truth=np.array([conversion[x] for x in train['labels']])
    # Fixed systematic subsampling reduces fitting cost, and includes both scenarios equally.
    index=np.arange(0,len(truth),5);index=index[truth[index]!=3]
    x=np.concatenate([temporal_features(train[s],blocks,25)[index] for s in SCENARIOS])
    y=np.tile(truth[index],len(SCENARIOS))
    base=Config();rows=[];selections={}
    for method in ('baseline_raw','baseline_smooth50','heuristic'):
        selections[method]={'method':method,'configuration':asdict(base)}
    for regularization in (0.001,0.01,0.1):
        model=fit(x,y,regularization=regularization,steps=300)
        for transition in (6.,18.):
            name=f'temporal_l2_{regularization}_switch_{transition:g}'
            selections[name]={'method':'temporal','configuration':asdict(replace(base,transition_cost=transition)),'model':model}
    for name,selection in selections.items():
        row={'name':name,'scores':benchmark(validation,selection)}
        rows.append(row);print(json.dumps(row),flush=True)
    best=choose_candidate(rows)
    selected=selections[best['name']]
    selected.update({'name':best['name'],'training_match':TRAIN,'validation_match':VALIDATION,
                     'held_out_match':TEST,'selection_rule':'mean spellF1 of clean and masked validation scenarios',
                     'dataset':'https://doi.org/10.6084/m9.figshare.28196177','license':'CC BY 4.0',
                     'benchmark_sha256':{game:sha256(Path(root)/(game+'-benchmark.npz')) for game in (TRAIN,VALIDATION)},
                     'validation':best['scores']})
    (out/'selection.json').write_text(json.dumps(selected,indent=2,sort_keys=True)+'\n')
    (out/'selected-config.json').write_text(json.dumps(selected['configuration'],indent=2,sort_keys=True)+'\n')
    (out/'validation.json').write_text(json.dumps(rows,indent=2)+'\n')
    print('SELECTED',best['name'],flush=True)


def held_out(root,out,selection_path):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    selection=json.loads(Path(selection_path).read_text())
    if (selection['training_match'],selection['validation_match'],selection['held_out_match'])!=(TRAIN,VALIDATION,TEST):
        raise ValueError('selection does not have the fixed external split')
    data=load_match(root,TEST)
    rows=[]
    for method in ('baseline_raw','baseline_smooth50','heuristic'):
        candidate={'method':method,'configuration':asdict(Config())}
        rows.append({'name':method,'scores':benchmark(data,candidate)})
    rows.append({'name':selection['name'],'scores':benchmark(data,selection)})
    report={'test_match':TEST,'selection_sha256':sha256(selection_path),'benchmark_sha256':sha256(Path(root)/(TEST+'-benchmark.npz')),'results':rows}
    (out/'held-out.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('fit','test'))
    parser.add_argument('--data-dir',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--selection',type=Path)
    args=parser.parse_args()
    if args.stage=='fit':train_select(args.data_dir,args.out)
    else:
        if args.selection is None:parser.error('--selection required for held-out test')
        held_out(args.data_dir,args.out,args.selection)

if __name__=='__main__':main()
