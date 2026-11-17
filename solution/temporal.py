"""Small supervised temporal model. Training is for EXTERNAL matches only."""
import json
from pathlib import Path
import numpy as np
from .evidence import validate_blocks, runs

WINDOW_SECONDS=(0.,0.2,1.,3.,6.)
FEATURES=('bias',)+tuple(f'{side}_fraction_{window:g}s' for window in WINDOW_SECONDS for side in ('home','away','missing'))+('missing_depth','previous_home','previous_away','next_home','next_away','same_owner_across_gap')
SWAP=np.array([FEATURES.index(name.replace('home','TEMP').replace('away','home').replace('TEMP','away')) for name in FEATURES])


def temporal_features(labels,blocks,fps):
    labels=np.asarray(labels)
    if labels.ndim!=1 or not np.isin(labels,['H','A','D','X']).all():raise ValueError('invalid temporal observations')
    if not np.isfinite(fps) or fps<=0:raise ValueError('fps must be positive')
    n=len(labels);validate_blocks(blocks,n)
    x=np.zeros((n,len(FEATURES)),dtype=np.float64);x[:,0]=1
    for start,stop in blocks:
        sequence=labels[start:stop];m=len(sequence)
        for w,seconds in enumerate(WINDOW_SECONDS):
            radius=round(seconds*fps)
            low=np.maximum(0,np.arange(m)-radius);high=np.minimum(m,np.arange(m)+radius+1)
            for side,label in enumerate(('H','A','D')):
                cumulative=np.r_[0,np.cumsum(sequence==label)]
                x[start:stop,1+3*w+side]=(cumulative[high]-cumulative[low])/(high-low)
        previous=np.full(m,-1,int);following=np.full(m,-1,int)
        known=-1
        for i,label in enumerate(sequence):
            if label in ('H','A'):known=0 if label=='H' else 1
            previous[i]=known
        known=-1
        for i in range(m-1,-1,-1):
            label=sequence[i]
            if label in ('H','A'):known=0 if label=='H' else 1
            following[i]=known
        for a,b in runs(sequence=='D'):
            depth=np.minimum(np.arange(1,b-a+1),np.arange(b-a,0,-1))/fps
            x[start+a:start+b,16]=np.log1p(np.minimum(depth,30))/np.log(31)
        x[start:stop,17]=(previous==0);x[start:stop,18]=(previous==1)
        x[start:stop,19]=(following==0);x[start:stop,20]=(following==1)
        x[start:stop,21]=(previous==following)&(previous>=0)&np.isin(sequence,['D','X'])
    return x


def swap_features(x):return np.asarray(x)[:,SWAP]


def _weights(model):
    if model.get('feature_names')!=list(FEATURES) or model.get('version')!=1:
        raise ValueError('unsupported temporal model schema')
    w=np.asarray(model.get('weights'),dtype=float)
    if w.shape!=(len(FEATURES),3) or not np.isfinite(w).all():raise ValueError('invalid temporal model weights')
    return w


def fit(x,y,regularization=0.01,steps=300):
    x=np.asarray(x,dtype=float);y=np.asarray(y)
    if x.ndim!=2 or x.shape[1]!=len(FEATURES) or not np.isfinite(x).all() or len(x)==0:
        raise ValueError('invalid training features')
    if y.shape!=(len(x),) or not np.isin(y,[0,1,2]).all():raise ValueError('training labels must be H/A/D codes')
    if not np.isfinite(regularization) or regularization<0 or not isinstance(steps,int) or steps<1:
        raise ValueError('invalid training settings')
    y=y.astype(int)
    # Projection each step enforces exact team exchange symmetry, avoiding team memorization.
    w=np.zeros((x.shape[1],3));moment=np.zeros_like(w);variance=np.zeros_like(w)
    counts=np.bincount(y,minlength=3)
    class_weights=1/np.sqrt(np.maximum(counts,1));sample_weights=class_weights[y]
    sample_weights/=sample_weights.mean()
    for step in range(1,steps+1):
        logits=x@w;logits-=logits.max(axis=1,keepdims=True)
        probability=np.exp(logits);probability/=probability.sum(axis=1,keepdims=True)
        probability[np.arange(len(x)),y]-=1
        gradient=x.T@(probability*sample_weights[:,None])/len(x)
        penalty=w.copy();penalty[0]=0
        gradient+=regularization*penalty
        gradient=0.5*(gradient+gradient[SWAP][:,[1,0,2]])
        moment=0.9*moment+0.1*gradient;variance=0.999*variance+0.001*gradient**2
        w-=0.04*(moment/(1-0.9**step))/(np.sqrt(variance/(1-0.999**step))+1e-8)
        w=0.5*(w+w[SWAP][:,[1,0,2]])
    return {'version':1,'feature_names':list(FEATURES),'weights':w.tolist(),'regularization':regularization,'steps':steps}


def predict_scores(x,model):
    w=_weights(model);x=np.asarray(x,dtype=float)
    if x.ndim!=2 or x.shape[1]!=len(FEATURES) or not np.isfinite(x).all():raise ValueError('invalid inference features')
    logits=x@w
    logits-=logits.max(axis=1,keepdims=True)
    return logits-np.log(np.exp(logits).sum(axis=1,keepdims=True))


def save_model(path,model):
    _weights(model)
    Path(path).write_text(json.dumps(model,indent=2,sort_keys=True,allow_nan=False)+'\n',encoding='utf-8')


def load_model(path):
    model=json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(model,dict):raise ValueError('model must be an object')
    _weights(model)
    return model


def calibrated_evidence(labels,blocks,config,model):
    from .evidence import Evidence
    features=temporal_features(labels,blocks,config.fps)
    return Evidence(predict_scores(features,model),np.isin(labels,['H','A','D']))
