"""Bounded observation rewards, not calibrated probabilities."""
from dataclasses import dataclass
import numpy as np

@dataclass
class Evidence:
    scores: np.ndarray
    informative: np.ndarray
    stationary: np.ndarray | None = None


def validate_blocks(blocks,n):
    previous=0
    for start,stop in blocks:
        if not isinstance(start,int) or not isinstance(stop,int) or not 0<=previous<=start<stop<=n:
            raise ValueError('blocks must be ordered, disjoint, nonempty half-open frame intervals')
        previous=stop


def runs(mask):
    edges=np.flatnonzero(np.diff(np.r_[False,mask,False]))
    return zip(edges[::2],edges[1::2])


def tracking_evidence(tracking,blocks,config,barriers=()):
    ball=tracking.ball.astype(np.float64).copy()
    n=len(ball);validate_blocks(blocks,n)
    blocked=np.zeros(n,dtype=bool)
    for start,stop in barriers: blocked[max(0,start):min(n,stop)]=True
    observed=np.isfinite(ball).all(axis=1)
    confidence=observed.astype(float)
    length,width=tracking.pitch
    for start,stop in blocks:
        # Reject isolated impossible jumps; do not derive player velocities across track slots.
        b=ball[start:stop]
        if len(b)>2:
            jumps=np.linalg.norm(np.diff(b,axis=0),axis=1)*config.fps>config.ball_speed_limit_mps
            spikes=np.flatnonzero(jumps[:-1]&jumps[1:])+start+1
            ball[spikes]=np.nan;confidence[spikes]=0
        for a,b in runs(~np.isfinite(ball[start:stop]).all(axis=1)):
            a+=start;b+=start
            if a==start or b==stop or b-a>round(config.interpolation_seconds*config.fps) or blocked[a-1:b+1].any(): continue
            left,right=ball[a-1],ball[b]
            if not (0<=left[0]<=length and 0<=right[0]<=length and 0<=left[1]<=width and 0<=right[1]<=width): continue
            if np.linalg.norm(right-left)*config.fps/(b-a+1)>config.ball_speed_limit_mps: continue
            ball[a:b]=left+(right-left)*np.arange(1,b-a+1)[:,None]/(b-a+1)
            confidence[a:b]=0.5
    visible=np.isfinite(ball).all(axis=1)
    scores=np.zeros((n,3),dtype=float)
    distances=np.linalg.norm(tracking.players.astype(float)-ball[:,None,:],axis=2)
    distances[~np.isfinite(distances)]=np.inf
    nearest=np.stack([distances[:,tracking.team==side].min(axis=1) for side in (0,1)],axis=1)
    for side in (0,1):
        own=nearest[:,side];other=nearest[:,1-side]
        advantage=np.zeros(n)
        both=np.isfinite(own)&np.isfinite(other)
        advantage[both]=np.clip((other[both]-own[both])/config.control_margin_m,0,1)
        advantage[np.isfinite(own)&~np.isfinite(other)]=0.25
        proximity=np.exp(-0.5*(own/config.control_radius_m)**2)
        scores[:,side]+=config.control_weight*proximity*advantage*confidence
    inside=visible&(ball[:,0]>=-config.boundary_margin_m)&(ball[:,0]<=length+config.boundary_margin_m)&(ball[:,1]>=-config.boundary_margin_m)&(ball[:,1]<=width+config.boundary_margin_m)
    scores[inside,:2]+=0.15*confidence[inside,None]
    scores[inside,2]-=0.15*confidence[inside]
    outside=visible&~inside
    scores[outside,:2]=0
    scores[outside,2]=config.outside_weight*confidence[outside]
    stationary=np.zeros(n,dtype=bool)
    for start,stop in blocks:
        speed=np.linalg.norm(np.diff(ball[start:stop],axis=0),axis=1)*config.fps
        stationary[start+1:stop]=np.isfinite(speed)&(speed<0.5)&(confidence[start+1:stop]==1)&(confidence[start:stop-1]==1)
    return Evidence(scores,visible,stationary)


def bootstrap_evidence(labels,blocks,config):
    n=len(labels);validate_blocks(blocks,n)
    if not np.isin(labels,['H','A','D','X']).all(): raise ValueError('unknown bootstrap labels')
    scores=np.zeros((n,3),float)
    informative=np.isin(labels,['H','A'])
    for side,label in enumerate(('H','A')):
        selected=labels==label
        scores[selected,side]=config.bootstrap_weight
        scores[selected,1-side]=-0.5*config.bootstrap_weight
        scores[selected,2]=-0.3*config.bootstrap_weight
    # D in the raw baseline means unavailable nearest-team evidence, not observed stoppage.
    # Only the interior of sustained D runs gets a WEAK stoppage prior.
    threshold=config.bootstrap_gap_seconds*config.fps
    for start,stop in blocks:
        for a,b in runs(labels[start:stop]=='D'):
            a+=start;b+=start
            depth=np.minimum(np.arange(1,b-a+1),np.arange(b-a,0,-1))
            reward=config.bootstrap_dead_weight*np.clip((depth-threshold)/threshold,0,1)
            scores[a:b,2]=reward
            informative[a:b]|=reward>0
    return Evidence(scores,informative)


def add_tags(evidence,tags,config):
    n=len(evidence.scores)
    support=np.zeros((n,2),float)
    restart=np.zeros(n,dtype=bool)
    for tag in tags:
        if not 0<=tag.start<tag.stop<=n: raise ValueError('tag interval outside evidence')
        if tag.kind=='restart':
            restart[tag.start:tag.stop]=True
            continue
        if tag.team not in (0,1) or tag.kind not in ('attack','defense'): continue
        side=1-tag.team if tag.kind=='defense' else tag.team
        support[tag.start:tag.stop,side]=1.0
    difference=support[:,0]-support[:,1]
    scores=evidence.scores.copy()
    scores[:,0]+=difference*config.tag_weight
    scores[:,1]-=difference*config.tag_weight
    # Consistent phase evidence weakly favours live play. Conflicts remain neutral.
    scores[:,2]-=np.abs(difference)*config.tag_weight*0.25
    informative=evidence.informative.copy()|(difference!=0)
    if evidence.stationary is not None:
        stopped=restart&evidence.stationary
        scores[stopped,2]+=config.outside_weight
        informative|=stopped
    return Evidence(scores,informative,evidence.stationary)
