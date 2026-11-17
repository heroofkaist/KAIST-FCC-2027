"""Exact age-expanded dynamic program with soft short-segment penalties.

Maximizes observation rewards minus one switch cost per transition and
short_segment_cost * max(0, mature_frames - duration) / mature_frames
for EVERY segment, including the final segment. Complexity O(N * 3 * K).
"""
import numpy as np
from .evidence import validate_blocks


def _block(scores,config):
    n=len(scores);k=config.mature_frames
    ages=np.arange(1,k+1)
    penalty=config.short_segment_cost*(k-ages)/k
    previous=np.full((3,k),-np.inf)
    previous[:,0]=scores[0]
    back=np.zeros((n,3,k),dtype=np.uint16)
    forward=np.empty((n,3,k),dtype=np.float64)
    forward[0]=previous
    flat=np.arange(3*k).reshape(3,k)
    for t in range(1,n):
        current=np.full((3,k),-np.inf)
        for state in range(3):
            exit_values=previous-penalty
            exit_values[state]=-np.inf
            predecessor=int(np.argmax(exit_values))
            current[state,0]=exit_values.flat[predecessor]-config.transition_cost
            back[t,state,0]=predecessor
            if k>1:
                current[state,1:]=previous[state,:-1]
                back[t,state,1:]=flat[state,:-1]
            # The mature bucket can continue forever; for K=1 this competes with a switch.
            if previous[state,-1]>=current[state,-1]:
                current[state,-1]=previous[state,-1]
                back[t,state,-1]=flat[state,-1]
        previous=current+scores[t,:,None]
        forward[t]=previous
    optimum=float(np.max(previous-penalty))
    state=int(np.argmax(previous-penalty))
    path=np.empty(n,dtype=np.int8)
    for t in range(n-1,-1,-1):
        path[t]=state//k
        state=int(back[t].flat[state])
    labels=np.array(['H','A','D'])[path]
    # Max-marginals identify ALL optimal ownership alternatives, including ties
    # in only part of a half and different paths with equal total reward.
    # Comparing just the local emissions or entire-block columns misses these.
    suffix=np.broadcast_to(-penalty,(3,k)).copy()
    next_age=np.minimum(np.arange(k)+1,k-1)
    for t in range(n-1,-1,-1):
        marginal=np.max(forward[t]+suffix,axis=1)
        if labels[t]!='D' and abs(marginal[0]-optimum)<1e-8 and abs(marginal[1]-optimum)<1e-8:
            labels[t]='X'
        if t:
            previous_suffix=np.empty((3,k),dtype=np.float64)
            for current in range(3):
                stay=scores[t,current]+suffix[current,next_age]
                switch=max(scores[t,other]+suffix[other,0] for other in range(3) if other!=current)-config.transition_cost
                previous_suffix[current]=np.maximum(stay,switch-penalty)
            suffix=previous_suffix
    return labels


def decode(evidence,blocks,config):
    scores=np.asarray(evidence.scores,dtype=np.float64)
    informative=np.asarray(evidence.informative)
    if scores.ndim!=2 or scores.shape[1]!=3 or informative.shape!=(len(scores),):
        raise ValueError('evidence must be N x 3 scores and N informative flags')
    if not np.isfinite(scores).all(): raise ValueError('evidence scores must be finite')
    validate_blocks(blocks,len(scores))
    labels=np.full(len(scores),'X',dtype='<U1')
    for start,stop in blocks:
        if not informative[start:stop].any(): continue
        block=scores[start:stop]
        path=_block(block,config)
        labels[start:stop]=path
    return labels
