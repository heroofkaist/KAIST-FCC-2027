"""Streaming adapter for the CC BY 4.0 Sportec/DFL dataset (Bassek et al., 2025).

Human ball status/ownership are returned SEPARATELY from observation features.
Coordinates are centered metres; speed is km/h in the source.
"""
from dataclasses import dataclass
import xml.etree.ElementTree as ET
import numpy as np
from solution.io import Tracking, xml_root

@dataclass
class ExternalMatch:
    tracking: Tracking
    labels: np.ndarray
    blocks: list
    height: np.ndarray


def _frames(path):
    current=None
    for event,element in ET.iterparse(path,events=('start','end')):
        tag=element.tag.rsplit('}',1)[-1]
        if event=='start' and tag=='FrameSet':current=dict(element.attrib)
        elif event=='end' and tag=='Frame':
            if current is not None:yield current,element.attrib
            element.clear()
        elif event=='end' and tag=='FrameSet':
            current=None;element.clear()


def read_sportec(meta_path,raw_path):
    root=xml_root(meta_path);general=root.find('.//General');environment=root.find('.//Environment')
    if general is None or environment is None:raise ValueError('Sportec metadata lacks General/Environment')
    team_ids=[general.get('HomeTeamId'),general.get('GuestTeamId')]
    if None in team_ids or len(set(team_ids))!=2:raise ValueError('invalid Sportec team IDs')
    names=(general.get('HomeTeamName'),general.get('GuestTeamName'))
    pitch=(float(environment.get('PitchX')),float(environment.get('PitchY')))
    if not all(np.isfinite(v) and v>0 for v in pitch):raise ValueError('invalid Sportec pitch')
    periods={'firstHalf':1,'secondHalf':2,'firstHalfExtra':3,'secondHalfExtra':4}
    slots={};records={}
    for frameset,frame in _frames(raw_path):
        section=frameset.get('GameSection')
        if section not in periods:continue
        team=frameset.get('TeamId');person=frameset.get('PersonId')
        if team in team_ids:
            slots.setdefault((team,person),len(slots))
        elif team=='BALL':
            key=(periods[section],int(frame['N']))
            if key in records:raise ValueError('duplicate Sportec ball frame')
            status=frame.get('BallStatus');owner=frame.get('BallPossession')
            if status not in ('0','1'):raise ValueError('unknown Sportec ball status')
            label='D' if status=='0' else {'1':'H','2':'A'}.get(owner,'X')
            records[key]=(float(frame['X'])+pitch[0]/2,float(frame['Y'])+pitch[1]/2,float(frame.get('Z','nan')),label)
    if not records or not slots:raise ValueError('Sportec file missing ball or player records')
    keys=sorted(records);index={key:i for i,key in enumerate(keys)};n=len(keys)
    ball=np.array([records[key][:2] for key in keys],dtype=np.float32)
    height=np.array([records[key][2] for key in keys])
    labels=np.array([records[key][3] for key in keys])
    players=np.full((n,len(slots),2),np.nan,dtype=np.float32)
    speed=np.full((n,len(slots)),np.nan,dtype=np.float32)
    team=np.array([team_ids.index(key[0]) for key in slots],dtype=np.int8)
    if set(team)!={0,1}:raise ValueError('Sportec requires players from both teams')
    for frameset,frame in _frames(raw_path):
        slot=slots.get((frameset.get('TeamId'),frameset.get('PersonId')))
        section=periods.get(frameset.get('GameSection'))
        if slot is None or section is None:continue
        i=index.get((section,int(frame['N'])))
        if i is None:continue
        if np.isfinite(players[i,slot]).any():raise ValueError('duplicate Sportec player frame')
        players[i,slot]=[float(frame['X'])+pitch[0]/2,float(frame['Y'])+pitch[1]/2]
        speed[i,slot]=float(frame.get('S','nan'))/3.6
    blocks=[];start=0
    for i in range(1,n):
        if keys[i][0]!=keys[i-1][0] or keys[i][1]!=keys[i-1][1]+1:
            blocks.append((start,i));start=i
    blocks.append((start,n))
    if np.isinf(ball).any() or np.isinf(players).any():raise ValueError('infinite Sportec coordinates')
    return ExternalMatch(Tracking(ball,players,team,speed,names,pitch),labels,blocks,height)


def nearest_labels(tracking):
    distance=np.linalg.norm(tracking.players-tracking.ball[:,None,:],axis=2)
    distance[~np.isfinite(distance)]=np.inf
    slot=distance.argmin(axis=1)
    labels=np.where(tracking.team[slot]==0,'H','A')
    labels[~np.isfinite(distance.min(axis=1))]='D'
    return labels


def mask_observations(match,seed=42,fps=25):
    """A stress scenario, not a fitted model of the competition camera.

    Ignore GT entirely. Mask outside-field/high-ball observations plus seeded
    one-second blocks (35%) to simulate correlated tracking loss.
    """
    t=match.tracking;labels=nearest_labels(t);n=len(labels)
    rng=np.random.default_rng(seed)
    for start,stop in match.blocks:
        length=stop-start
        missing=np.repeat(rng.random((length+fps-1)//fps)<0.35,fps)[:length]
        labels[start:stop][missing]='D'
    outside=(t.ball[:,0]<0)|(t.ball[:,0]>t.pitch[0])|(t.ball[:,1]<0)|(t.ball[:,1]>t.pitch[1])
    labels[outside|(match.height>2.0)]='D'
    return labels
