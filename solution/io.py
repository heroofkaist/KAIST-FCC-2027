"""Strict local input readers. Errors never echo restricted input rows."""
from dataclasses import dataclass
from pathlib import Path
import csv
import math
import xml.etree.ElementTree as ET
import numpy as np

@dataclass
class Tracking:
    ball: np.ndarray
    players: np.ndarray
    team: np.ndarray
    speed: np.ndarray
    team_names: tuple
    pitch: tuple

@dataclass(frozen=True)
class Tag:
    start: int
    stop: int
    team: int
    kind: str


def xml_root(path):
    root=ET.parse(path).getroot()
    for element in root.iter():
        element.tag=element.tag.rsplit('}',1)[-1]
    return root


def read_metadata(path):
    root=xml_root(path)
    teams=list(root.iter('Team'))
    ids=[t.get('id') for t in teams]
    names=tuple(t.findtext('Name','').strip() for t in teams)
    if len(ids)!=2 or None in ids or len(set(ids))!=2 or any(not n for n in names) or names[0].casefold()==names[1].casefold():
        raise ValueError('metadata must contain two distinct teams with IDs and names')
    slot_ids=[p.get('teamId') for p in root.iter('Player')]
    if not slot_ids or any(t not in ids for t in slot_ids) or set(slot_ids)!=set(ids):
        raise ValueError('metadata player slots must belong to the two known teams')
    return np.array([ids.index(t) for t in slot_ids],dtype=np.int8),names


def read_tracking(meta_path,raw_path,n_frames,pitch):
    if n_frames<=0: raise ValueError('expected frame count must be positive')
    if len(pitch)!=2 or not all(math.isfinite(x) and x>0 for x in pitch):
        raise ValueError('pitch must have two positive finite dimensions')
    team,names=read_metadata(meta_path)
    players=np.full((n_frames,len(team),2),np.nan,dtype=np.float32)
    ball=np.full((n_frames,2),np.nan,dtype=np.float32)
    speed=np.full((n_frames,len(team)),np.nan,dtype=np.float32)
    count=0
    with open(raw_path,encoding='utf-8') as f:
        for line_number,line in enumerate(f,1):
            if line_number>n_frames: raise ValueError('tracking has extra rows')
            parts=line.strip().split(':')
            if len(parts)!=3: raise ValueError(f'tracking row {line_number}: expected three sections')
            try: frame=int(parts[0])
            except ValueError: raise ValueError(f'tracking row {line_number}: invalid frame ID') from None
            if frame!=line_number: raise ValueError(f'tracking row {line_number}: frame IDs must be exactly 1..N')
            groups=parts[1].split(';')+[parts[2]]
            if len(groups)!=len(team)+1: raise ValueError(f'tracking row {line_number}: wrong slot count')
            try:
                values=np.array([[float(x) for x in group.split(',')] for group in groups],dtype=np.float64)
            except ValueError: raise ValueError(f'tracking row {line_number}: invalid numeric fields') from None
            if values.shape!=(len(team)+1,3) or np.isinf(values).any():
                raise ValueError(f'tracking row {line_number}: invalid shape or infinite value')
            xy=values[:,:2]
            if np.any(np.isnan(xy[:,0])!=np.isnan(xy[:,1])):
                raise ValueError(f'tracking row {line_number}: partially missing coordinates')
            if np.any(np.isfinite(values[:,2]) & (values[:,2]<0)):
                raise ValueError(f'tracking row {line_number}: negative speed')
            scaled=xy*np.asarray(pitch)
            if np.any(np.isfinite(scaled) & (np.abs(scaled)>10000)):
                raise ValueError(f'tracking row {line_number}: implausible coordinate magnitude')
            players[line_number-1]=scaled[:-1]
            ball[line_number-1]=scaled[-1]
            speed[line_number-1]=values[:-1,2]
            count=line_number
    if count!=n_frames: raise ValueError(f'incomplete tracking: {count} rows, expected {n_frames}')
    return Tracking(ball,players,team,speed,names,tuple(pitch))


def read_baseline(path,n_frames):
    p=Path(path)
    if any(part.lower()=='gt' or 'possession_gt' in part.lower() for part in p.resolve().parts):
        raise ValueError('ground truth cannot be used as algorithmic baseline evidence')
    labels=[]
    with p.open(newline='',encoding='utf-8') as f:
        reader=csv.reader(f)
        if next(reader,None)!=['frame','label']: raise ValueError('baseline header must be frame,label')
        for index,row in enumerate(reader):
            if len(row)!=2 or row[0]!=str(index) or row[1] not in ('H','A','D','X'):
                raise ValueError(f'baseline row {index}: invalid frame or label')
            labels.append(row[1])
            if len(labels)>n_frames: raise ValueError('baseline has extra rows')
    if len(labels)!=n_frames: raise ValueError('baseline has wrong row count')
    return np.asarray(labels,dtype='<U1')


def read_tags(path,n_frames,fps,team_names):
    attack={'build up','progression','creating chances','attacking transition','long ball','ball in final third','ball in the box','players in the box'}
    defense={'defending in attacking third','defending in middle third','defending in defensive third','defensive transition'}
    names={name.casefold():i for i,name in enumerate(team_names)}
    result=[]
    for event in xml_root(path).iter('instance'):
        try: start=float(event.findtext('start'));stop=float(event.findtext('end'))
        except (TypeError,ValueError): raise ValueError('tag contains invalid timing') from None
        if not math.isfinite(start) or not math.isfinite(stop) or not 0<=start<stop<=n_frames/fps+0.5/fps:
            raise ValueError('tag timing outside video or reversed')
        teams={(item.findtext('text') or '').strip().casefold() for item in event.findall('label') if (item.findtext('group') or '').strip().casefold()=='team'}
        if len(teams)>1: raise ValueError('tag contains contradictory Team labels')
        team=names.get(next(iter(teams),''),-1)
        code=(event.findtext('code') or '').strip().casefold()
        kind='attack' if code in attack else 'defense' if code in defense else 'restart' if code=='set pieces' else 'ignore'
        a,b=round(start*fps),min(n_frames,round(stop*fps))
        if a<b: result.append(Tag(a,b,team,kind))
    return result


def validate_homography(path,n_frames):
    columns=['frame']+[f'h{r}{c}' for r in range(3) for c in range(3)]+['reprojection_error','valid']
    count=0
    with open(path,newline='',encoding='utf-8') as f:
        reader=csv.DictReader(f)
        if not set(columns).issubset(reader.fieldnames or []): raise ValueError('homography missing required columns')
        for index,row in enumerate(reader):
            if index>=n_frames or row['frame']!=str(index) or row['valid'] not in ('0','1'):
                raise ValueError('homography frame sequence or valid flag invalid')
            try: values=np.array([float(row[c]) for c in columns[1:-1]])
            except (TypeError,ValueError): raise ValueError('homography numeric value invalid') from None
            if row['valid']=='1' and (not np.isfinite(values).all() or values[-1]<0):
                raise ValueError('valid homography contains invalid numeric values')
            count+=1
    if count!=n_frames: raise ValueError('homography has wrong row count')
