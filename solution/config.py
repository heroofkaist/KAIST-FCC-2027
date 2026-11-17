"""Fixed, interpretable parameters. Never tune against target-match human labels."""
from dataclasses import dataclass, fields
import json
import math

@dataclass(frozen=True)
class Config:
    fps: float = 25.0
    control_radius_m: float = 2.5
    control_margin_m: float = 1.5
    control_weight: float = 2.0
    ball_speed_limit_mps: float = 45.0
    interpolation_seconds: float = 0.4
    boundary_margin_m: float = 0.75
    outside_weight: float = 2.0
    tag_weight: float = 0.7
    bootstrap_weight: float = 1.0
    bootstrap_dead_weight: float = 0.65
    bootstrap_gap_seconds: float = 1.5
    transition_cost: float = 12.0
    short_segment_cost: float = 6.0
    mature_frames: int = 25

    def __post_init__(self):
        for field in fields(self):
            value=getattr(self,field.name)
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):
                raise ValueError(f'{field.name} must be finite numeric')
            if value < 0:
                raise ValueError(f'{field.name} cannot be negative')
        for key in ('fps','control_radius_m','control_margin_m','ball_speed_limit_mps','bootstrap_gap_seconds'):
            if getattr(self,key)<=0: raise ValueError(f'{key} must be positive')
        if not isinstance(self.mature_frames,int) or not 1<=self.mature_frames<=250:
            raise ValueError('mature_frames must be an integer in 1..250')


def load_config(path):
    with open(path,encoding='utf-8') as f: values=json.load(f)
    if not isinstance(values,dict): raise ValueError('config must be an object')
    unknown=set(values)-{f.name for f in fields(Config)}
    if unknown: raise ValueError(f'unknown configuration fields: {sorted(unknown)}')
    return Config(**values)
