from dataclasses import dataclass
from time import perf_counter
from ..ai_search.config import GeometryConfig, SearchConfig
from ..ai_search.search import BudgetStop

@dataclass(frozen=True)
class TaskConfig:
    seconds: float = 3.
    ray_pair_limit: int = 262144
    geometry_limit: int = 256
    candidate_limit: int = 32
    active_limit: int = 64
    starts: int = 2
    center_iterations: int = 1
    def __post_init__(self):
        if self.seconds < 0 or not __import__('math').isfinite(self.seconds):raise ValueError('seconds')
        for f in ('ray_pair_limit','geometry_limit','candidate_limit','active_limit','starts','center_iterations'):
            if type(getattr(self,f)) is not int or getattr(self,f)<0:raise ValueError(f)
        if self.active_limit>64 or self.starts>2 or self.center_iterations>1:raise ValueError('D1 finite mechanism cap')
    def to_record(self):return __import__('dataclasses').asdict(self)
    @classmethod
    def from_record(cls,record):return cls(**record)

class Ledger:
    """Charge before work. Limits are inclusive; clock overrun is descriptive."""
    def __init__(self,config=TaskConfig(),clock=None,stop=None):
        self.config=config;self.clock=clock or perf_counter;self.start=self.clock();self.deadline=self.start+config.seconds
        self.attempts=self.geometry_calls=self.ray_pair_checks=self.late_candidates=0
        self.complete_candidates=0
        self.reason=None;self.stop=stop;self.events=[]
        self.ray_cache={};self.ray_cache_hits=0;self.rays_constructed=0;self.ray_cache_lookup_seconds=0.
        self.work_counts={}
        self.construction_operations={}
    @property
    def elapsed(self):return self.clock()-self.start
    def check(self):
        reason='stop_requested' if self.stop and self.stop() else 'time_budget' if self.clock()>=self.deadline else None
        if reason:self.reason=reason;raise BudgetStop(reason)
    def geometry(self,pairs,phase='geometry'):
        self.check()
        if type(pairs) is not int or pairs<0:raise ValueError('pairs')
        reason='geometry_budget' if self.geometry_calls+1>self.config.geometry_limit else 'ray_pair_budget' if self.ray_pair_checks+pairs>self.config.ray_pair_limit else None
        if reason:self.reason=reason;raise BudgetStop(reason)
        self.geometry_calls+=1;self.ray_pair_checks+=pairs
        self.events.append({'phase':phase,'pairs':pairs,'at_seconds':self.elapsed,'cumulative_ray_pair_checks':self.ray_pair_checks,'cumulative_geometry_calls':self.geometry_calls,'charge_before':True})
    def attempt(self):
        self.check()
        if self.attempts+1>self.config.candidate_limit:self.reason='candidate_budget';raise BudgetStop(self.reason)
        self.attempts+=1
    def center_rays(self,centers,profile):
        import numpy as np
        from ..geometry import satellite_rays_numpy
        self.check();lookup_start=self.clock();centers=np.asarray(centers,dtype=np.float64);key=(profile,centers.shape,centers.tobytes())
        hit=key in self.ray_cache;self.ray_cache_lookup_seconds+=self.clock()-lookup_start
        if hit:self.ray_cache_hits+=1;return self.ray_cache[key]
        value=self.construct('center_ray_construction',len(centers),satellite_rays_numpy,centers,profile);value.setflags(write=False)
        self.ray_cache[key]=value;self.rays_constructed+=len(centers);return value
    def work(self,phase,units):
        self.check();self.work_counts[phase]=self.work_counts.get(phase,0)+int(units)
    def construct(self,phase,units,function,*args):
        self.work(phase,units);record=self.construction_operations.setdefault(phase,dict(input_count=0,calls=0,seconds=0.))
        record['input_count']+=int(units);record['calls']+=1;t=self.clock()
        try:return function(*args)
        finally:record['seconds']+=self.clock()-t
    def prefix(self):
        return dict(seconds=self.elapsed,attempts=self.attempts,complete_candidates=self.complete_candidates,ray_pair_checks=self.ray_pair_checks,geometry_calls=self.geometry_calls,event_count=len(self.events))
    def record(self):
        return dict(seconds=self.elapsed,seconds_limit=self.config.seconds,attempts=self.attempts,candidate_validation_started=self.attempts,complete_candidates=self.complete_candidates,candidate_limit=self.config.candidate_limit,geometry_calls=self.geometry_calls,geometry_limit=self.config.geometry_limit,ray_pair_checks=self.ray_pair_checks,ray_pair_limit=self.config.ray_pair_limit,late_candidates=self.late_candidates,stop_reason=self.reason,overrun_seconds=max(0.,self.elapsed-self.config.seconds),charge_events=list(self.events),center_rays_constructed=self.rays_constructed,ray_cache_hits=self.ray_cache_hits,ray_cache_lookup_seconds=self.ray_cache_lookup_seconds,construction_counts=dict(self.work_counts),construction_operations={k:dict(v) for k,v in self.construction_operations.items()})
