"""Sparse integral full-aircraft ownership model with a complete witness seed."""
from dataclasses import dataclass
import hashlib
import os
import numpy as np
from .pool import WIDTHS,omega
from ..ai_search.core import Layout
from ..ai_search_r2.validation import validate_owned

class ScaleGuard(ValueError):pass

@dataclass
class IntegerProblem:
    costs: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    integer: np.ndarray
    row_lower: np.ndarray
    row_upper: np.ndarray
    starts: np.ndarray
    indices: np.ndarray
    values: np.ndarray
    seed: np.ndarray
    slots: np.ndarray
    edge_aircraft: np.ndarray
    edge_slot: np.ndarray
    zindex: int
    stage: int

    def fingerprint(self):
        h=hashlib.sha256()
        for name,value in self.__dict__.items():
            array=np.asarray(value);h.update(name.encode());h.update(str(array.dtype).encode());h.update(str(array.shape).encode());h.update(array.tobytes())
        return dict(content_sha256=h.hexdigest(),rows=len(self.row_lower),columns=len(self.costs),nonzeros=len(self.values),
            legal_edges=len(self.edge_aircraft),stage=self.stage,complete_seed_sha256=hashlib.sha256(self.seed.tobytes()).hexdigest())

    def save(self,path):np.savez_compressed(path,**self.__dict__)
    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as f:
            return cls(**{k:(int(f[k]) if k in ('zindex','stage') else f[k].copy()) for k in cls.__dataclass_fields__})

def build_problem(p,slots,stage=1,stage1=None,max_edges=1000000):
    slots=np.asarray(slots,np.int64);nb=len(slots);n=len(p.scene.demands)
    edge_count=sum(int(p.coverage[int(s)//6,(int(s)//2)%3].sum()) for s in slots)
    if edge_count>max_edges:raise ScaleGuard('legal_edge_variable_guard')
    # Conservative Python-list/CSR/seed/edge peak guard before large allocations.
    estimated_build_bytes=edge_count*1536+n*2048+nb*8192+n*nb
    from .solver import own_rss
    parent_existing_rss=own_rss(os.getpid())
    if parent_existing_rss+estimated_build_bytes>2147483648:raise ScaleGuard('matrix_build_memory_estimate_guard')
    legal=p.coverage[slots//6,(slots//2)%3].T
    ia,bs=np.nonzero(legal)
    ne=len(ia);zi=nb+ne;nv=zi+1+(nb if stage==2 else 0)
    lo=np.zeros(nv);hi=np.ones(nv);hi[zi:]=np.inf;integer=np.zeros(nv,np.uint8);integer[:zi]=1
    costs=np.zeros(nv);costs[zi]=1 if stage==1 else 0
    if stage==2:costs[zi+1:]=1
    shares=p.scene.demands/p.scene.demands.sum() if p.scene.demands.sum() else np.zeros(n)
    rr=[];cc=[];vv=[];rl=[];ru=[]
    def row(columns,values,lower=-np.inf,upper=np.inf):
        r=len(rl);rr.extend([r]*len(columns));cc.extend(columns);vv.extend(values);rl.append(lower);ru.append(upper)
    aircraft_edges=[[] for _ in range(n)];beam_edges=[[] for _ in range(nb)]
    for e,(i,b) in enumerate(zip(ia,bs)):aircraft_edges[i].append(nb+e);beam_edges[b].append(nb+e)
    for edges in aircraft_edges:row(edges,[1]*len(edges),1,1)
    for e,b in enumerate(bs):row([nb+e,int(b)],[1,-1],upper=0)
    for b,edges in enumerate(beam_edges):
        row([b]+edges,[1]+[-1]*len(edges),upper=0)
        weights=shares[ia[np.array(edges)-nb]] if edges else []
        row(edges+[zi],list(weights)+[-1],upper=0)
        if stage==2:
            a=zi+1+b
            row(edges+[b,a],list(weights)+[-1/p.ncap,-1],upper=0)
            row(edges+[b,a],list(-np.asarray(weights))+[1/p.ncap,-1],upper=0)
    row(list(range(nb)),[1]*nb,upper=p.ncap)
    row(list(range(nb)),list(omega(WIDTHS[(slots//2)%3])/p.omega_cap),upper=1)
    lookup={int(s):b for b,s in enumerate(slots)}
    for sid,b in lookup.items():
        if sid%2 and sid-1 in lookup:
            first=lookup[sid-1];row([b,first],[1,-1],upper=0)
            e0=beam_edges[first];e1=beam_edges[b]
            row(e1+e0,list(shares[ia[np.array(e1,dtype=int)-nb]])+list(-shares[ia[np.array(e0,dtype=int)-nb]]),upper=0)
    if stage==2:
        if stage1 is None:raise ValueError('stage2_requires_valid_stage1_vector')
        stage1_loads=np.bincount(bs,weights=shares[ia]*stage1[nb:zi],minlength=nb)
        stage1_peak=float(stage1_loads.max(initial=0))
        row([zi],[1],upper=stage1_peak+1e-8)
    # Complete vector: active y, every legal x (including zeros), z and stage2 auxiliaries.
    seed=np.zeros(nv)
    if stage1 is not None:seed[:zi+1]=stage1[:zi+1];seed[zi]=stage1_peak
    else:
        for b,c in enumerate(p.witness.centers):
            ci=np.flatnonzero(np.all(p.centers==c,axis=1))[0];wi=int(np.flatnonzero(WIDTHS==p.witness.widths[b])[0]);sid=int(ci*6+wi*2)
            j=lookup[sid];seed[j]=1
            members=np.flatnonzero(p.witness.assignment==b)
            for i in members:
                es=[e for e in aircraft_edges[i] if bs[e-nb]==j]
                if len(es)!=1:raise ValueError('witness_seed_edge_missing')
                seed[es[0]]=1
        seed[zi]=max(np.bincount(p.witness.assignment,weights=shares,minlength=len(p.witness.centers)))
    if stage==2:
        loads=np.bincount(bs,weights=shares[ia]*seed[nb:zi],minlength=nb)
        seed[zi+1:]=np.abs(loads-seed[:nb]/p.ncap)
    # CSR row storage, no scipy optimization dependency.
    order=np.argsort(rr,kind='stable');r=np.asarray(rr,dtype=np.int64)[order]
    counts=np.bincount(r,minlength=len(rl));starts=np.r_[0,np.cumsum(counts)].astype(np.int64)
    problem=IntegerProblem(costs,lo,hi,integer,np.asarray(rl),np.asarray(ru),starts,
        np.asarray(cc,np.int32)[order],np.asarray(vv,float)[order],seed,slots,ia,bs,zi,stage)
    valid,reason=validate_vector(problem,seed)
    if not valid:raise ValueError('invalid_complete_witness_seed:'+reason)
    return problem

def validate_vector(problem,vector,tolerance=1e-6):
    x=np.asarray(vector,float)
    if x.shape!=problem.costs.shape or not np.isfinite(x).all():return False,'shape_or_finite'
    ix=problem.integer.astype(bool)
    if np.max(np.abs(x[ix]-np.round(x[ix])),initial=0)>tolerance:return False,'original_integer_values'
    if (x<problem.lower-tolerance).any() or (x>problem.upper+tolerance).any():return False,'bounds'
    rows=np.repeat(np.arange(len(problem.row_lower)),np.diff(problem.starts))
    activity=np.bincount(rows,weights=problem.values*x[problem.indices],minlength=len(problem.row_lower))
    if (activity<problem.row_lower-tolerance).any() or (activity>problem.row_upper+tolerance).any():return False,'row_constraints'
    return True,'accepted_original_vector'

def decode(p,problem,vector):
    valid,reason=validate_vector(problem,vector)
    if not valid:raise ValueError(reason)
    x=np.asarray(vector);active=np.flatnonzero(x[:len(problem.slots)]>.5)
    owner=np.full(len(p.scene.demands),-1,np.int64);lookup={int(b):j for j,b in enumerate(active)}
    for e in np.flatnonzero(x[len(problem.slots):problem.zindex]>.5):
        i=int(problem.edge_aircraft[e]);b=int(problem.edge_slot[e])
        if owner[i]!=-1 or b not in lookup:raise ValueError('duplicate_or_inactive_owner')
        owner[i]=lookup[b]
    ids=problem.slots[active];layout=Layout(p.centers[ids//6],WIDTHS[(ids//2)%3],owner,tuple('s'+str(s) for s in ids))
    audit=validate_owned(p.scene,layout)
    if not audit['valid']:raise ValueError('independent_geometry:'+str(audit['reasons']))
    if len(active)>p.ncap or omega(layout.widths).sum()>p.omega_cap+1e-12:raise ValueError('resource_caps')
    return layout,dict(active_slot_ids=ids.tolist(),original_vector_integrality=True,independent_audit=audit)
