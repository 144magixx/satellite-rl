"""Input-only streaming witness and bounded, reusable candidate geometry."""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import math
import time
import numpy as np
from ..ai_search.core import Scene, Layout, ray_center
from ..geometry import satellite_rays_numpy, ray_angle_numpy

WIDTHS = np.array([.3, .5, 1.], dtype=np.float64)

def omega(width):
    return 2*np.pi*(1-np.cos(np.deg2rad(np.asarray(width)/2)))

class PreparationIncomplete(RuntimeError):
    def __init__(self,reason,ledger):super().__init__(reason);self.ledger=ledger

def streaming_witness(scene,check=None):
    started=time.monotonic(); calls=0
    for width in (.5,1.):
        remaining=np.ones(len(scene.demands),bool); owner=np.full(len(remaining),-1); centers=[]
        order=sorted(range(len(remaining)),key=lambda i:scene.aircraft_ids[i])
        for i in order:
            if check is not None:check()
            if not remaining[i]: continue
            if len(centers)>=200: break
            distance=np.sum((scene.rays-scene.rays[i])**2,axis=1); calls+=len(remaining)
            members=remaining & (distance<=4*np.sin(np.deg2rad(width/2)/2)**2+1e-14)
            owner[members]=len(centers); centers.append(scene.positions[i]); remaining[members]=False
        if not remaining.any():
            return Layout(centers,np.full(len(centers),width),owner),dict(raypairs=calls,seconds=time.monotonic()-started)
    raise ValueError('bounded_witness_not_found_not_infeasibility_certificate')

@dataclass
class PreparedCase:
    scene: Scene
    case_id: str
    witness: Layout
    ncap: int
    omega_cap: float
    centers: np.ndarray
    origins: np.ndarray
    coverage: np.ndarray
    ledger: dict
    content_sha: str

    @property
    def slot_ids(self): return np.arange(len(self.centers)*6,dtype=np.int64)
    @property
    def mandatory(self): return np.flatnonzero(np.repeat(self.origins[:,0],6))
    @property
    def available(self): return np.flatnonzero(np.repeat(self.coverage.any(2).reshape(-1),2))
    @property
    def extra_budget(self): return min(self.ncap,len(np.setdiff1d(self.available,self.mandatory)))
    def save(self,path):
        path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
        metadata=dict(case_id=self.case_id,aircraft_ids=self.scene.aircraft_ids,profile=self.scene.profile,
            ncap=self.ncap,omega_cap=self.omega_cap,ledger=self.ledger,content_sha=self.content_sha,
            witness=self.witness.to_record(),schema='r3_prepared_case_v2')
        np.savez_compressed(path,positions=self.scene.positions,demands=self.scene.demands,
            centers=self.centers,origins=self.origins,coverage=self.coverage,metadata=json.dumps(metadata))
    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as d:
            m=json.loads(str(d['metadata'])); scene=Scene(d['positions'],d['demands'],tuple(m['aircraft_ids']),m['profile'])
            result=cls(scene,m['case_id'],Layout.from_record(m['witness']),m['ncap'],m['omega_cap'],
                d['centers'].copy(),d['origins'].copy(),d['coverage'].copy(),m['ledger'],m['content_sha'])
        if result.compute_sha()!=result.content_sha: raise ValueError('candidate_cache_hash_mismatch')
        return result
    def compute_sha(self):
        h=hashlib.sha256(self.scene.input_sha256.encode())
        for x in (self.centers,self.origins,self.coverage):h.update(np.ascontiguousarray(x).tobytes())
        h.update(json.dumps([self.witness.to_record(),self.ncap,self.omega_cap],sort_keys=True).encode())
        return h.hexdigest()

def prepare_case(scene,case_id,stop=None,seconds=10.):
    started=time.monotonic()
    def check():
        elapsed=time.monotonic()-started
        if (stop is not None and stop()) or elapsed>=seconds:
            raise PreparationIncomplete('requested_stop' if stop is not None and stop() else 'preparation_time_guard',dict(seconds=elapsed,state='incomplete_not_reusable'))
    witness,wledger=streaming_witness(scene,check)
    if not len(scene.demands): raise ValueError('empty_input_case')
    ncap=min(200,len(scene.demands),math.ceil(1.25*len(witness.centers)))
    cap=float(omega(witness.widths).sum()); centers=[]; origins=[]; index={}; rays=[]; calls=wledger['raypairs']; means=0
    def add(point,kind):
        check()
        if point is None:return
        key=tuple(np.asarray(point,dtype=np.float64))
        if key in index:origins[index[key]][kind]=1;return
        if len(centers)>=512:return
        index[key]=len(centers);centers.append(point);flags=[0]*4;flags[kind]=1;origins.append(flags)
        rays.append(satellite_rays_numpy(np.asarray(point)[None],scene.profile)[0])
    for c in witness.centers:add(c,0)
    for b in range(len(witness.centers)):
        members=np.flatnonzero(witness.assignment==b);means+=len(members);add(ray_center(scene,members),1)
    order=sorted(range(len(scene.demands)),key=lambda i:(-scene.demands[i],scene.aircraft_ids[i]))
    for i in order[:64]:add(scene.positions[i],2)
    nearest=np.full(len(scene.demands),np.inf)
    for ray in rays:nearest=np.minimum(nearest,np.sum((scene.rays-ray)**2,axis=1));calls+=len(nearest)
    idorder=sorted(range(len(nearest)),key=lambda i:scene.aircraft_ids[i])
    while len(centers)<512:
        check()
        eligible=[i for i in idorder if tuple(scene.positions[i]) not in index]
        if not eligible:break
        i=max(eligible,key=lambda j:nearest[j]);add(scene.positions[i],3)
        nearest=np.minimum(nearest,np.sum((scene.rays-rays[-1])**2,axis=1));calls+=len(nearest)
    centers=np.asarray(centers,dtype=np.float64); coverage=np.empty((len(centers),3,len(nearest)),bool)
    for c,ray in enumerate(rays):
        check()
        distance=np.sum((scene.rays-ray)**2,axis=1);calls+=len(nearest)
        coverage[c]=distance[None]<=4*np.sin(np.deg2rad(WIDTHS[:,None]/2)/2)**2+1e-14
    if calls>1424*len(nearest)+262144:raise ValueError('pool_raypair_guard')
    ledger=dict(witness=wledger,candidate_seconds=time.monotonic()-started-wledger['seconds'],
        seconds=time.monotonic()-started,raypairs=calls,ray_constructions=len(rays),mean_input_counts=means,
        sort_work=len(order)+len(idorder),peak_array_bytes=coverage.nbytes+centers.nbytes,cache_hits=0)
    p=PreparedCase(scene,str(case_id),witness,ncap,cap,centers,np.asarray(origins,dtype=np.uint8),coverage,ledger,'')
    p.content_sha=p.compute_sha();check();return p
