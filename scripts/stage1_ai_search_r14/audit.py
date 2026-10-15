"""Independent case-only ray cache and O(n+N) ownership audit for R14.

No producer, SceneAudit, solver, tree, teacher or fixed train partition import.
Real projection/owner geometry is executable only after root engineering admission.
"""
from dataclasses import dataclass
import hashlib
import json
import math
import numpy as np

REQUEST=21600000
OFFLINE_LAYOUTS=4116
OFFLINE_PAIRS=8232000
ONLINE_LAYOUTS=504
ONLINE_PAIRS=1008000
RAYS=280000
METRICS=('N','maxload','top10_mean','CV','actual_Omega','total_request')

class AuditError(ValueError):pass
def require(value,reason):
    if not value:raise AuditError(reason)
def encoded(value):return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode('utf8')
def sha(value):return hashlib.sha256(encoded(value)).hexdigest()

class AuditLedger:
    """Caller owns durable persist callback; debit before each actual entry.

    Failed entered operations stay charged and are not eligible for automatic retry.
    Online and offline positions/pairs are separate; producer costs never enter here.
    """
    def __init__(self,state=None,persist=None):
        self.state=dict(rays=0,offline_layouts=0,offline_pairs=0,online_layouts=0,online_pairs=0,projected_cases=[],inflight=None) if state is None else json.loads(json.dumps(state))
        self.persist=persist
        require(self.state.get('inflight') is None,'unresolved audit entry cannot replay')
        for key,cap in (('rays',RAYS),('offline_layouts',OFFLINE_LAYOUTS),('offline_pairs',OFFLINE_PAIRS),('online_layouts',ONLINE_LAYOUTS),('online_pairs',ONLINE_PAIRS)):
            require(type(self.state.get(key)) is int and 0<=self.state[key]<=cap,'valid cumulative nonrefunded audit ledger')
        require(isinstance(self.state.get('projected_cases'),list) and len(set(self.state['projected_cases']))==len(self.state['projected_cases']),'unique already-entered case projections')
    def _commit(self):
        if self.persist is not None:self.persist(json.loads(json.dumps(self.state)))
    def reserve_projection(self,key,n):
        require(self.state['inflight'] is None and key not in self.state['projected_cases'],'case projection already entered or unresolved')
        require(type(n) is int and 1<=n<=2000 and self.state['rays']+n<=RAYS,'projection hard cap')
        self.state['rays']+=n;self.state['projected_cases'].append(key);self.state['inflight']=dict(kind='projection',key=key,rays=n);self._commit()
    def reserve_layout(self,scope,key,n):
        require(scope in ('offline','online') and self.state['inflight'] is None,'audit scope or unresolved entry')
        lc,pc=(OFFLINE_LAYOUTS,OFFLINE_PAIRS) if scope=='offline' else (ONLINE_LAYOUTS,ONLINE_PAIRS)
        require(type(n) is int and 1<=n<=2000 and self.state[scope+'_layouts']+1<=lc and self.state[scope+'_pairs']+n<=pc,'separate layout/owner coverage ceiling')
        self.state[scope+'_layouts']+=1;self.state[scope+'_pairs']+=n;self.state['inflight']=dict(kind='layout',scope=scope,key=key,layouts=1,owner_pairs=n);self._commit()
    def complete(self):
        require(self.state['inflight'] is not None,'no entered audit to complete');self.state['inflight']=None;self._commit()

def _case_parts(case):
    ids=case['aircraft_ids'];n=len(ids)
    require(1<=n<=2000 and all(type(x) is str and x for x in ids) and len(set(ids))==n,'full stable aircraft ID domain')
    positions=np.asarray(case['positions'],dtype='<f8',order='C');demands=np.asarray(case['demands'])
    require(positions.shape==(n,2) and np.isfinite(positions).all(),'full finite original positions')
    require(demands.shape==(n,) and np.all(demands==REQUEST),'all original21600000bps requests')
    require(isinstance(case.get('identity'),dict) and bool(case['identity']),'original case/source identity retained')
    source=case['source_ref'];require(type(source.get('bytes')) is int and source['bytes']>=0 and type(source.get('sha256')) is str and len(source['sha256'])==64,'exact original case source SHA/bytes')
    p=case['physical_parameters']
    physics_version=p.get('version',p.get('environment_version'))
    require(type(physics_version) is str and bool(physics_version),'explicit original physics version')
    for k in ('earth_radius_km','orbit_radius_km','satellite_longitude_deg','min_elevation_deg'):
        require(type(p.get(k)) in (int,float) and math.isfinite(p[k]),'finite explicit physics '+k)
    require(0<p['earth_radius_km']<p['orbit_radius_km'],'physical radii')
    require(p.get('satellite_latitude_deg',0)==0,'original equatorial satellite physics only')
    for k in ('latitude_bounds_deg','longitude_bounds_deg'):
        require(len(p[k])==2 and all(type(v) in (int,float) and math.isfinite(v) for v in p[k]) and p[k][0]<p[k][1],'explicit source coordinate bounds')
    binding=dict(identity=case['identity'],source_content_identity=dict(sha256=source['sha256'],bytes=source['bytes']),ordered_aircraft_ids=list(ids),
        original_positions_SHA=sha(case['positions']),float64_positions_SHA=hashlib.sha256(positions.tobytes(order='C')).hexdigest(),uniform_request_bps=REQUEST,aircraft_count=n,physics=p)
    return ids,positions,p,binding

def _project_rays_visibility(positions,p):
    """Independent float64 O(n), includes original bounds/elevation visibility."""
    lat=np.deg2rad(positions[:,0]);lon=np.deg2rad(positions[:,1]);slon=np.deg2rad(p['satellite_longitude_deg'])
    c=np.cos(lat)*np.cos(lon-slon)
    elevation=np.rad2deg(np.arctan2(c-p['earth_radius_km']/p['orbit_radius_km'],np.sqrt(np.maximum(0.,1.-c*c))))
    visible=(positions[:,0]>=p['latitude_bounds_deg'][0])&(positions[:,0]<=p['latitude_bounds_deg'][1])&(positions[:,1]>=p['longitude_bounds_deg'][0])&(positions[:,1]<=p['longitude_bounds_deg'][1])&(elevation>=p['min_elevation_deg'])
    surface=p['earth_radius_km']*np.stack((np.cos(lat)*np.cos(lon),np.cos(lat)*np.sin(lon),np.sin(lat)),axis=1)
    satellite=p['orbit_radius_km']*np.array([np.cos(slon),np.sin(slon),0.],dtype=np.float64)
    rays=surface-satellite;norm=np.sqrt(np.sum(rays*rays,axis=1));require(np.all(norm>0),'nondegenerate satellite rays')
    return np.asarray(rays/norm[:,None],dtype='<f8',order='C'),np.asarray(visible,dtype=bool)

def build_ray_cache(case,ledger,stop=lambda:False):
    ids,positions,p,binding=_case_parts(case);key=sha(binding)
    require(not stop(),'stop before independent case projection');ledger.reserve_projection(key,len(ids))
    rays,visible=_project_rays_visibility(positions,p)
    require(np.isfinite(rays).all() and bool(visible.all()),'all original aircraft visible; no discard')
    record=dict(schema='R14IndependentCaseRayCacheV1',binding=binding,dtype='<f8',order='C',shape=[len(ids),3],rays=rays.tolist(),visible=visible.tolist(),rays_bytes_SHA=hashlib.sha256(rays.tobytes(order='C')).hexdigest(),actual_cost=dict(new_projections=len(ids),visibility_checks=len(ids)))
    record['cache_SHA']=sha(record)
    # Caller must persist this record before completing the ledger operation.
    return record

@dataclass(frozen=True)
class RayCache:
    binding_SHA: str
    cache_SHA: str
    rays: np.ndarray
    visible: np.ndarray

def bind_cache(case,record,*,expected_cache_sha):
    ids,positions,p,binding=_case_parts(case)
    require(record.get('schema')=='R14IndependentCaseRayCacheV1' and record.get('binding')==binding,'case-only cache full source/ID/position/physics binding')
    require(record.get('cache_SHA')==expected_cache_sha==sha({k:v for k,v in record.items() if k!='cache_SHA'}),'preteacher manifest exact cache SHA')
    require(record.get('dtype')=='<f8' and record.get('order')=='C' and record.get('shape')==[len(ids),3],'canonical endian/dtype/order/shape')
    rays=np.asarray(record['rays'],dtype='<f8',order='C');visible=np.asarray(record['visible'])
    require(rays.shape==(len(ids),3) and np.isfinite(rays).all() and visible.dtype==np.dtype(bool) and visible.shape==(len(ids),) and visible.all(),'complete visible float64 rays')
    require(np.all(np.abs(np.sum(rays*rays,axis=1)-1.)<=1e-12),'unit ray cache')
    require(hashlib.sha256(rays.tobytes(order='C')).hexdigest()==record['rays_bytes_SHA'],'exact little-endian ray bytes')
    rays.setflags(write=False);visible.setflags(write=False)
    return RayCache(sha(binding),expected_cache_sha,rays,visible)

def _owner_chord(rays,anchor_indices,owners):
    delta=rays-rays[np.asarray(anchor_indices,dtype=np.int64)[owners]]
    return np.sum(delta*delta,axis=1)

def _minimal_widths(max_chords):
    return np.maximum(.3,np.nextafter(np.rad2deg(4*np.arcsin(np.minimum(np.sqrt(max_chords)/2.,1.))),np.inf))

def audit_layout(case,layout,cache,ledger,*,scope,envelope,Ncap,Omega_cap,saved_metrics=None,stop=lambda:False):
    """One n-owner geometry pass; raw_W is a coverage witness, not minimum width.

    Distinct logical audit positions are charged even for identical layout hashes.
    No future teacher/quality certificate is accepted in online or offline input.
    """
    ids,positions,p,binding=_case_parts(case);n=len(ids)
    require(isinstance(cache,RayCache) and cache.binding_SHA==sha(binding),'bound original case cache required')
    require(scope in ('offline','online') and envelope in ('raw_W','minimal'),'explicit independent/online and width semantics')
    require(type(Ncap) is int and 1<=Ncap<=200 and type(Omega_cap) in (int,float) and math.isfinite(Omega_cap) and Omega_cap>0,'original finite N/Omega caps')
    beamids=layout['beam_ids'];anchors=layout['anchor_aircraft_ids'];b=len(beamids)
    require(1<=b<=min(n,Ncap,200) and all(type(v) is str and v for v in beamids) and len(set(beamids))==b,'nonempty unique beam IDs/actualN')
    require(len(anchors)==b and all(type(v) is str for v in anchors),'complete original anchor IDs')
    pindex={v:i for i,v in enumerate(ids)};require(all(v in pindex for v in anchors),'non-aircraft anchor forbidden before new projection')
    ai=[pindex[v] for v in anchors];centers=np.asarray(layout['centers'],dtype=np.float64);widths=np.asarray(layout['widths_deg'],dtype=np.float64);owners=np.asarray(layout['assignment'])
    require(centers.shape==(b,2) and np.isfinite(centers).all() and np.array_equal(centers,positions[ai]),'exact source original anchor centers')
    require(widths.shape==(b,) and np.isfinite(widths).all() and np.all((widths>=.3)&(widths<=1.)),'complete fullwidth envelope[.3,1]')
    require(owners.shape==(n,) and owners.dtype.kind in 'iu' and np.all((owners>=0)&(owners<b)),'every original aircraft exactly one integer owner')
    owners=owners.astype(np.int64,copy=False);counts=np.bincount(owners,minlength=b);require(np.all(counts>0),'all beams nonempty')
    require(not stop(),'stop before new owner coverage audit');layoutSHA=sha(layout);ledger.reserve_layout(scope,layoutSHA,n)
    chord=_owner_chord(cache.rays,ai,owners)
    require(chord.shape==(n,) and np.isfinite(chord).all() and np.all(chord>=0),'complete finite owner-angle witness')
    require(np.all(chord<=4*np.sin(np.deg2rad(widths[owners]/2.)/2.)**2+1e-14),'all original aircraft covered by actual owned beam')
    maxchord=np.zeros(b,dtype=np.float64);np.maximum.at(maxchord,owners,chord)
    if envelope=='minimal':require(np.all(np.abs(widths-_minimal_widths(maxchord))<=1e-12),'required original minimal fullwidth envelope')
    omega=float(np.sum(2.*np.pi*(1.-np.cos(np.deg2rad(widths)/2.))))
    require(omega<=Omega_cap+1e-12,'whole original Omega cap')
    loads=counts.astype(np.float64)*REQUEST
    # N<=200; sorting this bounded vector does not turn owner geometry into n*N.
    metrics=dict(N=b,maxload=int(counts.max())*REQUEST,top10_mean=float(np.sort(loads)[-max(1,math.ceil(.1*b)):].mean()),CV=float(loads.std()/loads.mean()),actual_Omega=omega,total_request=int(counts.sum())*REQUEST)
    require(metrics['total_request']==n*REQUEST,'full aircraft request conservation')
    if saved_metrics is not None:
        for key in METRICS:
            value=saved_metrics.get(key);require(type(value) in (int,float) and math.isfinite(value),'finite complete saved metrics')
            tol=0 if key in ('N','maxload','total_request') else 1e-12 if key=='actual_Omega' else (1e-9 if key=='CV' else 1e-6)+1e-12*abs(metrics[key])
            require(abs(value-metrics[key])<=tol,'independent saved metric mismatch '+key)
    receipt=dict(schema='R14IndependentLinearLayoutAuditV1',passed=True,scope=scope,envelope=envelope,case_binding_SHA=cache.binding_SHA,cache_SHA=cache.cache_SHA,layout_SHA=layoutSHA,metrics=metrics,aircraft_count=n,actual_cost=dict(layout_positions=1,owner_coverage_checks=n,new_projections=0),future_teacher_used=False)
    # Persist receipt before ledger.complete(), exactly as case-cache publication.
    return receipt

def engineering_fixture_definitions():
    """Only coordinates/IDs/layout definitions. No projection or geometry calls."""
    result=[]
    for j,n in enumerate((8,8,16,16,32,32,64,64)):
        ids=['r14_fixture_'+str(j)+'_aircraft_'+str(i) for i in range(n)]
        positions=[[45+.05*j+.01*(i//8),13+.01*(i%8)] for i in range(n)];anchors=list(range(0,n,4));N=n//4
        result.append(dict(fixture_index=j,aircraft_ids=ids,positions=positions,demands=[REQUEST]*n,Ncap=N+2,
            Omega_cap_expression='(N+2)*2*pi*(1-cos(radians(0.3)/2))',layout=dict(beam_ids=['r14_fixture_'+str(j)+'_beam_'+str(k) for k in range(N)],anchor_aircraft_ids=[ids[i] for i in anchors],centers=[positions[i] for i in anchors],widths_deg=[.3]*N,assignment=[i//4 for i in range(n)])))
    return result
