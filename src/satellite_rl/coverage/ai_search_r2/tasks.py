from dataclasses import dataclass
import numpy as np
from .initials import canonical,digest
from ..geometry import satellite_rays_numpy
from ..ai_search.core import ray_center,ray_location

@dataclass(frozen=True)
class Task:
    family: str
    operation: str
    region_ids: tuple
    target_k: int
    widths: tuple
    parent_sha: str
    active_users: tuple
    fixed_users: tuple
    features: dict
    status: str='ready'
    task_id: str=''
    def __post_init__(self):
        for f in ('region_ids','active_users','fixed_users'):object.__setattr__(self,f,tuple(int(v) for v in getattr(self,f)))
        object.__setattr__(self,'widths',tuple(float(v) for v in self.widths))
        if set(self.active_users)&set(self.fixed_users) or len(set(self.active_users))!=len(self.active_users) or len(set(self.fixed_users))!=len(self.fixed_users):raise ValueError('active/fixed overlap or duplicate')
        if len(self.widths)!=self.target_k:raise ValueError('task widths')
        if not self.task_id:object.__setattr__(self,'task_id',digest({k:v for k,v in self.to_record().items() if k!='task_id'}))
    def to_record(self):
        x={k:(list(v) if isinstance(v,tuple) else v) for k,v in self.__dict__.items()}
        x.update(slot=self.features.get('slot'),parent_sha256=self.parent_sha,region_beam_ids=self.features.get('region_beam_ids'),region_aircraft_ids=self.features.get('region_aircraft_ids'),kind=self.operation,retained_beam_ids=self.features.get('retained_beam_ids'),deleted_beam_id=self.features.get('deleted_beam_id'),recipient_beam_id=self.features.get('recipient_beam_id'),split_source_beam_id=self.features.get('split_source_beam_id'),new_beam_id=self.features.get('new_beam_id'),proposed_widths_deg=list(self.widths),center_seed_sets=self.features.get('center_seed_sets'),active_mandatory_aircraft_ids=self.features.get('active_mandatory_aircraft_ids'),source_feature_cost_ref='common_preparation')
        return x
    @classmethod
    def from_record(cls,x):
        from dataclasses import fields
        task=cls(**{k:v for k,v in x.items() if k in {f.name for f in fields(cls)}})
        if task.task_id!=digest({k:v for k,v in task.to_record().items() if k!='task_id'}):raise ValueError('task SHA')
        return task

def seed_sets(scene,parent,region,target,op,hot,deleted,recipient,split_seed,ledger):
    retained=list(region[:target]);base=parent.centers[retained].copy()
    if op=='split' and split_seed is not None:base=np.vstack((base,scene.positions[split_seed]))
    if len(base)!=target:return [{'index':i,'status':'unavailable','centers':None} for i in range(8)]
    focus=target-1 if op=='split' else retained.index(recipient) if op=='merge' and recipient is not None else retained.index(hot) if hot in retained else 0
    group=[hot] if op!='merge' else [recipient,deleted]
    users=np.flatnonzero(np.isin(parent.assignment,group));rows=[dict(index=0,status='available',centers=base.tolist())];used={digest(base.tolist())}
    if len(users):
        mean=ledger.construct('seed_mean_ray',int(len(users)),ray_center,scene,users) if ledger else ray_center(scene,users);candidate=base.copy()
        if mean is not None:candidate[focus]=mean
        sh=digest(candidate.tolist());status='unavailable' if mean is None else 'duplicate' if sh in used else 'available'
        rows.append(dict(index=1,status=status,centers=candidate.tolist() if mean is not None else None));used.add(sh)
        avg=scene.rays[users].mean(0);avg/=np.linalg.norm(avg)
        if ledger:ledger.geometry(int(len(users)),'seed_extreme_distance')
        distance=np.sum((scene.rays[users]-avg)**2,axis=1)
        extreme=sorted(range(len(users)),key=lambda i:(-distance[i],scene.aircraft_ids[users[i]]))[:8]
        boundary=sorted([int(users[i]) for i in extreme],key=lambda u:scene.aircraft_ids[u]);points=[]
        for i,u in enumerate(boundary):
            for v in boundary[i+1:]:
                center=ledger.construct('pair_midpoint_ray_construction',2,ray_location,scene,scene.rays[u]+scene.rays[v]) if ledger else ray_location(scene,scene.rays[u]+scene.rays[v])
                if center is not None and not any(np.array_equal(center,c) for c in points):points.append(center)
                if len(points)>=6:break
            if len(points)>=6:break
        for center in points:
            candidate=base.copy();candidate[focus]=center;sh=digest(candidate.tolist());status='duplicate' if sh in used else 'available';used.add(sh)
            rows.append(dict(index=len(rows),status=status,centers=candidate.tolist()))
    while len(rows)<8:rows.append(dict(index=len(rows),status='unavailable',centers=None))
    return rows[:8]

def make_pool(scene,common,config=None,ledger=None):
    from .config import TaskConfig
    config=config or TaskConfig();p=common.primary_parent;n=len(p.centers)
    if scene.input_sha256!=common.input_sha:raise ValueError('common input SHA')
    if ledger:ledger.work('request_aggregation',int(len(scene.demands)))
    loads=np.bincount(p.assignment,weights=scene.demands,minlength=n);counts=np.bincount(p.assignment,minlength=n)
    rays=ledger.center_rays(p.centers,scene.profile) if ledger else satellite_rays_numpy(p.centers,scene.profile);tasks=[]
    peak=min(range(n),key=lambda b:(-loads[b],p.beam_ids[b])) if n else None
    sparse=min(range(n),key=lambda b:(loads[b],p.beam_ids[b])) if n else None
    if n:
        if ledger:ledger.geometry(int(len(scene.demands)),'pool_owner_margin')
        owner_chord=np.sum((scene.rays-rays[p.assignment])**2,axis=1)
        margin=4*np.sin(np.deg2rad(p.widths[p.assignment]/2)/2)**2-owner_chord
    else:margin=np.empty(0)
    for family,anchor,size in [('peak',peak,4),('sparse',sparse,2)]:
        if n:
            if ledger:ledger.geometry(int(n),'region_anchor_distances')
            distances=np.sum((rays-rays[anchor])**2,axis=1)
            region=tuple(sorted(sorted(range(n),key=lambda b:(float(distances[b]),p.beam_ids[b]))[:size]))
        else:region=()
        members=np.flatnonzero(np.isin(p.assignment,region));k=len(region)
        for op in ('reassign','width_reallocate','merge','split'):
            task_region=region
            deleted=min(region,key=lambda b:(loads[b],p.beam_ids[b])) if region else None
            if op=='merge':task_region=tuple(b for b in region if b!=deleted)+(deleted,) if region else ()
            target=k-1 if op=='merge' else k+1 if op=='split' else k
            target=max(0,target)
            hot=min(region,key=lambda b:(-loads[b],p.beam_ids[b])) if region else None
            split_members=np.flatnonzero(p.assignment==hot) if hot is not None else np.array([],int)
            split_seed=min(split_members.tolist(),key=lambda u:(-owner_chord[u],scene.aircraft_ids[u])) if len(split_members) else None
            recipient=None
            if op=='merge' and k>1:
                if ledger:ledger.geometry(int(k-1),'merge_recipient_distances')
                recipient=min((b for b in region if b!=deleted),key=lambda b:(float(np.sum((rays[b]-rays[deleted])**2)),p.beam_ids[b]))
            required=[u for u in members if p.assignment[u]==deleted] if op=='merge' else [split_seed] if op=='split' and split_seed is not None else []
            ranked=sorted(required,key=lambda u:(-scene.demands[u],scene.aircraft_ids[u]))
            room=max(0,config.active_limit-len(ranked));hot_quota=(room+1)//2
            ranked+=sorted((u for u in members if p.assignment[u]==hot and u not in ranked),key=lambda u:(-scene.demands[u],scene.aircraft_ids[u]))[:hot_quota]
            ranked+=sorted((u for u in members if u not in ranked),key=lambda u:(margin[u],scene.aircraft_ids[u]))
            active=tuple(sorted(ranked[:config.active_limit]));fixed=tuple(sorted(set(members.tolist())-set(active)))
            status='unsupported_active_set' if len(required)>config.active_limit else 'ready'
            widths=tuple(p.widths[list(task_region[:target])])
            if op=='width_reallocate':
                menu=(.3,.5,1.);proposed=[]
                for b in task_region:
                    if b==hot:
                        larger=[w for w in menu if w>p.widths[b]+1e-12]
                        if not larger:status='not_supported_no_wider_menu'
                        proposed.append(larger[0] if larger else p.widths[b])
                    else:
                        us=np.flatnonzero(p.assignment==b)
                        if ledger:ledger.work('cached_margin_menu_thresholds',int(len(us)))
                        needed=float(owner_chord[us].max())
                        possible=[w for w in menu if needed<=4*np.sin(np.deg2rad(w/2)/2)**2+1e-14]
                        proposed.append(possible[0] if possible else 1.)
                widths=tuple(proposed)
            if op=='merge':widths=tuple(1. for _ in range(target))
            if op=='split':widths=tuple(p.widths[list(task_region)])+(.5,)
            duplicate=family=='sparse' and tuple(region)==tuple(tasks[0].region_ids) if tasks else False
            seeds=seed_sets(scene,p,task_region,target,op,hot,deleted,recipient,split_seed,ledger) if target>=1 else []
            slot=('A' if family=='peak' else 'B')+'_'+op
            new_id='r2_'+digest([common.parent_sha,slot,p.beam_ids[hot]]) if op=='split' and hot is not None else None
            tasks.append(Task(family,op,task_region,target,widths,common.parent_sha,active,fixed,dict(slot=slot,region_users=int(len(members)),region_peak=float(loads[list(region)].max()) if k else None,region_request=float(scene.demands[members].sum()),region_count=k,active_count=len(active),fixed_count=len(fixed),requested_region_size=size,insufficient_region_beams=k<size,duplicate_region=duplicate,deleted_beam=deleted if op=='merge' else None,heaviest_beam=hot,region_beam_ids=[p.beam_ids[b] for b in task_region],region_aircraft_ids=[scene.aircraft_ids[u] for u in members],retained_beam_ids=[p.beam_ids[b] for b in task_region[:target]],deleted_beam_id=p.beam_ids[deleted] if op=='merge' and deleted is not None else None,recipient_beam_id=p.beam_ids[recipient] if recipient is not None else None,split_source_beam_id=p.beam_ids[hot] if op=='split' and hot is not None else None,new_beam_id=new_id,active_mandatory_aircraft_ids=[scene.aircraft_ids[u] for u in required],center_seed_sets=seeds,parent_margin_by_user={str(u):float(margin[u]) for u in members}),status))
    if len({t.task_id for t in tasks})!=8:raise ValueError('pool task IDs not unique')
    return tasks
