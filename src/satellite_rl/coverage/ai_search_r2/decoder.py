"""R1 mechanism adapter D0 and bounded active-set decoder D1."""
import numpy as np
from ..ai_search import search as r1
from ..ai_search.core import Layout,metrics,ray_center,layout_sha256
from .envelope import minimum_enclosing_width
from ..ai_search.config import GeometryConfig,SearchConfig
from ..geometry import satellite_rays_numpy,visible_numpy
from .validation import validate_owned
from .config import BudgetStop

def _qualification(scene,users,centers,widths,geometry,ledger):
    rays=ledger.center_rays(centers,scene.profile)
    ledger.geometry(int(len(users)*len(centers)),'local_qualification')
    chord=np.sum((scene.rays[np.asarray(users,int),None]-rays[None])**2,axis=2)
    limits=4*np.sin(np.deg2rad(np.asarray(widths)/2)/2)**2
    return (chord<=limits[None]+geometry.tolerance)&visible_numpy(centers,scene.profile)[None]

def assignment_key(loads,beam_id):
    """Frozen provisional load objective, distinct from final candidate rank."""
    loads=np.asarray(loads,dtype=np.float64)
    return float(loads.max()),float(np.sum(loads*loads)),beam_id

def assemble_region(scene,parent,region,members,centers,widths,owners,task):
    if len(region)==len(centers):return r1._merge_region(scene,parent,region,members,centers,widths,owners,task)
    keep=[b for b in range(len(parent.centers)) if b not in region]
    remap=np.full(len(parent.centers),-1);remap[keep]=np.arange(len(keep))
    assignment=remap[parent.assignment].copy();assignment[members]=len(keep)+owners
    ids=[parent.beam_ids[b] for b in keep]+[parent.beam_ids[b] for b in region[:len(centers)]]
    for i in range(max(0,len(centers)-len(region))):
        identity=task.cheap_features.get('new_beam_id') or f'r2_{task.task_id}_{i}'
        while identity in ids:identity='r'+identity
        ids.append(identity)
    return Layout(np.vstack((parent.centers[keep],centers)),np.concatenate((parent.widths[keep],widths)),assignment,tuple(ids))

def _active_assign(scene,parent,task,centers,widths,geometry,ledger,start):
    region=list(task.region_ids);active=np.asarray(task.active_users,int);fixed=np.asarray(task.fixed_users,int);k=task.target_k
    local={b:i for i,b in enumerate(region)};fixed_owner=np.array([local[int(parent.assignment[u])] for u in fixed],int)
    if len(fixed_owner) and (fixed_owner>=k).any():return None,'deleted_beam_has_fixed_users'
    # Check all fixed users against their unchanged local owner, even when a
    # center moves/width changes; no fixed plane silently enters the active set.
    if len(fixed):
        legal_fixed=_qualification(scene,fixed,centers,widths,geometry,ledger)
        if not legal_fixed[np.arange(len(fixed)),fixed_owner].all():return None,'fixed_user_coverage'
    legal=_qualification(scene,active,centers,widths,geometry,ledger)
    if len(active) and not legal.any(1).all():return None,'active_user_coverage'
    loads=np.bincount(fixed_owner,weights=scene.demands[fixed],minlength=k);counts=np.bincount(fixed_owner,minlength=k)
    owners=np.full(len(active),-1,int)
    order=sorted(range(len(active)),key=lambda i:(int(legal[i].sum()),-scene.demands[active[i]] if start==0 else task.features['parent_margin_by_user'][str(active[i])],scene.aircraft_ids[active[i]]))
    outside=np.bincount(parent.assignment,weights=scene.demands,minlength=len(parent.centers))[[b for b in range(len(parent.centers)) if b not in region]]
    for i in order:
        ledger.check()
        def prospective(b):
            ledger.work('prospective_assignment_loads',int(len(loads)+len(outside)))
            trial=loads.copy();trial[b]+=scene.demands[active[i]];all_loads=np.concatenate((outside,trial))
            identity=parent.beam_ids[region[int(b)]] if b<len(region) else task.features['new_beam_id']
            return assignment_key(all_loads,identity)
        b=min(np.flatnonzero(legal[i]),key=prospective);ledger.work('assignment_updates',1);owners[i]=b;loads[b]+=scene.demands[active[i]];counts[b]+=1
    for b in np.flatnonzero(counts==0):
        possible=[i for i in range(len(active)) if legal[i,b] and counts[owners[i]]>1]
        if not possible:return None,'empty_local_beam'
        i=min(possible,key=lambda i:(scene.demands[active[i]],scene.aircraft_ids[active[i]]));ledger.work('assignment_updates',1);counts[owners[i]]-=1;owners[i]=b;counts[b]+=1
    members=np.array(sorted(task.active_users+task.fixed_users),int);mapping={int(u):int(o) for u,o in zip(active,owners)};mapping.update({int(u):int(o) for u,o in zip(fixed,fixed_owner)})
    return (members,np.array([mapping[int(u)] for u in members],int)),None

def _d1(scene,parent,task,ledger,geometry,candidates):
    config=ledger.config;region=list(task.region_ids);k=task.target_k
    all_members=np.flatnonzero(np.isin(parent.assignment,region))
    if sorted(task.active_users+task.fixed_users)!=all_members.tolist():raise ValueError('region users omitted/duplicated')
    if len(task.active_users)>config.active_limit:raise ValueError('active cap')
    if k<len(region) and any(parent.assignment[u] in region[k:] for u in task.fixed_users):return [],'not_supported_deleted_fixed'
    legacy=r1.Task(task.operation,tuple(region),k,task.widths,'menu',parent_layout_sha256=task.parent_sha,task_id=task.task_id,cheap_features=task.features)
    seen=set();reason='no_feasible_candidate'
    for start in range(config.starts):
        ledger.check();seed=task.features['center_seed_sets'][start]
        if seed['status']!='available':continue
        centers=np.asarray(seed['centers'],float)
        widths=np.asarray(task.widths,float)
        for iteration in range(config.center_iterations+1):
            assigned,why=_active_assign(scene,parent,task,centers,widths,geometry,ledger,start)
            if assigned is None:reason=why;break
            members,owners=assigned
            for shrink in (False,True):
                candidate_widths=widths.copy()
                if shrink:
                    for b in range(k):
                        needed=minimum_enclosing_width(scene,members[owners==b],centers[b],geometry,ledger)
                        if needed is None or needed>widths[b]+1e-12:break
                        candidate_widths[b]=min(widths[b],needed)
                    else:needed='ok'
                    if needed!='ok':continue
                candidate=assemble_region(scene,parent,region,members,centers,candidate_widths,owners,legacy)
                sh=layout_sha256(candidate)
                if sh in seen:continue
                seen.add(sh);ledger.attempt();e=validate_owned(scene,candidate,geometry,ledger)
                values=metrics(scene,candidate,geometry,e) if e['valid'] else None
                ledger.complete_candidates+=1
                record=dict(layout=candidate.to_record(),delta=r1.layout_delta(parent,candidate),sha256=sh,metrics=values,evidence=e,start=start,iteration=iteration,shrink=shrink,attempt=ledger.attempts,cost_prefix=ledger.prefix())
                completed=ledger.elapsed;late=completed>ledger.config.seconds
                if late:ledger.late_candidates+=1
                record.update(late=late,completed_seconds=completed);record['cost_prefix']['seconds']=completed
                candidates.append(record)
                if e['valid'] and not late:reason='completed'
            if iteration<config.center_iterations:
                moved=centers.copy()
                for b in range(k):
                    us=members[owners==b];c=ledger.construct('mean_ray_update',int(len(us)),ray_center,scene,us)
                    if c is not None:moved[b]=c
                centers=moved
    return candidates,reason

def solve_task(scene,common,task,variant,ledger,geometry=GeometryConfig()):
    from .tasks import Task
    Task.from_record(task.to_record())
    if scene.input_sha256!=common.input_sha or task.parent_sha!=common.parent_sha:raise ValueError('parent/input mismatch')
    parent=common.primary_parent;candidates=[];reason='not_supported'
    try:
        ledger.check()
        if not task.region_ids or task.target_k<1 or task.target_k>len(task.active_users)+len(task.fixed_users) or len(parent.centers)-len(task.region_ids)+task.target_k>geometry.max_beams:reason='not_supported_beam_count'
        elif task.status=='not_supported_no_wider_menu':reason='completed_no_applicable_width_step'
        elif variant=='D0':
            old=r1.Task(task.operation,task.region_ids,task.target_k,task.widths,'menu',task_id=task.task_id,parent_layout_sha256=task.parent_sha,preference_index=2,cheap_features=task.features)
            def retain(x):
                ledger.complete_candidates+=1
                record={**x,'layout':x['layout'].to_record(),'delta':r1.layout_delta(parent,x['layout']),'cost_prefix':ledger.prefix()}
                if ledger.clock()>ledger.deadline:
                    if not record['late']:ledger.late_candidates+=1
                    record['late']=True
                    x['late']=True
                record['completed_seconds']=ledger.elapsed;record['cost_prefix']['seconds']=record['completed_seconds'];candidates.append(record)
            from .d0_adapter import decode_task as decode
            result=decode(scene,parent,old,ledger,geometry,SearchConfig(task_candidate_limit=ledger.config.candidate_limit),retain)
            reason=result.reason
        elif variant=='D1':
            if task.status=='unsupported_active_set':reason='not_supported_active_set'
            elif task.status=='not_supported_no_wider_menu':reason=task.status
            else:_,reason=_d1(scene,parent,task,ledger,geometry,candidates)
        else:raise ValueError('unknown decoder variant')
    except BudgetStop as error:reason=str(error)
    censored=reason in ('time_budget','candidate_budget','geometry_budget','ray_pair_budget','stop_requested','region_user_limit')
    status='censored_time' if reason=='time_budget' else 'censored_region_limit' if reason=='region_user_limit' else 'censored_work' if censored else 'terminal_unsupported_active_set' if 'active_set' in reason or 'deleted_fixed' in reason else 'terminal_structurally_unsupported' if reason.startswith('not_supported') else 'completed_with_candidate' if any(c['metrics'] and not c['late'] for c in candidates) else 'completed_no_feasible_candidate'
    proof=dict(region_count=len(task.region_ids),target_k=task.target_k,region_user_count=len(task.active_users)+len(task.fixed_users),result_N=len(parent.centers)-len(task.region_ids)+task.target_k,beam_cap=min(200,len(scene.demands)),mandatory_count=len(task.features['active_mandatory_aircraft_ids']),mandatory_aircraft_ids=task.features['active_mandatory_aircraft_ids'],active_limit=ledger.config.active_limit,reason=reason)
    return dict(task=task.to_record(),variant=variant,parent_sha=common.parent_sha,reason=reason,status=status,predecode_proof=proof,candidates=candidates,budget=ledger.record(),complete_candidates=len(candidates),closed=not censored,supported=not reason.startswith('not_supported'))
