from dataclasses import asdict,replace
from time import perf_counter
from functools import cmp_to_key
import numpy as np
from .config import TaskConfig,Ledger,BudgetStop
from .initials import digest,CommonStart
from .tasks import make_pool
from .decoder import solve_task
from ..ai_search.search import metric_tolerance
from ..ai_search.config import SearchConfig
from ..ai_search.core import layout_sha256,Layout
from ..ai_search.search import layout_delta,add_archive

def archive_config(config,common_count):
    # The complete finite pool has at most this many points; never inherits
    # R1's default 64-point crowding cut into the R2 scientific archive.
    return replace(SearchConfig(),archive_limit=max(64,8*config.candidate_limit+common_count))

def mechanism(parent,candidate,form):
    before={identity:i for i,identity in enumerate(parent.beam_ids)};after={identity:i for i,identity in enumerate(candidate.beam_ids)};retained=sorted(set(before)&set(after))
    return dict(N_delta=len(candidate.centers)-len(parent.centers),retained_width_changed=[identity for identity in retained if abs(parent.widths[before[identity]]-candidate.widths[after[identity]])>1e-10],new_beam_widths={identity:float(candidate.widths[after[identity]]) for identity in after if identity not in before},deleted_beam_ids=[identity for identity in before if identity not in after],whole_layout_mixed_width=bool(len(candidate.widths) and candidate.widths.max()-candidate.widths.min()>1e-10),centers_changed=any(not np.array_equal(parent.centers[before[i]],candidate.centers[after[i]]) for i in retained),ownership_changed=any(parent.beam_ids[a]!=candidate.beam_ids[b] for a,b in zip(parent.assignment,candidate.assignment)),menu_vs_envelope=form)

def target_hit(values,parent,slice_metrics):
    cfg=SearchConfig()
    if parent['CV'] is None or values['CV'] is None:return False
    return (values['N']<=slice_metrics['N'] and
            values['solid_angle_sum_sr']<=slice_metrics['solid_angle_sum_sr']+metric_tolerance('solid_angle_sum_sr',values['solid_angle_sum_sr'],slice_metrics['solid_angle_sum_sr'],cfg) and
            values['maxload']<=.99*parent['maxload']+metric_tolerance('maxload',values['maxload'],.99*parent['maxload'],cfg) and
            values['top10_mean']<=parent['top10_mean']+metric_tolerance('top10_mean',values['top10_mean'],parent['top10_mean'],cfg) and
            values['CV']<=parent['CV']+metric_tolerance('CV',values['CV'],parent['CV'],cfg))

def prepare_pool(scene,common,config=TaskConfig(),stop=None,clock=None):
    if isinstance(config,dict):config=TaskConfig(**config)
    if isinstance(common,dict):common=CommonStart.from_record(common)
    cfg=replace(config,seconds=3.,ray_pair_limit=config.ray_pair_limit*8,geometry_limit=config.geometry_limit*8,candidate_limit=config.candidate_limit*8)
    ledger=Ledger(cfg,stop=stop,clock=clock)
    try:pool=make_pool(scene,common,config,ledger)
    except BudgetStop as error:
        return dict(status='incomplete_preparation',reason=str(error),input_sha=common.input_sha,parent_sha=common.parent_sha,archive_sha=common.archive_sha,tasks=[],pool_sha=None,preparation_cost=ledger.record(),config=asdict(config))
    rows=[t.to_record() for t in pool]
    return dict(status='prepared',reason=None,input_sha=common.input_sha,parent_sha=common.parent_sha,archive_sha=common.archive_sha,tasks=rows,pool_sha=digest(rows),preparation_cost=ledger.record(),config=asdict(config))

def evaluate_closed_pool(scene,common,variant,config=TaskConfig(),stop=None,progress=None,prepared_pool=None):
    if isinstance(config,dict):config=TaskConfig(**config)
    if isinstance(common,dict):common=CommonStart.from_record(common)
    start=perf_counter();initial_bytes=common.frozen_bytes
    # Shared feature preparation charged once, then deducted from the pool total.
    prep_cfg=replace(config,seconds=3.,ray_pair_limit=config.ray_pair_limit*8,geometry_limit=config.geometry_limit*8,candidate_limit=config.candidate_limit*8)
    prep=Ledger(prep_cfg,stop=stop)
    if prepared_pool is None:
        try:pool=make_pool(scene,common,config,prep)
        except BudgetStop as error:
            return dict(schema='s1_ai18_closed_pool_v1',variant=variant,input_sha=common.input_sha,archive_sha=common.archive_sha,parent_sha=common.parent_sha,pool_sha=None,tasks=[],results=[],closed=False,preparation_cost=prep.record(),reason=str(error),training_updates=0)
        prep_record=prep.record()
    else:
        from .tasks import Task
        prepared=prepared_pool
        if prepared['parent_sha']!=common.parent_sha or prepared['input_sha']!=common.input_sha or prepared['archive_sha']!=common.archive_sha or prepared['config']!=asdict(config):raise ValueError('prepared pool binding')
        if prepared.get('status')=='incomplete_preparation':
            cost=prepared['preparation_cost']
            return dict(schema='s1_ai18_closed_pool_v1',variant=variant,input_sha=common.input_sha,parent_sha=common.parent_sha,archive_sha=common.archive_sha,pool_sha=prepared['pool_sha'],tasks=prepared['tasks'],results=[],common=common.to_record(),closed=False,status='incomplete_preparation',reason=prepared['reason'],preparation_cost=cost,preparation_executed_here=False,preparation_reported_full_once_per_decoder=True,actual_cost={k:cost[k] for k in ('seconds','ray_pair_checks','geometry_calls','attempts')},actual_compute_cost=dict(end_to_end_seconds=perf_counter()-start,ray_pair_checks=0,geometry_calls=0,attempts=0),training_updates=0)
        if digest(prepared['tasks'])!=prepared['pool_sha']:raise ValueError('prepared pool SHA')
        pool=[Task.from_record(t) for t in prepared['tasks']];prep_record=prepared['preparation_cost']
    pool_sha=digest([t.to_record() for t in pool]);results=[]
    archive=[];parent_layout=common.primary_parent;archive_search=archive_config(config,len(common.archive))
    for row in common.archive:
        archive,_=add_archive(archive,dict(sha256=row['sha256'],metrics=row['metrics'],provenance='common'),archive_search)
    spent_pairs=prep_record['ray_pair_checks'];spent_calls=prep_record['geometry_calls'];spent_attempts=prep_record['attempts']
    for i,task in enumerate(pool):
        task_config=replace(config,ray_pair_limit=min(config.ray_pair_limit,config.ray_pair_limit*8-spent_pairs),geometry_limit=min(config.geometry_limit,config.geometry_limit*8-spent_calls),candidate_limit=min(config.candidate_limit,config.candidate_limit*8-spent_attempts))
        ledger=Ledger(task_config,stop=stop);r=solve_task(scene,common,task,variant,ledger)
        parent=common.record['primary_parent']['metrics'];sl=common.slice_ref['metrics'];hit=None
        for candidate_index,c in enumerate(r['candidates']):
            m=c['metrics'];c['within_primary_slice']=bool(m and m['N']<=sl['N'] and m['solid_angle_sum_sr']<=sl['solid_angle_sum_sr']+1e-12+1e-12*max(m['solid_angle_sum_sr'],sl['solid_angle_sum_sr']))
            c['within_joint_track']=bool(m and m['N']<=200)
            c['target_hit']=bool(m and not c['late'] and target_hit(m,parent,sl))
            if hit is None and c['target_hit']:
                hit={k:c[k] for k in ('sha256','completed_seconds','attempt')};hit['cost_prefix']=c['cost_prefix']
                hit['with_common_preparation']={k:c['cost_prefix'][k]+prep_record[k] for k in ('seconds','ray_pair_checks','geometry_calls','attempts')}
            layout=Layout.from_record(c.pop('layout'));c['delta']=layout_delta(parent_layout,layout)
            c['mechanism']=mechanism(parent_layout,layout,'envelope' if c.get('shrink') or c.get('width_form')=='shrink' else 'menu')
            if m and not c['late']:
                archive,retained=add_archive(archive,dict(sha256=c['sha256'],metrics=m,provenance=dict(task_id=task.task_id,candidate_index=candidate_index)),archive_search)
                c['archive_retained_at_trial']=retained
            else:c['archive_retained_at_trial']=False
        r['first_hit']=hit;results.append(r)
        spent_pairs+=ledger.ray_pair_checks;spent_calls+=ledger.geometry_calls;spent_attempts+=ledger.attempts
        if progress:progress(i+1,8,r)
        if common.frozen_bytes!=initial_bytes:raise ValueError('common bytes changed')
        if r['reason']=='stop_requested':break
    task_seconds=sum(r['budget']['seconds'] for r in results)
    sl=common.slice_ref['metrics'];slice_points=[r for r in archive if r['metrics']['N']<=sl['N'] and r['metrics']['solid_angle_sum_sr']<=sl['solid_angle_sum_sr']+metric_tolerance('solid_angle_sum_sr',r['metrics']['solid_angle_sum_sr'],sl['solid_angle_sum_sr'],SearchConfig())]
    def primary_compare(a,b):
        for field in ('maxload','top10_mean','CV','N','solid_angle_sum_sr'):
            x,y=a['metrics'][field],b['metrics'][field]
            if x is None or y is None:continue
            tolerance=metric_tolerance(field,x,y,SearchConfig())
            if x<y-tolerance:return -1
            if x>y+tolerance:return 1
        return (a['sha256']>b['sha256'])-(a['sha256']<b['sha256'])
    selected=min(slice_points,key=cmp_to_key(primary_compare)) if slice_points else dict(sha256=common.parent_sha,metrics=common.record['primary_parent']['metrics'],provenance='parent_fallback')
    final_hashes={r['sha256'] for r in archive}
    for task_result in results:
        for trial in task_result['candidates']:
            eligible=trial['metrics'] is not None and not trial['late']
            trial['layout_in_final_archive']=trial['sha256'] in final_hashes
            trial['archive_retained_final']=eligible and trial['layout_in_final_archive']
            trial['primary_slice_selected']=eligible and trial['sha256']==selected['sha256']
            trial['final_published']=trial['primary_slice_selected'] and not trial['late']
    return dict(schema='s1_ai18_closed_pool_v1',input_sha=common.input_sha,archive_sha=common.archive_sha,parent_sha=common.parent_sha,pool_sha=pool_sha,hashes_before=dict(input_sha=common.input_sha,archive_sha=common.archive_sha,parent_sha=common.parent_sha,pool_sha=pool_sha),hashes_after=dict(input_sha=common.input_sha,archive_sha=common.archive_sha,parent_sha=common.parent_sha,pool_sha=pool_sha),variant=variant,config=asdict(config),common=common.to_record(),tasks=[t.to_record() for t in pool],results=results,full_archive=archive,primary_slice_selected=selected,final_published=selected,preparation_cost=prep_record,preparation_reported_full_once_per_decoder=True,preparation_executed_here=prepared_pool is None,closed=len(results)==8 and all(r['closed'] for r in results),actual_cost=dict(seconds=task_seconds+prep_record['seconds'],ray_pair_checks=spent_pairs,geometry_calls=spent_calls,attempts=spent_attempts),actual_compute_cost=dict(end_to_end_seconds=perf_counter()-start,ray_pair_checks=spent_pairs-(prep_record['ray_pair_checks'] if prepared_pool else 0),geometry_calls=spent_calls-(prep_record['geometry_calls'] if prepared_pool else 0),attempts=spent_attempts),training_updates=0)
