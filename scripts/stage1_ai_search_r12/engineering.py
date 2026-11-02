"""Separate root-admitted synthetic8-plane E1/E2/E3/E4 driver; never science.

It calls production describe/evaluate with the fixture's explicit fixed region.
No real mother loader, tree exploration, historical solver or learning exists.
"""
import argparse
from datetime import datetime, timezone
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time
from .admission import DEADLINE, STOP_NEW, pause_inventory, require_execution_design
from .persistence import Journal, content_sha
from .worker import pinned, verify_imports, source_check, deadline_registration, design_documents, bound_parent, one_thread, _markers, _rss, _birth, _reap, _storage

def validate(plan,review,now):
    require_execution_design(plan)
    if plan.get('schema')!='R12EngineeringPlanV2' or plan.get('group')!='S1-AI17-R12':raise ValueError('separate engineering plan')
    if plan.get('executor_port') not in (20473,20474) or plan.get('pause_deadline_utc')!='2026-10-14T23:00:00Z':raise ValueError('candidate/pause binding')
    if now>=STOP_NEW or (DEADLINE-now).total_seconds()<2550:raise ValueError('full prospective envelope/pause')
    pause_inventory(plan)
    if any(v is not None for v in _markers(plan).values()):raise ValueError('fresh persistent pause markers')
    if review.get('schema')!='RootR12EngineeringExecutionAdmissionV2' or review.get('scientific_execution_authorized') is not False:raise ValueError('engineering-only root admission')
    if any(review.get(k) is not True for k in ('engineering_execution_authorized','source_accepted','fixture_accepted','runtime_accepted','pause_accepted','disk_accepted')):raise ValueError('engineering exact qualification')
    if review.get('plan_content_SHA')!=plan.get('content_SHA') or content_sha({k:v for k,v in plan.items() if k!='content_SHA'})!=plan['content_SHA']:raise ValueError('engineering content binding')
    if plan.get('budget')!=dict(kernels=3,flow_queries=24,seconds=30,RSS_bytes=2147483648):raise ValueError('exact engineering finite budget')
    route=review['fresh_route'];age=now.timestamp()-route['observed_unix_seconds']
    if not -5<=age<=120 or route['port']!=plan['executor_port'] or route.get('conflict_free') is not True or route.get('free_bytes',0)<8589934592:raise ValueError('fresh actual engineering owner/resource/space')

def _child(plan,root,expected_parent):
    bound_parent(expected_parent)
    one_thread()
    import numpy as np
    import psutil
    from types import SimpleNamespace
    from satellite_rl.coverage.geometry import satellite_rays_numpy
    from satellite_rl.coverage.ai_search_r9.core import audit
    from satellite_rl.coverage.ai_search_r12.core import describe,evaluate
    from satellite_rl.coverage.ai_search_r12.ledger import Ledger,MotherLedger
    from .geometry_reader import verify_flow_proofs
    verify_imports(plan)
    fixture=json.loads(pinned(plan['fixture_ref']))
    if fixture.get('schema')!='R12EngineeringFixtureV2' or len(fixture['aircraft_ids'])!=8 or [c['case'] for c in fixture['cases']]!=['E1','E2','E3','E4']:
        raise ValueError('exact original8-plane four-case fixture v2')
    journal=Journal(root);began=time.monotonic();kernels=flows=0
    provenance=dict(native_pid=os.getpid(),parent_pid=os.getppid(),native_birth_unix=psutil.Process().create_time(),
                    boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
    try:
        rays=satellite_rays_numpy(np.asarray(fixture['positions'],dtype=np.float64),'europe_v1')
        scene=SimpleNamespace(aircraft_ids=tuple(fixture['aircraft_ids']),positions=np.asarray(fixture['positions'],dtype=np.float64),
                    rays=rays,demands=np.asarray(fixture['demands'],dtype=np.float64),input_sha256=plan['fixture_ref']['sha256'])
        journal.write('fixture_raw',dict(fixture=fixture,rays=rays.tolist(),fixture_ref=plan['fixture_ref']))
        for case in fixture['cases']:
            def stop():return datetime.now(timezone.utc)>=STOP_NEW or any(v is not None for v in _markers(plan).values())
            if stop():raise RuntimeError('engineering pause; no next fixture case')
            identity=dict(case_id='R12-engineering-'+case['case'],mother_id='synthetic8',profile='uniform21600000',role='train',
                input_sha256=plan['fixture_ref']['sha256'],identity_sha256=content_sha(['fixture',case['case']]),
                shared_case_content_sha256=plan['fixture_ref']['sha256'],source_asset_SHA=plan['fixture_ref']['sha256'])
            preparation=MotherLedger(8,stop=stop)
            baseline=audit(scene,fixture['layout'],fixture['Ncap'],case['Omega_cap'],preparation)
            origin=dict(identity=identity,layout=fixture['layout'],layout_SHA=content_sha(fixture['layout']),
                        metrics=baseline,Ncap=fixture['Ncap'],Omega_cap=case['Omega_cap'])
            current=dict(layout=fixture['layout'],metrics=baseline)
            descriptor=describe(scene,current,origin,preparation,fixed_region=fixture['fixed_region'],fixed_target=case['target_k'])[0]
            journal.write('preparation_'+case['case'],dict(origin=origin,descriptor=descriptor,ledger=preparation.record()))
            journal.callback('entered',dict(path=case['case'],semantic_SHA=descriptor['semantic_SHA'],descriptor=descriptor,state='unknown',complete=False))
            try:node=evaluate(scene,origin,current,descriptor,stop=stop)
            except BaseException as error:
                journal.callback('unknown',dict(path=case['case'],node=getattr(error,'partial_node',dict(semantic_SHA=descriptor['semantic_SHA'],complete=False,state='unknown'))));raise
            journal.callback('completed',dict(path=case['case'],node=node))
            kernels+=int(node['kernel_entered']);flows+=node['ledger']['flow_queries']
            if kernels>3 or flows>24:raise RuntimeError('engineering aggregate budget')
            if node['state']!=case['expected_state'] or int(node['kernel_entered'])!=case['expected_kernel_entries'] or node['ledger']['flow_queries']!=case['expected_flow_calls']:
                raise AssertionError('frozen engineering state/cost '+case['case'])
            if case['case']=='E3':
                if descriptor['unavailable_reasons']!=case['expected_reason'] or node['terminal']!=fixture['layout']:raise AssertionError('legal B known floorOmega prune')
            else:
                matching=node['raw_local_matching']
                if matching['anchor_ids']!=case['expected_anchor_ids'] or matching['assignment']!=case['expected_matching_assignment'] or node['flow_capacity']!=case['expected_capacity']:raise AssertionError('frozen anchors/capacity/matching')
                if node['raw_adjacency']!=[[True]*case['target_k'] for _ in range(8)]:raise AssertionError('fixture full adjacency')
                verify_flow_proofs(matching['member_ids'],matching['anchor_ids'],node['raw_adjacency'],node['ledger']['flow_results'],node['flow_capacity'],matching['assignment'])
                if [q['q'] for q in node['ledger']['flow_results']]!=case['expected_flow_queries']:raise AssertionError('frozen flow query order')
                if np.bincount(matching['assignment'],minlength=case['target_k']).tolist()!=case['expected_counts']:raise AssertionError('frozen exact counts')
                if node['raw_layout']['widths_deg']!=case['expected_widths_deg'] or node['metrics']['maxload']!=case['expected_maxload'] or node['metrics']['total_request']!=case['expected_total_request']:raise AssertionError('frozen true width/request metrics')
            journal.write('assertion_'+case['case'],dict(passed=True,actual_state=node['state'],kernels_cumulative=kernels,flow_queries_cumulative=flows))
        if kernels!=3 or flows>24:raise AssertionError('exactly3 kernel extension calls')
        journal.write('native_exit',dict(complete=True,exit_code=0,kernels=kernels,flow_queries=flows,seconds=time.monotonic()-began,**provenance))
    except BaseException as error:
        journal.write('native_exit',dict(complete=False,error_type=type(error).__name__,error=str(error),kernels=kernels,flow_queries=flows,seconds=time.monotonic()-began,**provenance));raise
    finally:journal.close()

def run(plan,review,archive):
    began=time.monotonic()
    validate(plan,review,datetime.now(timezone.utc))
    design_documents(plan)
    deadline_registration(plan)
    if Path('/proc/sys/kernel/random/boot_id').read_text().strip()!=review['fresh_route']['boot_id'] or Path(sys.executable).resolve()!=Path(plan['python']).resolve():raise ValueError('actual engineering boot/interpreter')
    source_check(plan)
    fixture=json.loads(pinned(plan['fixture_ref']))
    if fixture['schema']!='R12EngineeringFixtureV2':raise ValueError('fixture version2 only')
    root=Path(archive)
    if root.resolve()!=Path(plan['archive_root']).resolve() or not root.resolve().is_relative_to(Path(plan['storage_root']).resolve()):raise ValueError('exact engineering owned output/storage')
    if root.exists():raise ValueError('one new engineering output; no replay')
    root.mkdir(parents=True);journal=Journal(root/'parent');process=None
    deadline=min(began+30,time.monotonic()+(DEADLINE-datetime.now(timezone.utc)).total_seconds());peak=0;samples=0;birth=None;reason=None
    try:
        process=mp.get_context('spawn').Process(target=_child,args=(plan,str(root/'child'),os.getpid()));process.start();birth=_birth(process.pid)
        while process.is_alive():
            peak=max(peak,_rss(process.pid,birth));samples+=1;_storage(plan['storage_root'])
            if time.monotonic()-began>=15:reason='engineering30s_includes15s_reap_margin'
            elif peak>2147483648:reason='engineering_parent_children_RSS2GiB'
            elif datetime.now(timezone.utc)>=STOP_NEW or any(v is not None for v in _markers(plan).values()):reason='engineering_pause'
            if reason:break
            process.join(timeout=.1)
    finally:
        if process is not None:_reap(process,deadline)
        complete=process is not None and process.exitcode==0 and reason is None and samples>0 and 0<peak<=2147483648 and time.monotonic()-began<=30
        if complete:
            native=json.loads((root/'child'/'native_exit.json').read_text(encoding='utf8'))
            complete=(native.get('complete') is True and native.get('kernels')==3 and native.get('flow_queries',25)<=24
                and native.get('native_pid')==process.pid and native.get('parent_pid')==os.getpid()
                and native.get('native_birth_unix')==birth and native.get('boot_id')==review['fresh_route']['boot_id'])
        journal.write('guard_receipt',dict(complete=complete,exit_code=None if process is None else process.exitcode,reason=reason,
            native_pid=None if process is None else process.pid,native_birth_unix=birth,actual_parent_pid=os.getpid(),
            boot_id=review['fresh_route']['boot_id'],seconds=time.monotonic()-began,RSS_samples=samples,
            peak_parent_children_RSS_bytes=peak,actual_process_reaped=process is not None and not process.is_alive(),automatic_retry=False))
        journal.close()
    if not complete:raise RuntimeError('engineering incomplete; preserve raw and no retries')

def main():
    parser=argparse.ArgumentParser(description='Separate frozen synthetic8-plane3-kernel engineering only')
    for key in ('plan','plan-sha256','review','review-sha256','archive'):parser.add_argument('--'+key,required=True)
    args=parser.parse_args()
    def load(path,sha):return json.loads(pinned(dict(path=path,sha256=sha,bytes=Path(path).stat().st_size)))
    run(load(args.plan,args.plan_sha256),load(args.review,args.review_sha256),args.archive)

if __name__=='__main__':main()
