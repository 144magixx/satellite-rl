"""Explicitly admitted one-shot production batch; never launches on import.

Input preparation must independently pin normalized full-array scene/origin JSON.
No SSH, POST, controller, legacy RULE/D5 or training route exists here.
"""
import argparse
import hashlib
import json
import multiprocessing as mp
import os
import shutil
import sys
from pathlib import Path
import time
from datetime import datetime, timezone
from .admission import check, STOP_NEW, GLOBAL_PAUSE
from .persistence import Journal, content_sha,identity
from .runtime_epoch import marker_values,stopped,absolute_deadline,validate as validate_epoch,prior_dispatch_absent

def pinned(ref):
    if type(ref.get('bytes')) is not int or not 0<=ref['bytes']<=134217728:raise ValueError('bounded exact content reference')
    path=Path(ref['path'])
    with path.open('rb') as stream:raw=stream.read(ref['bytes']+1)
    if len(raw)!=ref['bytes'] or hashlib.sha256(raw).hexdigest()!=ref['sha256']:
        raise ValueError('exact content mismatch: '+str(path))
    return raw

def load_case(adapter_ref):
    """Pinned old source -> explicit in-memory view, no new file/ray generation."""
    adapter=json.loads(pinned(adapter_ref))
    if adapter.get('schema')!='R12OriginalInputAdapterV1':raise ValueError('small root-qualified original-input adapter')
    original=json.loads(pinned(adapter['source_original_input_ref']))
    if original.get('schema')!='R10ExistingSceneAndBestOriginInputV1' or original['original_index']!=adapter['original_index']:raise ValueError('original self-contained R10 case')
    scene=original['scene'];origin=original['origin']
    if scene['profile']!=adapter['geometry_profile'] or len(scene['aircraft_ids'])!=adapter['aircraft_count']:raise ValueError('full original source geographic profile/count')
    if origin['identity']!=adapter['expected_origin_identity'] or origin['layout_SHA']!=adapter['expected_B_layout_SHA'] or content_sha(origin['layout'])!=origin['layout_SHA']:raise ValueError('original qualified B and identity')
    if origin['Ncap']!=adapter['expected_Ncap'] or origin['Omega_cap']!=adapter['expected_Omega_cap']:raise ValueError('original caps')
    return dict(adapter,scene={k:scene[k] for k in ('aircraft_ids','positions','demands','input_sha256')},origin=origin)

def audited_S_rows(plan):
    from .historical_s import audited_table
    ref=plan['historical_S_counts_ref']
    if ref.get('sha256')!='9537b72ecf7062c5f4de769b13032491db436db780cdcc3f55a266b48c22d68a' or ref.get('bytes')!=52037:raise ValueError('sole root-accepted original S count qualification asset')
    accepted_ref=plan['historical_S_counts_acceptance_ref']
    if accepted_ref.get('sha256')!='b4f0dec71d8f43e5e1ed463729b9ef20c47453cf2b7ef5c4a624ad20bf95fe60':raise ValueError('root actual old S acceptance')
    accepted=json.loads(pinned(accepted_ref))
    if (accepted.get('schema')!='RootR12ActualOldStrongSCountsAcceptanceV1' or accepted.get('accepted') is not True
        or identity(accepted['audited_counts_ref'])!=identity(ref) or accepted.get('reader_remaining_seconds')!=110):raise ValueError('accepted source identity/remaining original reader budget')
    value=json.loads(pinned(ref))
    return audited_table(value,plan['reader_prior_seconds'],plan['historical_control_ref'])

def remaining_reader_seconds(plan):
    prior=plan['reader_prior_seconds']
    if prior!=10:raise ValueError('original120 reader includes conservative10s S-source qualification')
    return 120-prior

def _markers(plan):
    return marker_values(plan)

def deadline_registration(plan):
    if 'runtime_epoch' in plan:
        validate_epoch(plan)
        return plan['runtime_epoch']
    value=json.loads(pinned(plan['deadline_registration_ref']))
    if (value.get('schema')!='S1AI17NewDeadlineRegisteredScopeV1' or value.get('root_accepted') is not True
        or value.get('group')!='S1-AI17' or value.get('deadline_unix_seconds')!=1792018800.
        or value.get('global_pause_path')!=GLOBAL_PAUSE or value.get('ssh_port')!=plan['executor_port']
        or value.get('run_id')!=plan['run_id'] or value.get('archive_root')!=plan['archive_root']
        or value.get('coverage')!='cooperative_watch_and_absolute_deadline'):
        raise ValueError('exact root registration to new15th deadline required')
    return value

def design_documents(plan):
    from .admission import require_execution_design,EXECUTION_DESIGN_VERSION
    require_execution_design(plan)
    design=json.loads(pinned(plan['design_ref']));accepted=json.loads(pinned(plan['root_design_acceptance_ref']))
    if design.get('schema')!=EXECUTION_DESIGN_VERSION or accepted.get('schema')!='RootR12ProspectiveDesignAcceptanceV2' or accepted.get('prospective_design_accepted') is not True:
        raise ValueError('actual exact accepted V2 design documents')
    if accepted.get('design_ref',{}).get('sha256')!=plan['design_ref']['sha256']:raise ValueError('root V2 design content binding')

def verify_imports(plan):
    repo=Path(__file__).resolve().parents[2]
    expected=plan['source_files']
    for name,module in list(sys.modules.items()):
        if not name.startswith(('scripts.stage1_ai_search_r12','satellite_rl')):continue
        filename=getattr(module,'__file__',None)
        if not filename:continue
        path=Path(filename).resolve()
        if not path.is_relative_to(repo):raise ValueError('cached scientific import outside admitted source')
        key=path.relative_to(repo).as_posix()
        if key not in expected or Path(expected[key]['path']).resolve()!=path:raise ValueError('actual imported source unpinned: '+key)
        pinned(expected[key])

def source_check(plan):
    from .dependency_closure import project_closure
    repo=Path(__file__).resolve().parents[2]
    for relative,ref in plan['source_files'].items():
        if '..' in Path(relative).parts or Path(ref['path']).resolve()!=(repo/relative).resolve():raise ValueError('actual frozen source location')
        pinned(ref)
    required={'scripts/stage1_ai_search_r12/'+name for name in ('__init__.py','admission.py','persistence.py','reader.py','geometry_reader.py','collect.py','worker.py','engineering.py','server.py','normalize.py')}
    for package in ('ai_search_r12','ai_search_r10','ai_search_r9'):
        required.update(p.relative_to(repo).as_posix() for p in (repo/'src/satellite_rl/coverage'/package).glob('*.py'))
    required.update(('src/satellite_rl/coverage/geometry.py','src/satellite_rl/envs/regions.py'))
    required.update(('src/satellite_rl/__init__.py','src/satellite_rl/coverage/__init__.py',
                     'src/satellite_rl/coverage/scenes.py','src/satellite_rl/coverage/environment.py',
                     'src/satellite_rl/coverage/baselines.py','src/satellite_rl/envs/__init__.py'))
    required.update(project_closure(repo))
    if not required<=plan['source_files'].keys():raise ValueError('complete frozen runtime source closure')

def bound_parent(expected):
    import ctypes
    import signal
    if ctypes.CDLL(None,use_errno=True).prctl(1,signal.SIGKILL,0,0,0)!=0 or os.getppid()!=expected:
        os._exit(126)

def one_thread():
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[key]='1'

def _child(input_ref, root, plan, expected_parent, expected_n):
    bound_parent(expected_parent)
    one_thread()
    # No geometry/solver import occurs before root admission in parent.
    import numpy as np
    from types import SimpleNamespace
    from satellite_rl.coverage.ai_search_r12.core import explore_tree,select_controls
    from satellite_rl.coverage import geometry  # pin lazy original primitive before work
    verify_imports(plan)
    if Path(sys.executable).resolve()!=Path(plan['python']).resolve():raise ValueError('actual admitted interpreter')
    value=load_case(input_ref);scene=dict(value['scene'])
    if 'rays' in scene:raise ValueError('R12 normalized IO must not precompute production geometry')
    if len(scene['aircraft_ids'])!=expected_n:raise ValueError('exact full original aircraft count before new rays')
    for key in ('positions','demands'):scene[key]=np.asarray(scene[key],dtype=np.float64)
    scene['aircraft_ids']=tuple(scene['aircraft_ids'])
    journal=Journal(root)
    def stop():
        return stopped(plan)
    began=time.monotonic()
    import psutil
    provenance=dict(native_pid=os.getpid(),parent_pid=os.getppid(),native_birth_unix=psutil.Process().create_time(),
                    boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
    import signal
    def timeout(signum,frame):raise TimeoutError('R12 per-kernel30s including durable save')
    signal.signal(signal.SIGALRM,timeout)
    def persist(event,payload):
        if event=='entered':
            if stopped(plan,force=True):raise RuntimeError('pause before new native node')
            signal.setitimer(signal.ITIMER_REAL,30)
        journal.callback(event,payload)
        if event in ('completed','unknown'):signal.setitimer(signal.ITIMER_REAL,0)
    try:
        from .historical_s import audited_selected
        verify_imports(plan)
        if stopped(plan,force=True):raise RuntimeError('pause before producer new scene geometry')
        saved_S=audited_selected(audited_S_rows(plan)[value['original_index']],value)
        from satellite_rl.envs.regions import get_profile
        profile=get_profile(value['geometry_profile'])
        actual=dict(earth_radius_km=profile.earth_radius_km,orbit_radius_km=profile.orbit_radius_km,
                    satellite_longitude_deg=profile.satellite_longitude_deg)
        if actual!=value['physical_parameters']:raise ValueError('original production profile exact frozen physical constants')
        journal.write('scene_preparation_entered',dict(new_rays_reserved=expected_n,input_ref=input_ref,
                    original_source_input_ref=value['source_original_input_ref'],
                    geometry_profile=value['geometry_profile'],physical_parameters=actual))
        ray_began=time.monotonic()
        if stop():raise RuntimeError('pause before producer ray construction')
        scene['rays']=geometry.satellite_rays_numpy(scene['positions'],profile=value['geometry_profile'])
        journal.write('scene_preparation_completed',dict(new_rays=expected_n,rays=scene['rays'].tolist(),
                    input_ref=input_ref,geometry_profile=value['geometry_profile'],physical_parameters=actual,
                    original_source_input_ref=value['source_original_input_ref'],
                    observed_seconds=time.monotonic()-ray_began))
        scene=SimpleNamespace(**scene)
        tree=explore_tree(scene,value['origin'],persist,stop=stop)
        if stop():raise RuntimeError('pause before post-tree exact control selection')
        controls=select_controls(value['origin'],tree,saved_S)
        # Canonical full raw nodes already committed by callback; index them
        # instead of serializing all aircraft/owners again in a second tree.
        index={k:tree[k] for k in ('complete','identity','origin_SHA','ledger','new_model_calls')}
        index['tree_ledger']=tree['ledger'];index['ledger']=controls['ordering_ledger']
        def compact(state):
            return {key:state[key] for key in ('metrics','exact_evidence','semantic_SHA','method') if key in state}
        index.update(schema='R12PersistedTreeIndexV2',original_tree_schema=tree['schema'],
            nodes=[dict(semantic_SHA=n['semantic_SHA'],source_content_id=n['source_content_id']) for n in tree['nodes']],
            paths={path:dict(semantic_SHA=v['semantic_SHA']) for path,v in tree['paths'].items()},
            controls={arm:compact(controls[arm]) for arm in ('S','H','G','Q','O1','O2')},
            fixed_controls={seq:compact(state) for seq,state in controls['F'].items()},logical_costs=controls['logical_costs'])
        journal.write('tree',index)
        journal.write('native_exit',dict(complete=True,exit_code=0,seconds=time.monotonic()-began,**provenance))
    except BaseException as error:
        journal.write('native_exit',dict(complete=False,error_type=type(error).__name__,
                                        error=str(error),seconds=time.monotonic()-began,**provenance))
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        journal.close()

def _rss(pid,expected_birth):
    """Linux parent+children RSS; missing/unreadable process accounting is unknown."""
    import psutil
    process=psutil.Process(pid)
    if expected_birth is None or process.create_time()!=expected_birth:raise RuntimeError('native RSS generation unknown')
    parent=psutil.Process(os.getpid())
    owned={p.pid:p for p in [parent]+parent.children(recursive=True)}
    owned[process.pid]=process
    total=0
    for p in owned.values():
        before=p.create_time();total+=p.memory_info().rss
        if psutil.Process(p.pid).create_time()!=before:raise RuntimeError('RSS descendant generation changed')
    if psutil.Process(pid).create_time()!=expected_birth:raise RuntimeError('native RSS generation changed after sample')
    return total

def _birth(pid):
    import psutil
    return psutil.Process(pid).create_time()

def _reap(proc,absolute_deadline,grace=0):
    """All cooperative grace/terminate/kill share the existing absolute budget."""
    remaining=lambda:max(0.,absolute_deadline-time.monotonic())
    if proc.is_alive() and grace:
        proc.join(timeout=min(grace,max(0.,remaining()-15)))
    if proc.is_alive():proc.terminate();proc.join(timeout=min(10.,remaining()))
    if proc.is_alive():proc.kill();proc.join(timeout=min(5.,remaining()))
    if proc.is_alive():raise RuntimeError('native reap unknown at absolute deadline')

def _reader_child(plan,archive,expected_parent):
    bound_parent(expected_parent)
    one_thread()
    from .collect import collect
    verify_imports(plan)
    collect(plan,archive)

def _storage(root):
    """Bounded metadata space observation, never a scientific payload audit."""
    import stat
    from .runtime_epoch import ROOT
    root=Path(root)
    if not stat.S_ISDIR(root.lstat().st_mode):raise RuntimeError('owned storage root must be regular directory')
    journals={Path(ROOT)/'service',Path(ROOT)/'science'/'batch',Path(ROOT)/'science'/'reader'}
    from .admission import INDICES
    journals.update(Path(ROOT)/'science'/('mother_'+str(index)) for index in INDICES)
    rescan=None
    def walk_error(error):raise error
    def volatile(path):
        if path.parent not in journals:return False
        if path.name=='owner.lock':return True
        key=path.name[:-8] if path.name.endswith('.pending') else ''
        return bool(key) and all(c in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in key)
    for attempt in range(2):
        persistent=temporary=members=0
        collision=None
        for directory,dirs,files in os.walk(root,followlinks=False,onerror=walk_error):
            for name in dirs:
                path=Path(directory)/name
                if not stat.S_ISDIR(path.lstat().st_mode):raise RuntimeError('linked/nonregular owned directory unknown')
            for name in files:
                path=Path(directory)/name;members+=1
                if members>16384:raise RuntimeError('owned inventory bounded regular files only')
                try:info=path.lstat()
                except FileNotFoundError:
                    if attempt or not volatile(path):raise RuntimeError('owned inventory missing or repeated volatile collision')
                    collision=str(path);break
                if not stat.S_ISREG(info.st_mode):raise RuntimeError('linked/nonregular owned file unknown')
                if path.suffix=='.pending':temporary+=info.st_size
                else:persistent+=info.st_size
            if collision is not None:break
        if collision is not None:
            rescan=dict(path=collision,reason='legal_Journal_pending_publish_or_owner_close')
            # Small metadata only; preserve the attempted observation even when
            # the second pass fails. No payload read, watcher, or extra process.
            print(json.dumps(dict(event='storage_metadata_rescan',**rescan)),flush=True)
            continue
        free=shutil.disk_usage(root).free
        if persistent>134217728 or temporary>67108864 or free<8589934592:
            raise RuntimeError('storage cap/reserve exceeded; preserve evidence and stop')
        return dict(persistent_bytes=persistent,temporary_bytes=temporary,mount_free_bytes=free,
                    quota='not inferred from mount free',inventory_passes=attempt+1,
                    rescan=rescan,scientific_payload_audited=False)
    raise RuntimeError('storage metadata second pass incomplete')

def run(plan,review,archive):
    now=datetime.now(timezone.utc)
    check(plan,review,now=now,marker_values=_markers(plan))
    if 'runtime_epoch' in plan:prior_dispatch_absent(plan)
    design_documents(plan)
    deadline_registration(plan)
    if Path('/proc/sys/kernel/random/boot_id').read_text().strip()!=review['fresh_route']['boot_id']:
        raise ValueError('actual boot changed after fresh root resource review')
    if Path(sys.executable).resolve()!=Path(plan['python']).resolve() or review['fresh_route']['python']!=plan['python']:
        raise ValueError('actual root-admitted interpreter route')
    body={k:v for k,v in plan.items() if k!='content_SHA'}
    if content_sha(body)!=plan['content_SHA']:raise ValueError('plan content self binding')
    source_check(plan)
    audited_S_rows(plan)
    if review.get('reader_accepted') is not True:raise ValueError('bounded automatic reader must be prospectively accepted')
    if len(plan['cases'])!=12 or [c['original_index'] for c in plan['cases']]!=plan['indices']:
        raise ValueError('full ordered12 normalized inputs')
    if any(type(c.get('aircraft_count')) is not int or c['aircraft_count']<1 for c in plan['cases']) or sum(c['aircraft_count'] for c in plan['cases'])!=4715:raise ValueError('exact original4715 ray construction budget')
    output=Path(archive);storage_root=Path(plan['storage_root'])
    if output.resolve()!=Path(plan['archive_root']).resolve() or not output.resolve().is_relative_to(storage_root.resolve()):raise ValueError('exact owned archive and shared storage inventory')
    if output.exists():raise ValueError('one-shot output must be new; unknown cannot replay')
    output.mkdir(parents=True)
    owner=Journal(output/'batch')
    owner.write('runtime_binding',dict(actual_pid=os.getpid(),python=sys.executable,
                boot_id=review['fresh_route']['boot_id'],task_id=review['fresh_route']['task_id'],
                plan_content_SHA=plan['content_SHA'],candidate_port=plan['executor_port']))
    total=0.;ctx=mp.get_context('spawn')
    try:
        for case in plan['cases']:
            owner.write('storage',_storage(storage_root))
            if stopped(plan,force=True):
                raise RuntimeError('cooperative pause before next mother')
            root=output/('mother_'+str(case['original_index']))
            owner.write('ray_reservation_'+str(case['original_index']),dict(new_rays_reserved=case['aircraft_count'],input_ref=case['input_adapter_ref']))
            proc=ctx.Process(target=_child,args=(case['input_adapter_ref'],str(root),plan,os.getpid(),case['aircraft_count']))
            began=time.monotonic();proc.start();reason=None;peak_rss=0;samples=0;birth=None
            deadline=min(began+180,began+2160-total,absolute_deadline(plan))
            try:
                birth=_birth(proc.pid)
                while proc.is_alive():
                    _storage(storage_root)
                    elapsed=time.monotonic()-began
                    # Kill margin is inside the180s mother envelope; unknown
                    # interrupted nodes remain charged and cannot be replayed.
                    peak_rss=max(peak_rss,_rss(proc.pid,birth));samples+=1
                    if elapsed>=165 or total+elapsed>=2145:reason='native_time_save_margin'
                    elif peak_rss>2147483648:reason='parent_children_RSS_ceiling'
                    elif stopped(plan):reason='pause_stop_new_and_save'
                    if reason:break
                    proc.join(timeout=.1)
            finally:
                _reap(proc,deadline,30 if reason=='pause_stop_new_and_save' else 0)
            total+=time.monotonic()-began
            owner.write('mother_'+str(case['original_index']),dict(exit_code=proc.exitcode,
                        native_pid=proc.pid,parent_pid=os.getpid(),peak_parent_children_RSS_bytes=peak_rss,
                        native_birth_unix=birth,RSS_samples=samples,boot_id=review['fresh_route']['boot_id'],
                        reason=reason,seconds=time.monotonic()-began,cumulative_seconds=total))
            if reason or proc.exitcode!=0 or samples<1 or peak_rss<=0 or time.monotonic()-began>180 or total>2160:raise RuntimeError('incomplete native exit/time/RSS; no retry')
        owner.write('production_complete',dict(complete=True,seconds=total,
                    reader_pending=True,scientific_gate_pending=True))
        if stopped(plan,force=True):raise RuntimeError('pause before reader; no implicit recovery')
        proc=ctx.Process(target=_reader_child,args=(plan,str(output),os.getpid()));began=time.monotonic();proc.start();reason=None;peak_rss=0;samples=0;birth=None
        reader_remaining=remaining_reader_seconds(plan)
        deadline=min(began+reader_remaining,absolute_deadline(plan))
        try:
            birth=_birth(proc.pid)
            while proc.is_alive():
                _storage(storage_root)
                peak_rss=max(peak_rss,_rss(proc.pid,birth));samples+=1
                if time.monotonic()-began>=reader_remaining-15:reason='reader_original_remaining110s_reap_margin'
                elif peak_rss>2147483648:reason='reader_RSS2GiB'
                elif stopped(plan):reason='pause_before_reader_checkpoint'
                if reason:break
                proc.join(timeout=.1)
        finally:
            _reap(proc,deadline)
        owner.write('reader_native_exit',dict(exit_code=proc.exitcode,reason=reason,seconds=time.monotonic()-began,
                    native_pid=proc.pid,parent_pid=os.getpid(),peak_parent_children_RSS_bytes=peak_rss,
                    native_birth_unix=birth,RSS_samples=samples,boot_id=review['fresh_route']['boot_id']))
        if reason or proc.exitcode!=0 or samples<1 or peak_rss<=0 or time.monotonic()-began>reader_remaining:raise RuntimeError('reader incomplete/time/RSS; no retries or next stage')
        owner.write('complete',dict(complete=True,production_seconds=total,reader_seconds=time.monotonic()-began,
                                  training_auto_start=False))
    finally:owner.close()

def main():
    parser=argparse.ArgumentParser(description='R12 exact root-admitted production only')
    parser.add_argument('--plan',required=True);parser.add_argument('--review',required=True)
    parser.add_argument('--plan-sha256',required=True);parser.add_argument('--review-sha256',required=True)
    parser.add_argument('--archive',required=True)
    args=parser.parse_args()
    def load(path,sha):return json.loads(pinned(dict(path=path,sha256=sha,bytes=Path(path).stat().st_size)))
    run(load(args.plan,args.plan_sha256),load(args.review,args.review_sha256),args.archive)

if __name__=='__main__':main()
