"""Single-worker finite matrix, exact current binding, no hidden PHY replay."""
import os
from pathlib import Path
import pickle
from time import perf_counter
from satellite_rl.t0_width.data import read_json_ref,file_sha
from satellite_rl.t0_width.logging import atomic_json,content_sha
from .transaction import Journal,UnknownCost,fsync_directory
from .episode import run_episode
from .workbench import episode_identity
from .proposals import frozen_s10_probabilities
from .loading import load_policy,load_environment
from .binding import pack_environment,unpack_environment

def save_pickle(path,value):
    path=Path(path);tmp=path.with_suffix('.tmp')
    with tmp.open('wb') as stream:pickle.dump(value,stream,protocol=5);stream.flush();os.fsync(stream.fileno())
    os.replace(tmp,path)
    fsync_directory(path.parent)

def run_matrix(plan,root,stop_requested,progress=lambda value:None,*,policy_loader=load_policy,environment_loader=load_environment,baseline_factory=None):
    """Runtime functions are injectable for pure tests; production is root gated."""
    if plan.get('runtime_approved') is not True:raise ValueError('draft runtime is forbidden')
    root=Path(root);root.mkdir(parents=True,exist_ok=True);cursor_path=root/'matrix_cursor.pkl'
    if cursor_path.with_suffix('.tmp').exists():raise UnknownCost('unconfirmed matrix cursor write; preserve and audit')
    if cursor_path.exists():
        cursor=pickle.loads(cursor_path.read_bytes())
        if cursor['plan_SHA']!=content_sha(plan):raise ValueError('immutable matrix plan changed')
    else:
        if (root/'episodes').exists() and any((root/'episodes').iterdir()):raise UnknownCost('matrix cursor absent beside existing episode evidence; never reset')
        cursor=dict(schema='W28MatrixCursorV1',plan_SHA=content_sha(plan),cursor=0,active=None,
            costs=dict(query=0,commit=0,forward=0,proposal=0,kernel_preparation=0,restore_PHY=0),
            completed_episodes=0,elapsed_active_seconds=0.,timing=[],unknown=[])
        save_pickle(cursor_path,cursor)
    if cursor['unknown']:raise UnknownCost('unknown matrix load/restore cost must be independently accounted')
    cursor.setdefault('timing_unknown',[])
    for finished in range(cursor['cursor']):
        folder=root/'episodes'/('%04d'%finished)
        if not (folder/'episode_timing.json').exists() and finished not in cursor['timing_unknown']:
            cursor['timing_unknown'].append(finished)
            atomic_json(folder/'timing_gap.json',dict(stage='complete_episode_wall_unknown',matrix_cursor=finished,
                missing_unique_timing=True,unknown_is_not_zero=True,reexecution_allowed=False,
                budget_accounting='Root accounted recovery receipt must debit unrecorded wall upper bound against common6h'))
            fsync_directory(folder)
    save_pickle(cursor_path,cursor)
    manifest=read_json_ref(plan['data_manifest_ref']);rows=manifest['rows'];jobs=plan['matrix']['jobs']
    if len(rows)!=(20 if plan['phase']=='development' else 40):raise ValueError('exact partition required')
    if cursor['cursor']==len(jobs):return dict(stage='completed',cursor=cursor['cursor'],costs=cursor['costs'],timing_unknown=cursor['timing_unknown'])
    started=perf_counter();agent=None;policy_timing=None;episode_started=None;episode_elapsed_before=0.
    def checkpoint():save_pickle(cursor_path,cursor)
    def stop():return stop_requested() or cursor['elapsed_active_seconds']+perf_counter()-started>=plan['wall_budget_seconds']
    try:
        while cursor['cursor']<len(jobs):
            if stop():return dict(stage='safe_stopped',cursor=cursor['cursor'],costs=cursor['costs'])
            index=cursor['cursor'];job=jobs[index];row=rows[job['mother']]
            episode_started=perf_counter();episode_elapsed_before=cursor['active'].get('elapsed_active_seconds',0.) if cursor['active'] else 0.
            if row['case_id']!=job['case_id']:raise ValueError('manifest order/case mapping changed')
            folder=root/'episodes'/('%04d'%index);folder.mkdir(parents=True,exist_ok=True)
            binding=folder/'binding.pkl';active=cursor['active']
            if active is None:
                # Durable intent includes setup PHY; failed/unknown setup is not free.
                cursor['unknown']=[dict(kind='input_kernel_setup',matrix_cursor=index)];cursor['costs']['kernel_preparation']+=1;checkpoint()
                env,timing=environment_loader(plan,row,job['N'])
                save_pickle(binding,pack_environment(env))
                cursor['active']=dict(index=index,binding_SHA=file_sha(binding),setup_timing=timing,elapsed_active_seconds=0.)
                cursor['unknown']=[];checkpoint()
            else:
                if active['index']!=index or file_sha(binding)!=active['binding_SHA']:raise ValueError('saved active environment binding differs')
                restore_started=perf_counter();env=unpack_environment(pickle.loads(binding.read_bytes()))
                cursor['timing'].append(dict(matrix_cursor=index,binding_restore_seconds=perf_counter()-restore_started,restore_PHY_calls=0))
            journal=Journal(folder/'current.pkl',episode_identity(plan,index),dict(query=4*job['N'] if job['method'] in ('P4','R4') else 0,
                commit=job['N'],forward=job['N'] if job['method'].startswith('P') else 0,proposal=job['N']))
            if job['method'].startswith('P') and agent is None:
                cursor['unknown']=[dict(kind='policy_load',matrix_cursor=index,physical_calls=0)];checkpoint()
                agent,policy_timing=policy_loader(plan);cursor['timing'].append(policy_timing);cursor['unknown']=[];checkpoint()
            proposal=(lambda obs,mask:frozen_s10_probabilities(agent,obs,mask)) if job['method'].startswith('P') else None
            baseline=None
            if job['method'] in ('fixed','greedy'):
                if baseline_factory is None:
                    from satellite_rl.t0_width.runner import BackgroundPolicy
                    baseline=BackgroundPolicy(job['method'],plan.get('device','cuda'))
                else:baseline=baseline_factory(job['method'])
                baseline.bind_environment(env)
            def baseline_action(obs,mask):return baseline.act(obs,action_mask=mask)
            # Initial binding reset observation is exact and has no served prefix.
            # Active transaction supplies its own saved next observation on resume.
            observation=env._observation()
            algorithm_started=perf_counter()
            result=run_episode(env,observation,job['method'],job['case_id'],job['repeat'],journal,proposal,
                folder/'steps.jsonl',stop,dict(root_approved=True,qualification_passed=True),baseline_action if baseline else None)
            cursor['active']['algorithm_active_seconds']=cursor['active'].get('algorithm_active_seconds',0.)+perf_counter()-algorithm_started
            if result['stage']!='completed':return dict(result,matrix_cursor=index)
            export_started=perf_counter()
            counts=journal.value['attempts'];newcosts=dict(cursor['costs'])
            for key in ('query','commit','forward','proposal'):newcosts[key]+=counts[key]
            if (newcosts['query']>plan['matrix']['queries'] or newcosts['commit']>plan['matrix']['commits']
                    or newcosts['forward']>plan['matrix']['model_forwards']):raise ValueError('global finite matrix cost exceeded')
            summary=dict(schema='W28EpisodeTerminalV1',identity=episode_identity(plan,index),costs=counts,
                metrics=env.metrics(),beam_results=env.beam_results(),log_offset=journal.value['log_offset'],
                initial_setup_timing=cursor['active']['setup_timing'],
                timing=dict(episode_active_wall_to_export_seconds=episode_elapsed_before+perf_counter()-episode_started,
                    algorithm_active_wall_including_decision_WAL_seconds=cursor['active']['algorithm_active_seconds'],
                    export_seconds_so_far=perf_counter()-export_started,
                    includes=['input/kernel','actor load if first needed','mask/distribution/draw','candidate forks and queries','real commit','WAL and evidence writes'],
                    excludes=['external resource waits','human hold'],
                    unrecorded_recovery_wall='Root receipt upper bounds are in dispatch evidence and common backend wall debit; not represented as exact measured seconds here.',
                    partition_warning='query/commit subitems include their state/digest overhead; subitems overlap full wall and must not be summed as independent timings'))
            terminal=folder/'terminal.json'
            if terminal.exists():
                existing=read_json_ref(dict(path=str(terminal),sha256=file_sha(terminal)))
                if content_sha({k:v for k,v in existing.items() if k!='timing'})!=content_sha({k:v for k,v in summary.items() if k!='timing'}):raise ValueError('terminal differs from already persisted result')
            else:atomic_json(terminal,summary);fsync_directory(terminal.parent)
            cursor['costs']=newcosts;cursor['cursor']+=1;cursor['completed_episodes']+=1;cursor['active']=None;finished_episode_start=episode_started;episode_started=None;checkpoint()
            timing_path=folder/'episode_timing.json'
            if not timing_path.exists():
                atomic_json(timing_path,dict(episode_active_wall_through_terminal_and_cursor_WAL_seconds=episode_elapsed_before+perf_counter()-finished_episode_start,
                    algorithm_active_wall_including_decision_WAL_seconds=summary['timing']['algorithm_active_wall_including_decision_WAL_seconds'],
                    timing_definition=summary['timing'],timing_record_write_excluded=True))
                fsync_directory(folder)
            progress(dict(stage='evaluating',cursor=cursor['cursor'],budget=len(jobs),costs=cursor['costs']))
            # Finished episode dynamic caches no longer serve recovery. Raw step,
            # candidate/service evidence and complete terminal arrays remain.
            for temporary in (binding,folder/'current.pkl',folder/'current.previous'):
                if temporary.exists():temporary.unlink()
        return dict(stage='completed',cursor=cursor['cursor'],costs=cursor['costs'])
    finally:
        if episode_started is not None and cursor['active'] is not None:cursor['active']['elapsed_active_seconds']=episode_elapsed_before+perf_counter()-episode_started
        cursor['elapsed_active_seconds']+=perf_counter()-started;checkpoint()
