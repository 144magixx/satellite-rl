"""Bounded root driver for one W27 known matrix on independent 20418:9073.

Default prepares only. --execute requires a root-pinned execution authority and
already installed immutable evaluation source. No old workbench is replaced.
"""
import argparse
import sys
# Append transport dependencies after DQN site-packages; never shadow its Torch.
try:
    import paramiko
except ModuleNotFoundError:
    sys.path.append("C:/Users/Admin/anaconda3/Lib/site-packages")
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time
import torch
from scripts.tsac_search27 import remote_ops as r
from scripts.tsac_search27.pilot_controller import SOFT,HARD,guard,filtered_status,remote_plan,load_probe
from scripts.tsac_search27.final_matrix import policy_reference_catalog,build_candidate,export_results
from satellite_rl.t0_width.logging import atomic_json,content_sha
from satellite_rl.t0_width.data import read_json_ref
from satellite_rl.tsac_search27.known_evaluation import validate_policy_metadata


LATEST_START=datetime.fromisoformat('2026-10-14T22:00:00+00:00')

def apply_resume_window(args,science):
    """Only a SHA-pinned root attestation may replace the expired ops window.

    This does not modify/resume a remote Gate. Root must separately supply the
    actually resumed Gate epochs in the newly pinned science state.
    """
    global LATEST_START,SOFT,HARD
    if not getattr(args,'resume_authority_ref',None):return
    authority=read_json_ref(read(args.resume_authority_ref))
    if (authority.get('schema')!='W27KnownResumeOpsAuthorityV1'
            or authority.get('root_approved') is not True
            or authority.get('root_attested_user_continue') is not True
            or authority.get('experiment_id')!='S2-W27'
            or authority.get('science_state_SHA')!=r.sha(args.science_state)
            or authority.get('controller_SHA')!=r.sha(__file__)
            or authority.get('evaluation_source_commit')!=args.evaluation_commit
            or authority.get('training_updates')!=0
            or authority.get('allowed_episode_budgets')!=[560,680]
            or authority.get('epochs')!=science['epochs']):
        raise ValueError('explicit root resume ops authority mismatch')
    times=[datetime.fromisoformat(authority[k]) for k in ('user_continue_UTC','latest_matrix_start_UTC','soft_UTC','hard_UTC')]
    user,start,soft,hard=times
    if (any(t.tzinfo is None for t in times) or user<=datetime.fromisoformat('2026-10-14T23:00:00+00:00')
            or not user<=now()<start<soft<hard or (hard-user).total_seconds()>7200):
        raise ValueError('resume must be post-pause, current and bounded to two hours')
    LATEST_START,SOFT,HARD=start,soft,hard

def guard():
    if now()>=SOFT:raise RuntimeError('authorized ops soft deadline reached')

def now():return datetime.now(timezone.utc)
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def eval_api(route,body=None):
    request=dict(url='http://127.0.0.1:9073'+route,body=body)
    code='R='+repr(request)+'''
import urllib.request,urllib.error,json
req=urllib.request.Request(R['url'],data=None if R['body'] is None else json.dumps(R['body']).encode(),headers={'Content-Type':'application/json'})
try:
 with urllib.request.urlopen(req,timeout=20) as response:print(json.dumps(dict(status=response.status,body=json.load(response))))
except urllib.error.HTTPError as error:print(json.dumps(dict(status=error.code,error=error.read().decode())))
'''
    value=r.remote_python(20418,code,timeout=25)
    if value['exit_code']:raise RuntimeError('independent known API observation failed')
    return json.loads(value['stdout'])


def inspect_selected(job,plan,node):
    code='JOB='+repr(job)+'\nPLAN='+repr(plan)+'\nNODE='+repr(node)+'''
import pathlib,json,hashlib,torch,sys
sys.path[:0]=[PLAN['source_root']+'/src',PLAN['source_root']]
from satellite_rl.tsac_search27.pause import proc_identity,same_identity
from satellite_rl.t0_width.logging import content_sha
D=pathlib.Path(JOB['root'])/'runs'/JOB['task_key'];c=json.loads((D/'config.json').read_text())
assert c==PLAN['tasks'][JOB['task_key']]['config'];status=json.loads((D/'progress.json').read_text())
assert status['env_step']==NODE and status['stage']==('completed' if NODE==150000 else 'pilot_complete')
owner=json.loads((D/'owner.json').read_text());assert not same_identity(proc_identity(owner['pid']),owner)
p=pathlib.Path(c['model_dir'])/('policy_%06d.pt'%NODE);raw=p.read_bytes();policy=torch.load(p,map_location='cpu',weights_only=False)
assert policy['config_sha']==content_sha(c) and policy['env_step']==NODE and policy['source_sha']==c['source_sha'] and policy['data_manifest_sha']==c['data_manifest_sha']
print(json.dumps(dict(policy_ref=dict(path=str(p),sha256=hashlib.sha256(raw).hexdigest()),schema=policy['schema'],config_SHA=policy['config_sha'],source_SHA=policy['source_sha'],data_manifest_SHA=policy['data_manifest_sha'],env_step=NODE,owner=owner,owner_exit_verified=True,policy_file_SHA_verified=True,stage=status['stage'])))
'''
    result=r.remote_python(job['port'],code,timeout=40)
    if result['exit_code']:raise RuntimeError('selected fixed-node provenance/owner inspection failed')
    return json.loads(result['stdout'])


def observe_all_terminal(science,plans,node,passed):
    # Rejected candidates stay at pilot; all original science workers must exit.
    values={}
    for variant,job in science['jobs'].items():
        expected=node if variant=='B' or variant in passed else 50000
        if not passed:expected=50000
        values[variant]=inspect_selected(job,plans[variant],expected)
    return values


def stage_policy(method,item,origin_port,local_history,out,remote_dir):
    guard();temp=out/('transport_'+method+'.pt')
    if method=='HIST':
        origin=Path(local_history);actual=r.sha(origin)
        if item['policy_ref']['sha256']!='REQUIRE_ACTUAL_FILE_SHA' and actual!=item['policy_ref']['sha256']:raise ValueError('history SHA differs from original W26 lock')
        policy=torch.load(origin,map_location='cpu',weights_only=False)
        validate_policy_metadata(policy,item,item['schema'])
        source=origin;original_ref=dict(path=str(origin),sha256=actual)
    else:
        original_ref=item['policy_ref'];raw=r.read_remote(origin_port,original_ref['path'])
        if hashlib.sha256(raw).hexdigest()!=original_ref['sha256']:raise ValueError('selected source bytes changed')
        if temp.exists() and r.sha(temp)!=original_ref['sha256']:raise ValueError('unique transport file differs')
        if not temp.exists():temp.write_bytes(raw)
        source=temp
    guard();staged=r.publish(20418,source,remote_dir+'/policies/'+method+'_'+original_ref['sha256']+'.pt')
    r.write_once(out/('policy_transport_'+method+'.json'),dict(original_ref=original_ref,staged_ref=staged,original_preserved=True))
    if method!='HIST':temp.unlink()
    return dict(item,policy_ref=staged,executor_port=20418,root_approved=True,policy_file_SHA_verified=True,owner_exit_verified=True)


START_SERVICE=r'''
import json,pathlib,sys,os,socket,subprocess,hashlib,time
sys.path[:0]=[SOURCE+'/src',SOURCE]
from satellite_rl.tsac_search27.pause import Gate,proc_identity,same_identity
Gate(CONTROL).validate(EPOCH)
for rel,sha in MANIFEST.items():assert hashlib.sha256((pathlib.Path(SOURCE)/rel).read_bytes()).hexdigest()==sha
server_owner=pathlib.Path(ROOT)/'evaluation_server_owner.json'
if server_owner.exists():
 prior=json.loads(server_owner.read_text());actual=proc_identity(prior['pid'])
 assert same_identity(actual,prior) and '--http-port' in actual['argv'] and '9073' in actual['argv'] and PLAN in actual['argv']
 print(json.dumps(dict(stage='adopted_same_service',owner=prior)))
else:
 sock=socket.socket()
 try:assert sock.connect_ex(('127.0.0.1',9073))!=0
 finally:sock.close()
 env=os.environ.copy();env['PYTHONPATH']=SOURCE+'/src'+os.pathsep+SOURCE;env['PYTHONDONTWRITEBYTECODE']='1'
 for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):env[name]='1'
 args=[sys.executable,'-B','-m','satellite_rl.tsac_search27.workbench','serve','--root',ROOT,'--plan',PLAN,'--port','20418','--control',CONTROL,'--http-port','9073']
 with (pathlib.Path(ROOT)/'server.log').open('ab') as log:child=subprocess.Popen(args,cwd=SOURCE,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
 owner=proc_identity(child.pid);assert owner
 from satellite_rl.tsac_search27.pause import atomic_json
 atomic_json(server_owner,owner);print(json.dumps(dict(stage='started_independent_service',owner=owner)))
'''


def closed_matrix_evidence(config,remote_dir,source_root):
    code='CONFIG='+repr(config)+'\nROOT='+repr(remote_dir+'/workbench')+'\nSOURCE='+repr(source_root)+"""
import json,pathlib,sys
sys.path[:0]=[SOURCE+'/src',SOURCE]
from satellite_rl.tsac_search27.pause import proc_identity,same_identity
from satellite_rl.t0_width.logging import content_sha
D=pathlib.Path(ROOT)/'runs'/CONFIG['run_id'];owner=json.loads((D/'owner.json').read_text())
assert not same_identity(proc_identity(owner['pid']),owner),'worker has not closed logs/exited'
cursor=json.loads((D/'known_cursor.json').read_text());assert cursor['task_sha']==content_sha(CONFIG) and cursor['cursor']==CONFIG['budget_episodes']
assert not (D/'known_wal.pt').exists()
for name,offset in cursor['log_offsets'].items():
 p=D/'logs'/(name+'.jsonl.gz');assert (p.stat().st_size if p.exists() else 0)==offset
print(json.dumps(dict(actual_owner_exit_verified=True,known_WAL_absent=True,task_SHA=cursor['task_sha'],cursor=cursor['cursor'],log_offsets=cursor['log_offsets'],owner=owner)))
"""
    answer=r.remote_python(20418,code,timeout=20)
    if answer['exit_code']:raise RuntimeError('matrix closure not yet independently verified: '+answer['stderr'][-800:])
    return json.loads(answer['stdout'])


def prepare(args,ready,decision,science,out):
    guard();plans={v:remote_plan(science,j['port']) for v,j in science['jobs'].items()};node=ready['node'];passed=decision['passed_variants']
    observations=observe_all_terminal(science,plans,node,passed)
    r.write_once(out/'science_terminal_observations.json',observations)
    oldplan=read(args.w26_science_plan);oldconfirm=read(args.w26_confirmation_plan)
    catalog=policy_reference_catalog(science,{str(science['jobs'][v]['port']):p for v,p in plans.items()},oldplan,oldconfirm,node)
    r.write_once(out/'reference_catalog.json',catalog)
    historical_ref=read(args.w26_science_plan_ref)
    if r.sha(args.w26_science_plan)!=historical_ref['sha256']:raise ValueError('local original W26 training plan differs from exact remote reference')
    catalog['inventory_candidates']['HIST']['original_plan_ref']=historical_ref
    producer=read_json_ref(read(args.history_producer_config_ref))
    job=read_json_ref(read(args.history_producer_job_ref))
    if (content_sha(producer)!=catalog['inventory_candidates']['HIST']['config_SHA']
            or job['config_sha256']!=content_sha(producer)
            or job['status']['plan_sha256']!=historical_ref['sha256']
            or job['source_manifest']!=oldplan['source_manifest']):
        raise ValueError('archived W26 effective producer/config/job mismatch')
    source_manifest=read(args.source_manifest);source_root=args.evaluation_source_root
    sourcecheck=r.remote_python(20418,'import pathlib,hashlib,json\nSOURCE='+repr(source_root)+'\nMANIFEST='+repr(source_manifest)+'''\nfor rel,sha in MANIFEST.items(): assert hashlib.sha256((pathlib.Path(SOURCE)/rel).read_bytes()).hexdigest()==sha
print(json.dumps(dict(source_bytes_passed=True)))
''',timeout=45)
    if sourcecheck['exit_code']:raise ValueError('future evaluation source not installed/qualified')
    r.write_once(out/'evaluation_source_observation.json',sourcecheck)
    methods=['B']+passed+['HIST'] if passed else ['B','E','LP','HIST'];inventory={}
    epoch=getattr(args,'evaluation_epoch','evaluation_01')
    if not epoch.startswith('evaluation_') or not epoch.replace('_','').isalnum():raise ValueError('invalid evaluation epoch directory')
    remote_dir=r.NODES[20418]['base']+'/'+epoch
    for method in methods:
        item=catalog['inventory_candidates'][method]
        if method!='HIST':item.update({k:v for k,v in observations[method].items() if k!='owner'})
        history=Path(args.history_root)/('policy_%06d.pt'%node)
        inventory[method]=stage_policy(method,item,science['jobs'].get(method,{}).get('port'),history,out,remote_dir)
    r.write_once(out/'staged_policy_inventory.json',inventory)
    decision_path=Path(args.pilot_output)/'decision.json';decision_ref=r.publish(20418,decision_path,remote_dir+'/decision.json')
    # Candidate requires proposed lock ref; replace with actual SHA after root-authorized signature.
    candidate=build_candidate(ready,decision,science,inventory,evaluation_commit=args.evaluation_commit,executor_port=20418,
        rootpauseepoch=science['epochs']['20418'],run_id='known_'+args.evaluation_commit[:8]+'_'+str(node),decision_ref=decision_ref,
        terminal_lock_ref={'path':remote_dir+'/terminal_lock.json','sha256':'PENDING'},known_manifest_ref=catalog['known_manifest_ref'],w26_unlock=catalog['w26_unlock'])
    candidate['lock_candidate'].update(soft_UTC=SOFT.isoformat(),hard_UTC=HARD.isoformat())
    for key,arg in [('historical_producer_config_ref',args.history_producer_config_ref),('historical_producer_job_ref',args.history_producer_job_ref)]:
        local_ref=read(arg);read_json_ref(local_ref)
        ref=r.publish(20418,Path(local_ref['path']),remote_dir+'/'+key+'.json')
        candidate['config'][key]=ref;candidate['lock_candidate'][key]=ref
    r.write_once(out/'candidate.json',candidate)
    return candidate,source_manifest,remote_dir


def validate_execution_authority(authority,args,candidate,out):
    if (authority.get('schema')!='W27KnownExecutionAuthorityV1' or authority.get('root_approved') is not True
            or authority.get('evaluation_source_commit')!=args.evaluation_commit
            or authority.get('executor_port')!=20418 or authority.get('http_port')!=9073):raise ValueError('root frozen driver/source/card authority required')
    if authority.get('mode','exact_candidate')=='exact_candidate':
        if authority.get('candidate_SHA')!=r.sha(out/'candidate.json'):raise ValueError('exact candidate authority mismatch')
    elif authority.get('mode')=='audited_recipe':
        bindings={'controller_SHA':r.sha(__file__),'pilot_controller_SHA':read(Path(args.pilot_output)/'decision.json').get('controller_source_SHA'),
            'pilot_decision_phase_state_SHA':read(Path(args.pilot_output)/'decision.json').get('phase_state_SHA'),'final_matrix_SHA':r.sha(Path(__file__).with_name('final_matrix.py')),
            'science_state_SHA':r.sha(args.science_state),'w26_science_plan_SHA':r.sha(args.w26_science_plan),
            'w26_confirmation_plan_SHA':r.sha(args.w26_confirmation_plan),'source_manifest_SHA':r.sha(args.source_manifest),
            'admission_refs_SHA':r.sha(args.admission_refs),'known_manifest_SHA':candidate['config']['known_manifest_ref']['sha256'],
            'latest_matrix_start_UTC':LATEST_START.isoformat(),'soft_UTC':SOFT.isoformat(),'hard_UTC':HARD.isoformat(),
            'gate_formula':{'SI_floor':-.002,'SGM_floor':-.005,'SI_improvement':.003,'SGM_improvement':.005,'power_ratio_cap':1.10},
            'allowed_episode_budgets':[560,680],'single_gpu_single_worker':True,'known_scenes_not_independent':True}
        for key,value in bindings.items():
            if authority.get(key)!=value:raise ValueError('audited recipe mismatch: '+key)
        history_shas=authority.get('historical_policy_SHAs',{})
        if set(history_shas)!={'50000','150000'} or history_shas[str(candidate['config']['node'])]!=candidate['config']['policies']['HIST']['policy_ref']['sha256']:
            raise ValueError('history fixed-node real file SHA differs from root recipe')
        derived=dict(schema='W27KnownDerivedCandidateAuthorityV1',root_approved=True,authority_ref=read(args.execution_authority_ref),
                     candidate_SHA=r.sha(out/'candidate.json'),derived_from_exact_audited_recipe=True,policy_file_and_owner_inspection=True)
        r.write_once(out/'derived_candidate_authority.json',derived)
    else:raise ValueError('unknown execution authority mode')


def submit_matrix_once(run_id,plan_ref,out):
    intent=out/'POST_intent.json'
    if intent.exists():return dict(stage='observe_existing_intent',replay_allowed=False)
    if now()>=LATEST_START:
        return dict(stage='deferred_until_explicit_user_resume',reason='latest_matrix_start_0600',submitted=False)
    guard();r.write_once(intent,dict(run_id=run_id,plan_ref=plan_ref,replay_allowed=False))
    try:answer=eval_api('/'+run_id+'/start',{})
    except Exception as exc:
        r.write_once(out/'POST_unknown.json',dict(error=type(exc).__name__,message=str(exc),replay_allowed=False))
        return dict(stage='unknown_observe_only',replay_allowed=False)
    r.write_once(out/'POST_answer.json',answer)
    if answer['status']!=200:raise RuntimeError('sole matrix rejected; no POST replay')
    return dict(stage='submitted',response=answer)


def execute(args,candidate,manifest,remote_dir,science,out):
    guard();authority=read_json_ref(read(args.execution_authority_ref))
    validate_execution_authority(authority,args,candidate,out)
    node=candidate['config']['node'];passed=candidate['lock_candidate']['passed_variants'];plans={v:remote_plan(science,j['port']) for v,j in science['jobs'].items()}
    observe_all_terminal(science,plans,node,passed)
    config=dict(candidate['config']);lock=dict(candidate['lock_candidate'],root_approved=True)
    path=out/'terminal_lock.json';r.write_once(path,lock);config['terminal_lock_ref']=r.publish(20418,path,remote_dir+'/terminal_lock.json')
    path=out/'evaluation_config.json';r.write_once(path,config)
    probe=load_probe(args.resource_probe);raw,value=probe.probe(20418,args.evaluation_source_root,science['control_dirs']['20418'],science['epochs']['20418'])
    r.write_once(out/'resource_observation.json',value);path=out/'resources.json';r.write_once(path,probe.receipt(20418,value));resource_ref=r.publish(20418,path,remote_dir+'/resources.json')
    refs=read(args.admission_refs);refs['resources']=resource_ref
    plan=dict(schema='W27AuthorizedPlanV1',experiment_id='S2-W27',authorized=True,max_workers=1,source_commit=args.evaluation_commit,
              source_root=args.evaluation_source_root,source_manifest=manifest,admission_refs=refs,stop_url_base='http://127.0.0.1:9073',
              tasks={config['run_id']:dict(mode='known_evaluation',executor_port=20418,config=config)})
    path=out/'evaluation_plan.json';r.write_once(path,plan);planref=r.publish(20418,path,remote_dir+'/plan.json')
    guard();r.write_once(out/'service_start_intent.json',dict(plan_ref=planref,source_commit=args.evaluation_commit,replay_allowed=False))
    params=dict(SOURCE=args.evaluation_source_root,CONTROL=science['control_dirs']['20418'],EPOCH=science['epochs']['20418'],ROOT=remote_dir+'/workbench',PLAN=planref['path'],MANIFEST=manifest)
    r.remote_python(20418,'import pathlib\npathlib.Path('+repr(params['ROOT'])+').mkdir(parents=True,exist_ok=True)',timeout=20)
    code='\n'.join(k+'='+repr(v) for k,v in params.items())+'\n'+START_SERVICE
    answer=r.remote_python(20418,code,timeout=40);r.write_once(out/'service_start_answer.json',answer)
    if answer['exit_code']:raise RuntimeError('independent service startup failed; preserve owner and no restart')
    for attempt in range(10):
        try:
            status=eval_api('/status');assert status['status']==200 and status['body']['plan_SHA']==planref['sha256'];break
        except Exception:time.sleep(1)
    else:raise ValueError('new exact frozen service not ready')
    submission=submit_matrix_once(config['run_id'],planref,out)
    if submission['stage']=='deferred_until_explicit_user_resume':
        atomic_json(out/'state.json',submission);return
    # Existing POST intents are observed/adopted only. No repeat submission path exists.
    while now()<HARD:
        try:
            answer=eval_api('/'+config['run_id']);detail=answer['body']
            light={k:detail.get(k) for k in ('stage','completed_episodes','budget','owner','phase')};atomic_json(out/'latest_status.json',dict(checked_UTC=now().isoformat(),status=light))
            if detail.get('stage')=='completed':
                guard();closed=closed_matrix_evidence(config,remote_dir,args.evaluation_source_root)
                r.write_once(out/'matrix_closed_evidence.json',closed)
                local=out/'matrix_raw';local.mkdir(exist_ok=True)
                for rel in ('known_cursor.json','config.json','logs/evaluations.jsonl.gz','logs/steps.jsonl.gz','logs/episodes.jsonl.gz'):
                    src=remote_dir+'/workbench/runs/'+config['run_id']+'/'+rel
                    try:raw=r.read_remote(20418,src)
                    except FileNotFoundError:
                        if rel in ('logs/episodes.jsonl.gz',):continue
                        raise
                    target=local/rel;target.parent.mkdir(parents=True,exist_ok=True)
                    if target.exists() and target.read_bytes()!=raw:raise ValueError('local canonical result differs')
                    target.write_bytes(raw)
                manifest_bytes=r.read_remote(20418,config['known_manifest_ref']['path'])
                if hashlib.sha256(manifest_bytes).hexdigest()!=config['known_manifest_ref']['sha256']:raise ValueError('known metadata SHA differs')
                metadata=out/'known_manifest_metadata.json';metadata.write_bytes(manifest_bytes)
                export_results(config,local,out/'report_analysis',metadata);atomic_json(out/'state.json',dict(stage='audited_known_results',formal_report_written=False,node=node,evaluation_budget=config['budget_episodes']));return
        except ValueError as exc:
            atomic_json(out/'audit_failure.json',dict(stage='scientific_audit_failed',error=type(exc).__name__,message=str(exc),raw_evidence_preserved=True,automatic_POST_replay=False));return
        except Exception as exc:atomic_json(out/'observation_error.json',dict(error=type(exc).__name__,message=str(exc),automatic_POST_replay=False))
        time.sleep(min(60,max(0,(HARD-now()).total_seconds())))
    atomic_json(out/'state.json',dict(stage='deadline_hold',explicit_human_continue_required=True,actual_exit_not_assumed=True,new_dispatch_allowed=False))


def run(args):
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True);science=read(args.science_state)
    apply_resume_window(args,science)
    pins={name:r.sha(getattr(args,name)) for name in ('science_state','source_manifest','admission_refs','w26_science_plan','w26_science_plan_ref','w26_confirmation_plan')}
    pins.update(controller_SHA=r.sha(__file__),final_matrix_SHA=r.sha(Path(__file__).with_name('final_matrix.py')))
    r.write_once(out/'inputs_SHA.json',pins)
    atomic_json(out/'state.json',dict(stage='waiting_for_locked_pilot_decision',inputs_SHA=pins,submitted=False,latest_matrix_start_UTC=LATEST_START.isoformat()))
    if args.execute:
        authority=read_json_ref(read(args.execution_authority_ref))
        if authority.get('root_approved') is not True or authority.get('schema')!='W27KnownExecutionAuthorityV1':raise ValueError('root execution authority missing')
        if authority.get('mode')=='audited_recipe':
            fixed={'controller_SHA':pins['controller_SHA'],'final_matrix_SHA':pins['final_matrix_SHA'],
                'science_state_SHA':pins['science_state'],'source_manifest_SHA':pins['source_manifest'],
                'admission_refs_SHA':pins['admission_refs'],'w26_science_plan_SHA':pins['w26_science_plan'],
                'w26_confirmation_plan_SHA':pins['w26_confirmation_plan'],'evaluation_source_commit':args.evaluation_commit,
                'latest_matrix_start_UTC':LATEST_START.isoformat(),'soft_UTC':SOFT.isoformat(),'hard_UTC':HARD.isoformat()}
            for key,value in fixed.items():
                if authority.get(key)!=value:raise ValueError('startup audited recipe mismatch: '+key)
    while now()<LATEST_START:
        state=Path(args.pilot_output)/'controller_state.json'
        if state.exists() and read(state).get('stage')=='ready_for_known_matrix':
            ready=read(state);decision=read(Path(args.pilot_output)/'decision.json');candidate,manifest,remote_dir=prepare(args,ready,decision,science,out)
            if not args.execute:atomic_json(out/'state.json',dict(stage='prepared_candidate_only',submitted=False,root_execution_authority_required=True));return
            execute(args,candidate,manifest,remote_dir,science,out);return
        time.sleep(min(60,max(0,(SOFT-now()).total_seconds())))
    atomic_json(out/'state.json',dict(stage='deferred_until_explicit_user_resume',new_dispatch_allowed=False,explicit_human_continue_required=True))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('science-state','pilot-output','output','evaluation-commit','evaluation-source-root','source-manifest','admission-refs','w26-science-plan','w26-science-plan-ref','w26-confirmation-plan'):p.add_argument('--'+name,required=True)
    p.add_argument('--history-root',default='D:/paper/models/runs/s2_width26_20261014/65d381d7d092de5b12c52c37')
    p.add_argument('--history-producer-config-ref',required=True)
    p.add_argument('--history-producer-job-ref',required=True)
    p.add_argument('--resume-authority-ref')
    p.add_argument('--evaluation-epoch',default='evaluation_01')
    p.add_argument('--resource-probe',default=str(r.LOCAL/'control/resource_probe.py'));p.add_argument('--execution-authority-ref');p.add_argument('--execute',action='store_true');args=p.parse_args()
    if args.execute and not args.execution_authority_ref:p.error('--execute requires --execution-authority-ref')
    from satellite_rl.tsac_search27.pause import file_lock
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    try:
        with file_lock(out/'controller.lock',nonblocking=True):run(args)
    except Exception as exc:
        atomic_json(out/'controller_error.json',dict(error=type(exc).__name__,message=str(exc),automatic_POST_replay=False));raise

if __name__=='__main__':main()
