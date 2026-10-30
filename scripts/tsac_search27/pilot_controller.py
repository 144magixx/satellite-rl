"""Finite W27 root pilot controller. No work starts merely by importing this file."""
import argparse
from datetime import datetime,timezone
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import time
from scripts.tsac_search27 import remote_ops as r
from satellite_rl.t0_width.logging import atomic_json,content_sha
from scripts.t0_width.report_analysis import audit_terminal

SOFT=datetime.fromisoformat('2026-10-14T22:58:00+00:00')
HARD=datetime.fromisoformat('2026-10-14T23:00:00+00:00')
def now():return datetime.now(timezone.utc)
def read(path):return json.loads(Path(path).read_text(encoding='utf8'))
def guard():
    if now()>=SOFT:raise RuntimeError('soft deadline: no new continuation dispatch')

def gate_statistics(rows,expected_cases):
    expected={(case,repeat) for case in expected_cases for repeat in range(3)};found={};means={}
    for row in rows:
        if row.get('node')!=50000 or row.get('phase')!='development':raise ValueError('only fixed development node50000')
        key=(row['identity']['case_id'],row['repeat'])
        if key not in expected or key in found:raise ValueError('development duplicate/missing mother identity')
        audited,_,_=audit_terminal(row)
        if audited['complete']!=1 or audited['constraint_violations'] or audited['unserved_beams']:
            raise ValueError('incomplete/illegal pilot development episode')
        found[key]=audited
    if set(found)!=expected:raise ValueError('exact20 mothers x3 repeats must be complete')
    for metric in ('SI_mean','SGM','total_power_w'):
        means[metric]=sum(sum(found[(c,k)][metric] for k in range(3))/3 for c in expected_cases)/len(expected_cases)
    return dict(means=means,episodes=len(found),mother_ids=expected_cases,all_finite_complete_legal=True,zero_execution_clipping=True)

def decide(stats):
    b=stats['B']['means'];passed=[];contrasts={}
    for variant in ('E','LP'):
        c=stats[variant]['means'];si=c['SI_mean']-b['SI_mean'];sgm=c['SGM']-b['SGM'];power=c['total_power_w']<=1.10*b['total_power_w']
        accept=si>=-.002 and sgm>=-.005 and (si>=.003 or sgm>=.005) and power
        contrasts[variant]=dict(delta_SI=si,delta_SGM=sgm,power_guard=power,passed=accept)
        if accept:passed.append(variant)
    return dict(schema='W27PilotDecisionV1',experiment_id='S2-W27',root_approved=True,node=50000,
                passed_variants=passed,contrasts=contrasts,statistics=stats,selection_data='development_only',known40_opened=False)

SNAPSHOT=r'''
import json,pathlib,hashlib,gzip,sys,torch
sys.path[:0]=[PLAN['source_root']+'/src',PLAN['source_root']]
from satellite_rl.tsac_search27.pause import proc_identity,same_identity,Gate
from satellite_rl.t0_width.logging import content_sha
D=pathlib.Path(JOB['root'])/'runs'/JOB['task_key']; config=json.loads((D/'config.json').read_text())
assert config==PLAN['tasks'][JOB['task_key']]['config'];status=json.loads((D/'progress.json').read_text());assert status['stage']=='pilot_complete' and status['env_step']==50000
owner=json.loads((D/'owner.json').read_text());assert not same_identity(proc_identity(owner['pid']),owner)
G=Gate(CONTROL);G.validate(JOB['rootpauseepoch']);current=pathlib.Path(config.get('model_dir',D/'models'))/'current.pt'
saved=torch.load(current,map_location='cpu',weights_only=False);assert saved['config_sha']==content_sha(config)
assert saved['progress']['env_step']==50000 and 50000 in saved['progress']['completed_nodes'] and not saved['progress'].get('completed')
assert saved['progress'].get('evaluation_cursor')==60
for name,offset in saved['log_offsets'].items():
 p=D/'logs'/(name+'.jsonl.gz');assert (p.stat().st_size if p.exists() else 0)==offset
assert not (D/'evaluation_wal.pt').exists()
w=torch.load(D/'training_decision_wal.pt',map_location='cpu',weights_only=False);assert w['stage']=='committed' and w['env_step']==50000 and w['config_sha']==content_sha(config)
rows=[];steps=0
with gzip.open(D/'logs/evaluations.jsonl.gz','rt') as stream:
 for line in stream:
  row=json.loads(line)
  if row.get('node')==50000:rows.append(row)
with gzip.open(D/'logs/steps.jsonl.gz','rt') as stream:
 for line in stream:
  row=json.loads(line)
  if row.get('node')==50000 and row.get('phase')=='development':
   a=row['requested_action'];actual=row['actual_action']
   assert actual['start']==a[0] and actual['count']==a[1]+1 and actual['power_w']==(a[2]+1)*5
   steps+=1
assert steps==sum(len(row['beams']) for row in rows)
for rel,sha in PLAN['source_manifest'].items():
 p=pathlib.Path(PLAN['source_root'])/rel;assert hashlib.sha256(p.read_bytes()).hexdigest()==sha
h=hashlib.sha256()
with current.open('rb') as stream:
 for block in iter(lambda:stream.read(1048576),b''):h.update(block)
print(json.dumps(dict(config_SHA=content_sha(config),source_SHA=config['source_sha'],data_manifest_SHA=config['data_manifest_sha'],checkpoint_SHA=h.hexdigest(),previous_worker=owner,contract_SHA=G.spec['contract_SHA'],rootpauseepoch=JOB['rootpauseepoch'],rows=rows,step_count=steps,dev_cases=[row['case_id'] for row in config['dev_cases']],log_offsets=saved['log_offsets'],all_source_bytes_checked=True)))
'''

def snapshot(job,plan,control):
    code='JOB='+repr(job)+'\nPLAN='+repr(plan)+'\nCONTROL='+repr(control)+'\n'+SNAPSHOT
    result=r.remote_python(job['port'],code,timeout=90)
    if result['exit_code']:raise RuntimeError('pilot snapshot failed: '+result['stderr'][-1500:])
    return json.loads(result['stdout'])

def remote_plan(phase,port):
    ref=phase['plans'][str(port)];raw=r.read_remote(port,ref['path'])
    if hashlib.sha256(raw).hexdigest()!=ref['sha256']:raise ValueError('original plan SHA changed')
    return json.loads(raw)

def filtered_status(job):
    answer=r.api(job['port'],'/'+job['task_key'])
    if answer.get('status')!=200:raise RuntimeError('status GET rejected')
    value=answer['body']
    return {key:value.get(key) for key in ('stage','env_step','update_step','episode','budget','run_id','phase','wall_seconds','disk_free_bytes','owner')}

def stop_evidence(job,plan,control):
    code="""
import pathlib,json,sys
sys.path[:0]=[SOURCE+'/src',SOURCE]
from satellite_rl.tsac_search27.pause import proc_identity,same_identity,Gate
D=pathlib.Path(ROOT)/'runs'/KEY
owner=json.loads((D/'owner.json').read_text())
config=json.loads((D/'config.json').read_text())
model=pathlib.Path(config.get('model_dir',D/'models'));index=model/'resume_index.json'
print(json.dumps(dict(actual_worker_alive=same_identity(proc_identity(owner['pid']),owner),owner=owner,
 checkpoint_index=json.loads(index.read_text()) if index.exists() else None,
 current_checkpoint_exists=(model/'current.pt').exists(),gate=Gate(CONTROL).snapshot())))
"""
    code='SOURCE='+repr(plan['source_root'])+'\nROOT='+repr(job['root'])+'\nKEY='+repr(job['task_key'])+'\nCONTROL='+repr(control)+'\n'+code
    result=r.remote_python(job['port'],code,timeout=20)
    if result['exit_code']:raise RuntimeError('stop/exit observation unavailable')
    value=json.loads(result['stdout']);value['gate']={k:value['gate'].get(k) for k in ('epoch','hold','hold_epoch','hold_UTC')}
    return value


def load_probe(path):
    spec=importlib.util.spec_from_file_location('w27_resource_probe',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def continuation(variant,job,plan,phase,out,decision_ref,evidence,probe):
    guard();port=job['port'];prefix=out/('continue_'+variant)
    intent=prefix.with_suffix('.intent.json');answer=prefix.with_suffix('.answer.json')
    if answer.exists():return read(answer)
    if intent.exists():
        status=filtered_status(job)
        # Adopt only positive advancement or exact accounted new-generation registration.
        adopted=status.get('env_step',0)>50000 or (status.get('stage') in ('running','development','completed') and status.get('owner')!=evidence['previous_worker'])
        result=dict(stage='adopted_existing_continuation' if adopted else 'unknown_requires_root_reconciliation',observed=status,replay_allowed=False)
        if adopted:atomic_json(answer,result)
        return result
    guard();raw,value=probe.probe(port,plan['source_root'],phase['control_dirs'][str(port)],job['rootpauseepoch'])
    r.write_once(prefix.with_suffix('.resource_observation.json'),value)
    resource_path=prefix.with_suffix('.resource.json');r.write_once(resource_path,probe.receipt(port,value))
    remote_dir=r.NODES[port]['base']+'/control/pilot_controller_01/'
    resource_ref=r.publish(port,resource_path,remote_dir+resource_path.name)
    receipt=dict(schema='W27PilotContinuationV1',root_approved=True,experiment_id='S2-W27',run_id=job['run_id'],
                 env_step=50000,gate_passed=True,decision_ref=decision_ref,**{k:evidence[k] for k in ('config_SHA','source_SHA','checkpoint_SHA','rootpauseepoch','previous_worker','contract_SHA','data_manifest_SHA','log_offsets')})
    path=prefix.with_suffix('.receipt.json');r.write_once(path,receipt);ref=r.publish(port,path,remote_dir+path.name)
    guard();r.write_once(intent,dict(task_key=job['task_key'],receipt_ref=ref,resource_ref=resource_ref,issued_UTC=now().isoformat(),replay_allowed=False))
    try:
        guard();result=r.api(port,'/'+job['task_key']+'/continue',dict(receipt_ref=ref,resource_ref=resource_ref,dispatch_epoch=job['rootpauseepoch']))
    except Exception as exc:
        r.write_once(prefix.with_suffix('.unknown.json'),dict(error=type(exc).__name__,replay_allowed=False,unknown_spend_not_zero=True));return dict(stage='unknown_requires_root_reconciliation')
    r.write_once(answer,result);return result


def run(args):
    phase=read(args.phase_state);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    r.write_once(out/'controller_inputs.json',dict(controller_source_SHA=r.sha(__file__),
        phase_state_SHA=r.sha(args.phase_state),root_data_ref_SHA=r.sha(args.root_data_ref),
        resource_probe_SHA=r.sha(args.resource_probe),soft_UTC=SOFT.isoformat(),hard_UTC=HARD.isoformat(),
        automatic_worker_restart=False,automatic_POST_replay=False))
    dataref=read(args.root_data_ref);data=read(dataref['path'])
    if r.sha(dataref['path'])!=dataref['sha256'] or data.get('root_approved') is not True:raise ValueError('root pinned logical data admission required')
    if phase.get('stage')!='submitted':raise ValueError('original science launch not completely submitted')
    if set(phase['jobs'])!={'B','E','LP'}:raise ValueError('all three original science jobs required')
    plans={v:remote_plan(phase,j['port']) for v,j in phase['jobs'].items()}
    for v,j in phase['jobs'].items():
        c=plans[v]['tasks'][j['task_key']]['config']
        if c['phase']!='scientific' or c['training']['total_steps']!=150000 or c['training']['pilot_stop_node']!=50000 or c['data_manifest_sha']!=data['logical_dataset_identity']:raise ValueError('frozen scientific/logical data identity mismatch')
    probe=load_probe(args.resource_probe)
    decision_path=out/'decision.json';stats={};evidence={};next_targets=None;statuses={}
    if not decision_path.exists():
        atomic_json(out/'controller_state.json',dict(stage='observing_pilots',checked_UTC=now().isoformat(),
            evaluation_started=False,automatic_worker_restart=False))
    if decision_path.exists() and now()<SOFT:
        decision=read(decision_path);next_targets=['B']+decision['passed_variants'] if decision['passed_variants'] else []
        if not next_targets:
            atomic_json(out/'controller_state.json',dict(stage='ready_for_known_matrix',node=50000,passed_variants=[],evaluation_started=False));return
        evidence={v:read(out/('pilot_evidence_'+v+'.json')) for v in next_targets}
        for v in next_targets:
            guard();job=phase['jobs'][v];remote=r.NODES[job['port']]['base']+'/control/pilot_controller_01/decision.json'
            ref=r.publish(job['port'],decision_path,remote)
            outcome=continuation(v,job,plans[v],phase,out,ref,evidence[v],probe)
            if outcome.get('stage')=='unknown_requires_root_reconciliation' or outcome.get('status',200)!=200:
                atomic_json(out/'controller_state.json',dict(stage='unknown_requires_root_reconciliation',variant=v,outcome=outcome));return
    while now()<HARD:
        statuses={}
        for variant,job in phase['jobs'].items():
            try:statuses[variant]=filtered_status(job)
            except Exception as exc:statuses[variant]=dict(stage='observation_error',error=type(exc).__name__)
        atomic_json(out/'latest_status.json',dict(checked_UTC=now().isoformat(),jobs=statuses,no_large_config_stored=True))
        if now()>=SOFT:
            exits={}
            for v,j in phase['jobs'].items():
                try:exits[v]=stop_evidence(j,plans[v],phase['control_dirs'][str(j['port'])])
                except Exception as exc:exits[v]=dict(observation_error=type(exc).__name__,actual_exit_not_verified=True)
            atomic_json(out/'latest_pause_exit_evidence.json',dict(checked_UTC=now().isoformat(),jobs=exits))
            atomic_json(out/'controller_state.json',dict(stage='deadline_hold_observing_exit',new_dispatch_allowed=False,statuses=statuses))
        elif next_targets is None:
            if all(s.get('stage')=='pilot_complete' and s.get('env_step')==50000 for s in statuses.values()):
                guard()
                if not decision_path.exists():
                    for v,j in phase['jobs'].items():
                        guard();ev=snapshot(j,plans[v],phase['control_dirs'][str(j['port'])]);evidence[v]=ev
                        stats[v]=gate_statistics(ev.pop('rows'),ev['dev_cases']);r.write_once(out/('pilot_evidence_'+v+'.json'),ev)
                    decision=decide(stats);decision.update(controller_source_SHA=r.sha(__file__),phase_state_SHA=r.sha(args.phase_state),root_data_ref=dataref,issued_UTC=now().isoformat());r.write_once(decision_path,decision)
                else:
                    decision=read(decision_path);evidence={v:read(out/('pilot_evidence_'+v+'.json')) for v in ('B','E','LP')}
                next_targets=['B']+decision['passed_variants'] if decision['passed_variants'] else []
                if not next_targets:
                    atomic_json(out/'controller_state.json',dict(stage='ready_for_known_matrix',node=50000,passed_variants=[],evaluation_started=False));return
                for v in next_targets:
                    guard();remote=r.NODES[phase['jobs'][v]['port']]['base']+'/control/pilot_controller_01/decision.json'
                    decision_ref=r.publish(phase['jobs'][v]['port'],decision_path,remote)
                    outcome=continuation(v,phase['jobs'][v],plans[v],phase,out,decision_ref,evidence[v],probe)
                    atomic_json(out/'controller_state.json',dict(stage='continuation_dispatch',variant=v,outcome=outcome))
                    if outcome.get('stage')=='unknown_requires_root_reconciliation' or outcome.get('status',200)!=200:return
        elif all(statuses[v].get('stage')=='completed' and statuses[v].get('env_step')==150000 for v in next_targets):
            atomic_json(out/'controller_state.json',dict(stage='ready_for_known_matrix',node=150000,passed_variants=read(decision_path)['passed_variants'],evaluation_started=False));return
        time.sleep(min(60,max(0,(HARD-now()).total_seconds())))
    atomic_json(out/'controller_state.json',dict(stage='deadline_monitor_finished',new_dispatch_allowed=False,explicit_human_continue_required=True,actual_exit_not_assumed=True,last_statuses=statuses))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('phase-state','output','root-data-ref'):p.add_argument('--'+name,required=True)
    p.add_argument('--resource-probe',default=str(r.LOCAL/'control/resource_probe.py'))
    args=p.parse_args()
    try:
        from satellite_rl.tsac_search27.pause import file_lock
        out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
        with file_lock(out/'controller.lock',nonblocking=True):run(args)
    except Exception as exc:
        out=Path(args.output);out.mkdir(parents=True,exist_ok=True);atomic_json(out/'controller_error.json',dict(error=type(exc).__name__,message=str(exc),automatic_worker_restart=False,automatic_POST_replay=False));raise

if __name__=='__main__':main()
