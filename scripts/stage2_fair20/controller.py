"""Persistent F20 API DAG coordinator; queued resources remain visible."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
from . import ops, resources
from .tunnel import atomic, singleton
from satellite_rl.workbench.runtime import process_start_time


def process_birth():
    value=process_start_time(os.getpid())
    if value is None:
        raise RuntimeError('controller process birth could not be verified')
    return value


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def tasks(value):
    rows=value.get('tasks',value) if isinstance(value,dict) else value
    if not isinstance(rows,list):raise ValueError('unknown workbench tasks schema')
    return rows


def jobrow(row):
    job=row.get('job',row);status=row.get('status',job)
    return job,status


def resolve(value, state):
    if isinstance(value,dict) and set(value)=={'$task'}:
        ref=value['$task'];v=state['tasks'][ref['id']]
        for key in ref['path']:v=v[key]
        return v
    if isinstance(value,dict):return {k:resolve(v,state) for k,v in value.items()}
    if isinstance(value,list):return [resolve(v,state) for v in value]
    return value


def validate_plan(plan):
    if plan.get('experiment_id')!='S2-F20' or len(plan.get('scientific_commit',''))!=40:raise ValueError('frozen F20 commit required')
    if plan.get('allowed_ports') != [20474]:raise ValueError('latest user allocation permits ONLY 20474')
    seen=set()
    for t in plan['tasks']:
        if t['id'] in seen or t['port'] not in plan['allowed_ports']:raise ValueError('task identity/lane invalid')
        if any(d not in seen for d in t.get('requires',[])):raise ValueError('DAG must be topologically ordered')
        if t['kind'] not in ('sourceprep','prepare','shorttune','train','evaluate','refine','matrix'):raise ValueError('unsupported task kind')
        if t['body']['request_id']!='s2f20-'+t['id']:raise ValueError('stable request identity required')
        seen.add(t['id'])
    return plan


class Controller:
    def __init__(self, plan, root, api=ops.api, probe=resources.probe):
        self.plan=validate_plan(plan);self.root=Path(root);self.api=api;self.probe=probe
        self.path=self.root/'state.json';self.sha=digest(plan)
        self.state=json.loads(self.path.read_text(encoding='utf-8-sig')) if self.path.exists() else {'experiment_id':'S2-F20','plan_sha256':self.sha,'tasks':{},'stage':'queued_resources'}
        if self.state['plan_sha256']!=self.sha:raise ValueError('controller plan changed after freeze')

    def save(self):
        stamp=resources.datetime.now(resources.timezone.utc).isoformat()
        self.state.update(observed_utc=stamp,heartbeat_at=stamp)
        atomic(self.path,self.state)
        atomic(self.root/'process.json',{'pid':os.getpid(),'birth':process_birth(),'plan_sha256':self.sha,'stage':self.state['stage'],'observed_utc':stamp,'heartbeat_at':stamp})

    def event(self, kind, **fields):
        path=self.root/'events.jsonl';path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('a',encoding='utf-8') as f:f.write(json.dumps({'utc':resources.datetime.now(resources.timezone.utc).isoformat(),'kind':kind,**fields},allow_nan=False)+'\n')

    def release(self, port):
        # Root provides the exact evidence schema after live old-controller audit.
        # This contract is evidence, never a discretionary human approval gate.
        contract=self.root/'release_contract.json'
        if not contract.exists():return None
        pin=json.loads(contract.read_text(encoding='utf-8-sig'))
        if pin.get('scientific_commit')!=self.plan['scientific_commit']:raise ValueError('release contract source differs')
        receipt=Path(pin['handoff_receipt_path'])
        observed=json.loads(receipt.read_text(encoding='utf-8-sig'))
        if observed.get('released_ssh_port')!=port or observed.get('released') is not True:return None
        if observed.get('old_coordinator_excluded_ports') is None or port not in observed['old_coordinator_excluded_ports']:return None
        if observed.get('active_jobs') or observed.get('pending_jobs') or observed.get('future_lane_reserved') is not False:return None
        if not observed.get('persistent_coordinator_exclusion_verified') or not observed.get('coordinator_evidence'):return None
        lane=pin['lanes'][str(port)]
        release={**lane,'port':port,'scientific_commit':self.plan['scientific_commit'],'old_lane_all_work_terminal':True,'old_future_training_reserved':False,'evidence_path':str(receipt),'handoff_receipt_sha256':hashlib.sha256(receipt.read_bytes()).hexdigest(),'observed_utc':resources.datetime.now(resources.timezone.utc).isoformat()}
        atomic(self.root/'releases'/f'{port}.json',release)
        return release

    def admit(self,t,record):
        release=self.release(t['port'])
        if release is None:return False
        observation=self.probe(t['port'])
        reasons=resources.admission(observation,release,self.plan['scientific_commit'])
        if reasons:
            record['resource_wait_reasons']=reasons;return False
        release_remote=f"{ops.instance(t['port'])['ops']}/release_{release['handoff_receipt_sha256']}.json"
        release_transfer=ops.upload(Path(release['evidence_path']),release_remote,t['port'])
        if release_transfer['remote']!=release['handoff_receipt_sha256']:raise ValueError('handoff receipt transfer SHA differs')
        checked={**observation,'checked_at_utc':observation['observed_utc'],'scientific_commit':self.plan['scientific_commit'],'worker_free':True,'gpu_free':True}
        proof={'experiment_id':'S2-F20','run_id':record['run_id'],'admitted':True,'scientific_commit':self.plan['scientific_commit'],'release':release,'release_receipt_path':release_remote,'release_receipt_sha256':release['handoff_receipt_sha256'],'resource_observation':checked}
        local=self.root/'admission'/f"{record['run_id']}.json";atomic(local,proof)
        remote=f"{ops.instance(t['port'])['ops']}/admission_{record['run_id']}.json"
        receipt=ops.upload(local,remote,t['port'])
        if receipt['local_sha256']!=receipt['remote']:raise ValueError('admission proof SHA changed')
        self.api(t['port'],'/api/fair20/admit',{'run_id':record['run_id'],'proof_path':remote,'proof_sha256':receipt['remote']},method='POST')
        record['admission_sha256']=receipt['remote'];self.event('admitted',id=t['id'],run_id=record['run_id'],port=t['port']);return True

    def return_resource_if_verified(self):
        proofpath=self.root/'computational_delivery_verification.json'
        if not proofpath.exists():return False
        proof=json.loads(proofpath.read_text(encoding='utf-8-sig'))
        if proof.get('scientific_commit')!=self.plan['scientific_commit']:raise ValueError('delivery source differs')
        gates=('all_training_complete','all_test_and_B_complete','isolated_timing_complete','remote_archive_complete','all_remote_compute_complete')
        if not all(proof.get(k) is True for k in gates):return False
        obs=self.probe(20474)
        if obs.get('compute_apps'):return False
        active=[]
        for row in tasks(self.api(20474,'/api/fair20/tasks')):
            job,status=jobrow(row)
            if status.get('stage') not in resources.TERMINAL:active.append(job['run_id'])
        if active:return False
        root=ops.instance(20474)['root']
        workers=[p for p in obs['processes'] if root in p['command'] and 'workbench.server' not in p['command'] and 'tunnel' not in p['command']]
        if workers:return False
        receipt={'experiment_id':'S2-F20','released':True,'released_ssh_port':20474,'scientific_commit':self.plan['scientific_commit'],'observed_utc':obs['observed_utc'],'remote_root':root,'active_jobs':active,'active_scientific_workers':workers,'resource_observation':obs,'delivery_verification_path':str(proofpath),'delivery_verification_sha256':hashlib.sha256(proofpath.read_bytes()).hexdigest(),'task_ids':{k:v['run_id'] for k,v in self.state['tasks'].items()},'return_notice_pending':True}
        atomic(ops.LOCAL/'delivery/20474_resource_return.json',receipt)
        self.state['stage']='resource_return_notice_pending'
        self.event('resource_released_return_notice_pending',port=20474)
        return True

    def tick(self):
        catalogs={p:tasks(self.api(p,'/api/fair20/tasks')) for p in self.plan['allowed_ports']}
        for t in self.plan['tasks']:
            if any(self.state['tasks'].get(d,{}).get('stage')!='completed' for d in t.get('requires',[])):continue
            body=resolve(t['body'],self.state);expected=digest(body)
            record=self.state['tasks'].get(t['id'])
            if record is not None and record.get('stage')=='completed':continue
            matches=[r for r in catalogs[t['port']] if jobrow(r)[0].get('request_id')==body['request_id']]
            if len(matches)>1:raise ValueError('duplicate stable request ID')
            if record is None:
                # Submit is idempotent; if its response was lost, next tick GET
                # recovers the published ID before considering another POST.
                if matches:job,status=jobrow(matches[0]);runid=job['run_id']
                else:
                    response=self.api(t['port'],'/api/fair20/'+t['kind'],body,method='POST')
                    job=response.get('job',response);runid=job['run_id'];status=response.get('status',{'stage':'queued_resources'})
                record={'run_id':runid,'request_id':body['request_id'],'body_sha256':expected,'port':t['port'],'stage':status.get('stage','queued_resources')}
                self.state['tasks'][t['id']]=record;self.save();self.event('registered',id=t['id'],run_id=runid)
            elif record['body_sha256']!=expected:raise ValueError('resolved task body changed')
            detail=self.api(t['port'],'/api/fair20/tasks/'+record['run_id'])
            job,status=jobrow(detail)
            if job.get('git_commit')!=self.plan['scientific_commit']:raise ValueError('task scientific commit mismatch')
            record['stage']=status.get('stage',status.get('status'));record['detail']={k:v for k,v in detail.items() if k!='result'}
            if record['stage']=='queued_resources':self.admit(t,record)
            if record['stage']=='completed':
                result=detail.get('result',{})
                record['result']={k:v for k,v in result.items() if k not in ('datasets','case_files')}
                record['result']['dataset_counts']={k:v.get('count') for k,v in result.get('datasets',{}).items()}
                record['result']['case_count']=len(result.get('case_files',[]))
            if record['stage'] in ('failed','interrupted','stopped','cancelled','invalidated','attention_required'):
                self.state['stage']='attention_required';self.save();return False
        if len(self.state['tasks'])==len(self.plan['tasks']) and all(r['stage']=='completed' for r in self.state['tasks'].values()):
            # Computation is not delivery. Report and both archives must be verified.
            self.state['stage']='report_archive_pending'
            self.state['resource_return_notice_pending']=True
            self.return_resource_if_verified()
        else:
            actual=[record.get('stage') for record in self.state['tasks'].values()]
            if any(stage in ('training','evaluating','preparing','refining','running','starting','validating') for stage in actual):
                self.state['stage']='running'
            elif any(stage=='queued_resources' for stage in actual):
                self.state['stage']='queued_resources'
            elif any(stage=='queued' for stage in actual):
                self.state['stage']='queued'
            else:
                self.state['stage']='waiting_dependencies'
        self.save();return True


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--root',type=Path,required=True);p.add_argument('--once',action='store_true');a=p.parse_args()
    plan=json.loads(a.plan.read_text(encoding='utf-8-sig'))
    with singleton(a.root):
        c=Controller(plan,a.root);atomic(a.root/'process.json',{'pid':os.getpid(),'birth':process_birth(),'plan_sha256':c.sha})
        failures=0
        while True:
            try:
                if not c.tick():return
                failures=0
            except (OSError,ConnectionError,TimeoutError) as e:
                failures+=1;c.event('transport_retry',attempt=failures,error_type=type(e).__name__)
                if failures>=8:c.state['stage']='transport_attention_required';c.save();return
            except Exception as error:
                c.state['stage']='attention_required';c.state['error_type']=type(error).__name__;c.save();c.event('attention_required',error_type=type(error).__name__);raise
            if a.once or c.state['stage'] in ('report_archive_pending','resource_return_notice_pending'):return
            time.sleep(min(60,15*max(1,failures)))

if __name__=='__main__':main()
