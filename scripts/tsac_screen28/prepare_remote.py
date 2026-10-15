"""Root-only future preparation: source/descriptor publication and distinct Gate.

This helper never starts a service, watchdog, worker, training or HTTP POST.
Do not execute until root freezes its preparation authority after W27 closure.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tarfile
sys.path.append('C:/Users/Admin/anaconda3/Lib/site-packages')
from scripts.tsac_search27 import remote_ops as transport
from scripts.tsac_search27.freeze import package
from satellite_rl.t0_width.data import read_json_ref
from satellite_rl.t0_width.logging import content_sha

BASE=Path('D:/paper/outputs/experiments/s2_tsac_screen28_20261015')
REMOTE='/home/u2024110224/second_stage_satellite_rl/s2_w28'
def read(path):return json.loads(Path(path).read_text(encoding='utf8'))

INSTALL=r'''
import pathlib,hashlib,tarfile,json,sys,os,ast,shutil,subprocess
root=pathlib.Path(SOURCE);root.mkdir(parents=True,exist_ok=True)
assert hashlib.sha256(pathlib.Path(ARCHIVE['path']).read_bytes()).hexdigest()==ARCHIVE['sha256']
with tarfile.open(ARCHIVE['path'],'r:gz') as archive:
 seen=set()
 for member in archive:
  assert member.isfile() and member.name in FILES and member.name not in seen
  target=(root/member.name).resolve();assert target.is_relative_to(root.resolve())
  raw=archive.extractfile(member).read();assert hashlib.sha256(raw).hexdigest()==FILES[member.name]
  target.parent.mkdir(parents=True,exist_ok=True)
  if target.exists():assert target.read_bytes()==raw
  else:
   with target.open('xb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
  if member.name.endswith('.py'):ast.parse(raw.decode('utf8'))
  seen.add(member.name)
 assert seen==set(FILES)
actual={p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
assert actual==FILES
fd=os.open(str(root),os.O_RDONLY);os.fsync(fd);os.close(fd)
sys.path[:0]=[str(root/'src'),str(root)]
from satellite_rl.tsac_search27.pause import initialize,Gate
import satellite_rl.tsac_screen28.backend as backend_module
assert pathlib.Path(backend_module.__file__).resolve().is_relative_to(root.resolve())
control=pathlib.Path(CONTROL);existed=(control/'state.json').exists()
spec=dict(experiment_id='S2-W28',node_id='20418',scope_root=SCOPE,uid=os.getuid(),
 soft_UTC='2026-10-14T22:58:00+00:00',hard_UTC='2026-10-14T23:00:00+00:00',contract_SHA=CONTRACT_SHA)
initialize(control,spec);gate=Gate(control);state=gate.snapshot()
receipt_id='w28-explicit-continue-20261015T005733Z-20418'
if existed:
 assert not state['hold'] and state['resumed_contract'] and receipt_id in state['resume_receipt_ids'], 'existing/new hold cannot be bypassed by old continuation'
else:
 assert state['hold']
 receipt=dict(receipt_id=receipt_id,user_message_reference=CONTINUE_REF,issued_UTC='2026-10-15T00:57:33+00:00',
  hold_epoch=state['hold_epoch'],contract_SHA=CONTRACT_SHA,root_attested_user_continue=True)
 gate.resume(receipt)
gate.validate(gate.snapshot()['epoch'])
def command(args):
 p=subprocess.run(args,capture_output=True,text=True,timeout=20);return dict(exit_code=p.returncode,stdout=p.stdout,stderr=p.stderr)
usage=shutil.disk_usage(SCOPE)
print(json.dumps(dict(source_passed=True,files=actual,python=sys.executable,uid=os.getuid(),
 actual_module=backend_module.__file__,existing_cost_artifacts=[str(p) for p in pathlib.Path(SCOPE).rglob('*cursor.pkl')],
 boot_id=pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip(),gate_spec=spec,gate=gate.snapshot(),
 gpu=command(['nvidia-smi','--query-gpu=uuid,name,memory.free,memory.used','--format=csv,noheader,nounits']),
 compute=command(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader']),
 disk=command(['df','-PT',SCOPE]),disk_usage=dict(total=usage.total,free=usage.free,used=usage.used),
 directory_fsync_available=True,actual_environment_constructions=0,model_forwards=0,capacity_reward_calls=0,
 watchdog_started=False,workbench_started=False,workers_started=0,POST_calls=0)))
'''

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--commit',required=True);parser.add_argument('--authority-ref',required=True)
    parser.add_argument('--draft',default=str(BASE/'contract_draft_01/workbench_plan_draft.json'))
    parser.add_argument('--output',default=str(BASE/'control/preparation_01'));args=parser.parse_args()
    authority=read_json_ref(read(args.authority_ref));commit=args.commit
    if (authority.get('schema')!='W28PreparationAuthorityV1' or authority.get('root_approved') is not True
            or authority.get('experiment_id')!='S2-W28' or authority.get('source_commit')!=commit
            or authority.get('root_attested_user_continue') is not True
            or authority.get('allow_source_descriptor_publish_and_distinct_gate') is not True
            or authority.get('worker_starts_allowed') is not False or authority.get('POST_calls_allowed')!=0):raise ValueError('root exact prepare-only authority required')
    closure=read_json_ref(authority['w27_closure_ref'])
    if closure.get('root_approved') is not True or closure.get('formal_report_closed') is not True:raise ValueError('W27 formal closure must precede preparation')
    read_json_ref(authority['explicit_continue_ref']);read_json_ref(authority['gate_contract_ref'])
    if len(commit)!=40 or any(c not in '0123456789abcdef' for c in commit):raise ValueError('exact Git commit required')
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    transport.LOCAL=BASE
    archive,members=package(commit)
    frozen=BASE/'control'/('frozen_ops_'+commit[:8]);frozen.mkdir(parents=True,exist_ok=True)
    with tarfile.open(archive,'r:gz') as tar:
        for member in tar:
            target=(frozen/member.name).resolve();assert member.isfile() and target.is_relative_to(frozen.resolve())
            raw=tar.extractfile(member).read();assert hashlib.sha256(raw).hexdigest()==members['files'][member.name]
            target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists():assert target.read_bytes()==raw
            else:target.write_bytes(raw)
    archive_ref=transport.publish(20418,archive,REMOTE+'/deployment/'+commit+'/'+archive.name)
    params=dict(SOURCE=REMOTE+'/source/'+commit,ARCHIVE=archive_ref,FILES=members['files'],SCOPE=REMOTE,
        CONTROL=REMOTE+'/control/epoch_01',CONTRACT_SHA=authority['gate_contract_ref']['sha256'],CONTINUE_REF=authority['explicit_continue_ref'])
    result=transport.remote_python(20418,'\n'.join(k+'='+repr(v) for k,v in params.items())+'\n'+INSTALL,timeout=90)
    transport.write_once(out/'source_gate_preparation.json',result)
    if result['exit_code']:raise ValueError('remote immutable source/Gate preparation failed; preserve evidence and do not replay unknown')
    witness=json.loads(result['stdout']);draft=read(args.draft)
    for filename in ('development_metadata.json','known_metadata.json','engineering_fixtures.json','engineering_catalog.json'):
        path=Path(args.draft).parent/filename;transport.publish(20418,path,REMOTE+'/contracts/'+filename)
    for task in draft['tasks'].values():
        task.update(source_commit=commit,source_root=params['SOURCE'],source_manifest=members['python_manifest'],
            rootpauseepoch=witness['gate']['epoch'],runtime_approved=False,root_approved=False,w27_closure_ref=authority['w27_closure_ref'])
    draft.update(source_root=params['SOURCE'],runtime_approved=False,root_approved=False)
    transport.write_once(out/'workbench_prepared_draft.json',draft)
    gate=dict(schema='W28SourceAdmissionV1',experiment_id='S2-W28',root_approved=True,passed=True,executor_port=20418,
        source_commit=commit,source_root=params['SOURCE'],source_manifest=members['python_manifest'],
        source_witness_ref=dict(path=str(out/'source_gate_preparation.json'),sha256=transport.sha(out/'source_gate_preparation.json')),
        zero_model_forwards=True,zero_capacity_reward_calls=True)
    transport.write_once(out/'source_admission.json',gate)
    source_ref=transport.publish(20418,out/'source_admission.json',REMOTE+'/admissions/source.json')
    resources=dict(schema='W28ResourceCandidateV1',experiment_id='S2-W28',root_approved=False,passed=False,
        reason='Root must start and verify live dedicated W28 watchdog, inspect current tasks/GPU, quota/mount and refresh resource age before any task.',
        actual=witness,source_ref=source_ref)
    transport.write_once(out/'resource_candidate.json',resources)
    transport.write_once(out/'data_admission_candidate.json',dict(schema='W28DataAdmissionV1',experiment_id='S2-W28',
        root_approved=False,passed=False,allowed_phases=['engineering'],
        fixture_manifest_ref=draft['tasks']['engineering']['fixture_manifest_ref'],
        scientific_data_generation=0,geometry_kernel_qualification_still_required=True,
        inherited_development_ref=draft['tasks']['development']['data_manifest_ref'],inherited_known_ref=draft['tasks']['known']['data_manifest_ref']))
    transport.write_once(out/'cost_admission_candidate.json',dict(schema='W28CostAdmissionV1',experiment_id='S2-W28',
        root_approved=False,passed=False,engineering_caps=dict(physical_calls_attempted=320,real_commit_attempts=64,candidate_restore_attempts=256,model_forward_attempts=64),
        science_capacity_reward_cap=179722,unknown_cost_not_zero=True,existing_cost_artifacts=witness['existing_cost_artifacts'],
        finite_builtin_catalog_bound=draft['tasks']['engineering']['catalog_bounds'],
        ledger_reconciliation_required=True,source_epoch=commit,worker_starts_allowed=False))
    transport.write_once(out/'next_root_actions.json',dict(schema='W28PrepareOnlyNextActionsV1',
        service_command=['-B','-m','satellite_rl.tsac_screen28.backend','serve','--root',REMOTE+'/20418/workbench',
            '--plan',REMOTE+'/contracts/approved_engineering_workbench.json','--control',params['CONTROL'],'--http-port','9080'],
        service_started=False,workers_started=0,POST_calls=0,remaining=['root exact engineering plan/admissions','fresh disk/quota/GPU/owner proof','dedicated live watchdog','root single workbench start','root once-only engineering POST'],
        frozen_ops=str(frozen),source_root=params['SOURCE'],gate_epoch=witness['gate']['epoch']))
    print(json.dumps(dict(stage='prepared_not_started',source_commit=commit,new_gate=True,service_started=False,workers_started=0,POST_calls=0)))

if __name__=='__main__':main()
