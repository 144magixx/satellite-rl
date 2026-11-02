"""SHA-bound source/data/recovery qualifications shared by server/controller."""
from pathlib import Path
import json,hashlib
from satellite_rl.workbench.storage import digest
from satellite_rl.workbench.t0compare_contract import fingerprint
REQUIRED={'source':'s2t24_source_qualification_v1','core':'s2t24_core_qualification_v1','ops':'s2t24_ops_qualification_v1','data':'s2t24_data_qualification_v1'}
def raw_pin(ref):
 if not isinstance(ref,dict) or not isinstance(ref.get('path'),str) or not isinstance(ref.get('sha256'),str):raise ValueError('typed actual qualification ref absent')
 p=Path(ref['path'])
 if not p.is_file() or p.is_symlink() or digest(p)!=ref['sha256']:raise ValueError('qualification original bytes changed')
 return p
def pin(ref):return json.loads(raw_pin(ref).read_text(encoding='utf8'))
def input_identity(job):
 if job.get('t0_mode')=='prepare':
  data=dict(job['data_config']);data.pop('admission_reference',None)
  return {'prepared_source_contract':data}
 return {role:[row['sha256'] for row in record['files']] for role,record in job.get('scene_sets',{}).items()}
def qualify(job,proof,mode):
 refs=proof.get('qualification_refs')
 if not isinstance(refs,dict) or not set(REQUIRED)<=set(refs):raise ValueError('nested actual source/core/ops/data qualification refs required')
 manifest=pin({'path':str(Path(job['source_root'])/'source_manifest.json'),'sha256':digest(Path(job['source_root'])/'source_manifest.json')});dataset=fingerprint(input_identity(job))
 for role,format in REQUIRED.items():
  q=pin(refs[role])
  if q.get('format')!=format or q.get('passed') is not True or q.get('git_commit')!=job['git_commit'] or q.get('design_contract_sha256')!=job['design_contract_sha256'] or q.get('purpose')!='explicit_historical_reproduction':raise ValueError('nested '+role+' qualification/source/design/purpose mismatch')
  for source in q.get('raw_evidence_refs',[]):raw_pin(source)
  if not q.get('raw_evidence_refs'):raise ValueError('qualification cannot be barepassed flag')
  if role=='source' and q.get('source_file_SHA')!=manifest['files']:raise ValueError('actual frozen source members not qualified')
  if role=='data' and (q.get('input_identity_sha256')!=dataset or q.get('mode')!=mode):raise ValueError('actual frozen entities/data scope not qualified')
  if role=='ops' and (q.get('no_internal_training_test') is not True or q.get('checkpoint_policy_namespace_RNG_qualified') is not True or q.get('Direct_only') is not True):raise ValueError('full lifecycle/Direct checkpoint qualification absent')
  if role=='core' and q.get('original_T0_environment_reward_unchanged') is not True:raise ValueError('T0 original scientific core not qualified')
 return {'source_commit':job['git_commit'],'input_identity_sha256':dataset,'mode':mode,'nested_refs_verified':list(REQUIRED)}


def qualify_original_leases(proof,port):
 prefix='/home/u2024110224/first_stage_satellite_rl/s1_ai17/resource_leases/'
 lease_ref=proof.get('original_lease_ref');future_ref=proof.get('original_future_reservation_ref')
 if not isinstance(lease_ref,dict) or lease_ref.get('path')!=prefix+'s2_t24_'+str(port)+'.json' or not isinstance(future_ref,dict) or future_ref.get('path')!=prefix+'S1_future_reservation_after_s2_t24.json':raise ValueError('fresh original remote lease/future paths required; snapshot not release')
 lease=pin(lease_ref);future=pin(future_ref)
 if lease.get('schema')!='S1AI17PersistentResourceLeaseV1' or lease.get('status')!='active_lent' or lease.get('port')!=port or lease.get('borrower')!='S2-T24' or lease.get('s1_new_science_allowed') is not False or lease.get('gpu_uuid')!=proof.get('gpu_uuid'):raise ValueError('actual original lease no longer owned by S2-T24')
 if future.get('schema')!='S1AI17FutureReservationUpdateV1' or port not in future.get('borrowed_ports',[]) or port in future.get('new_S1_scientific_dispatch_ports',[]) or future.get('new_S1_scientific_dispatch_ports')!=[20473,20474]:raise ValueError('actual old-owner future reservation includes borrowed lane')
 return {'lease_ref':lease_ref,'future_ref':future_ref,'original_bytes_checked':True}
