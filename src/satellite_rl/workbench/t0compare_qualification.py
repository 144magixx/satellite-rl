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
