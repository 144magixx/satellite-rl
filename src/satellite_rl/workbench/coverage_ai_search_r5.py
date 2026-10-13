"""Only the signed R5 P0 pilot:24case bundles, zero MILP/training."""
from copy import deepcopy
from datetime import datetime,timezone
from pathlib import Path
import os,sys
from .storage import read_json,validate_id,digest
from .coverage_ai_search import pin
from . import frozen_round_tasks
GROUP='S1-AI17-R5-P0';DESIGN_SHA='3555fdc7c64142f7af32fb6b32177db97e931aad52877270a97d1fce6ed74e90'
IDENTITY_SHA='373d1b959231640f764ad21af09e9ed188d1635d60de04812ae6b793e88637d7';DATA_SHA='9929e31c2ecd7adca912deeb16ad58abd6bd97d7cb2c631fc53cb4a2c57e6b5b'
REGISTRY_SHA='df568fc4bcf51ea6b22c0ebd18ef7e23ed95ef50dcdf8830e77030fdff8afc37'
PARENT=Path('/home/u2024110224/first_stage_satellite_rl/s1_ai21');PORTS=('20418','20424','20473','20474');METHODS=('D5_RULE','D5_RANDOM','D5_EXACT','D5_OWNER10','D5_JOINT10','D4_OWNER10','D4_JOINT10');MODES=('p0',)
FIELDS=('config','mode','dataset_role','case_indices','executor_port','methods');RUNTIME_FIELDS=FIELDS[1:];CANONICAL_DATASETS=True

def task_contract(b):return {k:deepcopy(b.get(k))for k in FIELDS}
def run_prefix(c):return 'p0_ai_search_r5'
def plan_references(p):return {'identity':p['identity_ref'],'registry':p['registry_ref']}
def external_references(c):return {}
def decorate_job(c,p,refs,commit):return {'phase':'P0','execution_commit':commit,'algorithm':'ai_search_r5','network':'group_decoder_p0','kind':'evaluate','execution_device':'cpu','methods':c['methods']}
def verify_metadata(j,p):
 if j.get('phase')!='P0'or j.get('execution_commit')!=j['scientific_commit']or j.get('algorithm')!='ai_search_r5':raise ValueError('P0 metadata/source differs')
def expected_results(c):return len(c['case_indices'])*7

def deployment(store):
 root=store.archive_root.resolve();port=root.parent.name
 if root.name!='archive'or root.parent.parent!=PARENT.resolve()or port not in PORTS or os.environ.get('SATELLITE_RL_SSH_PORT')!=port:raise ValueError('independent four-lane R5 root required')
 from .coverage import _require_remote_dqn
 _require_remote_dqn();return port

def design(p):
 if p['design']['sha256']!=DESIGN_SHA:raise ValueError('fixed P0 design required')
 d=read_json(pin(p['design']),{})
 if d['P0']['new_decoders_total']!=168 or d['P0']['new_prepare_calls']!=0 or d['P0']['new_teacher_calls']!=0:raise ValueError('P0 boundaries')
 return d


def validate_config(c):
 if set(c)!={'schema_version','research_stage','algorithm','name','protocol_ref','datasets','workers','resources','hard_watchdogs'}or (c['schema_version'],c['research_stage'],c['algorithm'])!=(13,'coverage','ai_search_r5'):raise ValueError('P0 schema13 exactconfig')
 if c['workers']!=1 or type(c['workers'])is not int:raise ValueError('one sequential case per physical lane')
 if c['datasets'].keys()!= {'development'} or {k:c['datasets']['development'].get(k)for k in ('role','count','sha256')}!= {'role':'development','count':80,'sha256':DATA_SHA}:raise ValueError('only pinned original development source')
 if c['protocol_ref']['sha256']!=DESIGN_SHA or not isinstance(c['name'],str)or not 1<=len(c['name'])<=100:raise ValueError('name/protocol')
 if c['hard_watchdogs']!={'decoder':15}or c['resources']!={'minimum_free_bytes':8589934592,'parent_children_rss_bytes':2147483648}:raise ValueError('signed ownedchild guards required')
 return deepcopy(c)

def validate_task(c,p,port):
 cfg=validate_config(c['config']);d=design(p)
 if p.get('phase')!='P0' or p.get('training_allowed') is not False or p.get('test_allowed') is not False or p['identity_ref']['sha256']!=IDENTITY_SHA:raise ValueError('P0 only')
 ids=read_json(pin(p['identity_ref']),{})['development'];indices=d['P0']['case_indices']
 if p.get('selected_mothers')!=d['data']['mother_ids'] or p.get('mother_order')!=sorted(set(ids['mother_ids'])):raise ValueError('actual development mother order')
 if p['selected_case_indices']!=indices or ids['sha256']!=DATA_SHA:raise ValueError('prescribed 24 identities')
 tasks=list(p['tasks'].values())
 if len(tasks)!=4 or {str(t['executor_port']) for t in tasks}!=set(PORTS) or sorted(i for t in tasks for i in t['case_indices'])!=indices:raise ValueError('four unique lanes and exact 24cases')
 for t in tasks:
  validate_config(t['config'])
  if t['mode']!='p0' or t['methods']!=list(METHODS) or t['dataset_role']!='development':raise ValueError('seven P0 slots only')
 if c['mode']!='p0' or c['dataset_role']!='development' or str(c['executor_port'])!=str(port) or c['methods']!=list(METHODS) or cfg['protocol_ref']!=p['design']:raise ValueError('mode/method/physical source')
 if not c['case_indices'] or any(type(i)is not int or i not in indices for i in c['case_indices']) or len(set(c['case_indices']))!=len(c['case_indices']):raise ValueError('case bounds')
 registry=read_json(pin(p['registry_ref']),{})
 if p['registry_ref']['sha256']!=REGISTRY_SHA or registry['case_indices']!=indices or registry['roles_read']!=['development']:raise ValueError('exact immutable registry')
 for case in registry['cases']:
  i=case['input_case_index']
  if case['case_id']!=ids['case_ids'][i] or case['mother_id']!=ids['mother_ids'][i] or case['profile']!=ids['profiles'][i] or case['role']!='development' or case['identity_sha256']!=IDENTITY_SHA:raise ValueError('registry original identity binding')
 if p.get('budget')!={'new_prepare_calls':0,'new_teacher_calls':0,'decoder_slots':168,'result_slots':168,'MILP_calls':0,'training_updates':0,'validation_test_algorithms':0}:raise ValueError('P0 fixed budget')
 return cfg


def validate_admission(p,plan_ref,review_ref):
 d=design(p);r=read_json(pin(review_ref),{})
 if p.get('group')!=GROUP or p.get('authorized')is not True or r.get('passed')is not True or r.get('group')!=GROUP or r.get('scientific_commit')!=p['scientific_commit']or r.get('plan_sha256')!=plan_ref['sha256']or r.get('design_sha256')!=DESIGN_SHA:raise ValueError('exact root plan/source approval')
 if set(p.get('qualification_refs',{}))!={'geometry','process'}:raise ValueError('8case geometry qualification required')
 q=read_json(pin(p['qualification_refs']['geometry']),{})
 if q.get('passed')is not True or q.get('scientific_commit')!=p['scientific_commit']or q.get('design_sha256')!=DESIGN_SHA or q.get('geometry_fixture_cases')!=8:raise ValueError('source-bound geometry qualification')
 base=Path(__file__).resolve().parents[3];folder=base/'src/satellite_rl/coverage/ai_search_r5';actual={f.relative_to(base).as_posix():digest(f)for f in folder.rglob('*.py')if '__pycache__'not in f.parts}
 if not actual or q.get('source_file_sha256')!=actual:raise ValueError('qualified R5 core members changed')
 process=read_json(pin(p['qualification_refs']['process']),{})
 if process.get('guard_sha256')!=digest(base/'src/satellite_rl/workbench/r5_stage_guard.py') or process.get('rss_provider_sha256')!=digest(base/'src/satellite_rl/workbench/r4_monitor_repair.py'):raise ValueError('qualified process guard/RSS source changed')
 if process.get('passed') is not True or process.get('scientific_commit')!=p['scientific_commit'] or process.get('process_stub_calls')!=3 or process.get('geometry_calls')!=0 or process.get('CUDA_calls')!=0:raise ValueError('exact source-bound three process stub qualification')

def resource_admission(store,j):
 port=deployment(store);r=read_json(store.archive_root.parent/'ops/resource_route.json',{});age=(datetime.now(timezone.utc)-datetime.fromisoformat(r.get('observed_utc','1970-01-01T00:00:00+00:00'))).total_seconds()
 if r.get('group')!=GROUP or str(r.get('ssh_port'))!=port or r.get('scientific_commit')!=j['scientific_commit']or r.get('ready')is not True or not -5<=age<=120:raise ValueError('fresh P0 source route required')
 from .ai_search_resources import observe
 from .runtime import pid_alive
 o=observe(store.archive_root)
 if o['foreign_workers']or o['gpu'].split(',')[0].strip()!=r['gpu_uuid']:raise ValueError('foreign worker/changedGPU')
 for other,status in store.jobs():
  launch=read_json(store.job_dir(other['run_id'])/'launch.json',{})
  if other['run_id']!=j['run_id']and pid_alive(status.get('pid'),launch.get('process_start_time')):raise ValueError('other case on lane')
 return o

def api(app,method,parts,body):
 policy=sys.modules[__name__]
 if method=='GET'and parts==['schema']:return {'group':GROUP,'schema_version':13,'modes':MODES,'methods':METHODS,'slots':168,'MILP_calls':0,'training_updates':0}
 if method=='GET'and parts==['tasks']:
  from .coverage_ai_search import observed_status
  return {'tasks':[{'job':j,'status':observed_status(app.store,j,s)}for j,s in app.store.jobs()if j.get('experiment_id')==GROUP]}
 if method=='POST'and parts==['tasks']:return frozen_round_tasks.create(app,body,policy)
 if len(parts)>=2 and parts[0]=='tasks':
  run=validate_id(parts[1]);job=read_json(app.store.job_dir(run)/'job.json',{})
  if job.get('experiment_id')!=GROUP:raise ValueError('othergroup')
  if method=='GET'and len(parts)==2:return app.run_detail(run)
  if method=='POST'and parts[2:]==['resume']:
   return frozen_round_tasks.resume(app.scheduler,run,policy)
  if method=='POST'and parts[2:]==['stop']:return app.scheduler.request_stop(run)
 raise FileNotFoundError('unknown P0 endpoint')