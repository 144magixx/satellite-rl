"""Only the signed R4 B0 pilot:24case bundles, zero MILP/training."""
from copy import deepcopy
from datetime import datetime,timezone
from pathlib import Path
import os,sys
from .storage import read_json,validate_id,digest
from .coverage_ai_search import pin
from . import frozen_round_tasks
GROUP='S1-AI17-R4-B0';DESIGN_SHA='512482fa8f350df120b0bb501224a9b433b4ffd1ad05099a90f75c8bd38891c7'
IDENTITY_SHA='373d1b959231640f764ad21af09e9ed188d1635d60de04812ae6b793e88637d7';DATA_SHA='9929e31c2ecd7adca912deeb16ad58abd6bd97d7cb2c631fc53cb4a2c57e6b5b'
PARENT=Path('/home/u2024110224/first_stage_satellite_rl/s1_ai20');PORTS=('20418','20424','20473');METHODS=('RULE','RANDOM','TEACHER_EXACT','TEACHER_PERTURBED');MODES=('b0',)
FIELDS=('config','mode','dataset_role','case_indices','executor_port','methods');RUNTIME_FIELDS=FIELDS[1:];CANONICAL_DATASETS=True

def task_contract(b):return {k:deepcopy(b.get(k))for k in FIELDS}
def run_prefix(c):return 'b0_ai_search_r4'
def plan_references(p):return {'identity':p['identity_ref']}
def external_references(c):return {}
def decorate_job(c,p,refs,commit):return {'phase':'B0','execution_commit':commit,'algorithm':'ai_search_r4','network':'group_decoder_b0','kind':'evaluate','execution_device':'cpu','methods':c['methods']}
def verify_metadata(j,p):
 if j.get('phase')!='B0'or j.get('execution_commit')!=j['scientific_commit']or j.get('algorithm')!='ai_search_r4':raise ValueError('B0 metadata/source differs')
def expected_results(c):return len(c['case_indices'])*5

def deployment(store):
 root=store.archive_root.resolve();port=root.parent.name
 if root.name!='archive'or root.parent.parent!=PARENT.resolve()or port not in PORTS or os.environ.get('SATELLITE_RL_SSH_PORT')!=port:raise ValueError('independent three-lane R4 root required')
 from .coverage import _require_remote_dqn
 _require_remote_dqn();return port

def design(p):
 if p['design']['sha256']!=DESIGN_SHA:raise ValueError('fixed B0 design required')
 d=read_json(pin(p['design']),{})
 if d['scope']['B0_only']is not True or d['scope']['MILP_calls']!=0 or d['scope']['training_updates']!=0:raise ValueError('B0 zeroMIP/update boundaries')
 return d

def validate_config(c):
 if set(c)!={'schema_version','research_stage','algorithm','name','protocol_ref','datasets','workers','resources','hard_watchdogs'}or (c['schema_version'],c['research_stage'],c['algorithm'])!=(12,'coverage','ai_search_r4'):raise ValueError('B0 schema12 exactconfig')
 if c['workers']!=1 or type(c['workers'])is not int:raise ValueError('one sequential case per physical lane')
 if c['datasets'].keys()!= {'development'} or {k:c['datasets']['development'].get(k)for k in ('role','count','sha256')}!= {'role':'development','count':80,'sha256':DATA_SHA}:raise ValueError('only pinned original development source')
 if c['protocol_ref']['sha256']!=DESIGN_SHA or not isinstance(c['name'],str)or not 1<=len(c['name'])<=100:raise ValueError('name/protocol')
 if c['hard_watchdogs']!={'preparation':20,'teacher':45,'decoder':15}or c['resources']!={'minimum_free_bytes':8589934592,'parent_children_rss_bytes':2147483648}:raise ValueError('signed ownedchild guards required')
 return deepcopy(c)

def validate_task(c,p,port):
 cfg=validate_config(c['config']);d=design(p)
 if p.get('phase')!='B0'or p.get('training_allowed')is not False or p.get('test_allowed')is not False or p['identity_ref']['sha256']!=IDENTITY_SHA:raise ValueError('B0 phase/data identity only')
 ids=read_json(pin(p['identity_ref']),{})['development'];order=sorted(set(ids['mother_ids']),key=lambda m:(datetime.fromisoformat(m.split(':',1)[1].replace('Z','+00:00')),m));chosen=[order[i]for i in d['data']['mother_indices']];indices=d['data']['case_indices']
 if p['mother_order']!=order or p['selected_mothers']!=chosen or chosen!=d['data']['mother_ids']or p['selected_case_indices']!=indices or ids['sha256']!=DATA_SHA:raise ValueError('actual prescribed24 identities differ')
 tasks=list(p['tasks'].values());owners={}
 if len(tasks)!=3 or {str(t['executor_port'])for t in tasks}!=set(PORTS)or sorted(i for t in tasks for i in t['case_indices'])!=indices:raise ValueError('exact3lane24 scope')
 for t in tasks:
  if t['mode']!='b0'or t['methods']!=list(METHODS)or t['dataset_role']!='development':raise ValueError('teacher+4decoder only')
  for i in t['case_indices']:
   if owners.setdefault(ids['mother_ids'][i],t['executor_port'])!=t['executor_port']:raise ValueError('mother profiles split')
 if c['mode']!='b0'or c['dataset_role']!='development'or str(c['executor_port'])!=str(port)or c['methods']!=list(METHODS)or cfg['protocol_ref']!=p['design']:raise ValueError('mode/method/physical source')
 if not c['case_indices']or any(type(i)is not int or i not in indices for i in c['case_indices'])or len(set(c['case_indices']))!=len(c['case_indices']):raise ValueError('unselected/repeated case')
 return cfg

def validate_admission(p,plan_ref,review_ref):
 d=design(p);r=read_json(pin(review_ref),{})
 if p.get('group')!=GROUP or p.get('authorized')is not True or r.get('passed')is not True or r.get('group')!=GROUP or r.get('scientific_commit')!=p['scientific_commit']or r.get('plan_sha256')!=plan_ref['sha256']or r.get('design_sha256')!=DESIGN_SHA:raise ValueError('exact root plan/source approval')
 if set(p.get('qualification_refs',{}))!={'geometry'}:raise ValueError('8case geometry qualification required')
 q=read_json(pin(p['qualification_refs']['geometry']),{})
 if q.get('passed')is not True or q.get('scientific_commit')!=p['scientific_commit']or q.get('design_sha256')!=DESIGN_SHA or q.get('geometry_fixture_cases')!=8:raise ValueError('source-bound geometry qualification')
 base=Path(__file__).resolve().parents[3];folder=base/'src/satellite_rl/coverage/ai_search_r4';actual={f.relative_to(base).as_posix():digest(f)for f in folder.rglob('*.py')if '__pycache__'not in f.parts}
 if not actual or q.get('source_file_sha256')!=actual:raise ValueError('qualified R4 core members changed')

def resource_admission(store,j):
 port=deployment(store);r=read_json(store.archive_root.parent/'ops/resource_route.json',{});age=(datetime.now(timezone.utc)-datetime.fromisoformat(r.get('observed_utc','1970-01-01T00:00:00+00:00'))).total_seconds()
 if r.get('group')!=GROUP or str(r.get('ssh_port'))!=port or r.get('scientific_commit')!=j['scientific_commit']or r.get('ready')is not True or not -5<=age<=120:raise ValueError('fresh B0 source route required')
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
 if method=='GET'and parts==['schema']:return {'group':GROUP,'schema_version':12,'modes':MODES,'methods':METHODS,'slots':120,'MILP_calls':0,'training_updates':0}
 if method=='GET'and parts==['tasks']:
  from .coverage_ai_search import observed_status
  return {'tasks':[{'job':j,'status':observed_status(app.store,j,s)}for j,s in app.store.jobs()if j.get('experiment_id')==GROUP]}
 if method=='POST'and parts==['tasks']:return frozen_round_tasks.create(app,body,policy)
 if len(parts)>=2 and parts[0]=='tasks':
  run=validate_id(parts[1]);job=read_json(app.store.job_dir(run)/'job.json',{})
  if job.get('experiment_id')!=GROUP:raise ValueError('othergroup')
  if method=='GET'and len(parts)==2:return app.run_detail(run)
  if method=='POST'and parts[2:]==['resume']:return frozen_round_tasks.resume(app.scheduler,run,policy)
  if method=='POST'and parts[2:]==['stop']:return app.scheduler.request_stop(run)
 raise FileNotFoundError('unknown B0 endpoint')