"""Frozen pilot-only R3 contracts over the shared publication machinery."""
from copy import deepcopy
from datetime import datetime,timezone
from pathlib import Path
import os
from .storage import read_json,validate_id,digest
from .coverage_ai_search import pin
from . import frozen_round_tasks
GROUP='S1-AI17-R3'
DESIGN_SHA='2ddacb750898005e20595b7c38745e9197435f58da76cd1c29743c892b420f07'
PARENT=Path('/home/u2024110224/first_stage_satellite_rl/s1_ai19')
PORTS=('20418','20424','20473')
METHODS=('WITNESS','RULE','RANDOM','FULL','REF20','ORACLE_RESTRICT')
MODES=('prepare_candidates','development_gate')
FIELDS=('config','mode','dataset_role','case_indices','executor_port','methods','prepared_ref','external_refs')
CANONICAL_DATASETS=True
RUNTIME_FIELDS=('mode','dataset_role','case_indices','executor_port','methods')

def deployment(store):
 root=store.archive_root.resolve();port=root.parent.name
 if root.name!='archive' or root.parent.parent!=PARENT.resolve() or port not in PORTS or os.environ.get('SATELLITE_RL_SSH_PORT')!=port:raise ValueError('independent R3 lane required')
 from .coverage import _require_remote_dqn
 _require_remote_dqn();return port

def task_contract(body):return {k:deepcopy(body.get(k)) for k in FIELDS}
def run_prefix(c):return 'prepare_ai_search_r3' if c['mode']=='prepare_candidates' else 'develop_ai_search_r3'
def plan_references(p):return {'identity':p['identity_ref']}
def decorate_job(c,p,refs,commit):return {'phase':'pilot','execution_commit':commit,'algorithm':'ai_search_r3','network':'candidate_count','kind':'prepare' if c['mode']=='prepare_candidates' else 'evaluate','execution_device':'cpu','methods':c['methods']}
def verify_metadata(j,p):
 if j.get('phase')!='pilot' or j.get('execution_commit')!=j['scientific_commit'] or j.get('algorithm')!='ai_search_r3':raise ValueError('R3 epoch metadata differs')
def design(p):
 if p['design']['sha256']!=DESIGN_SHA:raise ValueError('only signed R3 v2 design')
 return read_json(pin(p['design']),{})
def validate_vendor(reference,port):
 value=read_json(pin(reference),{});root=PARENT/str(port)/'vendor/highspy_1_15_1'
 if value.get('passed')is not True or value.get('version')!='1.15.1' or Path(value.get('vendor','')).resolve()!=root.resolve():raise ValueError('source-bound isolated native vendor required')
 files=value['vendor_manifest'];expected=set()
 for row in files:
  rel=Path(row['path']);p=root/rel
  if rel.is_absolute() or '..' in rel.parts or digest(p)!=row['sha256'] or p.stat().st_size!=row['bytes']:raise ValueError('vendor member changed')
  expected.add(rel.as_posix())
 actual={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
 if actual!=expected:raise ValueError('vendor extra or missing files')
 if any(r['options']['threads']['actual']!=1 for r in value['options_set_get']):raise ValueError('native threads1 not verified')
 return value

def validate_config(c):
 if set(c)!={'schema_version','research_stage','algorithm','name','protocol_ref','datasets','workers','resources','vendor_ref'} or (c['schema_version'],c['research_stage'],c['algorithm'])!=(11,'coverage','ai_search_r3'):raise ValueError('exact R3 schema11 config required')
 if c['workers']!=1 or type(c['workers'])is not int:raise ValueError('one formal case per lane')
 if set(c['datasets'])!={'development'} or c['datasets']['development']['role']!='development' or c['datasets']['development']['count']!=80 or c['datasets']['development']['sha256']!='9929e31c2ecd7adca912deeb16ad58abd6bd97d7cb2c631fc53cb4a2c57e6b5b':raise ValueError('pilot cannot access other data roles')
 if c['protocol_ref']['sha256']!=DESIGN_SHA or not isinstance(c['name'],str) or not 1<=len(c['name'])<=100:raise ValueError('name/protocol')
 if c['resources']!={'minimum_free_bytes':8589934592,'max_rss_bytes':4294967296}:raise ValueError('frozen resource guards required')
 return deepcopy(c)

def scope(p):
 d=design(p);ids=read_json(pin(p['identity_ref']),{})['development'];order=p['mother_order'];selected=d['data']['development_pilot_mother_indices'];chosen=[order[i] for i in selected] if len(order)==20 else []
 indices=sorted(i for i,m in enumerate(ids['mother_ids']) if m in chosen)
 actual_order=sorted(set(ids['mother_ids']),key=lambda m:(datetime.fromisoformat(m.split(':',1)[1].replace('Z','+00:00')),m))
 if order!=actual_order or len(ids['case_ids'])!=80 or len(set(ids['case_ids']))!=80:raise ValueError('original UTC identities/order differ')
 if p.get('phase')!='pilot' or len(order)!=20 or len(set(order))!=20 or len(indices)!=24 or p.get('selected_case_indices')!=indices or p.get('selected_mothers')!=chosen or p.get('training_allowed')is not False or p.get('test_allowed')is not False:raise ValueError('exact pilot24 only')
 # Each selected mother has all frozen profiles on one physical owner.
 tasks=list(p['tasks'].values())
 if len(tasks)!=3 or {str(t['executor_port']) for t in tasks}!=set(PORTS) or sorted(i for t in tasks for i in t['case_indices'])!=indices or len({t['mode'] for t in tasks})!=1:raise ValueError('exact three-lane pilot scope')
 owners={}
 for t in tasks:
  for i in t['case_indices']:
   m=ids['mother_ids'][i]
   if owners.setdefault(m,t['executor_port'])!=t['executor_port']:raise ValueError('mother split across lanes')
 for m in chosen:
  selected_rows=[i for i in indices if ids['mother_ids'][i]==m]
  if len(selected_rows)!=4 or {ids['profiles'][i] for i in selected_rows}!=set(d['data']['profiles']):raise ValueError('mother/profile identity mismatch')
 return indices

def validate_task(c,p,port):
 config=validate_config(c['config']);indices=scope(p)
 if config['datasets']['development']['sha256']!=read_json(pin(p['identity_ref']),{})['development']['sha256']:raise ValueError('development dataset identity SHA differs')
 if config['protocol_ref']!=p['design'] or c['mode']not in MODES or c['dataset_role']!='development' or str(c['executor_port'])!=str(port):raise ValueError('mode/role/physical owner/protocol differs')
 if not c['case_indices'] or len(set(c['case_indices']))!=len(c['case_indices']) or any(type(i)is not int or i not in indices for i in c['case_indices']):raise ValueError('unselected case')
 if set(c['external_refs'])!={str(i) for i in c['case_indices']}:raise ValueError('exact R2 anchor inventory required')
 for refs in c['external_refs'].values():
  if set(refs)!= {'parent','D0','D1','audit'}:raise ValueError('exact R2 parent/D0/D1/audit refs required')
 if c['mode']=='prepare_candidates':
  if c['methods']!=[] or c['prepared_ref']is not None:raise ValueError('preparation cannot run a method/solver')
 else:
  if c['methods']!=list(METHODS) or not c['prepared_ref']:raise ValueError('six frozen pilot methods and prepared artifacts required')
  prepared=read_json(pin(c['prepared_ref']),{})
  if (prepared.get('group')!=GROUP or prepared.get('completed')is not True or prepared.get('phase')!='pilot' or prepared.get('source_commit')!=p['scientific_commit'] or prepared.get('execution_commit')!=p['scientific_commit'] or prepared.get('design_ref',{}).get('sha256')!=DESIGN_SHA or set(prepared.get('cases',{}))!={str(i) for i in c['case_indices']}):raise ValueError('prepared epoch/role/scope differs')
  ids=read_json(pin(p['identity_ref']),{})['development']
  for i in c['case_indices']:
   row=prepared['cases'][str(i)]
   if row['input_case_index']!=i or row['case_id']!=ids['case_ids'][i] or row['mother_id']!=ids['mother_ids'][i] or row['input_dataset_sha256']!=config['datasets']['development']['sha256']:raise ValueError('prepared mother/index/input differs')
 validate_vendor(config['vendor_ref'],port)
 for r in external_references(c).values():pin(r)
 ids=read_json(pin(p['identity_ref']),{})['development']
 from satellite_rl.coverage.ai_search.core import Layout,layout_sha256
 for i,refs in c['external_refs'].items():
  audit=read_json(pin(refs['audit']),{})
  if audit.get('passed')is not True or audit.get('issues')!=[] or audit.get('case_id')!=ids['case_ids'][int(i)]:raise ValueError('R2 source audit identity/validity differs')
  for name in ('parent','D0','D1'):
   bundle=read_json(pin(refs[name]),{})
   if (refs[name].get('json_pointer')!=['layouts',name,'layout'] or bundle.get('schema')!='r3_three_fixed_R2_external_layouts_v1' or bundle.get('case_id')!=ids['case_ids'][int(i)] or bundle.get('input_case_index')!=int(i) or bundle.get('input_dataset_sha256')!=config['datasets']['development']['sha256'] or bundle.get('input_sha256')!=audit.get('input_sha256') or bundle.get('original_case_audit_ref',{}).get('sha256')!=refs['audit']['sha256']):raise ValueError('R2 bundle original SHA/case/source selector differs')
   point=bundle['layouts'][name]
   if layout_sha256(Layout.from_record(point['layout']))!=point['layout_sha256']:raise ValueError('R2 actual published layout SHA differs')
 return config

def validate_admission(p,plan_ref,review_ref):
 r=read_json(pin(review_ref),{});design(p)
 if p.get('group')!=GROUP or p.get('authorized')is not True or r.get('passed')is not True or r.get('group')!=GROUP or r.get('scientific_commit')!=p['scientific_commit'] or r.get('plan_sha256')!=plan_ref['sha256'] or r.get('design_sha256')!=DESIGN_SHA:raise ValueError('root exact source/plan/design approval required')
 refs=p.get('qualification_refs',{})
 if set(refs)!={'core'}:raise ValueError('core engineering qualification required')
 q=read_json(pin(refs['core']),{})
 if q.get('passed')is not True or q.get('scientific_commit')!=p['scientific_commit'] or q.get('design_sha256')!=DESIGN_SHA or not q.get('source_file_sha256'):raise ValueError('qualification epoch/design missing')
 base=Path(__file__).resolve().parents[3];folder=base/'src/satellite_rl/coverage/ai_search_r3'
 actual={f.relative_to(base).as_posix():digest(f) for f in folder.rglob('*.py') if '__pycache__' not in f.parts}
 if q['source_file_sha256']!=actual:raise ValueError('qualification core source member set/SHA differs')

def external_references(c):
 refs={'vendor':c['config']['vendor_ref']}
 if c['prepared_ref']:refs['prepared_manifest']=c['prepared_ref']
 for i,points in c['external_refs'].items():
  for key,r in points.items():refs['R2_'+i+'_'+key]={'path':r['path'],'sha256':r['sha256']}
 return refs

def expected_results(c):return len(c['case_indices'])*max(1,len(c['methods']))
def resource_admission(store,job):
 port=deployment(store);route=read_json(store.archive_root.parent/'ops/resource_route.json',{});age=(datetime.now(timezone.utc)-datetime.fromisoformat(route.get('observed_utc','1970-01-01T00:00:00+00:00'))).total_seconds()
 if route.get('group')!=GROUP or str(route.get('ssh_port'))!=port or route.get('scientific_commit')!=job['scientific_commit'] or route.get('ready')is not True or not -5<=age<=120:raise ValueError('fresh R3 same-source route required')
 from .ai_search_resources import observe
 from .runtime import pid_alive
 result=observe(store.archive_root)
 if result['foreign_workers'] or result['gpu'].split(',')[0].strip()!=route['gpu_uuid']:raise ValueError('foreign worker or changed GPU')
 for other,status in store.jobs():
  launch=read_json(store.job_dir(other['run_id'])/'launch.json',{})
  if other['run_id']!=job['run_id'] and pid_alive(status.get('pid'),launch.get('process_start_time')):raise ValueError('another task occupies lane')
 return result

def api(app,method,parts,body):
 import sys
 policy=sys.modules[__name__]
 if method=='GET' and parts==['schema']:return {'group':GROUP,'schema_version':11,'modes':MODES,'methods':METHODS,'pilot_cases':24,'training_allowed':False,'test_allowed':False}
 if method=='GET' and parts==['tasks']:
  from .coverage_ai_search import observed_status
  return {'tasks':[{'job':j,'status':observed_status(app.store,j,s)} for j,s in app.store.jobs() if j.get('experiment_id')==GROUP]}
 if method=='POST' and parts==['tasks']:return frozen_round_tasks.create(app,body,policy)
 if len(parts)>=2 and parts[0]=='tasks':
  run_id=validate_id(parts[1]);j=read_json(app.store.job_dir(run_id)/'job.json',{})
  if j.get('experiment_id')!=GROUP:raise ValueError('another group')
  if method=='GET' and len(parts)==2:return app.run_detail(run_id)
  if method=='POST' and parts[2:]==['resume']:return frozen_round_tasks.resume(app.scheduler,run_id,policy)
  if method=='POST' and parts[2:]==['stop']:return app.scheduler.request_stop(run_id)
 raise FileNotFoundError('unknown R3 endpoint')