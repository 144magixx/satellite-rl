"""Independent source-frozen learner/node APIs; later data roles remain closed."""
from copy import deepcopy
from pathlib import Path
from datetime import datetime,timezone
import os,sys
from .storage import read_json,digest,validate_id
from .coverage_ai_search import pin
from . import frozen_round_tasks
from scripts.stage1_ai_search_r6model.plan_builder import GROUP,DESIGN_SHA,NODE_INDICES,audit
MODES=('train_setup','train2000','train6000','node2000');CANONICAL_DATASETS=True
FIELDS=('config','mode','dataset_role','case_indices','executor_port','methods','model_inputs','model_identity');RUNTIME_FIELDS=FIELDS[1:]
def freeze_source(store):
 from .r6_model_freeze import freeze_source as complete_freeze
 return complete_freeze(store)

def task_contract(b):return {k:deepcopy(b.get(k)) for k in FIELDS}
def run_prefix(c):return 'model_ai_search_r6'
def plan_references(p):return {'identity':p['identity_ref'],'prerequisite_gate':p['prerequisite_gate_ref']}
def external_references(c):return {'model_'+k:v for k,v in c['model_inputs'].items()}
def decorate_job(c,p,refs,commit):return {'phase':p['phase'],'execution_commit':commit,'algorithm':'ai_search_r6_model','network':'single_set_model','kind':'prepare' if c['mode']=='train_setup' else 'train' if c['mode'].startswith('train') else 'evaluate','execution_device':'cuda','methods':c['methods'],'model_inputs':deepcopy(c['model_inputs']),'model_identity':c['model_identity']}
def verify_metadata(j,p):
 if j.get('phase')!=p['phase'] or j.get('execution_commit')!=j['scientific_commit'] or j.get('algorithm')!='ai_search_r6_model':raise ValueError('model source/phase changed')
def expected_results(c):return len(c['case_indices']) if c['mode']=='train_setup' else c['config']['training']['end_update'] if c['mode'].startswith('train') else len(c['case_indices'])*3

def deployment(store):
 root=store.archive_root.resolve();port=root.parent.parent.parent.name;expected=Path('/home/u2024110224/first_stage_satellite_rl/s1_ai17')/port/'round6/learning/archive'
 if root!=expected or port not in ('20418','20424','20473','20474') or os.environ.get('SATELLITE_RL_SSH_PORT')!=port:raise ValueError('isolated learning workbench')
 from .coverage import _require_remote_dqn
 _require_remote_dqn();return port

def validate_config(c):
 if c.get('schema_version')!=15 or c.get('research_stage')!='coverage' or c.get('algorithm')!='ai_search_r6_model' or c.get('workers')!=1 or c.get('protocol_ref',{}).get('sha256')!=DESIGN_SHA:raise ValueError('fixed schema15 single worker')
 if c.get('resources')!={'minimum_free_bytes':8589934592,'parent_children_rss_bytes':4294967296}:raise ValueError('model resource caps')
 return deepcopy(c)
def validate_task(c,p,port):
 audit(p);cfg=validate_config(c['config'])
 if str(c['executor_port'])!=str(port) or c['model_identity']!=p['model_identity'] or c not in p['tasks'].values():raise ValueError('signed exact model lane')
 if c['mode'].startswith('train'):
  if c['mode']!='train_setup' and cfg['training']!={'seed':51001,'batch':16,'microbatch':4,'start_update':0 if c['mode']=='train2000' else 2000,'end_update':2000 if c['mode']=='train2000' else 6000,'checkpoint_interval':100} or cfg['hard_watchdogs']!={'training_active_seconds':28800}:raise ValueError('frozen training mechanism')
  if set(cfg['datasets'])!={'train'} or cfg['datasets']['train'].get('role')!='train' or cfg['datasets']['train'].get('count')!=2304 or cfg['datasets']['train'].get('sha256')!='7bc2b1f8fdd583e136621e3d25bf8aed371fa72e8d4a399f21f4ed3cd27655af':raise ValueError('only true training data')
 else:
  if cfg['hard_watchdogs']!={'AI':90,'decoder':15} or set(cfg['datasets'])!={'development'} or cfg['datasets']['development'].get('sha256')!='9929e31c2ecd7adca912deeb16ad58abd6bd97d7cb2c631fc53cb4a2c57e6b5b' or cfg['datasets']['development'].get('role')!='development' or cfg['datasets']['development'].get('count')!=80:raise ValueError('node only frozen development input')
  registry=read_json(pin(c['model_inputs']['registry']),{})
  if registry.get('case_indices')!=NODE_INDICES or registry.get('roles_read')!=['development']:raise ValueError('exact original24 prepared registry')
 return cfg

def validate_admission(p,plan_ref,review_ref):
 audit(p);r=read_json(pin(review_ref),{})
 if not r.get('passed') or r.get('group')!=GROUP or r.get('scientific_commit')!=p['scientific_commit'] or r.get('plan_sha256')!=plan_ref['sha256'] or r.get('design_sha256')!=DESIGN_SHA:raise ValueError('root source/plan exact admission')
 pin(p['design']);gate=read_json(pin(p['prerequisite_gate_ref']),{})
 if not gate.get('passed') or gate.get('model_identity')!=p['model_identity']:raise ValueError('root prior-stage capability')
 expected='labels' if p['phase'] in ('train_setup','train2000') else 'node2000' if p['phase']=='train6000' else 'train2000'
 if gate.get('phase')!=expected:raise ValueError('dependent gate cannot be skipped')
 if expected=='train2000' and gate.get('updates')!=2000:raise ValueError('exact2000 checkpoint')
 for task in p['tasks'].values():
  refs=task['model_inputs']
  if p['phase'] in ('train_setup','train2000') and refs['labels_gate']!=p['prerequisite_gate_ref']:raise ValueError('qualified label gate is not the signed prerequisite')
  if p['phase']=='node2000' and any(refs[key]!=gate.get(field) for key,field in (('checkpoint','checkpoint_ref'),('normalization','normalization_ref'),('initialization_receipt','initialization_receipt_ref'))):raise ValueError('node task refs do not bind exact passed2000 endpoint')
  if p['phase']=='train6000' and (refs['node_gate']!=p['prerequisite_gate_ref'] or refs['checkpoint']['sha256']!=gate.get('checkpoint_sha256')):raise ValueError('continuation does not bind actual passed node checkpoint')
 if not p['qualification_refs']:raise ValueError('source-bound engineering qualification')
 root=Path(__file__).resolve().parents[3]
 for ref in p['qualification_refs'].values():
  q=read_json(pin(ref),{});members=q.get('source_file_sha256',{})
  if q.get('passed')is not True or q.get('scientific_commit')!=p['scientific_commit'] or q.get('design_sha256')!=DESIGN_SHA or not members or any(digest(root/n)!=s for n,s in members.items()):raise ValueError('qualified model runner source differs')

def resource_admission(store,j):
 port=deployment(store);r=read_json(store.archive_root.parent/'ops/resource_route.json',{});age=(datetime.now(timezone.utc)-datetime.fromisoformat(r.get('observed_utc','1970-01-01T00:00:00+00:00'))).total_seconds()
 if not r.get('ready') or r.get('group')!=GROUP or str(r.get('ssh_port'))!=port or r.get('scientific_commit')!=j['scientific_commit'] or not -5<=age<=120:raise ValueError('fresh unique learning route required')
 from .r6_resources import observe
 actual=observe(store.archive_root)
 if actual['foreign_workers'] or actual['gpu'].split(',')[0].strip()!=r['gpu_uuid']:raise ValueError('foreign worker/GPU changed')
 return actual

def api(app,method,parts,body):
 policy=sys.modules[__name__]
 if method=='GET' and parts==['schema']:return {'group':GROUP,'schema_version':15,'modes':MODES,'single_model':True,'science_updates_max':6000,'node2000_D5_slots':72,'validation_test_allowed':False}
 if method=='GET' and parts==['tasks']:
  from .coverage_ai_search import observed_status
  return {'tasks':[{'job':j,'status':observed_status(app.store,j,s)} for j,s in app.store.jobs() if j.get('experiment_id')==GROUP]}
 if method=='POST' and parts==['tasks']:return frozen_round_tasks.create(app,body,policy)
 if len(parts)>=2 and parts[0]=='tasks':
  run=validate_id(parts[1]);job=read_json(app.store.job_dir(run)/'job.json',{})
  if job.get('experiment_id')!=GROUP:raise ValueError('foreign model group')
  if method=='GET' and len(parts)==2:return app.run_detail(run)
  if method=='POST' and parts[2:]==['resume']:return frozen_round_tasks.resume(app.scheduler,run,policy)
  if method=='POST' and parts[2:]==['stop']:return app.scheduler.request_stop(run)
 raise FileNotFoundError('unknown model endpoint')
