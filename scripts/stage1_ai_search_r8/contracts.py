"""Exact-plan and durable owner contracts; no old endpoint dispatch."""
import datetime,hashlib,json,os,time
from pathlib import Path

GROUP='S1-AI17-R8'
DESIGN_SHA='e0e24cf00e873965ee585d77f5f1e38c7455196a19a5357de7c5732fcfa08bb4'
CHECKPOINT_SHA='10ff42e745ee0b1639cab9781d53ca43f5baa61b7e9e75aa12e49eec15bc9a40'
DEADLINE=1791932400.0
OPS_FILES=('__init__.py','contracts.py','deadline.py','server.py','worker.py','child.py','plan_builder.py','recovery.py')
CORE_FILES=('__init__.py','core.py','geometry.py','ledger.py')
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def pin(r):
 p=Path(r['path'])
 if digest(p)!=r['sha256']:raise ValueError('exact pinned bytes differ: '+str(p))
 return p
def read(r):return json.loads(pin(r).read_bytes())
def atomic(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp')
 with tmp.open('w',encoding='utf8') as f:json.dump(v,f,ensure_ascii=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 os.replace(tmp,p)
def immutable(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x',encoding='utf8') as f:json.dump(v,f,ensure_ascii=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 return dict(path=str(p),sha256=digest(p))
def identity(pid):
 p=Path('/proc')/str(pid)
 if not p.exists():return None
 try:raw=(p/'stat').read_text()
 except FileNotFoundError:return None
 fields=raw[raw.rfind(')')+2:].split()
 return dict(pid=pid,birth_ticks=int(fields[19]),state=fields[0],boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
def order(i):
 arms=['G','A','R'];return arms[i%3:]+arms[:i%3]
def validate_sources(p):
 for ref in p['source_files'].values():pin(ref)
 for name in OPS_FILES:
  key='scripts/stage1_ai_search_r8/'+name
  if key not in p['source_files'] or Path(p['source_files'][key]['path']).resolve()!=Path(__file__).with_name(name).resolve():raise ValueError('actual complete seven R8 operations source')
 for name in CORE_FILES:
  key='src/satellite_rl/coverage/ai_search_r8/'+name
  expected=Path(p['core_source_root'])/'satellite_rl/coverage/ai_search_r8'/name
  if key not in p['source_files'] or Path(p['source_files'][key]['path']).resolve()!=expected.resolve():raise ValueError('complete four actual R8 core sources')
 if p['core_ref']!=p['source_files']['src/satellite_rl/coverage/ai_search_r8/core.py']:raise ValueError('exact sole actual core ref')
def validate_imported_core(p,modules):
 for name in CORE_FILES:
  ref=p['source_files']['src/satellite_rl/coverage/ai_search_r8/'+name];pin(ref)
  if name not in modules or Path(modules[name].__file__).resolve()!=Path(ref['path']).resolve():raise ValueError('loaded R8 relative module differs: '+name)
def storage_inventory(p):
 inventory=read(p['storage_inventory_ref']);root=Path(p['archive_root']).resolve()
 if inventory.get('schema')!='R8OwnedStorageInventoryV1' or inventory.get('group')!=GROUP or inventory.get('archive_root')!=p['archive_root'] or inventory.get('root_accepted')is not True or inventory.get('new_logical_bytes_max')!=1073741824 or inventory.get('free_reserve_bytes')!=8589934592:raise ValueError('exact root-pinned R8 storage inventory/caps')
 growth=Path(inventory['owned_growth_root']).resolve()
 if '/round8' not in growth.as_posix() or not root.is_relative_to(growth):raise ValueError('own R8 growth tree must contain own archive')
 for path in inventory.get('additional_owned_roots',[]):
  if '/round8' not in Path(path).resolve().as_posix():raise ValueError('additional growth must be own R8 directory')
 if not isinstance(inventory.get('fixed_other_owned_bytes'),int) or inventory['fixed_other_owned_bytes']<0:raise ValueError('explicit immutable source/meta bytes outside growth roots')
 return inventory
def validate(plan_ref,review_ref):
 p=read(plan_ref);r=read(review_ref)
 if p.get('schema')!='R8ExactCachedOwnerPilotPlanV1' or p.get('group')!=GROUP or p.get('executor_port') not in (20473,20474):raise ValueError('own exact R8 plan/route')
 if not p.get('run_id','').startswith('r8_pilot_') or '/round8/' not in p['archive_root'] or '/round7/' in p['archive_root']:raise ValueError('independent R8 task/archive namespace')
 if r.get('plan_ref')!=plan_ref or any(r.get(k)is not True for k in ('passed','execution_authorized','scientific_execution_authorized')):raise ValueError('root exact R8 scientific execution admission')
 if p['design_ref']['sha256']!=DESIGN_SHA:raise ValueError('frozen R8v2 design')
 design=read(p['design_ref']);read(p['root_design_acceptance_ref'])
 if design['scientific_budgets']!=dict(cases=20,local_runs=60,local_candidates_max=7680,local_rank_items_max=3932160,D5_calls_max=60,D5_proposals_max=30720,D5_refits_max=480,W_calls=0,T_calls=0,model_forwards=0,updates=0,R7_unused4000_transfer=False):raise ValueError('budget exact design')
 if p['case_indices']!=list(range(20)) or p['arms']!=['G','A','R'] or p['checkpoint_ref']['sha256']!=CHECKPOINT_SHA:raise ValueError('full20/three arms/fixed model')
 qualification=read(p['input_qualification_ref'])
 if qualification.get('passed')is not True or qualification.get('case_indices')!=list(range(20)) or qualification.get('heldout_access')is not False:raise ValueError('original20 cache qualification required')
 if p['cases']!=qualification['cases']:raise ValueError('exact qualified original input refs')
 if [item['case_index'] for item in p['cases']]!=list(range(20)):raise ValueError('actual full20 original order')
 pin(p['checkpoint_ref'])
 for item in p['cases']:
  if item['checkpoint_ref']!=p['checkpoint_ref']:raise ValueError('same cached model checkpoint')
  pin(item['normalization_ref'])
  for ref in item['refs'].values():pin(ref)
 validate_sources(p);storage_inventory(p)
 sm=read(p['original_source_manifest_ref'])
 if sm['git_commit']!='523fbc2b1e2480dcc563d302ddcc85317e271aa1':raise ValueError('original D5 source epoch')
 source=Path(p['original_source_manifest_ref']['path']).parent
 for name,sha in sm['files'].items():
  if digest(source/name)!=sha:raise ValueError('original source changed: '+name)
 registration=read(p['deadline_registration_ref'])
 if registration.get('group')!=GROUP or registration.get('archive_root')!=p['archive_root'] or registration.get('deadline_unix_seconds')!=DEADLINE or registration.get('registered_modes')!=['r8_pilot','r8_local','r8_D5'] or registration.get('passed')is not True or registration.get('root_accepted')is not True:raise ValueError('own deadline modes/archive must be registered')
 if p.get('scientific_new_calls')!=dict(local=60,D5=60,W=0,T=0,forward=0,updates=0):raise ValueError('no hidden old train or extra scientific calls')
 if not registration.get('shared_pause_paths') or not any(path.endswith('/control/user_pause_20261014_0700.json') for path in registration['shared_pause_paths']):raise ValueError('original shared human pause contract must be included')
 return p
def fresh_route(plan):
 route=read(plan['actual_resource_route_ref']);age=time.time()-route['observed_unix_seconds']
 if not -5<=age<=120 or route.get('ready')is not True or route.get('root_accepted')is not True or route.get('ssh_port')!=plan['executor_port']:raise ValueError('fresh admitted own route')
 return route
