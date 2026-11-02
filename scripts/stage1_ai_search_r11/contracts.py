"""R11 exact Gh-only plan; original R10 common remains a historical wrapper."""
import importlib.util,os,sys,time
from pathlib import Path
from scripts.stage1_ai_search_r10 import contracts as c,control

GROUP='S1-AI17-R11'
DESIGN_SHA='6a4fd42f6510c86ef9bd5db881374a8a426e5493078ee8d7d159a0f73e67a2f1'
FILES=('__init__.py','contracts.py','storage.py','child.py','worker.py','server.py','controller.py','plan_builder.py')
PARTITION={20473:(0,12,24,42,54,72),20474:(6,18,36,48,60,78)}
DQN='/home/u2024110224/first_stage_satellite_rl/envs/DQN/bin/python'
CAPS=dict(method=12,global_matching_slots=36,global_solves=26,common=0,local=0,W=0,T=0,RULE=0,D5=0,model_initialization=0,forward=0,updates=0)
fref=lambda p:dict(path=str(p),sha256=c.digest(p))

def source_check(p):
 repo=Path(p['source_repo']).resolve()
 if Path(__file__).resolve().parents[2]!=repo or Path(c.__file__).resolve().parents[2]!=repo:raise ValueError('actual ops source root')
 required={'scripts/stage1_ai_search_r11/'+n for n in FILES}
 for package in ('ai_search_r11','ai_search_r10','ai_search_r9'):
  folder=repo/'src/satellite_rl/coverage'/package
  leaves=list(folder.glob('*.py'))
  if not leaves:raise ValueError('full math package absent')
  required.update(v.relative_to(repo).as_posix() for v in leaves)
 required.update('scripts/stage1_ai_search_r10/'+n for n in ('__init__.py','contracts.py','control.py','guard.py','costs.py'))
 if not required<=set(p['source_files']):raise ValueError('complete runtime source closure')
 for key,ref in p['source_files'].items():
  if '..' in Path(key).parts or c.pin(ref).resolve()!=(repo/key).resolve():raise ValueError('exact source member path')
 sm=c.read(p['original_source_manifest_ref']);original=Path(p['original_source_manifest_ref']['path']).parent.resolve()
 for name,sha in sm['files'].items():
  if c.digest(original/name)!=sha:raise ValueError('original frozen source changed')
 for name,module in list(sys.modules.items()):
  path=getattr(module,'__file__',None)
  if not path:continue
  if name.startswith(('scripts.stage1_ai_search_r11','scripts.stage1_ai_search_r10','satellite_rl.coverage.ai_search_r11','satellite_rl.coverage.ai_search_r10','satellite_rl.coverage.ai_search_r9')):
   real=Path(path).resolve()
   if not real.is_relative_to(repo):raise ValueError('cached runtime outside admitted source')
   key=real.relative_to(repo).as_posix()
   if key not in p['source_files'] or c.pin(p['source_files'][key]).resolve()!=real:raise ValueError('actual imported leaf unpinned')
  elif name=='satellite_rl' or name.startswith('satellite_rl.'):
   real=Path(path).resolve()
   if not real.is_relative_to(original) or sm['files'].get(real.relative_to(original).as_posix())!=c.digest(real):raise ValueError('actual original import not frozen')

def human_epoch(p):
 a=c.read(p['human_resume_authority_ref']);e=c.read(p['human_epoch_ref'])
 if p['human_resume_authority_ref']['sha256']!=c.AUTHORITY_SHA or a.get('user_message_verbatim')!='继续' or a.get('human_authorization')is not True or a.get('root_accepted')is not True:raise ValueError('explicit human continue')
 if e.get('schema')!='R11NewGhHumanEpochV1' or e.get('group')!=GROUP or e.get('root_accepted')is not True or e.get('new_deadline_unix_seconds')is not None:raise ValueError('new R11 epoch')
 if e.get('authority_ref')!=p['human_resume_authority_ref'] or any(e.get(k)!=p[k] for k in ('run_id','request_id','archive_root','execution_epoch_id')):raise ValueError('exact epoch/task binding')
 if not e.get('execution_epoch_id') or not e.get('watched_pause_paths') or not isinstance(e.get('historical_hold_exemptions'),list):raise ValueError('explicit pause inventory')
 return e

def check_pause(p,point):
 e=human_epoch(p);own={str(Path(p['archive_root'])/'stop.request'),str(Path(p['archive_root'])/'pause_state.json')};exempt={};held=[]
 for old in e['historical_hold_exemptions']:
  if old['path'] in own or old['reason']!='historical0700_pause' or old['archived_evidence_ref']['sha256']!=old['sha256']:raise ValueError('new pause never exempt')
  c.pin(old['archived_evidence_ref']);exempt[old['path']]=old
 for path in sorted(set(e['watched_pause_paths'])|own|set(exempt)):
  now=control.marker_snapshot(path);old=exempt.get(path)
  if old:
   if now is None or any(now[k]!=old[k] for k in control.FINGERPRINT):held.append(path)
  elif now is not None and (Path(path).name=='stop.request' or any(now['value'].get(k) for k in control.FLAGS) or str(now['value'].get('state','')).startswith('paused')):held.append(path)
 if held:raise control.Paused('new/changed pause '+point+': '+repr(held))
 return dict(passed=True,point=point)

def storage_check(p):
 s=c.read(p['storage_inventory_ref']);root=Path(p['archive_root']).resolve()
 if s.get('schema')!='R11OwnedStorageInventoryV1' or s.get('root_accepted')is not True or s.get('archive_root')!=p['archive_root'] or (s.get('new_logical_bytes_max'),s.get('temporary_bytes_max'),s.get('free_reserve_bytes'))!=(536870912,268435456,8589934592):raise ValueError('exact storage thresholds')
 growths=[Path(x).resolve() for x in s['owned_growth_roots']]
 for path in (p['archive_root'],p['source_repo'],p['outer_control_root'],p['worker_control_root']):
  if '/round11/' not in Path(path).as_posix()+'/' or not any(Path(path).resolve().is_relative_to(g) for g in growths):raise ValueError('owned R11 storage coverage')
 return s

def validate(plan_ref,review_ref):
 p=c.read(plan_ref);r=c.read(review_ref)
 if p.get('schema')!='R11ExactSavedCommonGhPlanV1' or p.get('group')!=GROUP or p.get('source_input_budget_frozen')is not True:raise ValueError('R11 candidate not admitted')
 if p.get('executor_port') not in PARTITION or p.get('workbench_port')!={20473:9050,20474:9051}[p['executor_port']] or not p.get('run_id','').startswith('r11_'):raise ValueError('new exact R11 task and candidate port')
 for key in ('archive_root','source_repo','outer_control_root','worker_control_root'):
  path=Path(p[key])
  if not path.is_absolute() or '..' in path.parts or path.resolve()!=path.absolute() or '/round11/' not in path.as_posix()+'/':raise ValueError('no old tree/linked target')
 if p['arms']!=['Gh'] or p['caps']!=CAPS or p['method_hard_seconds']!=300 or p['method_rss_bytes']!=2147483648 or p['outer_hard_seconds']!=2400:raise ValueError('fixed finite Gh budget')
 if r.get('schema')!='R11RootExactGhExecutionAdmissionV1' or r.get('exact_plan_ref')!=plan_ref or any(r.get(k)is not True for k in ('passed','execution_authorized','scientific_execution_authorized','source_accepted','input_accepted','runtime_accepted','engineering4_accepted')):raise ValueError('root source/input/toy/runtime scientific admission')
 for key in ('human_epoch_ref','input_qualification_ref','storage_inventory_ref','engineering_acceptance_ref'):
  if r.get(key)!=p[key]:raise ValueError('root exact '+key)
 if p['design_ref']['sha256']!=DESIGN_SHA:raise ValueError('unique Gh mathematics')
 c.read(p['design_ref']);accepted=c.read(p['root_design_acceptance_ref'])
 if accepted.get('mathematical_design_accepted')is not True or accepted['design_ref']['sha256']!=DESIGN_SHA:raise ValueError('root accepted mathematics')
 qualification=c.read(p['input_qualification_ref'])
 if qualification.get('schema')!='RootR11ExactSavedR10CommonQualificationV1' or qualification.get('passed')is not True or qualification.get('common_recomputations')!=0 or qualification.get('cases')!=p['cases']:raise ValueError('root exact original common reuse')
 if [v['original_index'] for v in p['cases']]!=list(c.MOTHERS):raise ValueError('full12 denominator/order')
 if sum(v['available_global_solves'] for v in p['cases'])>26:raise ValueError('global distinct solves26 ceiling')
 for case in p['cases']:
  if type(case['available_global_solves'])is not int or not 0<=case['available_global_solves']<=3:raise ValueError('frozen original available slots')
  c.check_identity(case['identity'])
  if case.get('uniform_request_bps')!=21600000 or not case.get('historical_R10_run_id','').startswith('r10_') or not case.get('historical_common_SHA') or set(case['historical_controls'])!={'Gr','S2','SH-C'}:raise ValueError('original full input/common/paid controls')
  for ref in [case['common_ref'],case['common_input_ref'],*case['historical_controls'].values()]:c.pin(ref)
 source_check(p);human_epoch(p);storage_check(p);return p

def fresh_route(p):
 r=c.read(p['actual_resource_route_ref']);age=time.time()-r['observed_unix_seconds']
 if not -5<=age<=120 or r.get('ssh_port')!=p['executor_port'] or r.get('group')!=GROUP or any(r.get(k)is not True for k in ('ready','root_accepted','own_task_absence_verified','old_owner_handoff_verified')):raise ValueError('fresh root owner/resource route')
 if r.get('boot_id')!=Path('/proc/sys/kernel/random/boot_id').read_text().strip() or r.get('python')!=DQN:raise ValueError('actual current boot/DQN')
 return r

def lifecycle(p):
 def load(name,ref):
  spec=importlib.util.spec_from_file_location(name,c.pin(ref));m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
 load('engineering_driver',p['lifecycle_dependencies']['engineering_driver.py'])
 return load('r11_accepted_lifecycle',p['lifecycle_dependencies']['engineering_persistent_launcher_once.py'])
