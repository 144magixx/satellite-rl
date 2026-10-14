"""R11 exact Gh-only plan; original R10 common remains a historical wrapper."""
import importlib.util,os,sys,time
from pathlib import Path
from scripts.stage1_ai_search_r10 import contracts as c,control

GROUP='S1-AI17-R11'
DESIGN_SHA='6a4fd42f6510c86ef9bd5db881374a8a426e5493078ee8d7d159a0f73e67a2f1'
FILES=('__init__.py','contracts.py','storage.py','child.py','worker.py','server.py','controller.py','plan_builder.py','reaper.py','outer.py','noop.py','noop_payload.py')
PARTITION={20473:(0,12,24,42,54,72),20474:(6,18,36,48,60,78)}
DQN='/home/u2024110224/first_stage_satellite_rl/envs/DQN/bin/python'
CAPS=dict(method=12,global_matching_slots=36,global_solves=26,common=0,local=0,W=0,T=0,RULE=0,D5=0,model_initialization=0,forward=0,updates=0)
fref=lambda p:dict(path=str(p),sha256=c.digest(p))

def source_check(p):
 repo=Path(p['source_repo']).resolve()
 if Path(__file__).resolve().parents[2]!=repo or Path(c.__file__).resolve().parents[2]!=repo:raise ValueError('actual ops source root')
 required={'scripts/stage1_ai_search_r11_prospective/'+n for n in FILES}
 required.update('scripts/stage1_ai_search_r11/'+n for n in ('__init__.py','contracts.py','storage.py','child.py','worker.py','server.py','controller.py','plan_builder.py'))
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
 if p['human_resume_authority_ref']['sha256']!=c.AUTHORITY_SHA or a.get('user_message_verbatim')!='\u7ee7\u7eed' or a.get('human_authorization')is not True or a.get('root_accepted')is not True:raise ValueError('explicit human continue')
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
 for path in (p['archive_root'],p['source_repo'],p['outer_control_root'],p['worker_control_root'],p['outer_observer_root']):
  if '/round11/' not in Path(path).as_posix()+'/' or not any(Path(path).resolve().is_relative_to(g) for g in growths):raise ValueError('owned R11 storage coverage')
 return s

def validate(plan_ref,review_ref):
 p=c.read(plan_ref);r=c.read(review_ref)
 if p.get('schema')!='R11RemainingFiveOnlyGhPlanV1' or p.get('group')!=GROUP or p.get('source_input_budget_frozen')is not True:raise ValueError('R11 candidate not admitted')
 if p.get('executor_port')!=20474 or p.get('workbench_port')!={20473:9050,20474:9051}[p['executor_port']] or not p.get('run_id','').startswith('r11_'):raise ValueError('new exact R11 task and candidate port')
 for key in ('archive_root','source_repo','outer_control_root','worker_control_root','outer_observer_root'):
  path=Path(p[key])
  if not path.is_absolute() or '..' in path.parts or path.resolve()!=path.absolute() or '/round11/' not in path.as_posix()+'/':raise ValueError('no old tree/linked target')
 if p['arms']!=['Gh'] or p['caps']!=CAPS or p['method_hard_seconds']!=300 or p['method_rss_bytes']!=2147483648 or p['outer_hard_seconds']!=RECOVERY_REMAINING[p['executor_port']]:raise ValueError('fixed finite Gh budget')
 if r.get('schema')!='R11RootRemainingFiveOnlyAdmissionV1' or r.get('exact_plan_ref')!=plan_ref or any(r.get(k)is not True for k in ('passed','execution_authorized','scientific_execution_authorized','source_accepted','input_accepted','runtime_accepted','four_toy_mathematics_accepted','prospective_lifecycle_accepted','noop_accepted','reader_preparation_accepted')):raise ValueError('root source/input/toy/runtime scientific admission')
 for key in ('human_epoch_ref','input_qualification_ref','storage_inventory_ref','mathematical_four_toy_acceptance_ref','prospective_lifecycle_acceptance_ref','noop_acceptance_ref','reader_preparation_ref'):
  if r.get(key)!=p[key]:raise ValueError('root exact '+key)
 math4=c.read(p['mathematical_four_toy_acceptance_ref']);prospective=c.read(p['prospective_lifecycle_acceptance_ref']);c.read(p['noop_acceptance_ref']);c.read(p['reader_preparation_ref'])
 if math4.get('mathematical_four_toy_accepted')is not True or math4.get('historical_complete_30s_gate_closed')is not False or p['prospective_lifecycle_acceptance_ref']['sha256']!='a26e12fcc51c5af9b042548187fe1947a74b75e03acd77e8fda2606651089d57' or prospective.get('accepted')is not True:raise ValueError('explicit mathematical4/prospective lifecycle conjunction; old30 unknown')
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
 validate_recovery_binding(p,r);source_check(p);human_epoch(p);storage_check(p);return p

def fresh_route(p):
 r=c.read(p['actual_resource_route_ref']);age=time.time()-r['observed_unix_seconds']
 if not -5<=age<=120 or r.get('ssh_port')!=p['executor_port'] or r.get('group')!=GROUP or any(r.get(k)is not True for k in ('ready','root_accepted','own_task_absence_verified','old_owner_handoff_verified')):raise ValueError('fresh root owner/resource route')
 if r.get('boot_id')!=Path('/proc/sys/kernel/random/boot_id').read_text().strip() or r.get('python')!=DQN:raise ValueError('actual current boot/DQN')
 return r

def lifecycle(p):
 def load(name,ref):
  spec=importlib.util.spec_from_file_location(name,c.pin(ref));m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
 load('engineering_driver',p['lifecycle_dependencies']['engineering_driver.py'])
 m=load('r11_accepted_lifecycle',p['lifecycle_dependencies']['engineering_persistent_launcher_once.py']);m.observe_descendants=observe_descendants;return m

# Fixed continuation after entered case6; never a generic recovery endpoint.
RECOVERY_REMAINING={20474:2117.8716288655996}
REMAINING_INDICES=(18,36,48,60,78)
def dispatch_cases(p):
 if p.get('executor_port')!=20474 or p.get('dispatch_indices')!=list(REMAINING_INDICES) or p.get('remaining_work_caps')!=dict(mothers=5,structural_slots=15,flows=9):raise ValueError('only proven-unentered74 five are dispatchable')
 cases=[v for v in p['cases'] if v['original_index'] in REMAINING_INDICES]
 if [v['original_index'] for v in cases]!=list(REMAINING_INDICES) or [v['available_global_solves'] for v in cases]!=[3,2,1,1,2]:raise ValueError('exact remaining nine flow opportunities')
 return cases

def validate_recovery_binding(p,r):
 b=p['recovery_binding'];dispatch_cases(p)
 pins={'design_ref':'11e09d433c63e9359e27a25eff3ccb1f42ee46b2714ee6a50c2d23e2d68bb138','root_design_acceptance_ref':'41f105299e60ca892d29b902d71e15dfd2e52a5ee7b920f15fa7b8a86427a248','failure_review_ref':'947fb73a156d7b105eac55d20501797b585800484e5af8f81d01faffb29190c8','known_generation_exit_ref':'f94056c1515c7ca8975f50efc5defb21596e019ac9b9a2ace1665921585fe259'}
 if b.get('schema')!='R11RemainingFiveFixedRecoveryBindingV1' or r.get('remaining_five_only_accepted')is not True or r.get('recovery_binding')!=b:raise ValueError('root exact remaining-five admission')
 docs={}
 for key,sha in pins.items():
  if b.get(key,{}).get('sha256')!=sha:raise ValueError('wrong original recovery proof '+key)
  docs[key]=c.read(b[key])
 design=docs['design_ref'];root=docs['root_design_acceptance_ref'];failure=docs['failure_review_ref'];exitproof=docs['known_generation_exit_ref'];host=failure['hosts']['20474'];details=host['details']
 if root.get('design_accepted')is not True or root.get('only_dispatch_port')!=20474 or root.get('only_remaining_indices')!=list(REMAINING_INDICES) or root.get('old_case6_and_73_six_replay_forbidden')is not True or root.get('original_complete_AND_3_plus_3_gate_not_waived')is not True or details.get('unentered_zero_entry_proven')is not True or details.get('unentered_indices')!=list(REMAINING_INDICES) or host.get('observed_algorithm_entry_indices')!=[6]:raise ValueError('only five certified zero-entry; case6 already entered')
 if exitproof.get('all_prior_generations_ended')is not True or not exitproof.get('observations') or not all(v.get('old_generation_ended')is True for v in exitproof['observations']):raise ValueError('all old74 known generations ended')
 if b['prior_dispatch_graph_ref']['sha256']!=failure['graph_ref']['sha256']:raise ValueError('actual prior graph')
 oldhost=c.read(b['prior_dispatch_graph_ref'])['hosts']['20474']
 if b['prior_plan_ref']!=oldhost['plan_ref'] or b['prior_root_review_ref']!=oldhost['review_ref']:raise ValueError('exact prior plan pair')
 old=c.read(b['prior_plan_ref']);c.read(b['prior_root_review_ref'])
 for key in ('run_id','request_id','executor_port','workbench_port','cases','arms','caps','method_hard_seconds','method_rss_bytes','host_compute_lock','lifecycle_dependencies','original_source_manifest_ref','human_resume_authority_ref'):
  if p[key]!=old[key]:raise ValueError('unchanged original scientific identity '+key)
 for key,ref in old['source_files'].items():
  if key not in ('scripts/stage1_ai_search_r11_prospective/'+n for n in ('contracts.py','reaper.py','worker.py')) and p['source_files'][key]['sha256']!=ref['sha256']:raise ValueError('unchanged scientific/other operations bytes '+key)
 base='/home/u2024110224/first_stage_satellite_rl/s1_ai17/20474/round11/scientific_recovery_02'
 if p['archive_root']!=base+'/archive' or p['outer_control_root']!=base+'/control/outer' or p['worker_control_root']!=base+'/control/worker' or p['outer_observer_root']!=base+'/control/observer' or p['execution_epoch_id']!=old['execution_epoch_id']+':remaining_five_proc_race_02' or p['source_repo']==old['source_repo']:raise ValueError('new exact single74 epoch/archive/source')
 if type(p.get('outer_hard_seconds'))not in(int,float) or p['outer_hard_seconds']!=2117.8716288655996 or root.get('remaining_lane_seconds')!=p['outer_hard_seconds'] or design['budget']['20474']['remaining_lane_seconds']!=p['outer_hard_seconds'] or b.get('prior_lane_debits_seconds')!=[5.438956959173083,276.6894141752273] or b.get('automatic_retry')is not False:raise ValueError('cumulative conservative debit; no reset or nested double count')
 for key in ('original_map_ref','final_inventory_ref'):
  if b[key]['sha256']!=host[key]['sha256']:raise ValueError('exact stopped original inventory')
  c.read(b[key])
 if b['original_map_ref']['sha256']!=exitproof['original_map_ref']['sha256']:raise ValueError('same stopped scope')
 mapping=c.read(b['original_map_ref']);paths=[row['remote_path'] for row in mapping['files']]
 for index in REMAINING_INDICES:
  if any('/mothers/'+str(index)+'/' in path or '/budget_reservations/'+str(index)+'/' in path for path in paths):raise ValueError('contradictory previous entry/reservation')
 if b.get('preserved_case6_control')!=host['method_controls']['6'] or b.get('preserved_73_original_map_ref',{}).get('sha256')!=failure['hosts']['20473']['original_map_ref']['sha256'] or b.get('preserved_case6_engineering_complete')is not None or p.get('original_engineering_complete')is not False or p.get('original_gate_status')!='incomplete_unknown_not_pass_or_valid_negative':raise ValueError('preserve six73 and unknowncase6 without forged qualification')
 for key in ('entry_ref','WAL_ref','typed_asset_ref'):c.pin(b['preserved_case6_control'][key])
 for absent in host['details']['missing_exit_qualification_refs']:
  if absent in paths:raise ValueError('do not forge historical completion')
 return old

def observe_descendants(c,owner,registry):
 pending=[owner];seen=set()
 while pending:
  parent=pending.pop();pk=(parent['pid'],parent['birth_ticks'],parent['boot_id'])
  if pk in seen:continue
  seen.add(pk);before=c.identity(parent['pid'])
  if not c.same_process(parent,before) or before['state']=='Z':continue
  proc=Path('/proc')/str(parent['pid']);children=set()
  try:
   for task in (proc/'task').iterdir():children.update(map(int,(task/'children').read_text().split()))
  except (FileNotFoundError,ProcessLookupError):continue
  middle=c.identity(parent['pid']);candidates=[]
  for pid in children:
   try:
    current=c.identity(pid)
    if current is None:continue
    known=c.same_process(registry.get((current['pid'],current['birth_ticks'],current['boot_id'])),current)
    if current['state']=='Z':
     if known:continue
     raise RuntimeError('unregistered ended descendant; unknown')
    raw=(Path('/proc')/str(pid)/'stat').read_text();fields=raw[raw.rfind(')')+2:].split();after=c.identity(pid)
   except (FileNotFoundError,ProcessLookupError):continue
   # An observed old generation that has exited is never a new live kill target.
   if after is None:continue
   if c.same_process(current,after) and after['state']=='Z':
    if known:continue
    raise RuntimeError('unregistered ended descendant; unknown')
   if not c.same_process(current,after) or int(fields[1])!=parent['pid'] or int(fields[19])!=current['birth_ticks'] or current['uid']!=owner['uid'] or current['boot_id']!=owner['boot_id']:raise RuntimeError('unproven live descendant generation')
   candidates.append(after)
  last=c.identity(parent['pid']);closed_parent=c.same_process(parent,middle) and middle['state']!='Z' and c.same_process(parent,last) and last['state']!='Z'
  if not closed_parent:
   # No candidates acquire ownership from an unclosed parent bracket. Existing
   # registrations retain their independent prior proof; ended candidates disappear.
   for current in candidates:
    now=c.identity(current['pid']);key=(current['pid'],current['birth_ticks'],current['boot_id'])
    if now is None:continue
    if c.same_process(current,now) and now['state']=='Z':
     if c.same_process(registry.get(key),now):continue
     raise RuntimeError('unregistered ended candidate across ended parent; unknown')
    if not c.same_process(current,now) or not c.same_process(registry.get(key),now):raise RuntimeError('unregistered live candidate across ended parent; unknown')
    pending.append(now)
   continue
  for current in candidates:
   registry[(current['pid'],current['birth_ticks'],current['boot_id'])]=current;pending.append(current)
