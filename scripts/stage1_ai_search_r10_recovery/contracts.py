"""Exact root recovery graph, original-byte reuse and conservative budget debits."""
import importlib.util,math,os,sys,time
from pathlib import Path
from scripts.stage1_ai_search_r10 import contracts as c
from scripts.stage1_ai_search_r10 import control
from scripts.stage1_ai_search_r10.worker import check_output
DESIGN_SHA='beadefb53834dbb0447df69fcafe916ea3bfb38373f814f1eb280ccd4138cd92'
ROOT_DESIGN_SHA='ac1bed4ab3693e9c6ce8bf6e75d0b0185acc4cc6ba5938492fb1981b95eeccd8'
REUSE_SHA='543a4adf5fe73d93b6c3ee8bc9a5124443686af44ef79af7b5abbef81d046706'
REVIEW_SHA='0741ef750edb12d923b7abaf58f3892efd77056d762a908805fb71ff1d0f5911'
FILES=('__init__.py','contracts.py','storage.py','worker.py','server.py','controller.py','plan_builder.py')
def fref(path):return dict(path=str(path),sha256=c.digest(path))
def same_ref(a,b):
 if a['path']!=b['path'] or a['sha256']!=b['sha256']:return False
 for ref in (a,b):
  if 'bytes' in ref and c.pin(ref).stat().st_size!=ref['bytes']:raise ValueError('declared original byte size differs')
 return True
def stage_key(row):return (row['original_index'],row['stage'])
def finite(value):
 if type(value) not in (int,float) or not math.isfinite(value) or value<0:raise ValueError('finite nonnegative numeric budget, not bool')
 return value
def original_matches(p,r,expected):
 original=c.origin_ref(p,r)
 if not (original['path']==expected['path'] and original['sha256']==expected['sha256']):return False
 for ref in (r,original,expected):
  if 'bytes' in ref and c.pin(r).stat().st_size!=ref['bytes']:raise ValueError('original transport byte size')
 return True
def source_import_check(p):
 repo=Path(p['source_repo']).resolve()
 if Path(c.__file__).resolve().parents[2]!=repo or Path(__file__).resolve().parents[2]!=repo:raise ValueError('actual original/recovery ops imported from different source root')
 declared={**p['source_files'],**p['recovery_source_files']}
 for name,module in list(sys.modules.items()):
  if not (name=='scripts.stage1_ai_search_r10' or name.startswith('scripts.stage1_ai_search_r10.') or name=='scripts.stage1_ai_search_r10_recovery' or name.startswith('scripts.stage1_ai_search_r10_recovery.')):continue
  path=getattr(module,'__file__',None)
  if path is None:raise ValueError('missing actual module source '+name)
  real=Path(path).resolve()
  if not real.is_relative_to(repo):raise ValueError('cached ops module outside new runtime '+name)
  key=real.relative_to(repo).as_posix()
  if key not in declared or c.pin(declared[key]).resolve()!=real:raise ValueError('actual ops module unpinned '+name)
def select_rows(reuse,port):
 rows=[r for r in reuse['rows'] if r['executor_port']==port]
 expected=[(i,s) for i in ([0,12,24,42,54,72] if port==20473 else [6,18,36,48,60,78]) for s in ('common',*c.ARMS)]
 if [stage_key(r) for r in rows]!=expected:raise ValueError('exact six-mother ordered36rows')
 return rows
def limits_for(row):
 if row['stage']=='common':return 30.0
 if row['executor_port']==20473 and stage_key(row)==(0,'S2'):
  if row['old_algorithm_spend_exact']is not None or row['algorithm_entry_exists'] or row['actual_stage_wall_seconds']!=1.6233237385749817:raise ValueError('failed S2 proof/debit differs')
  return 300.0-row['actual_stage_wall_seconds']
 return 300.0
def check_pause(p,point):
 """Read the same exact generations; outer never publishes into worker archive."""
 epoch=c.human_epoch(p);own={str(Path(p['archive_root'])/'stop.request'),str(Path(p['archive_root'])/'pause_state.json')};exempt={};held=[]
 for old in epoch['historical_hold_exemptions']:
  if old['path'] in own or old['reason']!='historical0700_pause' or old['archived_evidence_ref']['sha256']!=old['sha256']:raise ValueError('new control never exempt')
  c.pin(old['archived_evidence_ref']);exempt[old['path']]=old
 for path in sorted(set(epoch['watched_pause_paths'])|own|set(exempt)):
  current=control.marker_snapshot(path);old=exempt.get(path)
  if old:
   if current is None or any(current[k]!=old[k] for k in control.FINGERPRINT):held.append(path)
  elif current is not None:
   value=current['value']
   if Path(path).name=='stop.request' or any(value.get(k) for k in control.FLAGS) or str(value.get('state','')).startswith('paused'):held.append(path)
 if held:raise control.Paused('new/changed pause at '+point+': '+repr(held))
 return dict(passed=True,point=point)
def validate(plan_ref,review_ref):
 p=c.validate(plan_ref,review_ref);r=c.read(review_ref);x=c.read(p['recovery_contract_ref'])
 if x.get('schema')!='R10StorageRaceSameIDRecoveryContractV1' or any(r.get(k)is not True for k in ('recovery_execution_authorized','old_reuse_accepted','cumulative_budget_accepted')):raise ValueError('root exact recovery/reuse/budget admission')
 if r.get('recovery_contract_ref')!=p['recovery_contract_ref']:raise ValueError('root exact recovery contract')
 if x['design_ref']['sha256']!=DESIGN_SHA or x['root_design_acceptance_ref']['sha256']!=ROOT_DESIGN_SHA or x['independent_original_reuse_review_ref']['sha256']!=REVIEW_SHA:raise ValueError('accepted recovery design and original proof')
 if x['clarification_ref']['sha256']!='74eecbddd6e191fb7bde5b47a2f4d250e10d3ecb10bade9f2f42b7ed5d5eccc9':raise ValueError('quota/private-temp clarification')
 for name in ('design_ref','root_design_acceptance_ref','clarification_ref','independent_original_reuse_review_ref','zero_entry_source_fence_ref'):c.read(x[name])
 old=c.read(x['original_plan_ref']);c.read(x['original_root_review_ref'])
 if any(p[k]!=old[k] for k in ('run_id','request_id','executor_port','cases','mother_host_assignment','arms','caps','common_hard_seconds','method_hard_seconds','method_rss_bytes','original_source_manifest_ref','design_ref','numeric_origin_protocol_ref')):raise ValueError('original scientific run/input/identity/algorithm/budget differs')
 if any(x[k]!=p[k] for k in ('run_id','executor_port')) or x['recovery_archive_root']!=p['archive_root'] or x['execution_epoch_id']!=p['execution_epoch_id'] or x['original_archive_root']!=old['archive_root'] or p['original_plan_ref']!=x['original_plan_ref']:raise ValueError('exact same-ID new operations epoch')
 epoch_root=Path(old['archive_root']).absolute().parent/'operations_recovery/storage_race_01'
 expected_paths={p['archive_root']:epoch_root/'archive',x['outer_control_root']:epoch_root/'control/outer',x['worker_control_root']:epoch_root/'control/worker'}
 for path,expected in expected_paths.items():
  candidate=Path(path)
  if '..' in candidate.parts or candidate.absolute()!=expected or candidate.resolve()!=expected.absolute() or candidate.resolve().is_relative_to(Path(old['archive_root']).resolve()):raise ValueError('exact disjoint new epoch, no old-tree/symlink/.. alias')
 storage=c.storage_check(p);growths=[Path(v).resolve() for v in storage.get('owned_growth_roots',[storage['owned_growth_root']])]
 for path in [p['archive_root'],old['archive_root'],x['outer_control_root'],x['worker_control_root'],p['source_repo'],str(Path(plan_ref['path']).parent),str(Path(review_ref['path']).parent)]:
  if not any(Path(path).resolve().is_relative_to(g) for g in growths):raise ValueError('old/new/source/metadata/control not covered by same storage cap')
 if set(p['source_files'])!=set(old['source_files']):raise ValueError('same24 science/runtime members')
 for name,ref in old['source_files'].items():
  if name!='scripts/stage1_ai_search_r10/contracts.py' and p['source_files'][name]['sha256']!=ref['sha256']:raise ValueError('unapproved old source change '+name)
 repo=Path(__file__).resolve().parents[2]
 required={'scripts/stage1_ai_search_r10_recovery/'+name for name in FILES}
 if set(p['recovery_source_files'])!=required:raise ValueError('complete six recovery runtime leaves')
 for name,ref in p['recovery_source_files'].items():
  if c.pin(ref).resolve()!=(repo/name).resolve():raise ValueError('actual recovery helper import/source')
 source_import_check(p)
 for ref in x['lifecycle_dependencies'].values():c.pin(ref)
 inventory=c.read(x['original_canonical_inventory_ref']);c.read(x['original_collection_ref']);c.pin(x['original_status_snapshot_ref'])
 expected={(v['path'],v['sha256'],v['bytes']) for v in inventory};declared={(v['path'],v['sha256'],v['bytes']) for v in x['protected_original_refs']}
 if expected!=declared:raise ValueError('complete old raw set must be protected; no dropped file')
 for ref in x['protected_original_refs']:
  if c.pin(ref).stat().st_size!=ref['bytes']:raise ValueError('old raw byte size changed')
 roots={}
 for item in inventory:
  label,tail=item['relative'].split('/',1);base=item['path'][:-len(tail)].rstrip('/')
  if label in roots and roots[label]!=base:raise ValueError('old inventory scope ambiguous')
  roots[label]=base
 actual=set();scanbegan=time.monotonic()
 for base in roots.values():
  for path in Path(base).rglob('*'):
   if time.monotonic()-scanbegan>10:raise ValueError('bounded10s original absence inventory')
   if path.is_symlink():raise ValueError('linked old raw evidence')
   if path.is_file():actual.add(str(path))
 if actual!={v['path'] for v in inventory}:raise ValueError('old raw full member set changed, including unstarted paths')
 reuse=c.read(x['reuse_map_ref'])
 if x['reuse_map_ref']['sha256']!=REUSE_SHA or reuse['completed_common']!=5 or reuse['completed_methods']!=23 or reuse['remaining_common']!=7 or reuse['remaining_methods']!=37:raise ValueError('same72logicalstage map')
 rows=select_rows(reuse,p['executor_port']);host=next(h for h in reuse['hosts'] if h['executor_port']==p['executor_port']);ext=c.read(x['original_external_sameboot_ref'])
 if not same_ref(x['original_plan_ref'],host['old_plan_ref']) or not original_matches(p,x['original_external_sameboot_ref'],host['old_external_sameboot_ref']):raise ValueError('exact same-host original plan and external debit reference')
 if ext.get('execution_ended')is not True or ext.get('all_registered_descendants_execution_ended')is not True or ext.get('same_boot_verified')is not True:raise ValueError('original complete process exit evidence')
 if type(ext['ticks_per_second'])is not int or ext['ticks_per_second']<=0 or type(ext['actual_controller']['birth_ticks'])is not int or ext['actual_controller']['birth_ticks']<=0:raise ValueError('exact positive kernel ticks/HZ')
 for value in (ext['clock_boottime_after_gone_seconds'],ext['conservative_upper_seconds'],x['old_outer_debit_seconds'],x['outer_hard_seconds'],x['native_ceiling_seconds']):finite(value)
 debit=finite(ext['clock_boottime_after_gone_seconds']-(ext['actual_controller']['birth_ticks']-1)/ext['ticks_per_second'])
 if abs(debit-ext['conservative_upper_seconds'])>1e-9 or abs(x['old_outer_debit_seconds']-debit)>1e-9 or abs(x['outer_hard_seconds']-(10800-debit))>1e-9:raise ValueError('inclusive old outer debit not reset')
 newrows=[v for v in rows if not v['typed_output_exists']];native=sum(limits_for(v) for v in newrows)
 if abs(x['native_ceiling_seconds']-min(native,9180-debit))>1e-9 or x['new_native_max']!=len(newrows) or x['new_distinct_reservations']!=len(newrows)-(1 if p['executor_port']==20473 else 0):raise ValueError('native/logical remaining budgets')
 if p['executor_port']==20473 and r.get('zero_entry_reentry_accepted')is not True:raise ValueError('root explicit S2 zero-core-entry bridge admission')
 epoch=c.human_epoch(p);watched=set(epoch['watched_pause_paths']);required_watches={old['archive_root']+'/stop.request',old['archive_root']+'/pause_state.json',p['archive_root']+'/stop.request',p['archive_root']+'/pause_state.json'}
 for key in ('outer_control_root','worker_control_root'):required_watches.update({x[key]+'/stop.request',x[key]+'/storage_pause.json',x[key]+'/pause_state.json'})
 if not required_watches<=watched or epoch['execution_epoch_id']!=p['execution_epoch_id']:raise ValueError('old/new/control pause inventory incomplete')
 common={}
 for row in rows:
  if not row['typed_output_exists']:continue
  case=next(v for v in p['cases'] if v['original_index']==row['original_index']);guard=c.read(row['guard_ref']);request=c.read(row['request_ref']);entry=c.read(row['algorithm_entry_ref']);c.read(row['native_WAL_ref']);c.read(row['budget_reservation_ref']);terminal_ref=fref(Path(row['budget_reservation_ref']['path']).with_name('terminal.json'));terminal=c.read(terminal_ref)
  if guard['status']!='completed' or guard['exit_code']!=0 or not guard['subtree_exit_verified'] or not same_ref(request['plan_ref'],x['original_plan_ref']) or request['original_index']!=row['original_index'] or request['stage']!=row['stage'] or entry['identity']!=row['identity'] or terminal['result_ref']!=row['output_ref'] or not same_ref(terminal['guard_ref'],row['guard_ref']):raise ValueError('original completed native/request/budget chain')
  value=c.typed_asset(row['output_ref']);check_output(case,row['stage'],value,common.get(row['original_index']))
  if row['stage']=='common':common[row['original_index']]=row['output_ref']
 check_pause(p,'recovery_admission');return p,x,rows
def lifecycle(x):
 def load(name,ref):
  path=c.pin(ref);spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);return module
 load('engineering_driver',x['lifecycle_dependencies']['engineering_driver.py']);return load('r10_recovery_accepted_lifecycle',x['lifecycle_dependencies']['engineering_persistent_launcher_once.py'])
