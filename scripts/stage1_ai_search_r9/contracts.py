"""Exact byte/source/input admissions for the new R9 M contract."""
import errno,gzip,hashlib,json,os,time
from pathlib import Path

GROUP='S1-AI17-R9'
DESIGN_SHA='b6d17258c034e0dce0e47bf7183ddc5cd7478c3c640b260dcc98d7733f658666'
AUTHORITY_SHA='75098fdbe450e37d6865955b98f09188f39d95788a843ca0ae01844fc1f1956b'
RULE_FLOOR_SHA='5970d003999771e4eff111514f4634722359c4a5942aad438e31f7d104a39628'
M_INDICES=[0,6,12,18,24,36,42,48,54,60,72,78]
CAPS=dict(RULE=12,D5=12,candidate=384,W=0,T=0,forward=0,updates=0)
OPS_FILES=('__init__.py','contracts.py','control.py','guard.py','train_input.py','child.py','worker.py','server.py','plan_builder.py')
def digest(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(1048576),b''):h.update(block)
 return h.hexdigest()
def pin(ref):
 p=Path(ref['path'])
 if not p.is_file() or p.is_symlink() or digest(p)!=ref['sha256']:raise ValueError('pinned bytes differ: '+str(p))
 return p
def read(ref):return json.loads(pin(ref).read_bytes())
def asset_read(ref):
 if ref.get('compression')!='gzip' or not 0<=ref.get('uncompressed_bytes',-1)<=268435456:raise ValueError('bounded original typed asset')
 with gzip.open(pin(ref),'rb') as f:raw=f.read(ref['uncompressed_bytes']+1)
 if len(raw)!=ref['uncompressed_bytes'] or hashlib.sha256(raw).hexdigest()!=ref['canonical_json_sha256']:raise ValueError('exact canonical asset bytes')
 return json.loads(raw)
def atomic(path,value):
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp')
 with tmp.open('w',encoding='utf8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 os.replace(tmp,p)
 if os.name!='nt':
  fd=os.open(str(p.parent),os.O_DIRECTORY)
  try:os.fsync(fd)
  finally:os.close(fd)
def immutable(path,value):
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
 # Reservation files must never overwrite an old intent, even across crashes.
 with p.open('x',encoding='utf8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 if os.name!='nt':
  fd=os.open(str(p.parent),os.O_DIRECTORY)
  try:os.fsync(fd)
  finally:os.close(fd)
 return dict(path=str(p),sha256=digest(p))
def identity(pid):
 try:
  raw=(Path('/proc')/str(pid)/'stat').read_text();fields=raw[raw.rfind(')')+2:].split()
  return dict(pid=int(pid),birth_ticks=int(fields[19]),state=fields[0],boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
 except OSError as e:
  if e.errno in (errno.ENOENT,errno.ESRCH):return None
  raise
def same_process(a,b):return bool(a and b and all(a[k]==b[k] for k in ('pid','birth_ticks','boot_id')))
def io_snapshot(pid=None):
 """Linux cumulative process IO includes imports; logical chars != disk IO."""
 try:
  values={k:int(v) for k,v in (line.split(':',1) for line in (Path('/proc')/str(pid or os.getpid())/'io').read_text().splitlines())}
  return dict(observed_unix_seconds=time.time(),rchar=values['rchar'],wchar=values['wchar'],syscr=values['syscr'],syscw=values['syscw'],read_bytes=values['read_bytes'],write_bytes=values['write_bytes'],scope='process lifetime through snapshot; snapshot/receipt IO later excluded',logical_chars_not_physical_or_network_bytes=True)
 except OSError as e:
  if e.errno in (errno.ENOENT,errno.ESRCH):return None
  raise
def origin_ref(p,runtime_ref):
 """Root-pinned precise alias map preserves immutable local-reference bytes."""
 matches=[x for x in p.get('document_aliases',[]) if x['runtime_ref']==runtime_ref]
 if len(matches)>1:raise ValueError('ambiguous document alias')
 if not matches:return runtime_ref
 item=matches[0]
 if item['origin_ref']['sha256']!=runtime_ref['sha256']:raise ValueError('alias must preserve exact original bytes')
 pin(runtime_ref)
 return item['origin_ref']
def source_check(p):
 refs=p['source_files'];repo=Path(__file__).resolve().parents[2]
 for name in OPS_FILES:
  key='scripts/stage1_ai_search_r9/'+name
  if key not in refs or Path(refs[key]['path']).resolve()!=repo/key:raise ValueError('complete actual R9 operations inventory')
 core=repo/'src/satellite_rl/coverage/ai_search_r9'
 actual={str(q.relative_to(repo)).replace('\\','/'):q for q in core.glob('*.py')}
 if not actual:raise ValueError('missing R9 core source')
 for key,path in actual.items():
  if key not in refs or Path(refs[key]['path']).resolve()!=path.resolve():raise ValueError('complete actual R9 core inventory')
 for ref in refs.values():pin(ref)
 if any(k.startswith('src/satellite_rl/coverage/ai_search_r9/') and k not in actual for k in refs):raise ValueError('invented core member')
 sm=read(p['original_source_manifest_ref']);source=Path(p['original_source_manifest_ref']['path']).parent
 for name,sha in sm['files'].items():
  if digest(source/name)!=sha:raise ValueError('original frozen source changed: '+name)
def imported_check(p):
 import sys
 for key,ref in p['source_files'].items():
  if key.startswith('src/satellite_rl/coverage/ai_search_r9/'):
   name='satellite_rl.coverage.ai_search_r9'+('' if key.endswith('/__init__.py') else '.'+Path(key).stem)
   module=sys.modules.get(name)
   if module and Path(module.__file__).resolve()!=pin(ref).resolve():raise ValueError('actual imported R9 source differs')
 source=Path(p['original_source_manifest_ref']['path']).parent.resolve();sm=read(p['original_source_manifest_ref'])
 for name,module in list(sys.modules.items()):
  if not name.startswith('satellite_rl') or name.startswith('satellite_rl.coverage.ai_search_r9'):continue
  path=getattr(module,'__file__',None)
  if not path:continue
  real=Path(path).resolve()
  if not real.is_relative_to(source):raise ValueError('actual original import outside frozen source: '+name)
  rel=real.relative_to(source).as_posix()
  if rel not in sm['files'] or digest(real)!=sm['files'][rel]:raise ValueError('actual original import not frozen: '+name)
def human_epoch(p):
 authority=read(p['human_resume_authority_ref']);epoch=read(p['human_epoch_ref'])
 if p['human_resume_authority_ref']['sha256']!=AUTHORITY_SHA or authority.get('human_authorization')is not True or authority.get('root_accepted')is not True or authority.get('user_message_verbatim')!='继续':raise ValueError('original explicit post0700 human continue')
 if epoch.get('schema')!='R9NewScientificHumanEpochV1' or epoch.get('root_accepted')is not True or epoch.get('group')!=GROUP or epoch.get('stage')!='M' or epoch.get('new_deadline_unix_seconds')is not None:raise ValueError('new R9 scientific epoch; never reuse R8 remaining59 contract')
 if epoch.get('human_resume_authority_ref')!=p['human_resume_authority_ref'] or any(epoch.get(k)!=p[k] for k in ('run_id','request_id','archive_root')):raise ValueError('exact new epoch/task binding')
 if not epoch.get('execution_epoch_id') or not epoch.get('watched_pause_paths') or not isinstance(epoch.get('historical_hold_exemptions'),list):raise ValueError('explicit monitored pause inventory')
 return epoch
def storage_check(p):
 s=read(p['storage_inventory_ref']);root=Path(p['archive_root']).resolve()
 if s.get('schema')!='R9OwnedStorageInventoryV1' or s.get('root_accepted')is not True or s.get('archive_root')!=p['archive_root'] or s.get('new_logical_bytes_max')!=536870912 or s.get('temporary_bytes_max')!=268435456 or s.get('free_reserve_bytes')!=8589934592:raise ValueError('R9 exact storage caps')
 if not root.is_relative_to(Path(s['owned_growth_root']).resolve()) or '/round9/' not in root.as_posix()+'/':raise ValueError('own R9 archive tree')
 return s
def validate(plan_ref,review_ref):
 p=read(plan_ref);r=read(review_ref)
 if p.get('schema')!='R9MechanismMExactPlanV1' or p.get('group')!=GROUP or p.get('stage')!='M' or p.get('executor_port') not in (20473,20474):raise ValueError('sole R9 M and73/74')
 if not p.get('run_id','').startswith('r9_M_') or '/round9/' not in p['archive_root'] or p.get('caps')!=CAPS or p.get('case_indices')!=M_INDICES:raise ValueError('exact new M task/12indices/caps')
 if r.get('schema')!='R9RootMExecutionAdmissionV1' or r.get('plan_ref')!=plan_ref or any(r.get(k)is not True for k in ('passed','execution_authorized','scientific_execution_authorized','source_accepted','input_accepted','runtime_accepted')):raise ValueError('independent root source/input/runtime/scientific M admission')
 for key in ('human_epoch_ref','input_qualification_ref','storage_inventory_ref'):
  if r.get(key)!=p[key]:raise ValueError('review exact '+key)
 if p['design_ref']['sha256']!=DESIGN_SHA:raise ValueError('R9v3 design bytes')
 read(p['design_ref']);accepted=read(p['root_design_acceptance_ref'])
 if accepted.get('schema')!='R9RootFiniteDesignAcceptanceV1' or accepted.get('accepted')is not True or accepted.get('design_ref')!=origin_ref(p,p['design_ref']) or accepted.get('authority_ref')!=origin_ref(p,p['human_resume_authority_ref']):raise ValueError('root exact existing design admission with explicit runtime aliases')
 if p['RULE_floor_clarification_ref']['sha256']!=RULE_FLOOR_SHA:raise ValueError('result-before-only-success RULE floor clarification')
 read(p['RULE_floor_clarification_ref'])
 q=read(p['input_qualification_ref'])
 if q.get('passed')is not True or q.get('heldout_reads')is not False or q.get('cases')!=p['cases']:raise ValueError('root original train input qualification')
 if [c['original_index'] for c in p['cases']]!=M_INDICES:raise ValueError('all12 in original order')
 for c in p['cases']:
  if c['identity'].get('role')!='train' or c.get('uniform_request_bps')!=21600000 or not c.get('mechanism12'):raise ValueError('original train role/uniform request')
  for key in ('scene_ref','prepared_ref','shared_case_ref'):pin(c[key])
  if c.get('original_T_terminal_ref'):pin(c['original_T_terminal_ref'])
  if c.get('reused_RULE_ref'):pin(c['reused_RULE_ref'])
 source_check(p);human_epoch(p);storage_check(p)
 return p
def fresh_route(p):
 r=read(p['actual_resource_route_ref']);age=time.time()-r['observed_unix_seconds']
 if not -5<=age<=120 or any(r.get(k)is not True for k in ('ready','root_accepted','own_task_absence_verified','old_owner_handoff_verified')) or r.get('ssh_port')!=p['executor_port'] or r.get('group')!=GROUP:raise ValueError('fresh exact R9 owner/resource route')
 if r.get('boot_id')!=Path('/proc/sys/kernel/random/boot_id').read_text().strip():raise ValueError('route current boot differs')
 return r
