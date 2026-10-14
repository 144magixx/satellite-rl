"""Byte and identity primitives; execution intentionally gated pending frozen design v2."""
import errno,gzip,hashlib,json,os,time,sys
from pathlib import Path

GROUP='S1-AI17-R10'
ARMS=('Gd','Gr','Gw','S2','SH-C')
MOTHERS=(0,6,12,18,24,36,42,48,54,60,72,78)
IDENTITY_FIELDS=('case_id','mother_id','profile','role','input_sha256','identity_sha256','shared_case_content_sha256','source_asset_SHA')
AUTHORITY_SHA='75098fdbe450e37d6865955b98f09188f39d95788a843ca0ae01844fc1f1956b'
METHOD_HARD_SECONDS=300
METHOD_RSS_BYTES=2147483648
DESIGN_SHA='47cb6befc27519e18c8da8cf07513299d84c04b109db9aae6982d273febbb498'
NUMERIC_PROTOCOL_SHA='3088e9bb5a85032ec8b69b54eaffaf9714d51984dd1dc91f2cd61b276bb5521f'
OPS_FILES=('__init__.py','contracts.py','schedule.py','lifecycle.py','control.py','costs.py','guard.py','child.py','worker.py','server.py','plan_builder.py')
CAPS=dict(common=12,method=60,global_matching=108,sequential_kernel=768,W=0,T=0,RULE=0,D5=0,model_initialization=0,forward=0,updates=0)

def digest(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(1048576),b''):h.update(block)
 return h.hexdigest()

def pin(ref):
 p=Path(ref['path'])
 if p.is_symlink() or not p.is_file() or digest(p)!=ref['sha256']:raise ValueError('pinned original bytes differ')
 return p

def read(ref):return json.loads(pin(ref).read_bytes())
atomic=lambda path,value:publish(path,value)
immutable=lambda path,value:publish(path,value,exclusive=True)

def publish(path,value,exclusive=False):
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);raw=(json.dumps(value,ensure_ascii=False,allow_nan=False,indent=2)+'\n').encode()
 target=p if exclusive else p.with_name(p.name+'.tmp')
 with target.open('xb' if exclusive else 'wb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
 if not exclusive:os.replace(target,p)
 if os.name!='nt':
  fd=os.open(p.parent,os.O_DIRECTORY)
  try:os.fsync(fd)
  finally:os.close(fd)
 return dict(path=str(p),sha256=digest(p))

def check_identity(value,expected=None):
 if set(value)!=set(IDENTITY_FIELDS) or any(not isinstance(value[k],str) or not value[k] for k in IDENTITY_FIELDS):raise ValueError('full original eight-field string identity required')
 if value['role']!='train':raise ValueError('R10 original training diagnosis only')
 if expected is not None and value!=expected:raise ValueError('identity differs; no coercion/substitution')
 return value

def typed_asset(ref):
 required={'path','sha256','compression','uncompressed_bytes','canonical_json_sha256'}
 if not required<=set(ref) or ref['compression']!='gzip' or type(ref['uncompressed_bytes'])is not int or not 0<=ref['uncompressed_bytes']<=268435456:raise ValueError('fully qualified bounded gzip descriptor required')
 with gzip.open(pin(ref),'rb') as f:raw=f.read(ref['uncompressed_bytes']+1)
 if len(raw)!=ref['uncompressed_bytes'] or hashlib.sha256(raw).hexdigest()!=ref['canonical_json_sha256']:raise ValueError('original uncompressed bytes differ')
 return json.loads(raw)

def write_asset(path,value):
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);raw=json.dumps(value,ensure_ascii=False,allow_nan=False,sort_keys=True,separators=(',',':')).encode()
 if len(raw)>268435456:raise ValueError('bounded new typed output')
 compressed=gzip.compress(raw,mtime=0);temp=p.with_name(p.name+'.tmp')
 if p.exists() or temp.exists():raise FileExistsError('immutable native output already exists')
 with temp.open('xb') as f:f.write(compressed);f.flush();os.fsync(f.fileno())
 os.replace(temp,p)
 if os.name!='nt':
  fd=os.open(p.parent,os.O_DIRECTORY)
  try:os.fsync(fd)
  finally:os.close(fd)
 return dict(path=str(p),sha256=digest(p),compression='gzip',compressed_bytes=len(compressed),uncompressed_bytes=len(raw),canonical_json_sha256=hashlib.sha256(raw).hexdigest())

def identity(pid):
 try:
  proc=Path('/proc')/str(pid);raw=(proc/'stat').read_text();fields=raw[raw.rfind(')')+2:].split()
  return dict(pid=int(pid),birth_ticks=int(fields[19]),state=fields[0],boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),uid=proc.stat().st_uid)
 except OSError as e:
  if e.errno in (errno.ENOENT,errno.ESRCH):return None
  raise

def same_process(a,b):return bool(a and b and all(a.get(k)==b.get(k) for k in ('pid','birth_ticks','boot_id','uid')))

def io_snapshot(pid=None):
 try:
  values={k:int(v) for k,v in (line.split(':',1) for line in (Path('/proc')/str(pid or os.getpid())/'io').read_text().splitlines())}
  return dict(observed_unix_seconds=time.time(),**{k:values[k] for k in ('rchar','wchar','syscr','syscw','read_bytes','write_bytes')},logical_chars_not_disk_bytes=True,receipt_IO_and_process_exit_excluded=True)
 except OSError as e:
  if e.errno in (errno.ENOENT,errno.ESRCH):return None
  raise

def origin_ref(p,ref):
 matches=[v for v in p.get('document_aliases',[]) if v['runtime_ref']==ref]
 if len(matches)>1:raise ValueError('ambiguous metadata transport')
 if not matches:return ref
 item=matches[0]
 if item['origin_ref']['sha256']!=ref['sha256']:raise ValueError('transport changes original bytes')
 pin(ref);return item['origin_ref']

def source_check(p):
 repo=Path(__file__).resolve().parents[2];refs=p['source_files']
 required={str((Path('scripts/stage1_ai_search_r10')/n)).replace('\\','/') for n in OPS_FILES}
 for package in ('ai_search_r10','ai_search_r9'):
  directory=repo/'src/satellite_rl/coverage'/package;members=list(directory.glob('*.py'))
  if not members:raise ValueError('missing complete runtime package '+package)
  required.update(q.relative_to(repo).as_posix() for q in members)
 for key in required:
  if key not in refs or Path(refs[key]['path']).resolve()!=(repo/key).resolve():raise ValueError('full exact runtime source closure '+key)
 for key,r in refs.items():
  pin(r)
  if key.startswith(('scripts/stage1_ai_search_r10/','src/satellite_rl/coverage/ai_search_r10/','src/satellite_rl/coverage/ai_search_r9/')) and key not in required:raise ValueError('invented source leaf')
 sm=read(p['original_source_manifest_ref']);root=Path(p['original_source_manifest_ref']['path']).parent
 for name,sha in sm['files'].items():
  if digest(root/name)!=sha:raise ValueError('frozen original source differs '+name)

def imported_check(p):
 repo=Path(__file__).resolve().parents[2];original=Path(p['original_source_manifest_ref']['path']).parent.resolve();sm=read(p['original_source_manifest_ref'])
 for name,module in list(sys.modules.items()):
  path=getattr(module,'__file__',None)
  if not path or not name.startswith('satellite_rl'):continue
  real=Path(path).resolve()
  if name.startswith(('satellite_rl.coverage.ai_search_r10','satellite_rl.coverage.ai_search_r9')):
   key=real.relative_to(repo).as_posix()
   if key not in p['source_files'] or pin(p['source_files'][key]).resolve()!=real:raise ValueError('actual math dependency import differs '+name)
  else:
   if not real.is_relative_to(original):raise ValueError('original import outside frozen714 '+name)
   key=real.relative_to(original).as_posix()
   if key not in sm['files'] or digest(real)!=sm['files'][key]:raise ValueError('original imported source not pinned '+name)

def human_epoch(p):
 authority=read(p['human_resume_authority_ref']);epoch=read(p['human_epoch_ref'])
 if p['human_resume_authority_ref']['sha256']!=AUTHORITY_SHA or authority.get('human_authorization')is not True or authority.get('root_accepted')is not True or authority.get('user_message_verbatim')!='继续':raise ValueError('actual post0700 human continue')
 if epoch.get('schema')!='R10NewScientificHumanEpochV1' or epoch.get('group')!=GROUP or epoch.get('root_accepted')is not True or epoch.get('new_deadline_unix_seconds')is not None:raise ValueError('new exact R10 human epoch')
 if epoch.get('authority_ref')!=p['human_resume_authority_ref'] or any(epoch.get(k)!=p[k] for k in ('run_id','request_id','archive_root')):raise ValueError('new epoch/task binding')
 if not epoch.get('execution_epoch_id') or not epoch.get('watched_pause_paths') or not isinstance(epoch.get('historical_hold_exemptions'),list):raise ValueError('explicit pause inventory')
 return epoch

def storage_check(p):
 s=read(p['storage_inventory_ref']);root=Path(p['archive_root']).resolve()
 if s.get('schema')!='R10OwnedStorageInventoryV1' or s.get('root_accepted')is not True or s.get('archive_root')!=p['archive_root'] or s.get('new_logical_bytes_max')!=536870912 or s.get('temporary_bytes_max')!=268435456 or s.get('free_reserve_bytes')!=8589934592:raise ValueError('exact R10 storage caps')
 if not root.is_relative_to(Path(s['owned_growth_root']).resolve()) or '/round10/' not in root.as_posix()+'/':raise ValueError('owned R10 storage only')
 return s

def validate_execution(plan_ref,review_ref):
 """Only the final v2 finite plan can authorize its host's immutable six-mother task."""
 p=read(plan_ref);review=read(review_ref)
 if p.get('group')!=GROUP or p.get('source_input_budget_frozen')is not True:raise ValueError('R10 source/input/budget still preparation')
 if p.get('schema')!='R10ExactMechanismPlanV2' or p.get('executor_port') not in (20473,20474) or not p.get('run_id','').startswith('r10_') or '/round10/' not in p['archive_root']:raise ValueError('new R10 host/task namespace')
 if review.get('schema')!='R10RootExactExecutionAdmissionV1' or review.get('exact_plan_ref')!=plan_ref or any(review.get(k)is not True for k in ('passed','execution_authorized','scientific_execution_authorized','source_accepted','input_accepted','runtime_accepted','engineering8_accepted')):raise ValueError('exact root source/input/runtime/engineering/scientific execution admission required')
 for key in ('human_epoch_ref','input_qualification_ref','storage_inventory_ref'):
  if review.get(key)!=p[key]:raise ValueError('root exact '+key)
 if p['design_ref']['sha256']!=DESIGN_SHA:raise ValueError('unique accepted R10 mathematics v2')
 design=read(p['design_ref']);accepted=read(p['root_design_acceptance_ref'])
 if design['group']!=GROUP or accepted.get('schema')!='RootR10MathematicalDesignAcceptanceV2' or accepted.get('accepted_design')is not True or accepted.get('design_ref')!=origin_ref(p,p['design_ref']):raise ValueError('root exact mathematics acceptance')
 if p['numeric_origin_protocol_ref']['sha256']!=NUMERIC_PROTOCOL_SHA:raise ValueError('frozen original metric numeric qualification')
 read(p['numeric_origin_protocol_ref'])
 if p['method_hard_seconds']!=METHOD_HARD_SECONDS or p['method_rss_bytes']!=METHOD_RSS_BYTES:raise ValueError('root method safety ceilings differ')
 if p.get('common_hard_seconds')!=30 or p.get('arms')!=list(ARMS) or p.get('caps')!=CAPS:raise ValueError('common30/fivearms/finite caps')
 from .schedule import partition
 partition(p['cases'],p['mother_host_assignment'],p['executor_port'])
 qualification=read(p['input_qualification_ref'])
 if qualification.get('schema')!='RootR10Exact12ExistingInputsStaticQualificationV1' or qualification.get('qualified_static')is not True or qualification.get('heldout_reads')!=0 or [v['original_index'] for v in qualification['rows']]!=list(MOTHERS):raise ValueError('root existing exact twelve static input qualification')
 for case,qualified in zip(p['cases'],qualification['rows']):
  if case.get('uniform_request_bps')!=21600000 or case.get('origin_kind')!='original_R9_allpool_best':raise ValueError('original B/request source only')
  if case['identity']!=qualified['identity'] or case['aircraft_count']!=qualified['aircraft_count'] or case['common_input_ref']['sha256']!=qualified['input_ref']['sha256'] or case['B_layout_sha256']!=qualified['B_layout_sha256']:raise ValueError('root actual B/scene exact byte binding')
  pin(case['common_input_ref'])
 source_check(p);human_epoch(p);storage_check(p);return p

validate=validate_execution

def fresh_route(p):
 r=read(p['actual_resource_route_ref']);age=time.time()-r['observed_unix_seconds']
 if not -5<=age<=120 or r.get('ssh_port')!=p['executor_port'] or r.get('group')!=GROUP or any(r.get(k)is not True for k in ('ready','root_accepted','own_task_absence_verified','old_owner_handoff_verified')):raise ValueError('fresh exact owner/resource admission')
 if r.get('boot_id')!=Path('/proc/sys/kernel/random/boot_id').read_text().strip() or r.get('python')!='/home/u2024110224/first_stage_satellite_rl/envs/DQN/bin/python':raise ValueError('actual current boot/DQN interpreter')
 return r
