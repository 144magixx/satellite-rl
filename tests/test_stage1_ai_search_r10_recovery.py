"""Operational mocks only: no Scene, satellite solver, model, child or server."""
import errno,json,math,os,pathlib,random,tempfile,threading,time,types,unittest
from unittest.mock import patch
from scripts.stage1_ai_search_r10 import contracts as c
from scripts.stage1_ai_search_r10_recovery import contracts as r,worker,server,storage,controller
def identity(index):return {k:('train' if k=='role' else k+str(index)) for k in c.IDENTITY_FIELDS}
def rows_for(port):
 out=[]
 for index in ([0,12,24,42,54,72] if port==20473 else [6,18,36,48,60,78]):
  for stage in ('common',*c.ARMS):
   done=(port==20473 and index==0 and stage in ('common','Gd','Gr','Gw')) or (port==20474 and index in (6,18,36,48));failed=(port,index,stage)==(20473,0,'S2');out.append(dict(executor_port=port,original_index=index,stage=stage,identity=identity(index),typed_output_exists=done,native_stage_wal_exists=done or failed,algorithm_entry_exists=done,old_algorithm_spend_exact=None,actual_stage_wall_seconds=1.6233237385749817 if failed else .01))
 return out
class Publication(unittest.TestCase):
 def test_unique_mutable_concurrent_writers_and_rng(self):
  with tempfile.TemporaryDirectory() as tmp:
   target=pathlib.Path(tmp)/'latest.json';errors=[];state=random.getstate()
   def writer(k):
    try:
     for i in range(50):c.publish(target,dict(writer=k,index=i,payload='x'*1000))
    except BaseException as e:errors.append(e)
   # Native production is Linux. Windows lacks POSIX replacement of a pathname
   # while another thread briefly reads its old inode for the returned digest.
   # This test-only shim models that POSIX behavior; production never retries.
   replace=os.replace;digest=c.digest
   def posix_replace(src,dst):
    until=time.monotonic()+2
    while True:
     try:return replace(src,dst)
     except PermissionError:
      if os.name!='nt' or time.monotonic()>=until:raise
      time.sleep(.001)
   def posix_digest(path):
    until=time.monotonic()+2
    while True:
     try:return digest(path)
     except (PermissionError,FileNotFoundError):
      if os.name!='nt' or time.monotonic()>=until:raise
      time.sleep(.001)
   threads=[threading.Thread(target=writer,args=(k,)) for k in range(3)]
   with patch.object(c.os,'replace',side_effect=posix_replace),patch.object(c,'digest',side_effect=posix_digest):
    for t in threads:t.start()
    for t in threads:t.join()
   self.assertEqual(errors,[]);self.assertEqual(json.loads(target.read_bytes())['index'],49);self.assertFalse(list(target.parent.glob('*.tmp')));self.assertEqual(random.getstate(),state)
 def test_immutable_fence_never_overwrites(self):
  with tempfile.TemporaryDirectory() as tmp:
   target=pathlib.Path(tmp)/'algorithm_entry.json';c.immutable(target,dict(original=1));before=target.read_bytes()
   with self.assertRaises(FileExistsError):c.immutable(target,dict(replacement=1))
   self.assertEqual(target.read_bytes(),before)
 def test_failed_private_temp_preserved_with_true_diagnostics(self):
  with tempfile.TemporaryDirectory() as tmp:
   target=pathlib.Path(tmp)/'target.json'
   with patch.object(c.os,'replace',side_effect=OSError(errno.EDQUOT,'mock quota')):
    with self.assertRaises(OSError) as captured:c.publish(target,dict(value=7))
   info=captured.exception.publication_failure;self.assertFalse(info['cleanup_attempted']);self.assertFalse(info['replace_completed']);self.assertTrue(pathlib.Path(info['private_temp']).exists());self.assertFalse(target.exists());self.assertEqual(json.loads(pathlib.Path(info['private_temp']).read_bytes()),dict(value=7))
class Storage(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.root=pathlib.Path(self.tmp.name)/'round10/owned';self.archive=self.root/'archive';self.archive.mkdir(parents=True);self.ctl=self.root/'control';self.plan=dict(archive_root=str(self.archive));self.inventory=dict(owned_growth_root=str(self.root),owned_temporary_root=str(self.root/'tmp'),new_logical_bytes_max=536870912,temporary_bytes_max=268435456,free_reserve_bytes=8589934592)
 def tearDown(self):self.tmp.cleanup()
 def observer(self):return storage.Observer(self.plan,self.ctl,storage_check=lambda p:self.inventory,disk_usage=lambda p:types.SimpleNamespace(free=10**12))
 def test_old_worker_path_untouched_and_no_private_tmp_in_control(self):
  old=self.archive/'storage_latest.json';old.write_text('original');o=self.observer();o.observe('test');o.close();self.assertEqual(old.read_text(),'original');self.assertFalse(list(self.ctl.glob('*.tmp')))
 def test_walk_permission_is_unknown_not_empty(self):
  o=self.observer()
  def walk(*args,**kwargs):kwargs['onerror'](PermissionError(errno.EACCES,'mock denied',str(self.root)));return iter([])
  with patch.object(storage.os,'walk',side_effect=walk):
   with self.assertRaises(storage.StorageObservationUnknown):o._scan()
  o.close()
 def test_missing_walk_directory_rescans_then_unknown(self):
  o=self.observer();calls=[]
  def walk(*args,**kwargs):calls.append(1);kwargs['onerror'](FileNotFoundError(errno.ENOENT,'mock vanished',str(self.root)));return iter([])
  with patch.object(storage.os,'walk',side_effect=walk):
   with self.assertRaises(storage.StorageObservationUnknown):o._scan()
  o.close();self.assertEqual(len(calls),3);last=json.loads(o.path.read_text().splitlines()[-1]);self.assertFalse(last['accepted_complete_total']);self.assertEqual(len(last['partial_observations']),3)
 def test_stat_generation_change_is_not_accepted(self):
  path=self.root/'changing';path.write_text('data');o=self.observer();real=pathlib.Path.lstat;count=[0]
  def lstat(p,*args,**kwargs):
   result=real(p,*args,**kwargs)
   if p==path:
    count[0]+=1;values={k:getattr(result,k) for k in ('st_mode','st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns')};values['st_mtime_ns']+=count[0];return types.SimpleNamespace(**values)
   return result
  with patch.object(pathlib.Path,'lstat',lstat):
   with self.assertRaises(storage.StorageObservationUnknown):o._scan()
  o.close()
 def test_quota_unknown_is_explicit_but_reserve_low_stops(self):
  o=self.observer();value=o.observe('quotaunknown');self.assertIn('unknown',value['personal_quota']);o.disk_usage=lambda p:types.SimpleNamespace(free=1)
  with self.assertRaises(storage.StorageLimit):o.observe('lowreserve')
  o.close();self.assertTrue((self.ctl/'storage_pause.json').exists());self.assertFalse((self.archive/'pause_state.json').exists())
class BudgetsAndAdmission(unittest.TestCase):
 def test_rows_counts_and_failed_guard_debit(self):
  rows=rows_for(20473)+rows_for(20474);new=[x for x in rows if not x['typed_output_exists']];self.assertEqual(len(rows),72);self.assertEqual(sum(x['typed_output_exists'] for x in rows),28);self.assertEqual(len(new),44);self.assertEqual(sum(not x['native_stage_wal_exists'] for x in new),43);self.assertEqual(r.limits_for(next(x for x in new if x['native_stage_wal_exists'])),298.376676261425);self.assertAlmostEqual(sum(r.limits_for(x) for x in new if x['executor_port']==20473),8248.376676261425)
 def test_nonfinite_and_bool_budgets_rejected(self):
  for value in [float('nan'),float('inf'),-1,True]:
   with self.assertRaises(ValueError):r.finite(value)
 def test_reference_bytes_projection_and_wrong_host(self):
  with tempfile.TemporaryDirectory() as tmp:
   path=pathlib.Path(tmp)/'p';path.write_text('x');a=dict(path=str(path),sha256=c.digest(path));self.assertTrue(r.same_ref(a,dict(a,bytes=1)));self.assertFalse(r.same_ref(a,dict(a,path='/wronghost')))
   with self.assertRaises(ValueError):r.same_ref(a,dict(a,bytes=2))
 def test_final_outer_pause_and_debit_limits(self):
  with self.assertRaisesRegex(RuntimeError,'exhausted'):controller.final_limits(10548.42580656521,1,10548.42580656521,lambda:None)
  with self.assertRaisesRegex(RuntimeError,'newpause'):controller.final_limits(1,1,100,lambda:(_ for _ in ()).throw(RuntimeError('newpause')))
 def test_resume_only_and_unknown_no_retry(self):
  with tempfile.TemporaryDirectory() as tmp:
   endpoint=server.Endpoint.__new__(server.Endpoint);endpoint.root=pathlib.Path(tmp);endpoint.path='/api/ai-search-r10/tasks/same';endpoint.plan_ref=dict(path='/P',sha256='p');endpoint.review_ref=dict(path='/R',sha256='r');endpoint.plan=dict(request_id='original_request',archive_root=tmp,recovery_contract_ref=dict(path='/C',sha256='c'));body=dict(plan_ref=endpoint.plan_ref,root_review_ref=endpoint.review_ref,request_id='original_request');calls=[]
   with self.assertRaises(ValueError):endpoint.post(endpoint.path+'/start',body)
   with patch.object(server,'validate',return_value=(endpoint.plan,{},[])),patch.object(server,'check_pause'),patch.object(c,'fresh_route'),patch.object(c,'identity',return_value=None):
    with self.assertRaises(RuntimeError):endpoint.post(endpoint.path+'/resume',body,popen=lambda *a,**k:(calls.append(1) or types.SimpleNamespace(pid=123)))
    with self.assertRaises(ValueError):endpoint.post(endpoint.path+'/resume',body,popen=lambda *a,**k:calls.append(1))
   self.assertEqual(len(calls),1);self.assertTrue((endpoint.root/'submission_intent.json').exists())
 def test_readonly_pause_never_publishes_worker_archive(self):
  epoch=dict(historical_hold_exemptions=[],watched_pause_paths=['/new/stop.request']);p=dict(archive_root='/new/archive')
  with patch.object(c,'human_epoch',return_value=epoch),patch.object(r.control,'marker_snapshot',return_value=dict(value=dict(user_pause=True))),patch.object(c,'publish',side_effect=AssertionError('outer must not publish worker archive')):
   with self.assertRaises(r.control.Paused):r.check_pause(p,'unit')
 def test_actual_ops_import_origins(self):
  import scripts.stage1_ai_search_r10.child
  repo=pathlib.Path(c.__file__).resolve().parents[2];declared={}
  for package in ('stage1_ai_search_r10','stage1_ai_search_r10_recovery'):
   for path in (repo/'scripts'/package).glob('*.py'):declared[path.relative_to(repo).as_posix()]=dict(path=str(path),sha256=c.digest(path))
  p=dict(source_repo=str(repo),source_files={k:v for k,v in declared.items() if '/stage1_ai_search_r10/' in k},recovery_source_files={k:v for k,v in declared.items() if '/stage1_ai_search_r10_recovery/' in k});r.source_import_check(p)
  with patch.object(c,'__file__',str(repo.parent/'wrongroot/scripts/stage1_ai_search_r10/contracts.py')):
   with self.assertRaises(ValueError):r.source_import_check(p)
if __name__=='__main__':unittest.main()
