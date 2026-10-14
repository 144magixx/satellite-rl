"""Stable append-only observer; no publication into another worker's archive."""
import errno,json,os,pathlib,shutil,stat,time,threading
class StorageLimit(RuntimeError):pass
class StorageObservationUnknown(RuntimeError):pass
def generation(s):return (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
class Observer:
 def __init__(self,plan,control_root,*,storage_check,clock=time.monotonic,disk_usage=shutil.disk_usage):
  self.plan=plan;self.inventory=storage_check(plan);self.root=pathlib.Path(plan['archive_root']);self.control=pathlib.Path(control_root);self.clock=clock;self.disk_usage=disk_usage;self.lock=threading.Lock();self.log_max=2097152;s=self.inventory
  if (s['new_logical_bytes_max'],s['temporary_bytes_max'],s['free_reserve_bytes'])!=(536870912,268435456,8589934592):raise ValueError('unchanged frozen storage thresholds')
  self.growths=[pathlib.Path(v).resolve() for v in s.get('owned_growth_roots',[s['owned_growth_root']])];self.temps=[pathlib.Path(v).resolve() for v in s.get('owned_temporary_roots',[s['owned_temporary_root']])]
  if any('/round11/' not in p.as_posix()+'/' for p in self.growths+self.temps) or any(not any(t.is_relative_to(g) for g in self.growths) for t in self.temps):raise ValueError('only owned R11 growth/temp scope')
  if not any(self.control.resolve().is_relative_to(g) for g in self.growths):raise ValueError('control included in registered growth')
  self.control.mkdir(parents=True,exist_ok=True)
  if self.control.is_symlink():raise ValueError('linked control')
  self.path=self.control/'storage_observations.jsonl';self.log=self.path.open('xb');self._append(dict(schema='R11RecoveryStableStorageLogV1',actual_pid=os.getpid(),personal_quota='unknown',worker_storage_never_published=True));self._sync()
 def _sync(self):
  if os.name!='nt':
   fd=os.open(self.control,os.O_DIRECTORY)
   try:os.fsync(fd)
   finally:os.close(fd)
 def _append(self,value):
  raw=(json.dumps(value,ensure_ascii=False,allow_nan=False,separators=(',',':'))+'\n').encode()
  with self.lock:
   if self.log.tell()+len(raw)>self.log_max:raise StorageLimit('2MiB control journal; no rotation')
   self.log.write(raw);self.log.flush();os.fsync(self.log.fileno())
 def _scan(self):
  began=self.clock();history=[]
  for attempt in range(3):
   logical=temporary=0;seen={};directories={};unstable=[]
   def onerror(error):
    if error.errno==errno.ENOENT:unstable.append(dict(path=str(error.filename),reason='walk_ENOENT'))
    else:raise StorageObservationUnknown('walk IO/permission: '+repr(error))
   for growth in self.growths:
    for current,dirs,names in os.walk(growth,followlinks=False,onerror=onerror):
     if self.clock()-began>10:raise StorageObservationUnknown('bounded10s scan; partial='+repr(history))
     current=pathlib.Path(current)
     try:
      ds=current.lstat()
      if not stat.S_ISDIR(ds.st_mode):raise StorageObservationUnknown('linked/non-directory owned path')
      directories[current]=generation(ds)
      for name in dirs:
       if (current/name).is_symlink():raise StorageObservationUnknown('linked owned directory')
      for name in names:
       path=current/name
       if path in seen:continue
       before=path.lstat()
       if not stat.S_ISREG(before.st_mode):raise StorageObservationUnknown('linked/nonregular owned file')
       seen[path]=generation(before)
       if len(seen)>16384:raise StorageObservationUnknown('bounded16384 members')
       if any(path.is_relative_to(t) for t in self.temps):temporary+=before.st_size
       else:logical+=before.st_size
     except OSError as e:
      if e.errno==errno.ENOENT:unstable.append(dict(path=str(current),reason='member_ENOENT'))
      else:raise StorageObservationUnknown('member IO/permission: '+repr(e))
   for path,old in list(directories.items())+list(seen.items()):
    try:
     if generation(path.lstat())!=old:unstable.append(dict(path=str(path),reason='stat_generation_changed'))
    except OSError as e:
     if e.errno==errno.ENOENT:unstable.append(dict(path=str(path),reason='final_ENOENT'))
     else:raise StorageObservationUnknown('final IO/permission: '+repr(e))
   partial=dict(attempt=attempt+1,owned_logical_bytes=logical,owned_temporary_bytes=temporary,observed_files=len(seen),unstable=unstable[:16]);history.append(partial)
   if not unstable:return dict(owned_logical_bytes=logical,owned_temporary_bytes=temporary,observed_files=len(seen),scan_attempts=attempt+1)
  self._append(dict(point='storage_unknown',partial_observations=history,accepted_complete_total=False));raise StorageObservationUnknown('3 incomplete observations; no missing member counted zero')
 def observe(self,point):
  began=self.clock();disk=self.disk_usage(self.root);sizes=self._scan();s=self.inventory;value=dict(point=point,disk_free_mount_bytes=disk.free,**sizes,personal_quota='unknown unless separately verified',physical_occupancy_not_claimed=True,observed_seconds=self.clock()-began)
  limited=disk.free<s['free_reserve_bytes'] or sizes['owned_logical_bytes']>s['new_logical_bytes_max'] or sizes['owned_temporary_bytes']>s['temporary_bytes_max'];value['threshold_exceeded']=limited;self._append(value)
  if limited:
   with (self.control/'storage_pause.json').open('x',encoding='utf8') as f:json.dump(dict(paused=True,reason='storage',observation=value),f);f.flush();os.fsync(f.fileno())
   self._sync();raise StorageLimit('same storage cap/reserve; no deletion')
  return value
 def close(self):self.log.close()
