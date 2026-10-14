"""One caller-qualified common or method; no generators, teachers or model entrypoints."""
import argparse,os,sys,time,threading
from pathlib import Path
from .contracts import read,pin,publish,immutable,typed_asset,write_asset,validate,imported_check,identity,io_snapshot,ARMS,check_identity
from .control import plan_boundary as boundary,bind_parent
from .costs import measured_scope

class PauseProbe:
 """Every solver tick reads memory only; metadata polling belongs to this bounded probe."""
 def __init__(self,plan,interval=.25):self.plan=plan;self.interval=interval;self.error=None;self.finished=threading.Event();self.thread=None
 def start(self):
  boundary(self.plan,'child_before_algorithm');self.thread=threading.Thread(target=self._run,daemon=True);self.thread.start();return self
 def _run(self):
  while not self.finished.wait(self.interval):
   try:boundary(self.plan,'child_cooperative_pause_probe')
   except BaseException as e:self.error=e;return
 def check(self):
  if self.error is not None:raise self.error
  return False
 def close(self):
  self.finished.set()
  if self.thread:self.thread.join(timeout=1)

def execute(request,directory):
 began=time.monotonic();bind_parent('R10_PARENT_PID');p=validate(request['plan_ref'],request['root_review_ref']);boundary(p,'child_before_import');directory=Path(directory)
 stage=request['stage']
 if stage not in ('common',*ARMS):raise ValueError('only registered R10 common/method')
 case=next(v for v in p['cases'] if v['original_index']==request['original_index']);check_identity(case['identity'])
 reservation=read(request['budget_reservation_ref'])
 if any(reservation[k]!=request[k] for k in ('original_index','stage')) or reservation['run_id']!=p['run_id'] or reservation['identity']!=case['identity']:raise ValueError('exact independent budget WAL')
 original=Path(p['original_source_manifest_ref']['path']).parent;sys.path[:0]=[str(original/'src'),str(original)]
 import satellite_rl.coverage
 satellite_rl.coverage.__path__.append(str(Path(__file__).resolve().parents[2]/'src/satellite_rl/coverage'))
 from satellite_rl.coverage.ai_search_r10 import core
 from types import SimpleNamespace
 import numpy as np
 imported_check(p);publish(directory/'setup_cost.json',dict(observed_seconds=time.monotonic()-began,scope='bind/source/root/pause/import, included outer native stage',process=identity(os.getpid())))
 input_began=time.monotonic()
 with measured_scope(directory/'input_cost.json','pin/decode full original arrays and dtype restoration; no new rays'):
  if stage=='common':
   ref=case['common_input_ref'];value=typed_asset(ref) if ref.get('compression') else read(ref)
   if value.get('schema')!='R10ExistingSceneAndBestOriginInputV1' or value.get('original_index')!=case['original_index']:raise ValueError('root static original input')
   static=value['scene'];origin=value['origin'];check_identity(origin['identity'],case['identity'])
  else:
   common_wrapper=typed_asset(request['common_ref']);check_identity(common_wrapper['identity'],case['identity'])
   if common_wrapper['run_id']!=p['run_id'] or common_wrapper['original_index']!=case['original_index']:raise ValueError('same physical common source')
   static=common_wrapper['static'];origin=common_wrapper['common']['origin']
  scene=SimpleNamespace(**{**static,'aircraft_ids':tuple(static['aircraft_ids']),'positions':np.asarray(static['positions'],np.float64),'rays':np.asarray(static['rays'],np.float64),'demands':np.asarray(static['demands'],np.float64)})
  if len(scene.aircraft_ids)!=case['aircraft_count'] or scene.input_sha256!=case['identity']['input_sha256']:raise ValueError('full original scene denominator/input')
 immutable(directory/'algorithm_entry.json',dict(run_id=p['run_id'],original_index=case['original_index'],identity=case['identity'],stage=stage,actual_process=identity(os.getpid()),budget_reservation_ref=request['budget_reservation_ref'],entry_unix_seconds=time.time(),actual_work_until_completion_unknown=True))
 probe=PauseProbe(p).start();algorithm_began=time.monotonic()
 try:
  result=core.prepare_common(scene,origin,original_index=case['original_index'],stop=probe.check) if stage=='common' else core.run_method(scene,common_wrapper['common'],stage,stop=probe.check)
 finally:
  probe.close();publish(directory/'algorithm_scope_cost.json',dict(observed_seconds_to_finally=time.monotonic()-algorithm_began,scope='actual common/method plus cooperative probe; nested in outer nativeguard',receipt_IO_and_exit_excluded=True))
 algorithm_seconds=time.monotonic()-algorithm_began;imported_check(p);boundary(p,'child_before_output');publication=time.monotonic()
 with measured_scope(directory/'publication_cost.json','serialize/compress/publish one full native result; nested in native wall'):
  value=dict(run_id=p['run_id'],original_index=case['original_index'],identity=case['identity'],actual_stage=stage,observed_algorithm_seconds=algorithm_seconds,source_input_ref=case['common_input_ref'],new_RULE_D5_W_T_model_calls=0)
  if stage=='common':value.update(common=result,static=static)
  else:value.update(result=result,common_ref=request['common_ref'])
  asset=write_asset(directory/'stage_output.json.gz',value);immutable(directory/'stage_output.json',dict(asset_ref=asset));return asset

def main():
 began=time.monotonic();a=argparse.ArgumentParser();a.add_argument('--request',required=True);a.add_argument('--request-sha256',required=True);a.add_argument('--directory',required=True);v=a.parse_args();directory=Path(v.directory)
 try:execute(read(dict(path=v.request,sha256=v.request_sha256)),directory)
 except BaseException as e:
  publish(directory/'child_failure.json',dict(error=type(e).__name__+': '+str(e),partial_result=getattr(e,'partial_result',None),partial_ledger=getattr(e,'partial_ledger',None),partial_kernel=getattr(e,'partial_kernel',None),automatic_retry=False,hard_kill_may_leave_only_common_B=True));raise
 finally:publish(directory/'child_finally_cost.json',dict(observed_seconds_to_finally=time.monotonic()-began,process_IO=io_snapshot(),receipt_IO_and_process_exit_excluded=True,complete_worker_exit_cost_seconds=None))
if __name__=='__main__':main()
