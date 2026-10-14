"""One Gh call on the exact historical R10 common; no common producer."""
import argparse,os,sys,time,threading
from pathlib import Path
from scripts.stage1_ai_search_r10 import contracts as c
from scripts.stage1_ai_search_r10.control import bind_parent
from scripts.stage1_ai_search_r10.costs import measured_scope
from .contracts import validate,check_pause,source_check

class PauseProbe:
 def __init__(self,p):self.p=p;self.error=None;self.finished=threading.Event();self.thread=None
 def start(self):
  check_pause(self.p,'child_before_algorithm');self.thread=threading.Thread(target=self.loop,daemon=True);self.thread.start();return self
 def loop(self):
  while not self.finished.wait(.25):
   try:
    if time.clock_gettime(time.CLOCK_BOOTTIME)>=float(os.environ['R11_NATIVE_DEADLINE_BOOTTIME'])-.5:raise RuntimeError('shared native deadline cleanup reserve')
    check_pause(self.p,'child_pause_probe')
   except BaseException as e:self.error=e;return
 def check(self):
  if self.error is not None:raise self.error
  return False
 def close(self):
  self.finished.set()
  if self.thread:self.thread.join(timeout=1)

def execute(request,directory):
 began=time.monotonic();bind_parent('R10_PARENT_PID');p=validate(request['plan_ref'],request['root_review_ref']);check_pause(p,'child_import');directory=Path(directory)
 if request['stage']!='Gh':raise ValueError('no common or control reexecution')
 case=next(v for v in p['cases'] if v['original_index']==request['original_index'])
 if request['common_ref']!=case['common_ref']:raise ValueError('sole original common ref')
 reservation=c.read(request['budget_reservation_ref'])
 if reservation!=dict(schema='R11GhLogicalReservationV1',run_id=p['run_id'],execution_epoch_id=p['execution_epoch_id'],identity=case['identity'],original_index=case['original_index'],stage='Gh',common_ref=case['common_ref'],unknown_never_replayed=True):raise ValueError('exact single logical reservation')
 original=Path(p['original_source_manifest_ref']['path']).parent;sys.path[:0]=[str(original/'src'),str(original)]
 import satellite_rl.coverage
 satellite_rl.coverage.__path__.append(str(Path(p['source_repo'])/'src/satellite_rl/coverage'))
 from satellite_rl.coverage.ai_search_r11 import core
 from types import SimpleNamespace
 import numpy as np
 source_check(p);c.publish(directory/'setup_cost.json',dict(observed_seconds=time.monotonic()-began,scope='binding/source/root/pause/import inside native',actual_process=c.identity(os.getpid())))
 with measured_scope(directory/'input_cost.json','decode original R10 wrapper/static; no new rays/common'):
  wrapper=c.typed_asset(case['common_ref']);c.check_identity(wrapper['identity'],case['identity'])
  if wrapper['run_id']!=case['historical_R10_run_id'] or wrapper['original_index']!=case['original_index'] or wrapper.get('actual_stage')!='common' or wrapper['common']['common_SHA']!=case['historical_common_SHA']:raise ValueError('historical R10 common wrapper identity')
  static=wrapper['static'];scene=SimpleNamespace(**{**static,'aircraft_ids':tuple(static['aircraft_ids']),'positions':np.asarray(static['positions'],np.float64),'rays':np.asarray(static['rays'],np.float64),'demands':np.asarray(static['demands'],np.float64)})
  if len(scene.aircraft_ids)!=case['aircraft_count'] or scene.input_sha256!=case['identity']['input_sha256']:raise ValueError('all original aircraft/input')
 c.immutable(directory/'algorithm_entry.json',dict(run_id=p['run_id'],identity=case['identity'],original_index=case['original_index'],stage='Gh',common_ref=case['common_ref'],historical_common_run_id=wrapper['run_id'],actual_process=c.identity(os.getpid()),budget_reservation_ref=request['budget_reservation_ref'],entry_unix_seconds=time.time(),actual_work_until_completion_unknown=True))
 probe=PauseProbe(p).start();alg=time.monotonic()
 try:result=core.run_method(scene,wrapper['common'],'Gh',stop=probe.check)
 finally:
  probe.close();c.publish(directory/'algorithm_scope_cost.json',dict(observed_seconds_to_finally=time.monotonic()-alg,scope='Gh/probe nested in native',receipt_IO_and_exit_excluded=True))
 source_check(p);check_pause(p,'child_output')
 with measured_scope(directory/'publication_cost.json','full Gh result typed serialization nested in native'):
  value=dict(schema='R11GhNativeOutputV1',run_id=p['run_id'],identity=case['identity'],original_index=case['original_index'],actual_stage='Gh',result=result,common_ref=case['common_ref'],historical_common_run_id=wrapper['run_id'],source_input_ref=case['common_input_ref'],new_common_W_T_RULE_D5_model_calls=0)
  asset=c.write_asset(directory/'stage_output.json.gz',value);c.immutable(directory/'stage_output.json',dict(asset_ref=asset));return asset

def main():
 began=time.monotonic();a=argparse.ArgumentParser();a.add_argument('--request',required=True);a.add_argument('--request-sha256',required=True);a.add_argument('--directory',required=True);v=a.parse_args();directory=Path(v.directory)
 try:execute(c.read(dict(path=v.request,sha256=v.request_sha256)),directory)
 except BaseException as e:
  c.publish(directory/'child_failure.json',dict(error=type(e).__name__+': '+str(e),partial_result=getattr(e,'partial_result',None),partial_ledger=getattr(e,'partial_ledger',None),automatic_retry=False,hard_kill_unpublished_work_unknown=True));raise
 finally:c.publish(directory/'child_finally_cost.json',dict(observed_seconds_to_finally=time.monotonic()-began,process_IO=c.io_snapshot(),receipt_IO_and_exit_excluded=True,complete_worker_exit_cost_seconds=None))
if __name__=='__main__':main()
