"""One admitted native stage; never an implicit solver retry or model caller."""
import argparse,os,sys,time
from pathlib import Path
from .contracts import read,pin,atomic,immutable,digest,imported_check,identity,io_snapshot,validate
from .control import boundary,bind_parent

def execute(req,directory):
 began=time.monotonic();bind_parent('R9_PARENT_PID');p=validate(req['plan_ref'],req['root_review_ref']);boundary(p,'child_before_import');directory=Path(directory)
 if req['stage'] not in ('RULE','D5','common','pool'):raise ValueError('M stages only')
 case=next(c for c in p['cases'] if c['original_index']==req['original_index'])
 source=Path(p['original_source_manifest_ref']['path']).parent;sys.path[:0]=[str(source/'src'),str(source)]
 # Original packages remain pinned; only the new R9 package extends coverage.
 import satellite_rl.coverage
 satellite_rl.coverage.__path__.append(str(Path(__file__).resolve().parents[2]/'src/satellite_rl/coverage'))
 from satellite_rl.workbench.r6_io import read_asset,write_asset
 from .train_input import load_original,rule,d5,layout_delta
 from satellite_rl.coverage.ai_search_r9 import core
 imported_check(p);setup=time.monotonic()-began
 atomic(directory/'module_setup_timer.json',dict(observed_seconds=setup,scope='bound parent/plan/boundary/imports; included native stage wall',actual_process=identity(os.getpid()),no_free_geometry=True))
 stage=req['stage'];value=None
 def entry(kind,position=None):
  boundary(p,'before_algorithm_entry_'+kind)
  return immutable(directory/((('entry_'+str(position)) if position is not None else 'algorithm_entry')+'.json'),dict(stage=kind,position=position,actual_process=identity(os.getpid()),case_identity=case['identity'],entered_unix_seconds=time.time(),exact_work_until_completion_unknown=True))
 if stage in ('RULE','D5','common'):
  scene,prepared,inputcost=load_original(case);atomic(directory/'input_cost.json',inputcost)
 if stage=='RULE':
  entry('RULE');start=time.monotonic();proposal=rule(prepared,case['prepared_ref']);value=dict(identity=case['identity'],proposal=proposal,actual_RULE_calls=1,observed_algorithm_seconds=time.monotonic()-start,input_cost=inputcost)
 elif stage=='D5':
  proposal=read_asset(req['proposal_ref'])['proposal']
  if proposal['case_identity']!=case['identity'] or proposal['provenance']['source_method']!='RULE':raise ValueError('same train RULE only, no teacher')
  entry('D5');start=time.monotonic();result=d5(prepared,proposal)
  cost=result['cost'];n=len(scene.aircraft_ids)
  if cost['raypairs']>2048*n or cost['proposals']>512 or cost['refits']>8:raise ValueError('original D5 count ceiling')
  value=dict(identity=case['identity'],result=result,actual_D5_calls=1,nominal_D5_seconds=5.,observed_algorithm_seconds=time.monotonic()-start,input_cost=inputcost)
 elif stage=='common':
  start=time.monotonic();rulevalue=read_asset(req['RULE_ref']);teacher=read_asset(case['original_T_terminal_ref'])['result'] if case.get('original_T_terminal_ref') else None
  if rulevalue['result']['status']!='success':raise ValueError('unqualified mandatory RULE; fallback preserved but never floor')
  origin=core.select_origin(scene,prepared,rulevalue,teacher);pool=core.build_pool(scene,origin)
  if len(pool['descriptors'])>32:raise ValueError('fixed descriptor cap')
  value=dict(identity=case['identity'],origin=origin,pool=pool,input_cost=inputcost,common_cumulative_ledger=pool['common_ledger'],origin_and_pool_ledgers_not_additive=True,static=dict(positions=scene.positions.tolist(),demands=scene.demands.tolist(),aircraft_ids=list(scene.aircraft_ids),profile=scene.profile,rays=scene.rays.tolist(),input_sha256=scene.input_sha256),observed_common_after_input_seconds=time.monotonic()-start,common_physically_once=True,new_RULE_D5_T_W_model_calls=0)
 else:
  # Common owns the sole new scene rays. Reading its pinned static arrays does
  # not construct another Scene or recompute nearest tables/features.
  from types import SimpleNamespace
  import numpy as np
  common=read_asset(req['common_ref']);s=common['static'];scene=SimpleNamespace(**{**s,'positions':np.asarray(s['positions'],np.float64),'demands':np.asarray(s['demands'],np.float64),'rays':np.asarray(s['rays'],np.float64),'aircraft_ids':tuple(s['aircraft_ids'])});origin=common['origin'];pool=common['pool']
  if common['identity']!=case['identity'] or pool['origin_SHA']!=origin['layout_SHA'] or pool['pool_SHA']!=core.digest(pool['descriptors']) or len(pool['descriptors'])>32:raise ValueError('exact common identity/pool pin')
  results=[];index={};pos=None
  try:
   for descriptor in pool['descriptors']:
    pos=descriptor['position'];entry('candidate',pos);result=core.evaluate_candidate(scene,origin,descriptor);results.append(result)
    compact=dict(result)
    compact['terminal_delta']=layout_delta(compact.pop('terminal'),origin['layout']);compact['raw_layout_delta']=layout_delta(compact.pop('raw_layout'),origin['layout']);compact['common_ref']=req['common_ref']
    ref=write_asset(directory/('candidate_'+str(pos)+'.json.gz'),compact);index[str(pos)]=ref;atomic(directory/'candidate_index.json',index)
    immutable(directory/('finish_'+str(pos)+'.json'),dict(position=pos,result_ref=ref,descriptor_extractions=1,kernel_entries=int(result['kernel_entered'])))
   selected=core.best_terminal(origin,results)
   value=dict(identity=case['identity'],complete=True,common_ref=req['common_ref'],candidate_refs=index,selected_position=selected['position'],selected_metrics=selected['metrics'],descriptor_extractions=len(results),kernel_entries=sum(x['kernel_entered'] for x in results),new_RULE_D5_T_W_model_calls=0)
  except BaseException as e:
   atomic(directory/'partial_pool.json',dict(complete=False,completed_candidate_refs=index,incomplete_position=pos,remaining_positions=[d['position'] for d in pool['descriptors'] if pos is None or d['position']>=pos],partial_ledger=getattr(e,'partial_ledger',None),error=type(e).__name__+': '+str(e),uncompleted_are_missing_not_infeasible=True,unclosed_algorithm_cost=None,automatic_retry=False));raise
 imported_check(p);publish=time.monotonic();value.update(actual_stage=stage,operational_input_and_stage_seconds=time.monotonic()-began,original_input_refs={k:case[k] for k in ('scene_ref','prepared_ref','shared_case_ref')})
 asset=write_asset(directory/'stage_output.json.gz',value);immutable(directory/'stage_output.json',dict(asset_ref=asset));atomic(directory/'publication_timer.json',dict(observed_seconds=time.monotonic()-publish,nested_in_native_stage_wall=True));return asset
def main():
 began=time.monotonic();a=argparse.ArgumentParser();a.add_argument('--request',required=True);a.add_argument('--request-sha256',required=True);a.add_argument('--directory',required=True);v=a.parse_args()
 try:execute(read(dict(path=v.request,sha256=v.request_sha256)),v.directory)
 except BaseException as e:
  atomic(Path(v.directory)/'child_failure.json',dict(error=type(e).__name__+': '+str(e),partial_ledger=getattr(e,'partial_ledger',None),automatic_retry=False));raise
 finally:atomic(Path(v.directory)/'child_main_finally.json',dict(observed_seconds_since_main=time.monotonic()-began,process_IO=io_snapshot(),includes_this_receipt_IO_and_process_exit=False,complete_exit_cost_closed=False))
if __name__=='__main__':main()
