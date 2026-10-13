"""One case per physical lane, durable pilot rows, and no unknown-work repeat."""
import os,threading,time,traceback
from types import SimpleNamespace
from pathlib import Path
from satellite_rl.training.run_monitor import atomic_json
from .storage import Store,read_json,digest,now,clean_json
from .coverage_ai_search import pin
from .ai_search_runner import load_cases,result_key,recover_rows
from .ai_search_r2_runner import resource_guard
from .geoload_runner import exclusive_lock
from . import coverage_ai_search_r3 as policy,frozen_round_tasks

def method_order(index):
 base=list(policy.METHODS[:-1]);shift=index%len(base);order=base[shift:]+base[:shift];order.insert(order.index('REF20')+1,'ORACLE_RESTRICT');return order


def publication_accounting(method,base,provider,wrapper_seconds):
 total=None if base is None else base+provider+wrapper_seconds
 eligible=total is not None and (method in ('REF20','POOL') or total<=30)
 return {'core_comparable_seconds':base,'required_provider_seconds':provider,'wrapper_publication_seconds':wrapper_seconds,'comparable_publication_seconds':total,'regular_publication_eligible':eligible,'publication_state':'on_time_or_reference_budget' if eligible else 'late_or_unknown_candidate_retained_not_formal','endpoint':'row_case_index_status_visible','receipt_IO_not_recursively_charged':True}

def anchor_layout(reference):
 from satellite_rl.coverage.ai_search.core import Layout
 obj=read_json(pin(reference),{})
 for key in reference.get('json_pointer',['layout']):obj=obj[key]
 return Layout.from_record(obj)

def case_metadata(case,scene,job):return {'case_id':case['case_id'],'mother_id':case['mother_id'],'demand_profile':case['demand_profile'],'input_case_index':case['input_case_index'],'input_sha256':scene.input_sha256,'input_dataset_sha256':case['input_dataset_sha256'],'phase':'pilot','source_commit':job['scientific_commit'],'execution_commit':job['scientific_commit'],'mode':job['mode']}

def run_job(directory,resume=False):
 directory=Path(directory).resolve();job=read_json(directory/'job.json',{});config=read_json(directory/'config.json',{});status=read_json(directory/'status.json',{});done=threading.Event();lock=threading.RLock()
 def update(**kw):
  with lock:status.update(kw,pid=os.getpid(),heartbeat_at=now());atomic_json(directory/'status.json',status)
 def heartbeat():
  while not done.wait(10):update()
 def stopped():resource_guard(directory,config['resources']);return (directory/'stop.request').exists()
 def publish(row,key,index):
  start=time.perf_counter();path=directory/'cases'/(key+'.json');atomic_json(path,clean_json(row));index[key]={'path':str(path),'sha256':digest(path),'case_id':row['case_id'],'method':row['method'],'outcome':row['outcome']};atomic_json(directory/'case_index.json',index);update(completed_cases=len(index))
  wrapper_seconds=time.perf_counter()-start;base=row['result'].get('comparable_seconds');provider=row.get('required_provider_seconds',0.)
  receipt={'case_id':row['case_id'],'method':row['method'],'raw_row_ref':{'path':str(path),'sha256':digest(path)},**publication_accounting(row['method'],base,provider,wrapper_seconds)}
  cost=directory/'publication_costs'/(key+'.json');atomic_json(cost,receipt);index[key]['publication_cost_ref']={'path':str(cost),'sha256':digest(cost)}
  if row['method']!='POOL' and not receipt['regular_publication_eligible']:index[key]['outcome']='censored_publication'
  atomic_json(directory/'case_index.json',index)
 try:
  with exclusive_lock(directory/'worker.lock'),exclusive_lock(Path(job['archive_root']).parent/'ops/compute.lock',blocking=True):
   frozen_round_tasks.verify(Store(job['archive_root'],source_root=job['source_root']),job,policy)
   from .runner import strict_cuda_preflight
   atomic_json(directory/'preflight.json',strict_cuda_preflight(job));vendor=policy.validate_vendor(config['vendor_ref'],job['executor_port'])
   if any(os.environ.get(k)!='1' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')):raise ValueError('formal native thread env is not frozen1')
   (directory/'cases').mkdir(exist_ok=True);methods=job['methods'] or ['POOL'];ids=load_cases(config['datasets']['development'],job['case_indices']);expected={result_key(c['case_id'],m) for c in ids for m in methods};index=recover_rows(directory,expected)
   for key,ref in index.items():
    if 'publication_cost_ref' not in ref:
     cost=directory/'publication_costs'/(key+'.json')
     if not cost.exists():atomic_json(cost,{'case_id':ref['case_id'],'method':ref['method'],'raw_row_ref':{'path':ref['path'],'sha256':ref['sha256']},'comparable_publication_seconds':None,'wrapper_publication_seconds':None,'regular_publication_eligible':False,'unknown_publication_cost':True,'automatic_recompute_forbidden':True})
     c=read_json(cost,{})
     if c.get('raw_row_ref',{}).get('sha256')!=ref['sha256']:raise ValueError('publication receipt raw SHA changed')
     ref['publication_cost_ref']={'path':str(cost),'sha256':digest(cost)}
   update(stage='running',error=None,total_cases=len(expected),completed_cases=len(index),science_updates=0,case_workers=1);threading.Thread(target=heartbeat,daemon=True).start()
   from satellite_rl.coverage.ai_search.core import Scene
   from satellite_rl.coverage.ai_search_r3.pool import prepare_case,PreparedCase,PreparationIncomplete
   from satellite_rl.coverage.ai_search_r3.evaluate import evaluate_method
   from satellite_rl.coverage.ai_search_r3.projection import audit_projection
   from satellite_rl.coverage.ai_search_r3.teacher import build_teacher_label
   for case_id in job['case_indices']:
    if stopped():break
    read_start=time.perf_counter();case=load_cases(config['datasets']['development'],[case_id])[0];scene=Scene(case['positions'],case['demands'],case['aircraft_ids'],case['profile']);input_seconds=time.perf_counter()-read_start;meta=case_metadata(case,scene,job)
    if job['mode']=='prepare_candidates':
     key=result_key(case['case_id'],'POOL')
     if key in index:continue
     atomic_json(directory/'started'/(key+'.json'),{**meta,'method':'POOL','at':now(),'public_input_seconds':input_seconds,'unknown_work_preserved':True})
     try:
      p=prepare_case(scene,case['case_id'],stop=stopped,seconds=10.);poolfile=directory/'pools'/(key+'.npz');begin=time.perf_counter();p.save(poolfile);pool_write_seconds=time.perf_counter()-begin
      witness_path=directory/'witnesses'/(key+'.json');begin=time.perf_counter();atomic_json(witness_path,{'input_sha256':scene.input_sha256,'case_id':case['case_id'],'witness':p.witness.to_record(),'ncap':p.ncap,'omega_cap':p.omega_cap,'pool_content_sha256':p.content_sha,'ledger':{'witness':p.ledger['witness']}});witness_write_seconds=time.perf_counter()-begin;projections={}
      for name,ref in job['original_task_contract']['external_refs'][str(case_id)].items():
       if name!='audit':projections[name]={'source_ref':ref,**audit_projection(p,anchor_layout(ref))}
      value={'status':'prepared','pool_ref':{'path':str(poolfile),'sha256':digest(poolfile)},'pool_content_sha256':p.content_sha,'witness_ref':{'path':str(witness_path),'sha256':digest(witness_path)},'pool_cache_write_seconds':pool_write_seconds,'witness_cache_write_seconds':witness_write_seconds,'ledger':p.ledger,'public_input_seconds':input_seconds,'projection':projections,'projection_solver_calls':0,'new_solver_calls':0}
     except PreparationIncomplete as e:value={'status':'incomplete_preparation','reason':str(e),'ledger':e.ledger,'new_solver_calls':0,'automatic_repeat_forbidden':True}
     publish({**meta,'method':'POOL','outcome':'completed','result':value},key,index)
    else:
     manifest=read_json(pin(job['references']['prepared_manifest']),{});prepared=manifest['cases'][str(case_id)]
     if prepared['case_id']!=case['case_id'] or prepared['input_sha256']!=scene.input_sha256:raise ValueError('prepared identity differs')
     p=None;load_seconds=0;witness_p=None;witness_read_seconds=0
     if prepared['result']['status']=='prepared':
      start=time.perf_counter();p=PreparedCase.load(pin(prepared['result']['pool_ref']));load_seconds=time.perf_counter()-start
      if p.scene.input_sha256!=scene.input_sha256 or p.content_sha!=prepared['result']['pool_content_sha256']:raise ValueError('pool content changed')
      from satellite_rl.coverage.ai_search.core import Layout
      start=time.perf_counter();w=read_json(pin(prepared['result']['witness_ref']),{});witness_p=SimpleNamespace(scene=scene,case_id=case['case_id'],witness=Layout.from_record(w['witness']),ncap=w['ncap'],omega_cap=w['omega_cap'],ledger=w['ledger'],content_sha=w['pool_content_sha256']);witness_read_seconds=time.perf_counter()-start
      if w['input_sha256']!=scene.input_sha256 or witness_p.content_sha!=p.content_sha:raise ValueError('witness source changed')
     ref_record=None
     ref_key=result_key(case['case_id'],'REF20')
     if ref_key in index:ref_record=read_json(pin(index[ref_key]),{})['result']
     for ordinal,method in enumerate(method_order(case_id)):
      key=result_key(case['case_id'],method)
      if key in index:continue
      if stopped():break
      atomic_json(directory/'started'/(key+'.json'),{**meta,'method':method,'at':now(),'pool_ref':prepared['result'].get('pool_ref'),'method_order':method_order(case_id),'unknown_work_preserved':True})
      if p is None:value={'status':'preparation_unavailable','layout':None,'metrics':None,'solver_calls':[],'reason':prepared['result'].get('reason'),'actual_method_seconds':0,'comparable_seconds':None}
      else:
       label=None
       if method=='ORACLE_RESTRICT':
        begin=time.perf_counter();label=build_teacher_label(p,ref_record or {});label_path=directory/'labels'/(key+'.json');atomic_json(label_path,clean_json(label));label_seconds=time.perf_counter()-begin
       method_dir=directory/'methods'/key
       shared=p.ledger['witness']['seconds'] if method=='WITNESS' else p.ledger['seconds'];deadline=time.monotonic()+max(0,30-(prepared['result']['public_input_seconds']+shared)) if method!='REF20' else None
       stop=lambda:stopped() or (deadline is not None and time.monotonic()>=deadline)
       value=evaluate_method(witness_p if method=='WITNESS' else p,method,method_dir,oracle_label=label,vendor_dir=vendor['vendor'],stop=stop,public_input_seconds=prepared['result']['public_input_seconds'])
       if method=='ORACLE_RESTRICT':value['oracle_label_ref']={'path':str(label_path),'sha256':digest(label_path)};value['offline_label_derivation_seconds']=label_seconds
       if method=='REF20':ref_record=value
      row={**meta,'method':method,'outcome':'stopped' if (directory/'stop.request').exists() else 'completed','publication_state':'candidate_until_independent_cost_receipt','result':value,'prepared_ref':job['references']['prepared_manifest'],'pool_ref':prepared['result'].get('pool_ref'),'projection':prepared['result'].get('projection'),'method_order':method_order(case_id),'method_order_position':ordinal,'actual_pool_read_seconds':load_seconds,'actual_witness_read_seconds':witness_read_seconds,'required_provider_seconds':(witness_read_seconds+prepared['result'].get('witness_cache_write_seconds',0.)) if method=='WITNESS' else (load_seconds+prepared['result'].get('pool_cache_write_seconds',0.)),'physical_shared_preparation_once':True}
      publish(row,key,index)
   if (directory/'stop.request').exists():update(stage='stopped');return
   if set(index)!=expected:raise ValueError('frozen R3 rows incomplete')
   cases={}
   for r in index.values():
    row=read_json(pin(r),{});i=str(row['input_case_index']);entry=cases.setdefault(i,{k:row[k] for k in ('case_id','mother_id','demand_profile','input_case_index','input_sha256','input_dataset_sha256','phase','source_commit','execution_commit')});entry['role']='development';entry['scene_ref']={'packed':config['datasets']['development'],'case_index':row['input_case_index']};entry['external_refs']=job['original_task_contract']['external_refs'][i];entry.setdefault('raw_refs',{})[row['method']]={'path':r['path'],'sha256':r['sha256']};entry.setdefault('publication_cost_refs',{})[row['method']]=r['publication_cost_ref']
    if job['mode']=='prepare_candidates':entry['result']=row['result']
    else:entry['pool_ref']=row['pool_ref'];entry['prepared_ref']=row['prepared_ref']
   name='prepared_manifest.json' if job['mode']=='prepare_candidates' else 'result_manifest.json';path=directory/name;plan=read_json(pin(job['references']['plan']),{})
   manifest={'schema':'s1_ai19_case_manifest_v1','group':policy.GROUP,'phase':'pilot','completed':True,'cases':cases if job['mode']=='prepare_candidates' else list(cases.values()),'mother_order':plan['mother_order'],'case_indices':job['case_indices'],'source_commit':job['scientific_commit'],'execution_commit':job['scientific_commit'],'dataset_role':'development','identity_ref':job['references']['identity'],'qualification_refs':plan['qualification_refs'],'design_ref':job['references']['design'],'vendor_ref':job['references']['vendor'],'job_ref':{'path':str(directory/'job.json'),'sha256':digest(directory/'job.json')},'config_ref':{'path':str(directory/'config.json'),'sha256':digest(directory/'config.json')},'preflight_ref':{'path':str(directory/'preflight.json'),'sha256':digest(directory/'preflight.json')},'source_manifest_ref':{'path':str(Path(job['source_root'])/'source_manifest.json'),'sha256':job['source_manifest_sha256']},'science_updates':0,'gate_passed':None}
   atomic_json(path,manifest);atomic_json(directory/'artifacts.json',{'manifest':{'path':str(path),'sha256':digest(path)},'count':len(index),'science_updates':0});update(stage='completed',completed_at=now())
 except BaseException as e:
  atomic_json(directory/'failure.json',{'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc(),'unknown_work_preserved':True});update(stage='failed',error=str(e));raise
 finally:done.set()