"""P0 seven-method complete-case runner with immutable old input lineage."""
import os,sys,time,threading,traceback,hashlib,shutil,math
from pathlib import Path
from satellite_rl.training.run_monitor import atomic_json
from .storage import Store,read_json,digest,clean_json,now
from .coverage_ai_search import pin
from .ai_search_runner import result_key,recover_rows
from .geoload_runner import exclusive_lock
from .r5_stage_guard import run_stage
from .r5_io import identity,validate_output
from . import coverage_ai_search_r5 as policy,frozen_round_tasks

def method_order(case_index):
 methods=list(policy.METHODS);shift=case_index%7;return methods[shift:]+methods[:shift]

def cost_record(method,case,stage,publication_seconds,registry_read_seconds=0.):
 """Full restricted cold cost: original required work added once, current actual wall."""
 old=read_json(pin(case['teacher']['publication_cost']),{});components=old['components'];teacher_required=method.split('_',1)[1] in ('EXACT','OWNER10','JOINT10');input_seconds=components.get('input');prep_seconds=components.get('prepare');teacher_seconds=components.get('teacher') if teacher_required else 0.
 values=[input_seconds,prep_seconds,teacher_seconds,stage.get('actual_stage_wall_seconds'),publication_seconds,registry_read_seconds]
 known=all(isinstance(x,(int,float)) and math.isfinite(x) and x>=0 for x in values);unknown=not known or bool(old.get('unknown_cost') or old.get('unknown_algorithm_spend') or stage.get('unknown_algorithm_spend'));total=sum(values) if known else None
 return {'schema':'R5P0CostV1','complete_method_seconds':total,'complete_comparable_seconds':total,'components':{'input':input_seconds,'prepare':prep_seconds,'teacher':teacher_seconds,'new_registry_read':registry_read_seconds,'new_guarded_stage_wall':stage.get('actual_stage_wall_seconds'),'row_index_status_publication':publication_seconds},'unknown_cost':unknown,'unknown_algorithm_spend':unknown,'old_cold_cost_ref':case['teacher']['publication_cost'],'old_prepare_ref':case['prepared']['stage_output'],'old_teacher_ref':case['teacher']['stage_output'] if teacher_required else None,'old_unknown_failure_and_extra_operations_separate':True,'reused_asset_not_free_cold_pipeline':True,'physical_shared_work_separate':True,'restricted_full_cost_not_strict_cross_epoch_speed':True}

def stage_value(stage,request):
 if stage.get('hard_interface_error'):raise ValueError('hard stage interface/monitor error; successors stopped')
 if stage['status']!='completed':return {'schema':'DecoderResultV2' if request['method'].startswith('D5_') else 'DecoderResultV1','status':'unpublished','reason':stage['reason'],'unknown_cost':stage.get('unknown_algorithm_spend',True),'no_old_W_success_credit':True},{}
 child=read_json(pin(stage['output_ref']),{});return validate_output(child,request),child

def run_job(directory,resume=False):
 directory=Path(directory).resolve();job=read_json(directory/'job.json',{});config=read_json(directory/'config.json',{});status=read_json(directory/'status.json',{});done=threading.Event();lock=threading.RLock()
 def update(**kw):
  with lock:status.update(kw,pid=os.getpid(),heartbeat_at=now());atomic_json(directory/'status.json',status)
 def heartbeat():
  while not done.wait(10):update()
 def stopped():
  if shutil.disk_usage(directory).free<config['resources']['minimum_free_bytes']:raise OSError('P0 disk reserve')
  return (directory/'stop.request').exists()
 try:
  with exclusive_lock(directory/'worker.lock'),exclusive_lock(Path(job['archive_root']).parent/'ops/compute.lock',blocking=True):
   frozen_round_tasks.verify(Store(job['archive_root'],source_root=job['source_root']),job,policy)
   from .runner import strict_cuda_preflight
   atomic_json(directory/'preflight.json',strict_cuda_preflight(job));registry_start=time.perf_counter();registry=read_json(pin(job['references']['registry']),{});registry_read_seconds=time.perf_counter()-registry_start;by_index={c['input_case_index']:c for c in registry['cases']};expected={result_key(by_index[i]['case_id'],m) for i in job['case_indices'] for m in policy.METHODS};(directory/'cases').mkdir(exist_ok=True);index=recover_rows(directory,expected)
   # A recovered immutable row is retained. Missing cost endpoint is unknown, never rerun.
   for key,ref in index.items():
    cost_path=directory/'publication_costs'/(key+'.json')
    if not cost_path.exists():atomic_json(cost_path,{'schema':'R5P0CostV1','case_id':ref['case_id'],'method':ref['method'],'complete_method_seconds':None,'complete_comparable_seconds':None,'unknown_cost':True,'unknown_algorithm_spend':True,'raw_ref':{'path':ref['path'],'sha256':ref['sha256']},'reason':'recovered_row_missing_cost_endpoint','no_algorithm_replay':True})
    cost=read_json(cost_path,{})
    if cost['raw_ref']['sha256']!=ref['sha256']:raise ValueError('recovered cost/raw changed')
    ref['publication_cost_ref']={'path':str(cost_path),'sha256':digest(cost_path)}
   update(stage='running',completed_cases=len(index),total_cases=len(expected),science_updates=0,MILP_calls=0,new_prepare_calls=0,new_teacher_calls=0);threading.Thread(target=heartbeat,daemon=True).start()
   for case_index in job['case_indices']:
    case=by_index[case_index];ident=identity(case)
    for method in method_order(case_index):
     key=result_key(case['case_id'],method)
     if key in index:continue
     if stopped():break
     wrapper_start=time.perf_counter()
     target=directory/'bundles'/hashlib.sha256(case['case_id'].encode()).hexdigest()/method;target.mkdir(parents=True,exist_ok=True);request={'stage':'decoder','method':method,'registry_case':case,'source_sha256':job['source_manifest_sha256'],'contract_sha256':policy.DESIGN_SHA};request_path=target/'request.json'
     if request_path.exists() and read_json(request_path,{})!=request:raise ValueError('original request changed')
     if not request_path.exists():atomic_json(request_path,request)
     started=directory/'started'/(key+'.json')
     if not started.exists():atomic_json(started,{'case_id':case['case_id'],'method':method,'request_ref':{'path':str(request_path),'sha256':digest(request_path)},'at':now()})
     stage=run_stage([sys.executable,'-B','-u','-m','satellite_rl.workbench.ai_search_r5_stage','--request',str(request_path),'--directory',str(target)],target,15,2147483648,stopped);value,child=stage_value(stage,request)
     row={**ident,'schema':'R5P0RowV1','input_case_index':case_index,'selected_case_ordinal':case['selected_case_ordinal'],'demand_profile':case['profile'],'input_dataset_sha256':policy.DATA_SHA,'phase':'P0','mode':'p0','method':method,'decoder':method.split('_',1)[0],'proposal_method':method.split('_',1)[1],'source_commit':job['scientific_commit'],'execution_commit':job['scientific_commit'],'source_sha256':job['source_manifest_sha256'],'outcome':value['status'],'result':value,'stage_guard':stage,'method_order':method_order(case_index),'original_prepared_ref':case['prepared']['stage_output'],'original_teacher_ref':case['teacher']['stage_output'],'registry_ref':job['references']['registry'],'registry_case':case,'scene_ref':case['scene_source'],'proposal_ref':child.get('proposal_ref'),'derived_envelope_ref':child.get('derived_envelope_ref'),'decoder_invoked':child.get('decoder_invoked',False),'child_measured_costs':{k:v for k,v in child.items() if k.startswith('actual_')},'unknown_historical_operations':registry['historical_unknowns']}
     wrapper_prepublication_seconds=time.perf_counter()-wrapper_start-stage['actual_stage_wall_seconds']
     publication_start=time.perf_counter();path=directory/'cases'/(key+'.json');atomic_json(path,clean_json(row));index[key]={'case_id':case['case_id'],'method':method,'path':str(path),'sha256':digest(path),'outcome':value['status']};atomic_json(directory/'case_index.json',index);update(completed_cases=len(index));publication=time.perf_counter()-publication_start;cost=cost_record(method,case,stage,publication+wrapper_prepublication_seconds,registry_read_seconds);cost.update(raw_ref={'path':str(path),'sha256':digest(path)},case_id=case['case_id'],method=method,child_measured_costs=row['child_measured_costs']);cost_path=directory/'publication_costs'/(key+'.json');finalize_start=time.perf_counter();atomic_json(cost_path,cost);index[key]['publication_cost_ref']={'path':str(cost_path),'sha256':digest(cost_path)};atomic_json(directory/'case_index.json',index);update(completed_cases=len(index));finalize_seconds=time.perf_counter()-finalize_start;atomic_json(directory/'publication_endpoints'/(key+'.json'),{'schema':'R5PublicationEndpointV1','raw_ref':cost['raw_ref'],'cost_ref':index[key]['publication_cost_ref'],'actual_cost_index_status_finalization_seconds':finalize_seconds,'complete_method_seconds_with_endpoint':None if cost['complete_method_seconds'] is None else cost['complete_method_seconds']+finalize_seconds,'unknown_cost':cost['unknown_cost'],'endpoint_receipt_IO_separate_metadata':True})
    if stopped():break
   if (directory/'stop.request').exists():update(stage='stopped');return
   if set(index)!=expected:raise ValueError('P0 exact result slots incomplete')
   cases={}
   for ref in index.values():
    row=read_json(pin(ref),{});entry=cases.setdefault(str(row['input_case_index']),{k:row[k] for k in ('case_id','mother_id','profile','role','input_sha256','identity_sha256','input_case_index','selected_case_ordinal','scene_ref','registry_case')});entry.setdefault('raw_refs',{})[row['method']]={'path':ref['path'],'sha256':ref['sha256']};entry.setdefault('publication_cost_refs',{})[row['method']]=ref['publication_cost_ref'];endpoint=directory/'publication_endpoints'/(result_key(row['case_id'],row['method'])+'.json');entry.setdefault('publication_endpoint_refs',{})[row['method']]={'path':str(endpoint),'sha256':digest(endpoint)} if endpoint.exists() else None;entry['prepared_ref']=row['original_prepared_ref'];entry['teacher_ref']=row['original_teacher_ref']
   plan=read_json(pin(job['references']['plan']),{});manifest={'schema':'R5P0ResultManifestV1','group':policy.GROUP,'phase':'P0','complete_result_slots':len(index),'cases':list(cases.values()),'case_indices':job['case_indices'],'mother_order':plan['mother_order'],'source_commit':job['scientific_commit'],'job_ref':{'path':str(directory/'job.json'),'sha256':digest(directory/'job.json')},'config_ref':{'path':str(directory/'config.json'),'sha256':digest(directory/'config.json')},'source_manifest_ref':{'path':str(Path(job['source_root'])/'source_manifest.json'),'sha256':job['source_manifest_sha256']},'identity_ref':job['references']['identity'],'registry_ref':job['references']['registry'],'design_ref':job['references']['design'],'qualification_refs':{k:job['references']['qualification_'+k] for k in ('geometry','process')},'MILP_calls':0,'training_updates':0,'new_prepare_calls':0,'new_teacher_calls':0,'gates_passed':None,'future_stage_dispatch_allowed':False};path=directory/'result_manifest.json';atomic_json(path,manifest);atomic_json(directory/'artifacts.json',{'result_manifest':{'path':str(path),'sha256':digest(path)}});update(stage='completed',completed_at=now())
 except BaseException as error:atomic_json(directory/'failure.json',{'type':type(error).__name__,'message':str(error),'traceback':traceback.format_exc(),'unknown_spend_preserved':True});update(stage='failed',error=str(error));raise
 finally:done.set()
