"""Durable B0 case bundles; prepare+one teacher+four decoder result slots."""
import os,sys,json,time,threading,traceback,hashlib,shutil
from pathlib import Path
import numpy as np
from satellite_rl.training.run_monitor import atomic_json
from .storage import Store,read_json,digest,clean_json,now
from .coverage_ai_search import pin
from .ai_search_runner import load_cases,result_key,recover_rows
from .geoload_runner import exclusive_lock
from .r4_stage_guard import run_stage
from . import coverage_ai_search_r4 as policy,frozen_round_tasks

def method_order(ordinal):
 methods=list(policy.METHODS);shift=ordinal%4;return methods[shift:]+methods[:shift]

def check_case_identity(case,case_index,identities):
 for field,key in (('case_id','case_ids'),('mother_id','mother_ids'),('demand_profile','profiles')):
  if case.get(field)!=identities[key][case_index]:raise ValueError('loaded development '+field+' differs from frozen identity')
 if case.get('input_dataset_sha256')!=identities['sha256']:raise ValueError('loaded development dataset SHA differs from frozen identity')
 return case

def tagged(stage,kind,identity=None,source_sha256=None):
 if stage['status']!='completed':return {'status':'unavailable','reason':stage['reason'],'unknown_cost':stage.get('unknown_algorithm_spend',True),'evidence_ref':stage.get('output_ref')}
 value=read_json(pin(stage['output_ref']),{})['result'];expected=('available','unavailable')if kind in ('prepared','teacher')else ('success','fallback','unavailable','unpublished')
 if not isinstance(value,dict)or value.get('status')not in expected:raise ValueError('invalid '+kind+' union discriminator')
 if identity is not None:
  from .r4_io import check_record
  check_record(value,kind,identity,source_sha256)
 return value

def execute_bundle(ordinal,perform,emit,identity=None,source_sha256=None):
 """Testable orchestration only. perform is guarded and never an implicit retry."""
 prep=perform('preparation',None);prepared=tagged(prep,'prepared',identity,source_sha256)
 if prep.get('hard_interface_error'):raise ValueError('hard preparation source/data error')
 if prepared['status']=='available':
  teacher_stage=perform('teacher',None);teacher=tagged(teacher_stage,'teacher',identity,source_sha256)
  if teacher_stage.get('hard_interface_error'):raise ValueError('hard teacher source/data error')
 else:teacher_stage={'status':'not_invoked','actual_stage_wall_seconds':0.,'unknown_algorithm_spend':prep.get('unknown_algorithm_spend',False)};teacher={'status':'unavailable','reason':'preparation_unavailable','cost':{'teacher_not_invoked':True}}
 emit('TEACHER',teacher,teacher_stage,prep,teacher_stage)
 for method in method_order(ordinal):
  diagnostic=method.startswith('TEACHER_')
  if prepared['status']!='available':stage={'status':'not_invoked','actual_stage_wall_seconds':0.,'unknown_algorithm_spend':prep.get('unknown_algorithm_spend',False)};value={'status':'unavailable','reason':'preparation_unavailable','decoder_invoked':False}
  elif diagnostic and teacher['status']!='available':stage={'status':'not_invoked','actual_stage_wall_seconds':0.,'unknown_algorithm_spend':teacher_stage.get('unknown_algorithm_spend',False)};value={'status':'unavailable','reason':'unavailable_teacher','decoder_invoked':False}
  else:
   stage=perform('decoder',method);value=tagged(stage,'decoder',identity,source_sha256)
   if stage.get('hard_interface_error'):raise ValueError('hard proposal/decoder source/data error')
  emit(method,value,stage,prep,teacher_stage)
 return {'prepared_status':prepared['status'],'teacher_status':teacher['status']}

def cost_record(method,input_seconds,prep,teacher,stage,publication_seconds):
 prep_seconds=prep.get('actual_stage_wall_seconds');teacher_required=method=='TEACHER'or method.startswith('TEACHER_');teacher_seconds=teacher.get('actual_stage_wall_seconds')if teacher_required else 0.;stage_seconds=0. if method=='TEACHER'else stage.get('actual_stage_wall_seconds');unknown=bool(prep.get('unknown_algorithm_spend')or (teacher_required and teacher.get('unknown_algorithm_spend'))or stage.get('unknown_algorithm_spend'))
 known=all(x is not None for x in [input_seconds,prep_seconds,teacher_seconds,stage_seconds,publication_seconds]);total=input_seconds+prep_seconds+teacher_seconds+stage_seconds+publication_seconds if known else None
 return {'complete_method_seconds':total,'components':{'input':input_seconds,'prepare':prep_seconds,'teacher':teacher_seconds,'proposal':None,'decoder':stage_seconds,'wrapper':publication_seconds},'unknown_algorithm_spend':unknown or not known,'input_read_seconds':input_seconds,'common_preparation_once_seconds':prep_seconds,'teacher_once_seconds':teacher_seconds,'proposal_decoder_wall_seconds':stage_seconds,'row_index_status_publication_seconds':publication_seconds,'complete_comparable_seconds':total,'unknown_cost':unknown or not known,'algorithm_budget_is_nominal_not_R3_30sec_eligibility':True,'physical_shared_work_separate':True}

def run_job(directory,resume=False):
 directory=Path(directory).resolve();job=read_json(directory/'job.json',{});config=read_json(directory/'config.json',{});status=read_json(directory/'status.json',{});done=threading.Event();lock=threading.RLock()
 def update(**kw):
  with lock:status.update(kw,pid=os.getpid(),heartbeat_at=now());atomic_json(directory/'status.json',status)
 def heartbeat():
  while not done.wait(10):update()
 def stopped():
  if shutil.disk_usage(directory).free<config['resources']['minimum_free_bytes']:raise OSError('B0 disk reserve')
  return (directory/'stop.request').exists()
 try:
  with exclusive_lock(directory/'worker.lock'),exclusive_lock(Path(job['archive_root']).parent/'ops/compute.lock',blocking=True):
   frozen_round_tasks.verify(Store(job['archive_root'],source_root=job['source_root']),job,policy)
   from .runner import strict_cuda_preflight
   atomic_json(directory/'preflight.json',strict_cuda_preflight(job));plan=read_json(pin(job['references']['plan']),{});identities=read_json(pin(job['references']['identity']),{})['development'];selected_ord={i:n for n,i in enumerate(plan['selected_case_indices'])};(directory/'cases').mkdir(exist_ok=True)
   expected={result_key(identities['case_ids'][i],m)for i in job['case_indices']for m in ('TEACHER',*policy.METHODS)};index=recover_rows(directory,expected)
   for key,reference in index.items():
    if 'publication_cost_ref'not in reference:
     cost_path=directory/'publication_costs'/(key+'.json')
     if not cost_path.exists():atomic_json(cost_path,{'case_id':reference['case_id'],'method':reference['method'],'complete_method_seconds':None,'components':{'input':None,'prepare':None,'teacher':None,'proposal':None,'decoder':None,'wrapper':None},'unknown_cost':True,'unknown_algorithm_spend':True,'reason':'atomic_row_recovered_without_publication_endpoint_cost','raw_ref':{'path':reference['path'],'sha256':reference['sha256']},'no_algorithm_replay':True})
     cost=read_json(cost_path,{})
     if cost['raw_ref']['sha256']!=reference['sha256']:raise ValueError('recovered cost/raw SHA changed')
     reference['publication_cost_ref']={'path':str(cost_path),'sha256':digest(cost_path)}
   update(stage='running',completed_cases=len(index),total_cases=len(expected),science_updates=0,MILP_calls=0,case_workers=1);threading.Thread(target=heartbeat,daemon=True).start()
   for case_index in job['case_indices']:
    if stopped():break
    keys={m:result_key(identities['case_ids'][case_index],m)for m in ('TEACHER',*policy.METHODS)}
    if all(k in index for k in keys.values()):continue
    begin=time.perf_counter();case=load_cases(config['datasets']['development'],[case_index])[0];check_case_identity(case,case_index,identities);case_dir=directory/'bundles'/hashlib.sha256(case['case_id'].encode()).hexdigest();case_dir.mkdir(parents=True,exist_ok=True);scene_path=case_dir/'raw_scene.npz'
    if not scene_path.exists():np.savez_compressed(scene_path,positions=case['positions'],demands=case['demands'],aircraft_ids=np.asarray(case['aircraft_ids']),profile=case['profile'])
    from satellite_rl.coverage.ai_search.core import Scene
    scene=Scene(case['positions'],case['demands'],case['aircraft_ids'],case['profile']);input_seconds=time.perf_counter()-begin;identity={'case_id':case['case_id'],'mother_id':case['mother_id'],'profile':case['demand_profile'],'input_sha256':scene.input_sha256,'identity_sha256':job['references']['identity']['sha256'],'role':'development'};phase_refs={}
    public_input_cost=case_dir/'input_read_receipt.json'
    if public_input_cost.exists():input_seconds=read_json(public_input_cost,{})['actual_seconds']
    else:atomic_json(public_input_cost,{'actual_seconds':input_seconds,'raw_scene_ref':{'path':str(scene_path),'sha256':digest(scene_path)},'case_id':case['case_id'],'input_sha256':scene.input_sha256})
    def perform(stage,method):
     phase=stage if method is None else method;target=case_dir/phase;target.mkdir(exist_ok=True);request_path=target/'request.json';request={'stage':stage,'identity':identity,'source_sha256':job['source_manifest_sha256'],'scene_path':str(scene_path)}
     if stage!='preparation':request['prepared_path']=phase_refs['preparation']['output_ref']['path']
     if stage=='decoder':
      request['method']=method
      if method.startswith('TEACHER_'):request['teacher_path']=phase_refs['teacher']['output_ref']['path']
     if request_path.exists()and read_json(request_path,{})!=request:raise ValueError('original stage request changed')
     if not request_path.exists():atomic_json(request_path,request)
     # Save the method boundary before any child algorithm; all unknown stages fail closed.
     method_key=keys.get(method or ('TEACHER'if stage=='teacher'else''))
     if method_key and method_key not in index and not (directory/'started'/(method_key+'.json')).exists():atomic_json(directory/'started'/(method_key+'.json'),{'case_id':case['case_id'],'method':method or'TEACHER','stage':stage,'request_ref':{'path':str(request_path),'sha256':digest(request_path)},'at':now()})
     r=run_stage([sys.executable,'-B','-u','-m','satellite_rl.workbench.ai_search_r4_stage','--request',str(request_path),'--directory',str(target)],target,config['hard_watchdogs'][stage],config['resources']['parent_children_rss_bytes'],stopped);phase_refs[stage if method is None else method]=r;return r
    def emit(method,value,stage,prep,teacher):
     key=keys[method]
     if key in index:return
     row={**identity,'input_case_index':case_index,'selected_case_ordinal':selected_ord[case_index],'demand_profile':case['demand_profile'],'input_dataset_sha256':case['input_dataset_sha256'],'phase':'B0','mode':'b0','method':method,'source_commit':job['scientific_commit'],'execution_commit':job['scientific_commit'],'outcome':value['status'],'result':value,'stage_guard':stage,'method_order':method_order(selected_ord[case_index]),'input_read_receipt_ref':{'path':str(public_input_cost),'sha256':digest(public_input_cost)},'scene_ref':{'packed':config['datasets']['development'],'case_index':case_index},'prepared_stage_ref':prep.get('output_ref'),'teacher_stage_ref':teacher.get('output_ref')if method=='TEACHER'or method.startswith('TEACHER_')else None,'proposal_ref':read_json(pin(stage['output_ref']),{}).get('proposal_ref')if stage.get('status')=='completed'and method!='TEACHER'else None,'decoder_invoked':read_json(pin(stage['output_ref']),{}).get('decoder_invoked',False)if stage.get('status')=='completed'else False}
     start=time.perf_counter();path=directory/'cases'/(key+'.json');atomic_json(path,clean_json(row));index[key]={'case_id':case['case_id'],'method':method,'path':str(path),'sha256':digest(path),'outcome':row['outcome']};atomic_json(directory/'case_index.json',index);update(completed_cases=len(index));publication=time.perf_counter()-start;cost=cost_record(method,input_seconds,prep,teacher,stage,publication);child=read_json(pin(stage['output_ref']),{})if stage.get('status')=='completed'and stage.get('output_ref')else{}
     if method=='TEACHER'or stage.get('status')=='not_invoked':cost['components'].update(proposal=0.,decoder=0.)
     elif stage.get('status')=='completed':
      proposal_seconds=child['actual_proposal_generation_seconds'];cost['components'].update(proposal=proposal_seconds,decoder=stage['actual_stage_wall_seconds']-proposal_seconds)
     cost.update(raw_ref={'path':str(path),'sha256':digest(path)},case_id=case['case_id'],method=method);cost_path=directory/'publication_costs'/(key+'.json');atomic_json(cost_path,cost);index[key]['publication_cost_ref']={'path':str(cost_path),'sha256':digest(cost_path)};atomic_json(directory/'case_index.json',index)
    execute_bundle(selected_ord[case_index],perform,emit,identity,job['source_manifest_sha256'])
   if (directory/'stop.request').exists():update(stage='stopped');return
   if set(index)!=expected:raise ValueError('B0120slots incomplete')
   cases={}
   for ref in index.values():
    row=read_json(pin(ref),{});k=str(row['input_case_index']);entry=cases.setdefault(k,{k:row[k]for k in ('case_id','mother_id','profile','role','input_sha256','identity_sha256','input_case_index','selected_case_ordinal','scene_ref')});entry.setdefault('raw_refs',{})[row['method']]={'path':ref['path'],'sha256':ref['sha256']};entry.setdefault('publication_cost_refs',{})[row['method']]=ref['publication_cost_ref']
   manifest={'schema':'R4B0ResultManifestV1','group':policy.GROUP,'phase':'B0','complete_result_slots':len(index),'cases':list(cases.values()),'case_indices':job['case_indices'],'mother_order':plan['mother_order'],'source_commit':job['scientific_commit'],'job_ref':{'path':str(directory/'job.json'),'sha256':digest(directory/'job.json')},'config_ref':{'path':str(directory/'config.json'),'sha256':digest(directory/'config.json')},'source_manifest_ref':{'path':str(Path(job['source_root'])/'source_manifest.json'),'sha256':job['source_manifest_sha256']},'identity_ref':job['references']['identity'],'design_ref':job['references']['design'],'qualification_ref':job['references']['qualification_geometry'],'MILP_calls':0,'training_updates':0,'gates_passed':None};path=directory/'result_manifest.json';atomic_json(path,manifest);atomic_json(directory/'artifacts.json',{'result_manifest':{'path':str(path),'sha256':digest(path)}});update(stage='completed',completed_at=now())
 except BaseException as error:atomic_json(directory/'failure.json',{'type':type(error).__name__,'message':str(error),'traceback':traceback.format_exc(),'unknown_spend_preserved':True});update(stage='failed',error=str(error));raise
 finally:done.set()
