"""One durable learner; no task retries or gate-driven model polling."""
import os,time,json,traceback,shutil
from pathlib import Path
import torch
from satellite_rl.training.run_monitor import atomic_json
from .storage import read_json,digest,now
from .coverage_ai_search import pin
from .geoload_runner import exclusive_lock
from .r7_model_io import signed_json,qualified_rows,load_feature,load_cached_target,validate_cache,verify_model_source_equivalence
from .r6_io import pin_tree

MODEL_OWNER_ROOT=Path("/home/u2024110224/first_stage_satellite_rl/s1_ai17/round7_model_ownership")

def train(job,config,directory):
 context={}
 try:return _train(job,config,directory,context)
 finally:
  session=context.get('session')
  if session is not None and session.active_started is not None:
   session.pause_activity(reason='resource_queue')
   directory=Path(directory)
   atomic_json(directory/'user_restart_activity_pause.json',dict(reason='user_restart_0700' if (directory/'stop.request').exists() else 'resource_withdrawal_or_failed_worker',core_pause_reason='resource_queue',
               charged_active_seconds=session.elapsed(),updates=session.updates,checkpoint_index=read_json(directory/'checkpoint_index.json',{}),optimizer_completion=read_json(directory/'optimizer_completion.json',{}),
               requires_root_hashed_verified_clock_recovery=True,no_automatic_resume=True,not_a_scientific_node_gate=True))

def _train(job,config,directory,activity_context):
 from .r7_model_limits import check
 from satellite_rl.coverage.ai_search_r6.runtime_v3 import assert_v3_runtime
 from satellite_rl.coverage.ai_search_r6.model import initialize_model,model_tensor_sha
 from satellite_rl.coverage.ai_search_r6.features import FeatureStats,record_sha
 from satellite_rl.coverage.ai_search_r6.learning import CaseSampler,TrainingSession
 setup_began=time.monotonic();directory=Path(directory);model_directory=Path(job["model_dir"]);model_directory.mkdir(parents=True,exist_ok=True);refs=job["model_inputs"];plan=signed_json(job["references"]["plan"]);manifest,rows=qualified_rows(refs["qualified"],refs["labels_gate"]);initial=signed_json(refs["initialization_receipt"])
 from .r7_train_binding import validate_setup,validate_owner
 setup_ref=refs['train_setup_endpoint'];setup_endpoint=signed_json(setup_ref)
 gate=signed_json(plan['prerequisite_gate_ref'])
 if job['mode']=='train2000':
  if gate.get('phase')!='train_setup' or gate.get('passed')is not True or gate.get('model_identity')!=job['model_identity'] or gate.get('train_setup_endpoint_ref')!=setup_ref:raise ValueError('actual plan prerequisite exact original setup endpoint')
 elif job['mode']=='train6000':
  from .r7_successor_binding import gate_requirements,continuation
  gate_requirements('train6000',gate,job['model_identity'],refs['checkpoint'])
  if refs['node_gate']!=plan['prerequisite_gate_ref'] or continuation(signed_json(refs['train2000_endpoint']),refs,job['model_identity'],job['case_indices'],config['datasets']['train'])!=job['model_dir']:raise ValueError('same original learner root and2000 checkpoint/node gate')
 else:raise ValueError('two original training segments only')
 complete_setup_charge=validate_setup(setup_endpoint,setup_ref,refs,job['model_identity'],job['executor_port'],job['case_indices'],config['datasets']['train'])
 if initial.get("passed")is not True or not initial.get("model_initial_state_SHA") or not initial.get("model_graph_source_SHA") or not initial.get("feature_graph_source_SHA"):raise ValueError("root accepted engineering initialization qualification")
 runtime=signed_json({'path':str(directory/'runtime_v3_identity.json'),'sha256':digest(directory/'runtime_v3_identity.json')})
 if runtime!=initial['training_runtime_identity']:raise ValueError('qualification/training/resume actual GPU/library/runtime identity differs')
 if runtime['runtime_profile_SHA']!=initial['runtime_profile_SHA']:raise ValueError('engineering/runtime normative profile differs')
 assert_v3_runtime(runtime)
 actual_sources=verify_model_source_equivalence(refs["source_equivalence"],refs["initialization_receipt"],initial)
 model_sources={k:actual_sources[k] for k in ("model_graph_source_SHA","feature_graph_source_SHA")};stats=FeatureStats.from_record(signed_json(refs["normalization"]));cache=signed_json(refs["feature_cache"]);supply_charge=validate_cache(cache,stats,rows,model_sources)
 supply_endpoint=signed_json(refs['supply_endpoint'])
 if supply_endpoint['cache_ref']!=refs['feature_cache'] or supply_endpoint['normalization_ref']!=refs['normalization'] or supply_endpoint['qualified_ref']!=refs['qualified']:raise ValueError('complete supply endpoint exact source refs')
 endpoint_charge=supply_endpoint['charged_supply_seconds_including_publication']
 if not isinstance(endpoint_charge,(int,float)) or not 0<=supply_charge<=endpoint_charge:raise ValueError('complete supply charge missing')
 if complete_setup_charge<endpoint_charge or setup_endpoint['runtime_identity']!=runtime:raise ValueError('complete setup clock/runtime origin differs')
 supply_charge=float(complete_setup_charge)
 if [r["case_index"] for r in rows]!=job["case_indices"]:raise ValueError("qualified indices exact signed plan")
 if refs["initial_state"]["sha256"]!=initial["model_initial_state_file_SHA"]:raise ValueError("original unupdated initialization")
 provenance={"model_identity":job["model_identity"],"qualified_table_sha":record_sha(rows),"normalization_SHA":stats.sha,"model_initial_state_SHA":initial["model_initial_state_SHA"],"model_initial_state_file_SHA":refs["initial_state"]["sha256"],**model_sources,"model_scientific_source_commit":initial["model_scientific_source_commit"],"training_seed":51001,"runtime_profile_SHA":runtime["runtime_profile_SHA"],"training_runtime_identity_SHA":record_sha(runtime),"train_setup_endpoint_SHA":setup_ref['sha256'],"setup_training_refs_SHA":record_sha(setup_endpoint['training_refs']),"setup_activity_origin_SHA":record_sha(setup_endpoint['activity_origin'])}
 start=config["training"]["start_update"];end=config["training"]["end_update"]
 owner_root=MODEL_OWNER_ROOT;owner_root.mkdir(parents=True,exist_ok=True);owner_key=record_sha(job["model_identity"]);owner_path=owner_root/(owner_key+".json")
 with exclusive_lock(owner_root/(owner_key+".lock")):
  owner=read_json(owner_path,{})
  validate_owner(owner,setup_endpoint,setup_ref)
  if owner.get("executor_port") is not None and owner.get("executor_port")!=job["executor_port"]:raise ValueError("single model GPU owner changed")
  if owner and owner.get("model_identity")!=job["model_identity"]:raise ValueError("another model owns global budget")
  owner=owner or {"model_identity":job["model_identity"],"executor_port":job["executor_port"],"science_updates_reserved":0,"last_completed_update":0,"initial_state_ref":refs["initial_state"]};owner.setdefault("science_updates_reserved",0);owner.setdefault("last_completed_update",0);owner["executor_port"]=job["executor_port"]
  if owner.get("provenance_sha",record_sha(provenance))!=record_sha(provenance):raise ValueError("same model identity changed label/normalization/initialization source")
  owner.setdefault("initial_state_ref",refs["initial_state"]);owner["provenance_sha"]=record_sha(provenance);owners=owner.setdefault("segment_runs",{})
  if owners.get(job["mode"],job["run_id"])!=job["run_id"]:raise ValueError("same model segment already has another run owner; no retry as new task")
  owners[job["mode"]]=job["run_id"];atomic_json(owner_path,owner)
  if start==0:
   if owner.get('original_learner_run_id',job['run_id'])!=job['run_id'] or owner.get('model_dir',job['model_dir'])!=job['model_dir']:raise ValueError('original learner root already owned')
   owner.update(original_learner_run_id=job['run_id'],model_dir=job['model_dir']);atomic_json(owner_path,owner)
  elif owner.get('original_learner_run_id')!=signed_json(refs['train2000_endpoint'])['run_id'] or owner.get('model_dir')!=job['model_dir']:raise ValueError('continuation cannot acquire a fresh learner root')
  model=initialize_model();model.load_state_dict(torch.load(pin(refs["initial_state"]),map_location='cpu',weights_only=True),strict=True)
  if model_tensor_sha(model)!=initial["model_initial_state_SHA"]:raise ValueError("initial tensor identity mismatch")
  sampler=CaseSampler(rows);session=TrainingSession(model,stats,sampler,provenance,'cuda');clock=directory/'activity_clock.json';activity_context['session']=session
  checkpoint_index=read_json(directory/'checkpoint_index.json',{});resume_ref=None;recovery=None
  if checkpoint_index:
   receipt=signed_json({"path":str(directory/'root_recovery.json'),"sha256":read_json(directory/'root_recovery_pin.json',{})['sha256']})
   if receipt.get("run_id")!=job["run_id"] or receipt.get("model_identity")!=job["model_identity"]:raise ValueError("explicit same-ID root clock recovery required")
   resume_ref=receipt.get('checkpoint_ref',checkpoint_index['latest']);recovery=receipt['recovery_clock'];pin(resume_ref)
   if resume_ref!=checkpoint_index['latest'] and receipt.get('original_checkpoint_index_sha256')!=digest(directory/'checkpoint_index.json'):raise ValueError('root orphan-state recovery does not bind original checkpoint index')
  elif start:
   resume_ref=refs['checkpoint'];recovery_record=signed_json(refs['recovery_clock']);recovery=dict(recovery_record['recovery_clock'])
   if recovery_record.get('train2000_endpoint_ref')!=refs['train2000_endpoint'] or recovery_record.get('checkpoint_ref')!=resume_ref or recovery_record.get('model_identity')!=job['model_identity']:raise ValueError('root original2000 checkpoint clock continuation evidence')
   if owner['last_completed_update']!=2000 or owner['science_updates_reserved']!=2000 or owner.get('last_completed_checkpoint_ref',{}).get('sha256')!=resume_ref['sha256']:raise ValueError('continuation original global2000 ledger/state; no unknown replay')
   state=torch.load(pin(resume_ref),map_location='cpu',weights_only=False)
   if state.get('updates')!=2000 or state.get('provenance')!=provenance:raise ValueError('exact original 2000 checkpoint chain')
  elif (directory/'step_intent.json').exists() or owner['last_completed_update']!=0:raise ValueError('unknown or previously completed learner; no blind reinitialization')
  if resume_ref:
   saved=torch.load(pin(resume_ref),map_location='cpu',weights_only=False)
   saved_provenance=saved['provenance']
   if start and saved_provenance.get('node2000_gate_pass'):
    if {k:v for k,v in saved_provenance.items() if k not in ('node2000_gate_pass','node2000_gate_sha256')}!=provenance or saved_provenance.get('node2000_gate_sha256')!=refs['node_gate']['sha256']:raise ValueError('continued provenance gate changed')
    session.provenance=saved_provenance
   recovery=dict(recovery);original_clock=recovery['clock_path'];recovery['clock_path']=str(clock)
   atomic_json(directory/'recovery_clock_destination.json',dict(original_root_recovery_clock_path=original_clock,new_phase_clock_path=str(clock),original_recovery_clock=recovery,original_checkpoint_ref=resume_ref,prior_phase_clock_not_overwritten=True))
   assert_v3_runtime(runtime);session.restore(pin(resume_ref),recovery_clock=recovery)
  else:session.active_seconds=supply_charge
  if start:
   gate=signed_json(refs['node_gate'])
   if gate.get('passed')is not True or gate.get('phase')!='node2000' or gate.get('checkpoint_sha256')!=refs['checkpoint']['sha256'] or gate.get('model_identity')!=job['model_identity']:raise ValueError('original node gate checkpoint capability')
   session.provenance['node2000_gate_pass']=True;session.provenance['node2000_gate_sha256']=refs['node_gate']['sha256'];atomic_json(directory/'node_gate_transition.json',{'gate_ref':refs['node_gate'],'original_checkpoint_ref':refs['checkpoint'],'operation':'restore original provenance then admitted gate upgrade','model_identity':job['model_identity']})
  if not resume_ref and session.updates!=0:raise ValueError("fresh learner must start at zero")
  if session.updates<start or session.updates>end:raise ValueError('checkpoint outside signed update segment')
  if read_json(directory/'optimizer_completion.json',{}).get('completed_optimizer_update',0)>session.updates:raise ValueError('known completed optimizer step has no restored durable state')
  if owner['last_completed_update']>session.updates:raise ValueError('known completed optimizer steps lack recoverable state; replay forbidden')
  setup_seconds=time.monotonic()-setup_began;session.active_seconds+=setup_seconds;atomic_json(directory/"setup_activity_receipt.json",{"charged_setup_seconds":setup_seconds,"offline_supply_seconds_added_only_at_initial":supply_charge if not resume_ref else 0.,"includes_manifest_initialization_checkpoint_load":True})
  session.start_activity(clock_path=clock)
  if not checkpoint_index:
   initial_checkpoint=model_directory/'checkpoints'/'initial_000000.pt'
   if initial_checkpoint.exists():raise ValueError('orphan initial checkpoint retained; root same-ID reconciliation required')
   receipt=session.checkpoint(initial_checkpoint)
   checkpoint_index=dict(latest=dict(path=str(initial_checkpoint),sha256=receipt['checkpoint_file_sha']),previous=None,updates=0,charged_active_seconds=receipt['charged_active_seconds'],model_identity=job['model_identity'],clock_path=str(clock))
   atomic_json(directory/'checkpoint_index.json',checkpoint_index);atomic_json(directory/'optimizer_completion.json',dict(completed_optimizer_update=0,recoverable_state_published=True,checkpoint_ref=checkpoint_index['latest'],model_identity=job['model_identity'],new_optimizer_updates=0))
   owner['last_completed_checkpoint_ref']=checkpoint_index['latest'];atomic_json(owner_path,owner)
  while session.updates<end:
   from .r7_deadline import stop_boundary
   stop_boundary(job,directory)
   check(directory,gpu=True)
   if session.elapsed()>=28800 or owner['science_updates_reserved']>=6000:raise RuntimeError('global activity/update cap reached')
   update=session.updates+1;owner['science_updates_reserved']+=1;atomic_json(owner_path,owner);atomic_json(directory/'step_intent.json',{'step':update,'reserved_total':owner['science_updates_reserved'],'sampler_before':session.sampler.state_dict(),'at':now(),'unknown_tail_preserved':True})
   pending=session.sampler.assemble();cases=[load_feature(rows[i],cache) for i in pending];targets=[load_cached_target(rows[i],case,cache) for i,case in zip(pending,cases)];check(directory,gpu=True);assert_v3_runtime(runtime);result=session.step(cases,targets,microbatch=4)
   atomic_json(directory/'optimizer_completion.json',{'completed_optimizer_update':session.updates,'recoverable_state_published':False,'model_identity':job['model_identity']})
   if result.get('update')!=update or session.updates!=update:raise ValueError('optimizer step progression not exact one')
   check(directory,gpu=True)
   # Every published completed optimizer step has a durable exact state first.
   rolling=model_directory/'checkpoints'/('rolling_%d.pt'%(update%2));receipt=session.checkpoint(rolling)
   checkpoint_index={'latest':{'path':str(rolling),'sha256':receipt['checkpoint_file_sha']},'previous':checkpoint_index.get('latest'),'updates':session.updates,'charged_active_seconds':receipt['charged_active_seconds'],'model_identity':job['model_identity'],'clock_path':str(clock)}
   atomic_json(directory/'checkpoint_index.json',checkpoint_index)
   atomic_json(directory/'optimizer_completion.json',{'completed_optimizer_update':update,'recoverable_state_published':True,'checkpoint_ref':checkpoint_index['latest'],'model_identity':job['model_identity']})
   owner['last_completed_update']=update;owner['last_completed_checkpoint_ref']=checkpoint_index['latest'];atomic_json(owner_path,owner)
   atomic_json(directory/'updates'/('%06d.json'%update),result);atomic_json(directory/'status_progress.json',{'update':session.updates,'active_seconds':session.elapsed(),'heartbeat_at':now()})
   stop_boundary(job,directory) # only after optimizer marker/checkpoint/index/owner are durable
  # Only the two contracted scientific nodes get immutable weight copies.
  immutable=model_directory/'nodes'/('%06d.pt'%end);immutable.parent.mkdir(parents=True,exist_ok=True)
  if immutable.exists():
   if digest(immutable)!=checkpoint_index['latest']['sha256']:raise ValueError('immutable node checkpoint already differs')
  else:
   temporary=immutable.with_suffix('.tmp');shutil.copyfile(pin(checkpoint_index['latest']),temporary)
   with temporary.open('rb+') as stream:os.fsync(stream.fileno())
   os.replace(temporary,immutable)
  checkpoint_index['fixed_node_ref']={'path':str(immutable),'sha256':digest(immutable)};atomic_json(directory/'checkpoint_index.json',checkpoint_index)
  if session.elapsed()>28800:raise RuntimeError('complete cumulative training charge exceeds8h; durable final weights not an admitted endpoint')
  session.pause_activity(reason='explicit_node_evaluation');atomic_json(directory/'training_endpoint.json',{'group':'S1-AI17-R7','phase':job['mode'],'passed':session.updates==end,'updates':session.updates,'active_seconds':session.elapsed(),'checkpoint_ref':checkpoint_index['fixed_node_ref'],'model_tensor_SHA':model_tensor_sha(session.model),'normalization_ref':refs['normalization'],'initialization_receipt_ref':refs['initialization_receipt'],'model_identity':job['model_identity'],'source_commit':job['scientific_commit'],'science_updates_reserved':owner['science_updates_reserved'],'future_phase_dispatch_allowed':False,
                  'run_id':job['run_id'],'original_learner_run_id':owner['original_learner_run_id'],'model_dir':job['model_dir'],'training_refs':setup_endpoint['training_refs'],'train_setup_endpoint_ref':setup_ref,'dataset_ref':config['datasets']['train'],'case_indices':job['case_indices'],'new_optimizer_updates':end-start,
                  'original_job_ref':dict(path=str(directory/'job.json'),sha256=digest(directory/'job.json')),'activity_clock_ref':dict(path=str(clock),sha256=digest(clock))});atomic_json(directory/'artifacts.json',{'training_endpoint':{'path':str(directory/'training_endpoint.json'),'sha256':digest(directory/'training_endpoint.json')},'fixed_checkpoint':checkpoint_index['fixed_node_ref']});return checkpoint_index
