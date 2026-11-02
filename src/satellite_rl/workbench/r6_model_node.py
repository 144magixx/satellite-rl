"""24 reused development bundles, three new D5 calls each, no new preparation/T."""
import os,sys,time,hashlib,shutil
from pathlib import Path
import numpy as np
from satellite_rl.training.run_monitor import atomic_json
from .storage import read_json,digest,now
from .coverage_ai_search import pin
from .r6_io import read_asset,pin_tree
from .r5_stage_guard import run_stage
from .r5_io import identity,original_records
from .r6_model_io import signed_json,verify_model_source_equivalence
from .r6_node_audit import audit_row

def method_order(i):
 methods=['AI','RULE','RANDOM'];shift=i%3;return methods[shift:]+methods[:shift]
def node(job,config,directory):
 directory=Path(directory);refs=job['model_inputs'];registry=signed_json(refs['registry']);cases={c['input_case_index']:c for c in registry['cases']};initial=signed_json(refs['initialization_receipt']);index=read_json(directory/'case_index.json',{});bycase={}
 if registry.get('roles_read')!=['development']:raise ValueError('original development source only')
 actual_sources=verify_model_source_equivalence(refs['source_equivalence'],refs['initialization_receipt'],initial)
 if any(actual_sources[k]!=initial[k] for k in ('model_graph_source_SHA','feature_graph_source_SHA')):raise ValueError('frozen model/feature sources differ from engineering admission')
 provenance={k:actual_sources[k] for k in ('model_graph_source_SHA','feature_graph_source_SHA','adapter_source_SHA','b7_decoder_source_SHA')};provenance['model_initial_state_SHA']=initial['model_initial_state_SHA'];provenance['source_epoch']=job['scientific_commit']
 def stop():
  if shutil.disk_usage(directory).free<8589934592:raise OSError('8GiB reserve')
  return (directory/'stop.request').exists()
 for i in job['case_indices']:
  supply_started=time.monotonic()
  case=cases[i];ident=identity(case);prepared,_=original_records(case,teacher_required=False);rowset={}
  if ident['role']!='development':raise ValueError('node true development role')
  with np.load(pin(case['scene_source']['raw_scene']),allow_pickle=False) as z:scene={'positions':z['positions'].tolist(),'demands':z['demands'].tolist(),'aircraft_ids':list(map(str,z['aircraft_ids'])),'profile':str(z['profile'])}
  old=signed_json(case['teacher']['publication_cost']);prep_seconds=old['components']['prepare'];input_seconds=old['components']['input']
  actual_parent_scene_prepared_supply=time.monotonic()-supply_started
  atomic_json(directory/'supply_ledger'/(str(i)+'.json'),{'identity':ident,'actual_parent_scene_prepared_supply_seconds':actual_parent_scene_prepared_supply,'physical_shared_work_paid_once':True,'complete_cold_each_method_adds_same_supply_seconds':True,'raw_scene_prepared_and_cost_metadata_read_only':True,'teacher_terminal_read_online':False})
  for method in method_order(i):
   key=hashlib.sha256((ident['case_id']+'\0'+method).encode()).hexdigest()
   if key in index:
    pin_tree(index[key]);row=read_json(pin(index[key]),{})
    if row['identity']!=ident or row['method']!=method or row['source_commit']!=job['scientific_commit'] or row['method_order']!=method_order(i):raise ValueError('recovered node identity/source differs')
    if row['result_ref']:
     stored=read_asset(row['result_ref']);checked=audit_row(scene,prepared,stored,row['request'])
     if checked!=row['audit']:raise ValueError('recovered independent node audit differs')
    if not index[key].get('cost_endpoint_ref'):raise ValueError('recovered missing cost endpoint; no method replay')
    endpoint=signed_json(index[key]['cost_endpoint_ref'])
    if endpoint['raw_ref']['sha256']!=index[key]['sha256'] or endpoint['identity']!=ident:raise ValueError('node cost backlink mismatch')
    rowset[method]=row;continue
   if stop():raise RuntimeError('requested stop')
   target=directory/'bundles'/str(i)/method;target.mkdir(parents=True,exist_ok=True)
   if (target/'parent_WAL.json').exists():raise ValueError('unknown node call retained; root reconciliation required')
   request={'method':method,'identity':ident,'scene_ref':case['scene_source']['raw_scene'],'prepared_ref':case['prepared']['stage_output'],'producer_sha256':job['source_manifest_sha256'],'model_identity':job['model_identity'],'b7_decoder_source_SHA':actual_sources['b7_decoder_source_SHA']}
   if method=='AI':request.update(checkpoint_ref=refs['checkpoint'],normalization_ref=refs['normalization'],AI_provenance=provenance,runtime_profile_SHA=initial['runtime_profile_SHA'])
   begin=time.monotonic();atomic_json(target/'request.json',request);atomic_json(target/'parent_WAL.json',{'request_sha':digest(target/'request.json'),'source_commit':job['scientific_commit'],'unknown_spend_preserved':True})
   guard=run_stage([sys.executable,'-B','-u','-m','satellite_rl.workbench.r6_model_eval_stage','--request',str(target/'request.json'),'--directory',str(target)],target,90 if method=='AI' else 15,4294967296 if method=='AI' else 2147483648,stop)
   if guard.get('hard_interface_error'):raise ValueError('hard interface/monitor error stops dependent phases')
   if guard['status']=='completed':value=read_asset(read_json(pin(guard['output_ref']),{})['asset_ref']);audit=audit_row(scene,prepared,value,request);raw_ref=read_json(pin(guard['output_ref']),{})['asset_ref']
   else:value={'result':{'status':'unpublished','reason':guard['reason']}};audit={'valid':True,'published':False,'nonfallback':False,'status':'unpublished','metrics':None,'identity_closed':True,'budget_closed':False};raw_ref=None
   cost_known=all(isinstance(x,(int,float)) and np.isfinite(x) and x>=0 for x in (prep_seconds,input_seconds)) and not old.get('unknown_cost') and not guard.get('unknown_algorithm_spend')
   row={'identity':ident,'input_case_index':i,'method':method,'method_order':method_order(i),'audit':audit,'result_ref':raw_ref,'guard':guard,'request':request,'reused_original_prepared_ref':case['prepared']['stage_output'],'checkpoint_ref':refs['checkpoint'] if method=='AI' else None,'new_prepare_calls':0,'new_T_calls':0,'new_D5_calls':1,'source_commit':job['scientific_commit'],'cost_components':{'reused_original_input':input_seconds,'reused_original_preparation':prep_seconds,'actual_parent_scene_prepared_supply':actual_parent_scene_prepared_supply,'actual_new_parent_pipeline_until_publication':time.monotonic()-begin},'complete_cold_seconds':None,'cost_known':cost_known,'historical_preparation_cost_not_free':True,'strict_speed_claim_requires_separate_host_epoch_receipt':True}
   path=directory/'rows'/(key+'.json');publish_start=time.monotonic();atomic_json(path,row);index[key]={'path':str(path),'sha256':digest(path)};atomic_json(directory/'case_index.json',index);publication=time.monotonic()-publish_start
   total=sum(row['cost_components'].values())+publication if cost_known else None;endpoint=target/'cost_endpoint.json';atomic_json(endpoint,{'identity':ident,'method':method,'raw_ref':index[key],'complete_cold_seconds':total,'row_index_publication_seconds':publication,'endpoint_IO_separate_metadata':True,'cost_known':cost_known});index[key]['cost_endpoint_ref']={'path':str(endpoint),'sha256':digest(endpoint)};atomic_json(directory/'case_index.json',index);rowset[method]=row
  bycase[str(i)]={'case_index':i,'identity':ident,'methods':{m:r['audit'] for m,r in rowset.items()},'raw_refs':{m:index[hashlib.sha256((ident['case_id']+'\0'+m).encode()).hexdigest()] for m in rowset},'teacher_target_ref':case['teacher']['stage_output'],'preparation_seconds':prep_seconds,'total_request':sum(scene['demands'])}
 atomic_json(directory/'result_manifest.json',{'group':job['experiment_id'],'phase':'node2000','scientific_commit':job['scientific_commit'],'cases':bycase,'checkpoint_ref':refs['checkpoint'],'model_identity':job['model_identity'],'new_prepare_calls':0,'new_T_calls':0,'D5_slots':len(index),'future_dispatch_allowed':False});return bycase
