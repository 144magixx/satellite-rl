"""Native bounded local or unchanged original D5 child; no model or teacher import."""
import argparse,os,sys,time
from pathlib import Path
import numpy as np
from .contracts import read,pin,atomic,digest,validate_imported_core
from .deadline import boundary,bind_parent,Paused

def execute(req,directory):
 bind_parent('R8_PARENT_PID');plan=read(req['plan_ref']);boundary(plan,'child_before_inputs')
 source=Path(plan['original_source_manifest_ref']['path']).parent;sys.path[:0]=[str(source/'src'),str(source)]
 from satellite_rl.coverage.ai_search.core import Scene
 from satellite_rl.coverage.ai_search_r5.core import PreparedCase,decode_groups
 from satellite_rl.workbench.r6_io import read_asset,write_asset
 case=plan['cases'][req['case_index']];refs=case['refs'];began=time.monotonic()
 with np.load(pin(refs['original_scene']),allow_pickle=False) as z:scene=Scene(z['positions'],z['demands'],tuple(map(str,z['aircraft_ids'])),str(z['profile']))
 prepared=PreparedCase.from_dict(scene,read_asset(refs['original_prepared_Wenv'])['result'])
 atomic(Path(directory)/'input_timer.json',dict(observed_input_seconds=time.monotonic()-began,scope='original scene and prepared read, Scene/PreparedCase reconstruction',includes_prior_imports=False))
 if prepared.identity!=case['identity']:raise ValueError('full original case/prepared identity')
 if req['stage']=='local':
  import satellite_rl.coverage
  satellite_rl.coverage.__path__.append(str(Path(plan['core_source_root'])/'satellite_rl/coverage'))
  from satellite_rl.coverage.ai_search_r8 import core
  import satellite_rl.coverage.ai_search_r8 as package
  from satellite_rl.coverage.ai_search_r8 import geometry,ledger
  validate_imported_core(plan,{'__init__.py':package,'core.py':core,'geometry.py':geometry,'ledger.py':ledger})
  core_ref=plan['core_ref'];pin(core_ref)
  if Path(core.__file__).resolve()!=Path(core_ref['path']).resolve():raise ValueError('actual core source differs')
  owner=None
  if req['arm']=='A':
   with np.load(pin(refs['actual_model_outputs_NPZ']),allow_pickle=False) as z:owner=z['owner'].copy()
  provenance=dict(design_sha256=plan['design_ref']['sha256'],actual_arm=req['arm'],source_epoch=plan['scientific_commit'],adapter_source_SHA=core_ref['sha256'],case_identity=case['identity'],checkpoint_sha256=plan['checkpoint_ref']['sha256'],source_refs={key:refs[key] for key in ('shared_case','original_scene','original_prepared_Wenv','AI_envelope','actual_model_outputs_NPZ','R7_RULE_final','R7_T_final')})
  if req['arm']=='A':provenance['owner_array_sha256']=read_asset(refs['AI_envelope'])['valid_aircraft_pointer_owner_tensor_refs_and_SHAs']['sha256']['owner']
  def stopped():
   try:boundary(plan,'local_candidate_boundary');return False
   except Paused:return True
  value=core.local_transfer(prepared,owner,arm=req['arm'],provenance=provenance,state_path=Path(req['state_path']),stop_callback=stopped,resume_admission=req.get('resume_admission'))
 else:
  boundary(plan,'before_D5_algorithm')
  proposal=read_asset(req['proposal_container_ref'])['proposal']
  if proposal['provenance']['source_method']!='EXACT' or proposal['case_identity']!=case['identity']:raise ValueError('EXACT compatibility with actual full R8 identity')
  result=decode_groups(prepared,proposal,time_limit=5.)
  value=dict(status='completed',result=result,actual_arm=req['arm'],identity=case['identity'],actual_D5_calls=1,new_model_calls=0,unchanged_decoder_source=plan['original_source_manifest_ref'],outer_proposal_container_ref=req['proposal_container_ref'])
 value.update(operational_input_and_stage_seconds=time.monotonic()-began,original_input_refs=refs,actual_arm=req['arm'],actual_stage=req['stage'])
 publish_started=time.monotonic();asset=write_asset(Path(directory)/'stage_output.json.gz',value);atomic(Path(directory)/'stage_output.json',dict(asset_ref=asset));atomic(Path(directory)/'publication_timer.json',dict(observed_publish_seconds=time.monotonic()-publish_started,scope='canonical gzip asset and small asset pointer',timer_receipt_IO_additional=True));return asset
def main():
 began=time.monotonic();p=argparse.ArgumentParser();p.add_argument('--request',required=True);p.add_argument('--directory',required=True);a=p.parse_args()
 try:execute(read(dict(path=a.request,sha256=digest(a.request))),a.directory)
 finally:atomic(Path(a.directory)/'child_epoch_finally.json',dict(observed_seconds_since_main=time.monotonic()-began,excludes_prior_module_imports=True,provisional_until_finally_sample=True,complete_child_exit_bound_closed=False,excludes_this_receipt_IO_and_interpreter_exit=True))
if __name__=='__main__':main()
