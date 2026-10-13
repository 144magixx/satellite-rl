"""Native bounded local or unchanged original D5 child; no model or teacher import."""
import argparse,os,sys,time
from pathlib import Path
import numpy as np
from .contracts import read,pin,atomic,digest
from .deadline import boundary,bind_parent

def execute(req,directory):
 bind_parent('R8_PARENT_PID');plan=read(req['plan_ref']);boundary(plan,'child_before_inputs')
 source=Path(plan['original_source_manifest_ref']['path']).parent;sys.path[:0]=[str(source/'src'),str(source)]
 from satellite_rl.coverage.ai_search.core import Scene
 from satellite_rl.coverage.ai_search_r5.core import PreparedCase,decode_groups
 from satellite_rl.workbench.r6_io import read_asset,write_asset
 case=plan['cases'][req['case_index']];refs=case['refs'];began=time.monotonic()
 with np.load(pin(refs['original_scene']),allow_pickle=False) as z:scene=Scene(z['positions'],z['demands'],tuple(map(str,z['aircraft_ids'])),str(z['profile']))
 prepared=PreparedCase.from_dict(scene,read_asset(refs['original_prepared_Wenv'])['result'])
 if prepared.identity!=case['identity']:raise ValueError('full original case/prepared identity')
 if req['stage']=='local':
  import satellite_rl.coverage
  satellite_rl.coverage.__path__.append(str(Path(plan['core_source_root'])/'satellite_rl/coverage'))
  from satellite_rl.coverage.ai_search_r8 import core
  core_ref=plan['core_ref'];pin(core_ref)
  if Path(core.__file__).resolve()!=Path(core_ref['path']).resolve():raise ValueError('actual core source differs')
  owner=None
  if req['arm']=='A':
   with np.load(pin(refs['actual_model_outputs_NPZ']),allow_pickle=False) as z:owner=z['owner'].copy()
  provenance=dict(designSHA=plan['design_ref']['sha256'],actualarm=req['arm'],sourceepoch=plan['scientific_commit'],sourcefileSHA=core_ref['sha256'],original_case_ref=refs['shared_case'],original_prepared_ref=refs['original_prepared_Wenv'],AI_envelope_ref=refs['AI_envelope'],owner_tensor_ref=refs['actual_model_outputs_NPZ'],checkpointSHA=plan['checkpoint_ref']['sha256'],identity8=case['identity'])
  def stopped():
   try:boundary(plan,'local_candidate_boundary');return False
   except Exception:return True
  value=core.local_transfer(prepared,owner,arm=req['arm'],provenance=provenance,state_path=Path(req['state_path']),stop_callback=stopped,resume_admission=req.get('resume_admission'))
 else:
  boundary(plan,'before_D5_algorithm')
  proposal=read_asset(req['proposal_ref'])
  if proposal['provenance']['source_method']!='EXACT' or proposal['case_identity']!=case['identity']:raise ValueError('EXACT compatibility with actual full R8 identity')
  result=decode_groups(prepared,proposal,time_limit=5.)
  value=dict(status='completed',result=result,actual_arm=req['arm'],identity=case['identity'],actual_D5_calls=1,new_model_calls=0,unchanged_decoder_source=plan['original_source_manifest_ref'],outer_proposal_ref=req['proposal_ref'])
 value.update(operational_input_and_stage_seconds=time.monotonic()-began,original_input_refs=refs,actual_arm=req['arm'],actual_stage=req['stage'])
 asset=write_asset(Path(directory)/'stage_output.json.gz',value);atomic(Path(directory)/'stage_output.json',dict(asset_ref=asset));return asset
def main():
 p=argparse.ArgumentParser();p.add_argument('--request',required=True);p.add_argument('--directory',required=True);a=p.parse_args();execute(read(dict(path=a.request,sha256=digest(a.request))),a.directory)
if __name__=='__main__':main()
