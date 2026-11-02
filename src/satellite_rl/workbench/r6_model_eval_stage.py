"""One cold raw-input inference and one unchanged D5 call; no teacher payload."""
import argparse,sys,time,os,subprocess
from pathlib import Path
import numpy as np
from satellite_rl.training.run_monitor import atomic_json
from .storage import read_json,digest
from .coverage_ai_search import pin
from .r6_io import write_asset,read_asset

def execute(req,directory):
 from satellite_rl.coverage.ai_search.core import Scene
 from satellite_rl.coverage.ai_search_r5.core import PreparedCase,build_proposal
 directory=Path(directory);began=time.monotonic();method=req['method']
 if method not in ('AI','RULE','RANDOM') or req['identity']['role']!='development' or req.get('teacher_ref'):raise ValueError('node inference has no T inputs')
 with np.load(pin(req['scene_ref']),allow_pickle=False) as z:scene=Scene(z['positions'],z['demands'],tuple(map(str,z['aircraft_ids'])),str(z['profile']))
 if scene.input_sha256!=req['identity']['input_sha256']:raise ValueError('original raw input SHA')
 prepared_record=read_json(pin(req['prepared_ref']),{})['result'];prepared=PreparedCase.from_dict(scene,prepared_record)
 if prepared.identity!=req['identity']:raise ValueError('reused original prepared identity')
 input_seconds=time.monotonic()-began;proposal_started=time.monotonic();envelope=None
 if method=='AI':
  from satellite_rl.coverage.ai_search_r6.runtime_v3 import configure_v3_runtime
  runtime=configure_v3_runtime();atomic_json(directory/'runtime_v3_identity.json',runtime)
  if runtime['runtime_profile_SHA']!=req['runtime_profile_SHA']:raise ValueError('fixed actual runtime profile')
  import torch
  from satellite_rl.coverage.ai_search_r6.model import initialize_model,model_tensor_sha
  from satellite_rl.coverage.ai_search_r6.features import FeatureStats,build_features,collate_features
  from satellite_rl.coverage.ai_search_r6.ai_adapter import build_ai_proposal
  if prepared.status!='available':raise ValueError('signed node24 expects original successful prepared')
  load_started=time.monotonic();state=torch.load(pin(req['checkpoint_ref']),map_location='cpu',weights_only=False)
  if state.get('updates')!=2000 or state.get('provenance',{}).get('model_identity')!=req['model_identity']:raise ValueError('fixed exact2000 model')
  stats=FeatureStats.from_record(read_json(pin(req['normalization_ref']),{}))
  if state['stats']!=stats.to_record():raise ValueError('checkpoint training normalization differs')
  model=initialize_model();model.load_state_dict(state['model'],strict=True);model=model.to('cuda');model.eval();model_load_seconds=time.monotonic()-load_started
  feature=build_features(scene);forward_started=time.monotonic()
  with torch.no_grad():output=model(**collate_features([feature],stats,'cuda'))
  torch.cuda.synchronize();forward_seconds=time.monotonic()-forward_started
  if torch.cuda.max_memory_reserved()>12884901888:raise MemoryError('12GiB GPU budget')
  provenance={**req['AI_provenance'],'checkpoint_file_SHA':req['checkpoint_ref']['sha256'],'model_tensor_SHA':model_tensor_sha(model),'normalization_SHA':stats.sha}
  inference_cost={'model_forward_calls':1,'model_load_seconds':model_load_seconds,'feature_seconds':feature.cost['seconds'],'NN_ray_pairs':feature.cost['NN_ray_pairs'],'forward_seconds':forward_seconds,'raw_prepared_read_seconds':input_seconds,'cold':True}
  envelope=build_ai_proposal(prepared,output,provenance,inference_cost=inference_cost,tensor_asset_path=directory/'actual_model_outputs.npz');proposal=envelope['inner'];envelope_ref=write_asset(directory/'AI_envelope.json.gz',envelope)
 else:
  proposal=build_proposal(prepared,method,producer_sha256=req['producer_sha256'],evidence_refs=[req['prepared_ref']]);envelope_ref=None
 proposal_ref=write_asset(directory/'proposal.json.gz',proposal);proposal_seconds=time.monotonic()-proposal_started;decode_started=time.monotonic()
 from .r5_stage_guard import run_stage
 target=directory/'D5_guard';target.mkdir();decoder_request={'identity':req['identity'],'scene_ref':req['scene_ref'],'prepared_ref':req['prepared_ref'],'proposal_ref':proposal_ref,'original_prepared_source_sha256':prepared.source_sha256,'b7_decoder_source_SHA':req['AI_provenance']['b7_decoder_source_SHA'] if method=='AI' else req['b7_decoder_source_SHA']};atomic_json(target/'request.json',decoder_request)
 def owned_child(command,**kwargs):
  env=os.environ.copy();env['R6_D5_PARENT_PID']=str(os.getpid());return subprocess.Popen(command,env=env,**kwargs)
 guard=run_stage([sys.executable,'-B','-u','-m','satellite_rl.workbench.r6_model_d5_stage','--request',str(target/'request.json'),'--directory',str(target)],target,15,2147483648,popen=owned_child)
 if guard['status']!='completed':raise RuntimeError('nested D5 hard guard failed; original unknown result/cost retained')
 decoded=read_asset(read_json(pin(guard['output_ref']),{})['asset_ref']);result=decoded['result'];decoder_seconds=time.monotonic()-decode_started
 asset=write_asset(directory/'stage_output.json.gz',{'stage':'decoder','actual_method':method,'identity':req['identity'],'result':result,'proposal_ref':proposal_ref,'AI_envelope_ref':envelope_ref,'input_seconds':input_seconds,'proposal_seconds':proposal_seconds,'decoder_seconds':decoder_seconds,'cold':True,'one_D5_call':True,'new_T_calls':0,'D5_guard_ref':{'path':str(target/'guard_receipt.json'),'sha256':digest(target/'guard_receipt.json')} });atomic_json(directory/'stage_output.json',{'asset_ref':asset,'actual_method':method,'identity':req['identity']});return asset

def main():
 p=argparse.ArgumentParser();p.add_argument('--request',required=True);p.add_argument('--directory',required=True);a=p.parse_args();execute(read_json(a.request,{}),a.directory)
if __name__=='__main__':main()
