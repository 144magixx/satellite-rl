"""Independently guarded unchanged D5; dies if its owning AI process dies."""
import os,signal,ctypes,sys

def bind_parent():
 if sys.platform!='linux':raise ValueError('Linux parent-death guard required')
 expected=int(os.environ['R7_D5_PARENT_PID']);libc=ctypes.CDLL(None,use_errno=True)
 if libc.prctl(1,signal.SIGKILL,0,0,0)!=0:raise OSError(ctypes.get_errno(),'PR_SET_PDEATHSIG failed')
 if os.getppid()!=expected:raise RuntimeError('owning parent changed before D5 admission; no algorithm call')

def main():
 bind_parent()
 import argparse,time
 from pathlib import Path
 import numpy as np
 from satellite_rl.training.run_monitor import atomic_json
 from .storage import read_json
 from .coverage_ai_search import pin
 from .r6_io import read_asset,write_asset
 p=argparse.ArgumentParser();p.add_argument('--request',required=True);p.add_argument('--directory',required=True);a=p.parse_args();request=read_json(a.request,{});directory=Path(a.directory);began=time.monotonic()
 from satellite_rl.coverage.ai_search.core import Scene
 from satellite_rl.coverage.ai_search_r5.core import PreparedCase,decode_groups
 with np.load(pin(request['scene_ref']),allow_pickle=False) as z:scene=Scene(z['positions'],z['demands'],tuple(map(str,z['aircraft_ids'])),str(z['profile']))
 if scene.input_sha256!=request['identity']['input_sha256']:raise ValueError('inner D5 raw identity')
 prepared=PreparedCase.from_dict(scene,read_asset(request['prepared_ref'])['result'])
 if prepared.identity!=request['identity'] or prepared.source_sha256!=request['original_prepared_source_sha256']:raise ValueError('original prepared source unchanged')
 proposal=read_asset(request['proposal_ref']);started=time.monotonic();result=decode_groups(prepared,proposal,time_limit=5.)
 from .r5_stage_guard import birth
 asset=write_asset(directory/'stage_output.json.gz',{'identity':request['identity'],'owning_AI_process_pid':os.getppid(),'owning_AI_process_birth':birth(os.getppid()),'result':result,'nominal_D5_algorithm_seconds':time.monotonic()-started,'full_D5_child_seconds':time.monotonic()-began,'actual_D5_calls':1,'original_prepared_source_sha256':prepared.source_sha256,'immutable_b7_decoder_source_SHA':request['b7_decoder_source_SHA']});atomic_json(directory/'stage_output.json',{'asset_ref':asset})
if __name__=='__main__':main()
