"""P0 decoder child; original PreparedCase/T bytes are never rewritten."""
import argparse,time,sys,inspect
from pathlib import Path
import numpy as np
from satellite_rl.training.run_monitor import atomic_json
from .storage import read_json,digest,clean_json
from .coverage_ai_search import pin
from .r5_io import identity,original_records,validate_output

def main():
 p=argparse.ArgumentParser();p.add_argument('--request',required=True);p.add_argument('--directory',required=True);a=p.parse_args();directory=Path(a.directory);req=read_json(a.request,{});began=time.monotonic()
 if req.get('stage')!='decoder' or req['method'] not in ('D5_RULE','D5_RANDOM','D5_EXACT','D5_OWNER10','D5_JOINT10','D4_OWNER10','D4_JOINT10'):raise ValueError('P0 decoder-only exact method')
 from satellite_rl.coverage.ai_search.core import Scene
 from satellite_rl.coverage.ai_search_r4.core import PreparedCase
 from satellite_rl.coverage.ai_search_r5.core import build_proposal,decode_groups
 from satellite_rl.coverage.ai_search_r5.adapter import decode_d4
 case=req['registry_case'];ident=identity(case);raw=pin(case['scene_source']['raw_scene'])
 with np.load(raw,allow_pickle=False) as z:scene=Scene(z['positions'],z['demands'],tuple(str(x)for x in z['aircraft_ids']),str(z['profile']))
 if scene.input_sha256!=ident['input_sha256']:raise ValueError('original scene input SHA mismatch')
 decoder,method=req['method'].split('_',1)
 prepared_record,teacher=original_records(case,teacher_required=method in ('EXACT','OWNER10','JOINT10'));prepared=PreparedCase.from_dict(scene,prepared_record);read_seconds=time.monotonic()-began
 proposal_start=time.monotonic();proposal=build_proposal(prepared,method,teacher=teacher if method in ('EXACT','OWNER10','JOINT10') else None,producer_sha256=req['source_sha256'],evidence_refs=[case['prepared']['stage_output'],case['teacher']['stage_output']]);proposal_seconds=time.monotonic()-proposal_start
 proposal_path=directory/'proposal.json';serialization_begin=time.monotonic();atomic_json(proposal_path,clean_json(proposal));proposal_publish_seconds=time.monotonic()-serialization_begin
 # Durable callbacks remain inside nominal D5 wall clock and algorithm ledger.
 accepted=directory/'accepted';accepted.mkdir(exist_ok=True);accepted_refs=[]
 def persist(event):
  path=accepted/(str(len(accepted_refs)).zfill(5)+'.json');atomic_json(path,clean_json({'identity':ident,'source_sha256':req['source_sha256'],'event':event}));accepted_refs.append({'path':str(path),'sha256':digest(path)});atomic_json(directory/'accepted_snapshot_index.json',{'schema':'R5AcceptedSnapshotsV1','identity':ident,'source_sha256':req['source_sha256'],'accepted_refs':accepted_refs,'partial_evidence_not_final_publication':True})
 decode_begin=time.monotonic()
 if decoder=='D5':result=decode_groups(prepared,proposal,time_limit=5.,on_accepted=persist)
 else:result=decode_d4(prepared,proposal,contract_sha256=req['contract_sha256'],adapter_source_sha256=digest(inspect.getsourcefile(decode_d4)))
 decoder_seconds=time.monotonic()-decode_begin
 envelope={'schema':'R5P0DerivedEnvelopeV1','identity':ident,'actual_method':req['method'],'new_source_sha256':req['source_sha256'],'contract_sha256':req['contract_sha256'],'original_prepared_ref':case['prepared']['stage_output'],'original_teacher_ref':case['teacher']['stage_output'],'original_prepared_source_sha256':case['prepared']['source_sha256'],'original_teacher_source_sha256':case['teacher']['source_sha256'],'proposal_ref':{'path':str(proposal_path),'sha256':digest(proposal_path)},'result_derivation':result.get('derivation',result.get('derivation_envelope')),'interpretation':'new_same_domain_diagnostic_not_original_R4_PERTURBED' if decoder=='D4' else 'new_D5_with_original_R4_inputs'}
 envelope_path=directory/'derivation_envelope.json';atomic_json(envelope_path,clean_json(envelope));value={'stage':'decoder','identity':ident,'source_sha256':req['source_sha256'],'method':req['method'],'result':result,'decoder_invoked':proposal.get('status')=='ready','actual_provider_read_seconds':read_seconds,'actual_proposal_generation_seconds':proposal_seconds,'actual_proposal_publication_seconds':proposal_publish_seconds,'actual_decoder_algorithm_seconds':decoder_seconds,'actual_package_source':str(Path(__file__).resolve()),'python':sys.executable,'original_prepared_ref':case['prepared']['stage_output'],'original_teacher_ref':case['teacher']['stage_output'],'proposal_ref':envelope['proposal_ref'],'derived_envelope_ref':{'path':str(envelope_path),'sha256':digest(envelope_path)},'accepted_snapshot_index_ref':{'path':str(directory/'accepted_snapshot_index.json'),'sha256':digest(directory/'accepted_snapshot_index.json')} if accepted_refs else None}
 validate_output(value,req);start=time.monotonic();atomic_json(directory/'stage_output.json',clean_json(value));atomic_json(directory/'serialization_receipt.json',{'actual_seconds':time.monotonic()-start})
if __name__=='__main__':main()
