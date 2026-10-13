"""Actual R4 stage entry. Called only by a reviewed remote workbench job."""
import argparse,json,time,sys
from pathlib import Path
import numpy as np
from satellite_rl.training.run_monitor import atomic_json
from .storage import read_json,digest,clean_json
from .r4_io import check_record

def persist_accepted(directory,identity,source_sha256):
 directory=Path(directory);snapshots=directory/'accepted';snapshots.mkdir(parents=True,exist_ok=True);index_path=directory/'accepted_snapshot_index.json'
 if index_path.exists()or any(snapshots.iterdir()):raise ValueError('existing accepted teacher snapshots forbid implicit replay')
 refs=[]
 def write_index():atomic_json(index_path,{'schema':'R4TeacherAcceptedSnapshotsV1','identity':identity,'source_sha256':source_sha256,'accepted_refs':list(refs),'count':len(refs),'partial_evidence_not_final_publication':True})
 write_index()
 def save(snapshot):
  path=snapshots/(str(len(refs)).zfill(5)+'.json');atomic_json(path,clean_json({'identity':identity,'source_sha256':source_sha256,'accepted_ordinal':len(refs),'snapshot':snapshot}));refs.append({'path':str(path),'sha256':digest(path)});write_index()
 return save

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--request',required=True);parser.add_argument('--directory',required=True);args=parser.parse_args();directory=Path(args.directory);request=read_json(args.request,{});began=time.monotonic()
 from satellite_rl.coverage.ai_search.core import Scene
 from satellite_rl.coverage.ai_search_r4.core import prepare_case,PreparedCase,build_teacher,build_proposal,decode_groups
 with np.load(request['scene_path'],allow_pickle=False)as z:scene=Scene(z['positions'],z['demands'],tuple(str(x)for x in z['aircraft_ids']),str(z['profile']))
 read_seconds=time.monotonic()-began;stage=request['stage'];start=time.monotonic();proposal=None;decode_called=False;proposal_seconds=0.;decoder_seconds=0.
 if stage=='preparation':result=prepare_case(scene,request['identity'],request['source_sha256'],time_limit=10.).to_dict()
 else:
  prepared_record=read_json(request['prepared_path'],{})['result'];check_record(prepared_record,'prepared',request['identity'],request['source_sha256']);prepared=PreparedCase.from_dict(scene,prepared_record)
  if stage=='teacher':
   persist=persist_accepted(directory,request['identity'],request['source_sha256']);result=build_teacher(prepared,on_accepted=persist);check_record(result,'teacher',request['identity'],request['source_sha256'])
  else:
   teacher=read_json(request['teacher_path'],{})['result'] if request.get('teacher_path')else None
   if request['method'].startswith('TEACHER_') and teacher is not None:check_record(teacher,'teacher',request['identity'],request['source_sha256'])
   if request['method']in ('TEACHER_EXACT','TEACHER_PERTURBED')and (teacher is None or teacher['status']=='unavailable'):
    proposal={'status':'unavailable','reason':'unavailable_teacher','provenance':{**request['identity'],'source_sha256':request['source_sha256'],'source_method':request['method']},'cost':{'seconds':0.,'skipped_before_proposal_generation':True}}
   else:
    proposal_started=time.monotonic();proposal=build_proposal(prepared,request['method'],teacher=teacher);proposal_seconds=time.monotonic()-proposal_started
   if proposal is None:proposal={'status':'unavailable','reason':'typed_unavailable_missing_payload','provenance':{**request['identity'],'source_sha256':request['source_sha256'],'source_method':request['method']},'cost':{'seconds':0.,'payload_missing':True}}
   check_record(proposal,'proposal',request['identity'],request['source_sha256'])
   proposal_path=directory/'proposal.json';atomic_json(proposal_path,clean_json(proposal))
   if proposal['status']=='ready':
    decode_called=True;decoder_started=time.monotonic();result=decode_groups(prepared,proposal);decoder_seconds=time.monotonic()-decoder_started
   else:result={'schema':'DecoderResultV1','status':'unavailable','reason':proposal['reason'],'provenance':{**request['identity'],'source_sha256':request['source_sha256']},'cost':{'seconds':0.,'decoder_not_invoked':True}}
 result.setdefault('source_sha256',request['source_sha256']);check_record(result,'prepared'if stage=='preparation'else'teacher'if stage=='teacher'else'decoder',request['identity'],request['source_sha256'])
 algorithm_seconds=time.monotonic()-start;value={'stage':stage,'result':result,'decoder_invoked':decode_called,'actual_provider_read_seconds':read_seconds,'actual_algorithm_seconds':algorithm_seconds,'actual_proposal_generation_seconds':proposal_seconds,'actual_decoder_algorithm_seconds':decoder_seconds,'python':sys.executable,'actual_package_source':str(Path(__file__).resolve()),'proposal_ref':None if proposal is None else {'path':str(directory/'proposal.json'),'sha256':digest(directory/'proposal.json')},'accepted_snapshot_index_ref':{'path':str(directory/'accepted_snapshot_index.json'),'sha256':digest(directory/'accepted_snapshot_index.json')}if stage=='teacher'else None};serialization_start=time.monotonic();atomic_json(directory/'stage_output.json',clean_json(value));atomic_json(directory/'serialization_receipt.json',{'actual_seconds':time.monotonic()-serialization_start})
if __name__=='__main__':main()
