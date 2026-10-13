"""Root-approved Linux pure-process guard qualification, exactly three stubs."""
import argparse,sys,json
from pathlib import Path
from satellite_rl.training.run_monitor import atomic_json
from satellite_rl.workbench.storage import digest
from satellite_rl.workbench.r5_stage_guard import run_stage

def qualify(directory,commit,approval):
 if sys.platform!='linux':raise ValueError('real Linux process qualification only')
 receipt=json.loads(Path(approval).read_text('utf-8'));source=Path(__file__).resolve().parents[2]
 guard=source/'src/satellite_rl/workbench/r5_stage_guard.py';fixed=source/'src/satellite_rl/workbench/r4_monitor_repair.py'
 if receipt.get('passed') is not True or receipt.get('scientific_commit')!=commit or receipt.get('guard_sha256')!=digest(guard) or receipt.get('rss_provider_sha256')!=digest(fixed) or receipt.get('process_stub_calls_max')!=3:raise ValueError('root source-bound process qualification admission required')
 root=Path(directory);root.mkdir(parents=True,exist_ok=True)
 if any(root.iterdir()):raise ValueError('qualification directory used; no implicit 3stub replay')
 outcomes=[]
 for n,behavior in enumerate(('publish_exit0','exit0_no_output','exit7')):
  target=root/str(n);target.mkdir();payload="import json,pathlib;pathlib.Path('"+str(target/'stage_output.json')+"').write_text(json.dumps({'stub':True}))" if n==0 else "raise SystemExit("+('0' if n==1 else '7')+")"
  outcomes.append(run_stage([sys.executable,'-B','-c',payload],target,15,2147483648))
 passed=outcomes[0]['status']=='completed' and outcomes[1]['reason']=='missing_output' and outcomes[2]['reason']=='child_error' and not any(x.get('monitor_error') for x in outcomes)
 value={'schema':'R5LinuxPureProcessQualificationV1','passed':passed,'scientific_commit':commit,'process_stub_calls':3,'geometry_calls':0,'CUDA_calls':0,'data_reads':0,'guard_sha256':digest(guard),'rss_provider_sha256':digest(fixed),'approval_ref':{'path':str(approval),'sha256':digest(approval)},'outcomes':outcomes};atomic_json(root/'qualification.json',value);return value
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--directory',required=True);p.add_argument('--commit',required=True);p.add_argument('--approval',required=True);a=p.parse_args();print(json.dumps(qualify(a.directory,a.commit,a.approval)))
