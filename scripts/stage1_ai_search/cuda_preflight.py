"""Two synthetic CUDA updates per admitted lane; separate engineering ledger."""
from concurrent.futures import ThreadPoolExecutor
import argparse
import json
from satellite_rl.training.run_monitor import atomic_json
from . import ops


def verify(port,commit):
    if port not in (20418,20424,20473):raise ValueError('74 is not admitted for engineering updates')
    info=ops.instance(port)
    code=f'''import sys,hashlib,json,pathlib,subprocess,datetime
repo=pathlib.Path({info['repo']!r});commit={commit!r}
actual=subprocess.run(['git','-C',str(repo),'rev-parse','HEAD'],capture_output=True,text=True,check=True).stdout.strip()
if actual!=commit:raise ValueError('engineering check requires exact new frozen source')
uuid=subprocess.run(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True).stdout.strip()
path=pathlib.Path({info['ops']!r})/('cuda_updates_'+commit+'_'+hashlib.sha256(uuid.encode()).hexdigest()[:12]+'.json')
if path.is_file():
 record=json.loads(path.read_text())
 if record.get('source_commit')==commit and record.get('gpu_uuid')==uuid and record.get('passed') is True and record.get('engineering_updates')==2:
  print(json.dumps({{'reused_same_source_same_GPU':True,'receipt':record,'path':str(path)}}));sys.exit(0)
sys.path.insert(0,str(repo/'src'))
import torch,satellite_rl
from satellite_rl.coverage.ai_search.learning import make_model,loss,MODEL_FORMAT
from satellite_rl.coverage.ai_search.features import FEATURE_SCHEMA
if str(pathlib.Path(satellite_rl.__file__).resolve())!=str(repo/'src/satellite_rl/__init__.py') or not torch.cuda.is_available() or torch.version.cuda is None:raise ValueError('new frozen package/DQN/CUDA required')
if 'DQN' not in sys.prefix or MODEL_FORMAT!='s1_ai17_task_ranker_v2':raise ValueError('wrong DQN/model schema')
torch.set_num_threads(1);torch.manual_seed(51001);torch.cuda.manual_seed_all(51001)
model=make_model().cuda();before={{k:v.detach().cpu().clone() for k,v in model.state_dict().items()}}
optimizer=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.0001);rows=[]
for step in (1,2):
 batch={{'preferred':torch.randn(4,53,device='cuda'),'other':torch.randn(4,53,device='cuda'),
 'auxiliary':torch.randn(4,53,device='cuda'),'auxiliary_target':torch.tensor([0.,1.,0.,1.],device='cuda')}}
 optimizer.zero_grad(set_to_none=True);terms=loss(model,batch)
 if not all(torch.isfinite(t).all().item() for t in terms):raise ValueError('nonfinite synthetic loss')
 terms[0].backward()
 if not all(torch.isfinite(p.grad).all().item() for p in model.parameters() if p.grad is not None):raise ValueError('nonfinite synthetic gradient')
 norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();torch.cuda.synchronize()
 if not all(torch.isfinite(p).all().item() for p in model.parameters()):raise ValueError('nonfinite updated parameters')
 rows.append({{'update':step,'loss':float(terms[0]),'gradient_norm':float(norm)}})
changed=[k for k,v in model.state_dict().items() if not torch.equal(before[k],v.detach().cpu())]
if not changed:raise ValueError('two updates did not change any model tensor')
source_hashes={{name:hashlib.sha256((repo/'src/satellite_rl/coverage/ai_search'/name).read_bytes()).hexdigest() for name in ('learning.py','features.py','config.py')}}
record={{'group':'S1-AI17','passed':True,'scope':'synthetic_engineering_only_no_scientific_model_or_test_data',
 'source_commit':commit,'gpu_uuid':uuid,'ssh_port':{port},'engineering_updates':2,'source_hashes':source_hashes,
 'python':sys.executable,'prefix':sys.prefix,'package':satellite_rl.__file__,'torch':torch.__version__,
 'cuda_build':torch.version.cuda,'gpu_name':torch.cuda.get_device_name(0),'feature_schema':FEATURE_SCHEMA,
 'model_format':MODEL_FORMAT,'parameters':sum(p.numel() for p in model.parameters()),'changed_tensors':changed,
 'updates':rows,'all_finite':True,'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}}
from satellite_rl.training.run_monitor import atomic_json
atomic_json(path,record);print(json.dumps({{'receipt':record,'path':str(path),'reused_same_source_same_GPU':False}}))
'''
    result=ops.remote_python(code,port,timeout=65)
    if result['exit_code']:raise RuntimeError(f'{port}: '+result['stderr'][-1200:])
    return json.loads(result['stdout'].splitlines()[-1])


def main(commit,ports=(20418,20424,20473)):
    with ThreadPoolExecutor(max_workers=3) as pool:rows=dict(zip(map(str,ports),pool.map(lambda p:verify(p,commit),ports)))
    value={'group':'S1-AI17','source_commit':commit,'lanes':rows,
           'engineering_updates_newly_executed':sum(0 if row['reused_same_source_same_GPU'] else 2 for row in rows.values()),
           'scientific_updates':0,'test_accessed':False}
    atomic_json(ops.LOCAL/'preflight'/('cuda_new_source_'+commit+'.json'),value)
    return value


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--commit',required=True)
    args=parser.parse_args();print(json.dumps(main(args.commit),ensure_ascii=False))
