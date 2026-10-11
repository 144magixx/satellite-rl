"""Bounded engineering CUDA updates on all authorized, frozen source lanes."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from . import ops
from .data_controller import save


def lane(port, commit):
    repo = ops.instance(port)['repo']
    code = f'''
import json,os,pathlib,subprocess
repo=pathlib.Path({repo!r})
head=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
if head!={commit!r} or subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True).strip():raise ValueError('source mismatch')
env=os.environ.copy();env.update(PYTHONPATH=str(repo/'src')+':'+str(repo/'tests'),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',SATELLITE_RL_OUTPUT_ROOT=str(repo.parent/'ops/engineering_smoke'))
script="""
import json,torch,satellite_rl,time
from satellite_rl.coverage.nondrl.models import make_model,supervised_loss
from satellite_rl.coverage.nondrl.learning import collate
from test_nondrl_learning import examples
assert torch.cuda.is_available()
torch.set_num_threads(1);torch.manual_seed(51001)
rows=[]
for route,sample in examples().items():
 torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
 model=make_model(route).cuda();batch=collate([sample,sample],route,'cuda')
 optimizer=torch.optim.Adam(model.parameters());before=[p.detach().clone() for p in model.parameters()]
 started=time.perf_counter();optimizer.zero_grad();loss=supervised_loss(route,model(batch),batch,[sample,sample]);assert torch.isfinite(loss)
 loss.backward();optimizer.step();torch.cuda.synchronize();assert any(not torch.equal(p,q) for p,q in zip(model.parameters(),before))
 rows.append(dict(route=route,parameters=sum(p.numel() for p in model.parameters()),updates=1,loss=float(loss.detach()),seconds=time.perf_counter()-started,peak_bytes=torch.cuda.max_memory_allocated()))
 del optimizer,model,batch,before,loss
print(json.dumps(dict(passed=True,package=satellite_rl.__file__,torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0),checks=rows,scope='five actual default-network CUDA Adam updates; engineering only; no effect claim')))
"""
r=subprocess.run([{ops.DQN!r},'-B','-c',script],cwd=repo,env=env,capture_output=True,text=True,timeout=120)
print(json.dumps(dict(source_commit=head,port={port},exit_code=r.returncode,stdout=r.stdout,stderr=r.stderr)))
'''
    value = ops.remote_python(code, port, timeout=150)
    if value['exit_code']:
        return {'port': port, 'passed': False, 'error': value['stderr'][-1000:]}
    result = json.loads(value['stdout'].splitlines()[-1])
    result['passed'] = result['exit_code'] == 0
    if result['passed']:
        result['checks'] = json.loads(result['stdout'].splitlines()[-1])
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--commit', required=True)
    args = parser.parse_args()
    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(lambda p: lane(p, args.commit), ops.SSH_PORTS))
    result = {'scientific_commit': args.commit, 'passed': all(r['passed'] for r in rows),
              'engineering_updates_total': sum(5 for r in rows if r['passed']), 'lanes': rows}
    save(ops.LOCAL / 'preflight/remote_cuda_smoke.json', result)
    print(json.dumps(result, ensure_ascii=False))
    raise SystemExit(0 if result['passed'] else 1)


if __name__ == '__main__':
    main()
