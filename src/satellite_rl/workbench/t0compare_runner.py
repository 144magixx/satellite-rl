"""S2-T24 original T0 Direct-only matrix worker; 560 committed rollouts."""
from pathlib import Path
from copy import deepcopy
import argparse,json,time,hashlib
import numpy as np
from satellite_rl.workbench.t0compare_training import _evaluation_episode,_load_scene,_atomic_json,_sha256,isolated_rng,run_training_job
from satellite_rl.workbench.t0compare_checkpoints import load_policy
from satellite_rl.t0_comparison.environment import make_env
from satellite_rl.workbench.datasets import scene_set_files
from satellite_rl.workbench.t0compare_contract import REPEATS,reject_third_stage
class DirectBaseline:
 def __init__(self,name,env):self.name=name;self.env=env;self.last_action_info={}
 def reset_episode(self):pass
 def act(self,state,deterministic=False,action_mask=None):
  mask=action_mask or self.env.action_mask();pairs=np.argwhere(mask['frequency_slots']);powers=np.flatnonzero(mask['power']);levels=self.env.config['environment']['power_levels_w'];raw=self.env.raw_env
  if self.name=='random':
   f,k=pairs[np.random.randint(len(pairs))];p=powers[np.random.randint(len(powers))];return int(f),int(k),int(p)
  from satellite_rl.baselines.fixed import fixed_policy
  from satellite_rl.baselines.greedy import greedy_policy
  group,wantedf,wantedn,db=(fixed_policy if self.name=='fixed' else greedy_policy)(raw);wantedW=10**(db/10);clamped=max(0,min(99,int(wantedf)))
  f,k=min(pairs.tolist(),key=lambda pair:(abs(pair[1]+1-wantedn),abs(pair[0]-clamped),pair[0],pair[1]));p=min(powers.tolist(),key=lambda p:(abs(levels[p]-wantedW),levels[p],p));self.last_action_info={'original_preference':{'group':int(group),'frequency':int(wantedf),'slots':int(wantedn),'power_w':float(wantedW)},'legal_mapping':[f,k,p],'direct_only':True};return f,k,p

def run_matrix(directory):
 d=Path(directory);job=json.loads((d/'job.json').read_text());reject_third_stage(job.get('request_body',{}));config=json.loads((d/'config.json').read_text());refs=scene_set_files(job['scene_sets']['test']);assert len(refs)==40;models={}
 for method,ref in job['endpoint_models'].items():
  for row in ref['model_files']:
   if _sha256(row['path'])!=row['sha256']:raise ValueError('frozen endpoint input SHA differs')
  policy=load_policy(ref['model_path'],device='cuda');metadata=policy.metadata
  if metadata.get('provenance',{}).get('env_step',metadata.get('env_step'))!=150000:raise ValueError('matrix requires full150k endpoint')
  models[method]=(policy,metadata)
 methods=list(REPEATS);rows=[]
 for i,(path,digest) in enumerate(refs):
  scene=_load_scene(path);order=methods[i%6:]+methods[:i%6]
  for method in order:
   for repeat in range(REPEATS[method]):
    if (d/'stop.flag').exists():_atomic_json(d/'status.json',{'run_id':job['run_id'],'stage':'stopped','cases_done':len(rows)});return
    case=d/'direct_cases'/f'{i:03d}_{method}_r{repeat}.json';control=case.with_suffix('.control.json');case.parent.mkdir(exist_ok=True)
    if control.exists():
     c=json.loads(control.read_text());assert c['scene_sha256']==digest and c['method']==method and c['repeat']==repeat and _sha256(case)==c['direct_sha256'];rows.append(c);continue
    if case.exists():raise ValueError('uncommitted Direct file requires root byte/budget recovery audit; no replay')
    seed=275000+i+repeat*1000000;began=time.perf_counter()
    with isolated_rng(seed):
     env=make_env(config)
     if method in models:
      policy=models[method][0];saved=policy.agent.rng.getstate();policy.agent.rng.seed(seed+7321)
     else:policy=DirectBaseline(method,env);saved=None
     try:result=_evaluation_episode(env,policy,scene,f'test_{i}_{method}_r{repeat}',seed,'stochastic' if REPEATS[method]>1 else 'deterministic',policy_name=method)
     finally:
      if saved is not None:policy.agent.rng.setstate(saved)
    result.update(method=method,repeat=repeat,scene_index=i,scene_sha256=digest,seed=seed,phase='Direct',timing={'direct_total_seconds':time.perf_counter()-began});_atomic_json(case,result);c={'scene_index':i,'scene_sha256':digest,'method':method,'repeat':repeat,'seed':seed,'direct_path':str(case),'direct_sha256':_sha256(case)};_atomic_json(control,c);rows.append(c);_atomic_json(d/'status.json',{'run_id':job['run_id'],'stage':'evaluating','cases_done':len(rows),'cases_total':560,'phase':'Direct','refinement_calls':0})
 if len(rows)!=560:raise ValueError('authorized Direct560 count differs')
 _atomic_json(d/'results.json',{'run_id':job['run_id'],'case_files':rows,'cases_done':560,'phase':'Direct','refinement_calls':0,'training_interactions':0});_atomic_json(d/'status.json',{'run_id':job['run_id'],'stage':'completed','cases_done':560,'phase':'Direct','refinement_calls':0})
def main():
 p=argparse.ArgumentParser();p.add_argument('--job-dir',type=Path,required=True);p.add_argument('--resume',action='store_true');a=p.parse_args();job=json.loads((a.job_dir/'job.json').read_text());mode=job['t0_mode']
 if mode=='train':run_training_job(a.job_dir,resume=a.resume)
 elif mode=='matrix':run_matrix(a.job_dir)
 else:raise ValueError('S2-T24 rejects non Direct/train modes')
if __name__=='__main__':main()
