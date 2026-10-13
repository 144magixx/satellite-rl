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


def model_identity(method,job):
 if method in ('sac','ppo','dqn'):
  ref=job['endpoint_models'][method];return {'algorithm':method,'model_run_id':ref['model_run_id'],'env_step':150000,'seed':51001,'model_files':[{k:row[k] for k in ('sha256',)}|{'name':Path(row['path']).name} for row in ref['model_files']]}
 return {'algorithm':method,'source_code_sha256':_sha256(__file__),'baseline_design':job['design_contract']['baselines'],'no_learned_model':True}
def validate_direct_result(result,scene,index,digest,method,repeat,seed,identity):
 from satellite_rl.workbench.environment import _sgm
 expected={'scene_index':index,'scene_sha256':digest,'method':method,'repeat':repeat,'seed':seed,'phase':'Direct','model_identity':identity}
 if any(result.get(k)!=v for k,v in expected.items()):raise ValueError('Direct committed case source/model/repetition identity differs')
 beams=result['beams'];trace=result['trace'];metrics=result['metrics']
 if len(beams)!=100 or len(trace)!=100 or metrics.get('complete') is not True or metrics.get('processed_count')!=100:raise ValueError('all100 Direct case incomplete')
 rates=np.array([row['rate_bps'] for row in beams]);demands=np.asarray(scene['rate']).reshape(-1);sgm=_sgm(rates,demands,float(demands.mean()))
 if not np.isfinite(rates).all() or (rates<0).any() or (demands<=0).any():raise ValueError('nonfinite/invalid all100 physical arrays')
 used={g:np.zeros(100,dtype=bool) for g in range(8)};power=0.
 for i,(beam,row) in enumerate(zip(beams,trace)):
  if beam['beam_id']!=i or beam['original_index']!=int(scene['original_indices'][i]) or beam['lat']!=float(scene['beam_location'][i,0]) or beam['lon']!=float(scene['beam_location'][i,1]) or beam['required_bps']!=float(demands[i]):raise ValueError('Direct beam/plane source identity differs')
  group=beam['group'];f=beam['freq'];n=beam['slots'];p=beam['power_w']
  if group!=i%8 or type(f) is not int or type(n) is not int or not 1<=n<=10 or not 0<=f or f+n>100 or p<5-1e-9 or p>50+1e-9 or not any(abs(p-v)<1e-8 for v in range(5,51,5)) or used[group][f:f+n].any():raise ValueError('Direct discrete/power/group feasibility violated')
  used[group][f:f+n]=True;power+=p
  req=row['requested_action'];actual=row['actual_action']
  if req['freq']!=actual['freq'] or req['slots']!=actual['slots'] or abs(req['power_w']-actual['power_w'])>1e-9:raise ValueError('hidden clipping of masked Direct action')
  if abs(beam['si']-min(rates[i]/demands[i],1))>1e-9 or abs(beam['sgm']-sgm[i])>1e-9:raise ValueError('Direct allN SI/SGM differs')
 if power>3000+1e-6 or abs(metrics['final_si_mean']-np.minimum(rates/demands,1).mean())>1e-9 or abs(metrics['final_sgm_mean']-sgm.mean())>1e-9:raise ValueError('Direct final all100 summary/resource differs')
 return True

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
    seed=275000+i+repeat*1000000;identity=model_identity(method,job)
    case=d/'direct_cases'/f'{i:03d}_{method}_r{repeat}.json';control=case.with_suffix('.control.json');case.parent.mkdir(exist_ok=True)
    if control.exists():
     c=json.loads(control.read_text());expected={'scene_index':i,'scene_sha256':digest,'method':method,'repeat':repeat,'seed':seed,'model_identity':identity,'phase':'Direct','direct_path':str(case)}
     if any(c.get(k)!=v for k,v in expected.items()) or _sha256(case)!=c['direct_sha256']:raise ValueError('committed control/model/source/path SHA differs')
     validate_direct_result(json.loads(case.read_text(encoding='utf8')),scene,i,digest,method,repeat,seed,identity);rows.append(c);continue
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
    result.update(method=method,repeat=repeat,scene_index=i,scene_sha256=digest,seed=seed,phase='Direct',model_identity=identity,timing={'direct_total_seconds':time.perf_counter()-began});validate_direct_result(result,scene,i,digest,method,repeat,seed,identity);_atomic_json(case,result);c={'scene_index':i,'scene_sha256':digest,'method':method,'repeat':repeat,'seed':seed,'direct_path':str(case),'direct_sha256':_sha256(case),'model_identity':identity,'phase':'Direct'};_atomic_json(control,c);rows.append(c);_atomic_json(d/'status.json',{'run_id':job['run_id'],'stage':'evaluating','cases_done':len(rows),'cases_total':560,'phase':'Direct','refinement_calls':0})
 if len(rows)!=560:raise ValueError('authorized Direct560 count differs')
 _atomic_json(d/'results.json',{'run_id':job['run_id'],'case_files':rows,'cases_done':560,'phase':'Direct','refinement_calls':0,'training_interactions':0});_atomic_json(d/'status.json',{'run_id':job['run_id'],'stage':'completed','cases_done':560,'phase':'Direct','refinement_calls':0})
def run_prepare(directory):
 from satellite_rl.workbench.t0compare_data import prepare_test_data
 d=Path(directory);job=json.loads((d/'job.json').read_text());config=deepcopy(job['data_config']);admission=d/'data_preparation_admission.json';config['admission_reference']={'path':str(admission),'sha256':_sha256(admission)};reject_third_stage(config)
 def progress(stage,done,total,detail):_atomic_json(d/'status.json',{'run_id':job['run_id'],'stage':stage,'data_items_done':done,'data_items_total':total,'detail':detail,'learning_interactions':0,'refinement_calls':0})
 result=prepare_test_data(config,d/'prepared',progress=progress,stop_requested=lambda:(d/'stop.flag').exists())
 if result.get('status')!='completed' or result['scene_sets']['test']['count']!=40:raise ValueError('prepare actual40 complete proof absent')
 _atomic_json(d/'results.json',result);_atomic_json(d/'status.json',{'run_id':job['run_id'],'stage':'completed','data_scene_count':40,'learning_interactions':0,'refinement_calls':0})


def strict_source_preflight(job,require_cuda):
 import satellite_rl,sys,torch,subprocess
 root=Path(job['source_root']).resolve(strict=True);manifest=json.loads((root/'source_manifest.json').read_text(encoding='utf8'))
 if manifest['git_commit']!=job['git_commit'] or not manifest['files']:raise ValueError('frozen source manifest/commit missing')
 for rel,digest in manifest['files'].items():
  p=(root/rel).resolve(strict=True)
  if not p.is_relative_to(root) or _sha256(p)!=digest:raise ValueError('frozen source member changed')
 if not Path(satellite_rl.__file__).resolve().is_relative_to(root/'src'):raise ValueError('newnamespace import not frozen source')
 if Path(sys.prefix).name.lower()!='dqn' or 'envs' not in [x.lower() for x in Path(sys.prefix).parts]:raise ValueError('DQN actual interpreter required')
 if require_cuda and (torch.version.cuda is None or not torch.cuda.is_available()):raise ValueError('actualCUDA unavailable')
 resource=json.loads((Path(job['archive_root'])/'outputs/workbench/jobs'/job['run_id']/'resource_admission.json').read_text(encoding='utf8'));q=subprocess.run(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=15)
 if q.returncode or q.stdout.strip()!=resource['gpu_uuid']:raise ValueError('actualGPUidentity not admitted lane')
 return {'actual_python':sys.executable,'prefix':sys.prefix,'package_file':satellite_rl.__file__,'source_manifest_sha256':_sha256(root/'source_manifest.json'),'CUDA_required':require_cuda,'GPU_query_exit':q.returncode,'GPU_uuid':q.stdout.strip(),'torch':torch.__version__}

def validate_worker_identity(directory):
 from satellite_rl.workbench.t0compare_service import BASE,PORTS
 from satellite_rl.workbench.storage import digest
 d=Path(directory).resolve(strict=True);job=json.loads((d/'job.json').read_text());port=job.get('executor_port')
 if port not in PORTS or job.get('experiment_id')!='S2-T24' or not d.is_relative_to(BASE/str(port)/'archive/outputs/workbench/jobs') or d.name!=job.get('run_id'):raise ValueError('worker exact ownlane job containment required')
 owner_ref=job['unique_owner_ref'];owner=json.loads(Path(owner_ref['path']).read_text(encoding='utf8'))
 if digest(owner_ref['path'])!=owner_ref['sha256'] or any(owner.get(k)!=job.get(k) for k in ('run_id','request_id','executor_port','git_commit','request_hash')):raise ValueError('unique scientific owner identity differs')
 authority=job['request_body']['uniform_request_or_explicit_historical_reproduction_authority_ref']
 if digest(authority['path'])!=authority['sha256']:raise ValueError('historical/data purpose authority changed')
 for row in job.get('scientific_reference_files',[]):
  if not Path(row['path']).resolve().is_relative_to(d/'scientific_references') or digest(row['path'])!=row['sha256']:raise ValueError('frozen cleaner/generator ref changed')
 return job

def main():
 p=argparse.ArgumentParser();p.add_argument('--job-dir',type=Path,required=True);p.add_argument('--resume',action='store_true');a=p.parse_args();job=validate_worker_identity(a.job_dir);mode=job['t0_mode'];worker_lock=(a.job_dir/'worker.lock').open('a+b')
 import fcntl
 fcntl.flock(worker_lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
 try:
  runtime=strict_source_preflight(job,require_cuda=mode!='prepare');_atomic_json(a.job_dir/'runtime_bootstrap.json',runtime)
  if mode=='prepare':run_prepare(a.job_dir)
  elif mode=='engineering':
   from satellite_rl.workbench.t0compare_engineering import run_engineering
   run_engineering(a.job_dir)
  elif mode=='train':run_training_job(a.job_dir,resume=a.resume)
  elif mode=='matrix':run_matrix(a.job_dir)
  else:raise ValueError('S2-T24 rejects non Direct/train modes')
 except InterruptedError as error:_atomic_json(a.job_dir/'status.json',{'run_id':job['run_id'],'stage':'stopped','error':str(error),'refinement_calls':0})
 except Exception as error:
  _atomic_json(a.job_dir/'failure.json',{'run_id':job['run_id'],'error_type':type(error).__name__,'error':str(error),'new_science_not_automatically_replayed':True});_atomic_json(a.job_dir/'status.json',{'run_id':job['run_id'],'stage':'attention_required','error':str(error),'refinement_calls':0});raise
 finally:worker_lock.close()
if __name__=='__main__':main()
