"""Strict S2-T24 Direct-only contract adapters; no third-stage execution."""
from copy import deepcopy
import hashlib,json
from pathlib import Path
from satellite_rl.workbench.config import validate_config as original_validate
from satellite_rl.t0_comparison.environment import environment_config,summary
from satellite_rl.t0_comparison.algorithms import validate_contract
from satellite_rl.t0_comparison.contract import canonical_config
ALGORITHMS=('sac','ppo','dqn');MODES=('prepare','train','matrix');NODES=[0,2000,5000]+list(range(10000,150001,10000))
REPEATS={'sac':3,'ppo':3,'dqn':3,'random':3,'greedy':1,'fixed':1}
FORBIDDEN={'refine','refinement','power_refine','full_b','full_B','A+B','B','optimizer','sca','sca_si_lp'}
def fingerprint(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def reject_third_stage(value):
 if isinstance(value,dict):
  for key,item in value.items():
   if key in FORBIDDEN:raise ValueError('S2-T24 Direct-only rejects third-stage field '+key)
   reject_third_stage(item)
 elif isinstance(value,list):
  for item in value:reject_third_stage(item)
 elif isinstance(value,str) and value in FORBIDDEN:raise ValueError('S2-T24 Direct-only rejects third-stage mode')
def training_config(reference,algorithm):
 if algorithm not in ALGORITHMS:raise ValueError('one of three authorized algorithms required')
 c=deepcopy(reference);c['algorithm']=algorithm;c['name']='S2-T24 '+algorithm;c['training'].update(seed=51001,total_steps=150000,checkpoint_interval=10000,log_interval=50,device='cuda',scene_order='shuffle_each_cycle',scene_order_seed=51001,gamma=.99,actor_lr=1e-5,critic_lr=1e-4,batch_size=256,replay_capacity=20000,warmup_steps=2000,update_every=1,gradient_steps=1,tau=.005,alpha_mode='fixed',fixed_alpha=.01,gradient_clip=10,rollout_steps=256,ppo_epochs=4,gae_lambda=.95,clip_ratio=.2,entropy_coef=.01,value_coef=.5,eps_start=1,eps_end=.05,eps_decay_steps=5000,target_update_interval=500)
 c['evaluation'].update(validation_count=20,test_count=40,validation_seed=274000,test_seed=275000,interval_steps=10000,extra_steps=[0,2000,5000],repeats=3,action_mode='stochastic')
 return c
def validate_t0_config(raw,reference=None):
 reject_third_stage(raw);c=canonical_config(raw,purpose='science');algorithm=c['algorithm']
 if algorithm not in ALGORITHMS:raise ValueError('unsupported training algorithm')
 # Only environment validation uses SAC view, externally stored configs remain actual PPO/DQN.
 original_validate(environment_config(c));dims=summary(c);validate_contract(c,dims['state_dim'],dims['action_dims'])
 if c['training']['total_steps']!=150000 or c['training']['seed']!=51001 or c['evaluation']['validation_count']!=20 or c['evaluation']['repeats']!=3 or c['evaluation']['extra_steps']!=[0,2000,5000] or c['evaluation']['interval_steps']!=10000:raise ValueError('finite seed/node/evaluation contract changed')
 if reference is not None and c!=training_config(reference,algorithm):raise ValueError('actual T0 reference/config differs')
 return c
