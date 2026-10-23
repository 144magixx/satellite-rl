"""W30 adapters around unchanged WorkbenchAgent factorized masked SAC math."""
from copy import deepcopy
import hashlib
import math
import random
import numpy as np
import torch
from torch import nn
from satellite_rl.agents.workbench_algorithms import WorkbenchAgent, _joint_q, _gather, _diagnostics, _state_tensor
from torch.nn import functional as F
from satellite_rl.t0_width.agent import build_config as t0_config
from satellite_rl.t0_width.agent import WidthEncoder
from satellite_rl.agents.action_masking import copy_action_mask, mask_batch, masked_policy, feasible_target_entropy, entropy_ratio
from .menus import power_table
from .schema import GROUPS, observation_schema, state_width, schema_identity


def build_config(menu, group, device='cpu', phase='scientific', seed=51001):
    table=power_table(menu)
    if group not in GROUPS:raise ValueError('Unknown W30 summary group')
    config=t0_config('S10',device,phase)
    config.update(name='S2-W30 '+menu+' '+group,menu=menu,group=group)
    config['environment']['power_levels_w']=list(table)
    config['training'].update(seed=seed,scene_order_seed=seed)
    config['evaluation'].update(interval_steps=50000,extra_steps=[])
    return config


def state_dimension(group):return state_width(group)


class Power30Encoder(nn.Module):
    def __init__(self, base, group, seed, encoder='transformer'):
        super().__init__()
        self.group=group;self.state_dim=state_width(group);self.output_dim=base.output_dim
        self.encoder=encoder
        for name in ('geo_embed','rate_embed','power_embed','slot_embed','net','global_embed'):
            setattr(self,name,deepcopy(getattr(base,name)))
        self.pos_embed=nn.Parameter(base.pos_embed.detach().clone())
        self.extra_pos_embed=nn.Parameter(base.extra_pos_embed.detach().clone())
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed+17011)
            self.geo_embed=WidthEncoder._extend(self.geo_embed,3)
            if GROUPS[group]:
                torch.random.default_generator.manual_seed(seed+30011)
                self.slot_embed=WidthEncoder._extend(self.slot_embed,30)
        self.last_token_count=14
        if encoder == 'mlp':
            width = self.pos_embed.shape[-1]
            target = sum(p.numel() for p in self.net.parameters())
            hidden = round((target - self.output_dim) / (14*width + self.output_dim + 1))
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(seed + 61001)
                self.net = nn.Sequential(nn.Linear(14*width, hidden), nn.ReLU(),
                                         nn.Linear(hidden, self.output_dim))

    def forward(self,states,lengths=None):
        if lengths is not None or states.ndim!=2 or states.shape[1]!=self.state_dim:
            raise ValueError('W30 fixed-width state schema mismatch')
        if not bool(torch.isfinite(states).all()):raise ValueError('Nonfinite W30 state')
        batch=states.shape[0]
        geo=torch.cat((states[:,:2],states[:,207:208]),dim=-1)
        channels=[states[:,4:104].reshape(batch,10,10),states[:,104:204].reshape(batch,10,10)]
        if GROUPS[self.group]:channels.append(states[:,208:308].reshape(batch,10,10))
        tokens=torch.cat((self.geo_embed(geo).unsqueeze(1),self.rate_embed(states[:,2:3]).unsqueeze(1),
                          self.power_embed(states[:,3:4]).unsqueeze(1),
                          self.slot_embed(torch.cat(channels,dim=-1))),dim=1)+self.pos_embed
        global_token=self.global_embed(states[:,204:207]).unsqueeze(1)+self.extra_pos_embed
        sequence=torch.cat((tokens,global_token),dim=1).permute(1,0,2)
        if self.encoder == 'mlp':
            return self.net(sequence.permute(1,0,2).reshape(batch,-1))
        return self.net(sequence).permute(1,0,2)[:,0]


class Power30Agent(WorkbenchAgent):
    def __init__(self,config,device=None,call_hook=None):
        menu=config['menu'];group=config['group'];table=power_table(menu)
        expected=build_config(menu,group,device or config['training']['device'],config.get('phase','scientific'),config['training']['seed'])
        expected['network'].update(config.get('network_overrides',{}))
        for section,keys in {
            'network':('kind','embedding_dim','num_heads','num_layers','feedforward_dim','activation','dropout',
                       'sac_policy_structure','sac_critic_structure','temporal_mode','graph_mode',
                       'context_beam_count','readout_mode','transformer_layout','beam_count_conditioning'),
            'training':('gamma','tau','alpha_mode','fixed_alpha','actor_lr','critic_lr','alpha_lr',
                        'initial_alpha','target_entropy','progress_strata','batch_size','replay_capacity',
                        'warmup_steps','update_every','gradient_steps','gradient_clip'),
            'state':('features','demand_visibility','context_level'),
            'reward':('preset',),
            'environment':('action_mask','power_levels_w','power_budget_w')}.items():
            if any(config[section][key]!=expected[section][key] for key in keys):
                raise ValueError('W30 frozen contract differs in '+section)
        if config['algorithm']!='sac':raise ValueError('W30 requires T0 SAC')
        if any(config['reward'].get('shaping',{}).get(k,0)!=0 for k in ('w_throughput','w_spectrum','w_shape')):
            raise ValueError('W30 does not add reward shaping')
        # A ten-row base keeps shared encoders/frequency/slot heads on the same stream
        # for every menu. Only the final power head is replaced, by physical watt row.
        super().__init__(config,207,(100,10,10),device=device or config['training']['device'])
        self.menu=menu;self.group=group;self.power_table=table;self.action_dims=(100,10,len(table))
        self.target_entropy_ratio=entropy_ratio(self.train["target_entropy"],self.action_dims)
        self.call_hook=call_hook;self.call_counts={}
        cpu_before=torch.get_rng_state().clone()
        cuda_before=torch.cuda.get_rng_state(self.device).clone() if self.device.type=='cuda' else None
        for index,name in enumerate(('actor','critic1','critic2')):
            network=getattr(self,name)
            component_seed=self.train['seed']+100003*(index+1)
            network.encoder=Power30Encoder(network.encoder,group,component_seed,
                                          config.get('encoder','transformer')).to(self.device)
            if menu!='P00':
                old=network.heads[2]
                # Explicit independent per-watt rows preserve matching new rows across
                # menus as well as matching old 5W rows. fork restores policy RNG.
                with torch.random.fork_rng(devices=[]):
                    torch.random.default_generator.manual_seed(component_seed+40009)
                    head=nn.Linear(old.in_features,len(table),dtype=old.weight.dtype)
                    with torch.no_grad():
                        for row,watt in enumerate(table):
                            if watt in power_table('P00'):
                                old_row=power_table('P00').index(watt)
                                head.weight[row].copy_(old.weight[old_row].cpu());head.bias[row].copy_(old.bias[old_row].cpu())
                            else:
                                torch.random.default_generator.manual_seed(component_seed+50021+int(watt*2)*997)
                                bound=1/math.sqrt(old.in_features)
                                head.weight[row].uniform_(-bound,bound);head.bias[row].uniform_(-bound,bound)
                network.heads[2]=head.to(self.device)
        self.target1=deepcopy(self.critic1).eval();self.target2=deepcopy(self.critic2).eval()
        cpu_after=torch.get_rng_state()
        cuda_after=torch.cuda.get_rng_state(self.device) if self.device.type=='cuda' else None
        if not torch.equal(cpu_before,cpu_after) or (cuda_before is not None and not torch.equal(cuda_before,cuda_after)):
            raise RuntimeError('W30 additions shifted action RNG')
        self.extension_rng = {} if config.get('s2fix') else dict(
            cpu_unchanged=True,cuda_unchanged=None if cuda_before is None else True,
            cpu_SHA=hashlib.sha256(cpu_after.cpu().numpy().tobytes()).hexdigest(),
            cuda_SHA=None if cuda_after is None else hashlib.sha256(cuda_after.cpu().numpy().tobytes()).hexdigest())
        self.actor_opt=torch.optim.Adam(self.actor.parameters(),lr=self.train['actor_lr'])
        self.critic1_opt=torch.optim.Adam(self.critic1.parameters(),lr=self.train['critic_lr'])
        self.critic2_opt=torch.optim.Adam(self.critic2.parameters(),lr=self.train['critic_lr'])
        self.schema=schema_identity(menu,group,plain=config.get('s2fix',False))
        if config.get('s2fix'):
            self.schema.update(version='s2fix_tsac_v1',encoder=config.get('encoder','transformer'))
        # Pre-hooks count every whole-network execution, including direct act() paths.
        self._meter_handles=[]
        for name in ('actor','critic1','critic2','target1','target2'):
            self._meter_handles.append(getattr(self,name).register_forward_pre_hook(self._forward_meter(name)))

    def _forward_meter(self,name):
        def meter(module,inputs):
            metadata=dict(network=name,batch_size=int(inputs[0].shape[0]))
            if self.call_hook is not None:self.call_hook('netforward',**metadata)
            self.call_counts['netforward']=self.call_counts.get('netforward',0)+1
        return meter

    def _upload(self,tensor):
        # 锁页内存异步上传：主机不必等GPU。多个进程共用一张卡时，每次等待都要排一轮GPU时间片。
        return tensor.pin_memory().to(self.device,non_blocking=True) if self.device.type=='cuda' else tensor

    def act(self,state,deterministic=False,env_step=None,action_mask=None,**kwargs):
        # 计算与采样顺序同 WorkbenchAgent.act 的分解式掩码路径（结果逐位相同），
        # 只是输入异步上传、动作和记录量一次取回。
        mask=copy_action_mask(action_mask,self.action_dims)
        was_training=self.actor.training
        self.actor.eval()
        with torch.no_grad():
            x=self._upload(_state_tensor(state,'cpu').reshape(1,-1))
            frequency_slots,power=(self._upload(t) for t in mask_batch([mask],self.action_dims,'cpu'))
            fl,p,log_fl,log_p,entropy_fl,entropy_p,entropy_f,entropy_slots=masked_policy(
                self.actor(x),frequency_slots,power,check=False)
            flat_fl=fl.flatten(1)
            fl_index=(flat_fl.argmax(1) if deterministic else
                      torch.distributions.Categorical(probs=flat_fl,validate_args=False).sample())
            power_index=(p.argmax(1) if deterministic else
                         torch.distributions.Categorical(probs=p,validate_args=False).sample())
            frequency,slots=fl_index//self.action_dims[1],fl_index%self.action_dims[1]
            values=torch.stack((frequency[0].float(),slots[0].float(),power_index[0].float(),
                                log_fl[0,frequency[0],slots[0]]+log_p[0,power_index[0]],
                                (entropy_fl+entropy_p)[0],entropy_f[0],entropy_slots[0],entropy_p[0],
                                feasible_target_entropy(frequency_slots,power,self.target_entropy_ratio)[0])).cpu().tolist()
        self.actor.train(was_training)
        result=tuple(int(v) for v in values[:3])
        legal=int(np.count_nonzero(mask['power']));maximum=math.log(legal);entropy=values[7]
        self.last_action_info=dict(log_prob=values[3],entropy=values[4],entropy_frequency=values[5],
                                   entropy_slots=values[6],entropy_power=entropy,target_entropy=values[8],
                                   legal_power_count=legal,power_maximum_entropy=maximum,
                                   power_normalized_entropy=entropy/maximum if maximum else 0.,
                                   alpha_times_power_entropy=float(self.train['fixed_alpha'])*entropy,
                                   requested_power_w=self.power_table[result[2]])
        return result

    def observe(self,state,action,reward,next_state,done,action_mask=None,next_action_mask=None,**kwargs):
        for value in (state,next_state):
            array=np.asarray(value)
            if array.shape!=(self.schema['dimension'],) or not np.isfinite(array).all():
                raise ValueError('W30 replay observation schema mismatch')
        if len(action)!=3 or any(isinstance(x,(bool,np.bool_)) or not isinstance(x,(int,np.integer)) for x in action):
            raise ValueError('W30 replay requires three integer indices')
        mask=copy_action_mask(action_mask,self.action_dims)
        f,l,p=map(int,action)
        if not (0<=f<100 and 0<=l<10 and 0<=p<len(self.power_table) and mask['frequency_slots'][f,l] and mask['power'][p]):
            raise ValueError('W30 replay action outside menu/mask')
        return super().observe(state,action,reward,next_state,done,action_mask,next_action_mask,**kwargs)

    def _update_masked_sac(self):
        if self.temporal_enabled:
            return self._update_temporal_sac()
        if self.sac_policy_structure != "factorized" or self.sac_critic_structure != "additive":
            return self._update_joint_action_sac()
        # 算术与原分解式掩码 SAC 更新逐位相同。改动只在主机与GPU的往来：回放在CPU组批后异步上传，
        # 终止样本的行号在CPU上算好（原来用GPU布尔索引，每次都要等GPU），逐步的有限性检查
        # 合并到最后一次取回诊断量时做。
        batch = self.replay.sample(self.train["batch_size"], "cpu", self.action_dims)
        active = torch.nonzero(batch[4] < 0.5).squeeze(1)
        (s, a, r, ns, d, lengths, next_lengths, (current_fl, current_power),
         (next_fl, next_power)) = (tuple(self._upload(t) for t in item) if isinstance(item, tuple)
                                   else self._upload(item) for item in batch)
        alpha = self.log_alpha.exp().detach()
        with torch.no_grad():
            target = r.clone()
            if len(active):
                active = self._upload(active)
                (next_prob_fl, next_prob_power, _, _, next_entropy_fl,
                 next_entropy_power, _, _) = masked_policy(
                    self._sac_actor_logits(ns[active], next_lengths[active]),
                    next_fl[active], next_power[active], check=False)
                next_min_q = torch.minimum(_joint_q(self._net(self.target1, ns[active], next_lengths[active])),
                                           _joint_q(self._net(self.target2, ns[active], next_lengths[active])))
                next_expected_q = torch.einsum("bfl,bp,bflp->b", next_prob_fl,
                                               next_prob_power, next_min_q)
                target[active] += self.train["gamma"] * (
                    next_expected_q + alpha * (next_entropy_fl + next_entropy_power))
        q1 = _gather(self._net(self.critic1, s, lengths), a)
        q2 = _gather(self._net(self.critic2, s, lengths), a)
        loss1, loss2 = F.mse_loss(q1, target), F.mse_loss(q2, target)
        clip = self.train["gradient_clip"]
        self.critic1_opt.zero_grad(set_to_none=True)
        loss1.backward()
        grad1 = nn.utils.clip_grad_norm_(self.critic1.parameters(), clip)
        self.critic1_opt.step()
        self.critic2_opt.zero_grad(set_to_none=True)
        loss2.backward()
        grad2 = nn.utils.clip_grad_norm_(self.critic2.parameters(), clip)
        self.critic2_opt.step()
        (prob_fl, prob_power, _, _, entropy_fl, entropy_power,
         entropy_frequency, entropy_slots) = masked_policy(
             self._sac_actor_logits(s, lengths), current_fl, current_power, check=False)
        with torch.no_grad():
            min_q_actor = torch.minimum(_joint_q(self._net(self.critic1, s, lengths)),
                                        _joint_q(self._net(self.critic2, s, lengths)))
        expected_q_actor = torch.einsum("bfl,bp,bflp->b", prob_fl, prob_power, min_q_actor)
        entropy_actor = entropy_fl + entropy_power
        actor_loss = -(expected_q_actor + alpha * entropy_actor).mean()
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        actor_grad = nn.utils.clip_grad_norm_(self.actor.parameters(), clip)
        self.actor_opt.step()
        target_entropy = feasible_target_entropy(current_fl, current_power,
                                                  self.target_entropy_ratio)
        if self.alpha_mode == "adaptive":
            alpha_loss = self.log_alpha * (entropy_actor.detach() - target_entropy).mean()
            self.alpha_opt.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.alpha_opt.step()
        else:
            alpha_loss = torch.zeros_like(self.log_alpha.detach())
        self.update_step += 1
        tau = self.train["tau"]
        with torch.no_grad():
            for target_net, source_net in ((self.target1, self.critic1), (self.target2, self.critic2)):
                for target_p, source_p in zip(target_net.parameters(), source_net.parameters()):
                    target_p.lerp_(source_p, tau)
            finite = torch.isfinite(torch.stack((loss1 + loss2, actor_loss, alpha_loss.reshape(()),
                                                 grad1, grad2, actor_grad))).all().float()
        diagnostics = _diagnostics({"finite": finite, "actor_loss": actor_loss, "critic_1_loss": loss1,
                             "critic_2_loss": loss2, "alpha_loss": alpha_loss,
                             "alpha": self.log_alpha.exp(), "entropy": entropy_actor.mean(),
                             "entropy_frequency": entropy_frequency.mean(),
                             "entropy_slots": entropy_slots.mean(),
                             "entropy_power": entropy_power.mean(),
                             "power_maximum_entropy": current_power.sum(1).float().log().mean(),
                             "power_normalized_entropy": (entropy_power / current_power.sum(1).float().log().clamp_min(1e-30)).mean(),
                             "alpha_times_power_entropy": alpha * entropy_power.mean(),
                             "target_entropy": target_entropy.mean(),
                             "legal_frequency_slot_count": current_fl.sum((1, 2)).float().mean(),
                             "legal_power_count": current_power.sum(1).float().mean(),
                             "q_mean": (q1.mean() + q2.mean()) / 2,
                             "q_min": torch.minimum(q1, q2).min(),
                             "q_max": torch.maximum(q1, q2).max(), "target_mean": target.mean(),
                             "target_std": target.std(unbiased=False),
                             "td_abs_mean": ((q1 - target).abs().mean() + (q2 - target).abs().mean()) / 2,
                             "actor_grad_norm": actor_grad, "critic_1_grad_norm": grad1,
                             "critic_2_grad_norm": grad2,
                             **({"alpha_update_applied": 0} if self.alpha_mode == "fixed" else {})})
        if diagnostics.pop("finite") != 1.0:
            raise FloatingPointError("Nonfinite SAC loss or gradient")
        return diagnostics

    def state_dict(self):
        data=super().state_dict()
        data['w30_schema']=deepcopy(self.schema)
        data['torch_rng_cpu']=torch.get_rng_state().clone()
        data['torch_rng_cuda']=torch.cuda.get_rng_state(self.device).clone() if self.device.type=='cuda' else None
        data['numpy_rng']=np.random.get_state();data['python_rng']=random.getstate()
        return data

    def load_state_dict(self,data):
        if data.get('w30_schema')!=self.schema:raise ValueError('W30 checkpoint schema mismatch')
        required={'actor','critic1','critic2','target1','target2'}
        if set(data['networks'])!=required or not {'actor_opt','critic1_opt','critic2_opt','alpha_opt'}<=set(data['optimizers']):
            raise ValueError('W30 resume lacks complete networks/optimizers')
        for row in data['replay']['rows']:
            if any(np.asarray(row[i]).shape!=(self.schema['dimension'],) for i in (0,3)):
                raise ValueError('W30 checkpoint replay width')
            copy_action_mask(row[5],self.action_dims)
            if not row[4]:copy_action_mask(row[6],self.action_dims)
            f,l,p=row[1]
            if not (0<=f<100 and 0<=l<10 and 0<=p<len(self.power_table) and row[5]['frequency_slots'][f,l] and row[5]['power'][p]):
                raise ValueError('W30 checkpoint replay action/mask')
        for key in ('torch_rng_cpu','torch_rng_cuda','numpy_rng','python_rng'):
            if key not in data:raise ValueError('W30 resume lacks RNG '+key)
        super().load_state_dict(data)
        torch.set_rng_state(data['torch_rng_cpu'].cpu())
        if self.device.type=='cuda':
            if data['torch_rng_cuda'] is None:raise ValueError('W30 CUDA resume lacks CUDA RNG')
            torch.cuda.set_rng_state(data['torch_rng_cuda'].cpu(),self.device)
        np.random.set_state(data['numpy_rng']);random.setstate(data['python_rng'])

    def policy_state(self):
        return dict(schema=deepcopy(self.schema),actor=self.actor.state_dict(),
                    critic1=self.critic1.state_dict(),critic2=self.critic2.state_dict(),update_step=self.update_step)

    def parameter_counts(self):
        return {name:sum(p.numel() for p in getattr(self,name).parameters())
                for name in ('actor','critic1','critic2','target1','target2')}
