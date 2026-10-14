"""W27 S10 SAC: unchanged B/E mathematics and a frequency + slot-power block."""
from copy import deepcopy
import hashlib
import json
import random

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from satellite_rl.agents.action_masking import copy_action_mask, mask_batch, feasible_target_entropy
from satellite_rl.agents.workbench_algorithms import WorkbenchAgent, _diagnostics, _finite_loss, _grad_norm
from satellite_rl.t0_width.agent import WidthEncoder, build_config as width_config, schema_identity as width_schema

VARIANTS = ('B', 'E', 'LP')


def state_dimension(variant):
    if variant not in VARIANTS:
        raise ValueError('unknown W27 variant')
    return 208


def build_config(variant, device='cpu', phase='scientific', total_steps=None):
    state_dimension(variant)
    config = width_config('S10', device, phase)
    config.update(name='S2-W27 ' + variant, experiment_id='S2-W27', variant=variant)
    seed = 73027 if phase == 'engineering' else 51001
    config['training'].update(seed=seed, scene_order_seed=seed, pilot_stop_node=50000,
                              initial_alpha=.003 if variant == 'E' else .01,
                              fixed_alpha=.003 if variant == 'E' else .01)
    if total_steps is not None and total_steps != config['training']['total_steps']:
        raise ValueError('W27 immutable total cap is 150000 scientific / 2048 other phases; pilot is a stop node')
    if variant == 'LP':
        config['network'].update(sac_policy_structure='frequency_slot_power_block',
                                  sac_critic_structure='frequency_slot_power_block')
    return config


def schema_identity(variant):
    state_dimension(variant)
    result = deepcopy(width_schema('S10'))
    result.pop('sha256')
    result.update(version='W27SearchAgentV1', variant=variant,
                  policy_structure='frequency_plus_slot_power_joint' if variant == 'LP' else 'factorized_FL_times_P',
                  critic_structure='QF_plus_QLP' if variant == 'LP' else 'QF_plus_QL_plus_QP',
                  head_shapes=[100, [10, 10]] if variant == 'LP' else [100, 10, 10],
                  lp_index_order='l_major_p_minor' if variant == 'LP' else None,
                  fixed_alpha=.003 if variant == 'E' else .01)
    result['sha256'] = hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest()
    return result


class SlotPowerNetwork(nn.Module):
    """Merge old L/P heads without consuming any CPU or available CUDA RNG stream."""
    def __init__(self, base):
        super().__init__()
        self.encoder = deepcopy(base.encoder)
        self.frequency = deepcopy(base.heads[0])
        slots, power = base.heads[1:]
        devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
        with torch.random.fork_rng(devices=devices):
            self.slot_power = nn.Linear(slots.in_features, slots.out_features * power.out_features,
                                        device=slots.weight.device, dtype=slots.weight.dtype)
            with torch.no_grad():
                self.slot_power.weight.copy_((slots.weight[:, None, :] + power.weight[None, :, :]).flatten(0, 1))
                self.slot_power.bias.copy_((slots.bias[:, None] + power.bias[None, :]).flatten())
        self.slot_count, self.power_count = slots.out_features, power.out_features

    def forward(self, states, lengths=None):
        encoded = self.encoder(states, lengths=lengths)
        return self.frequency(encoded), self.slot_power(encoded).reshape(-1, self.slot_count, self.power_count)


def lp_joint_q(heads):
    frequency, slot_power = heads
    return frequency[:, :, None, None] + slot_power[:, None, :, :]


def lp_gather(heads, actions):
    frequency, slot_power = heads
    batch = torch.arange(actions.shape[0], device=actions.device)
    return frequency[batch, actions[:, 0]] + slot_power[batch, actions[:, 1], actions[:, 2]]


def lp_masked_policy(logits, frequency_slots, power):
    """Exact global masked normalization; FL marginal then conditional P given L.

    Returns FL, P|L, safe_log_FL, safe_log_P|L, H(FL), H(P|FL),
    H(F), H(L|F). Only valid entries have nonzero probabilities.
    """
    frequency, lp = logits
    if (frequency.ndim != 2 or lp.ndim != 3 or lp.shape[0] != frequency.shape[0]
            or frequency_slots.shape != (frequency.shape[0], frequency.shape[1], lp.shape[1])
            or power.shape != (lp.shape[0], lp.shape[2])
            or frequency_slots.dtype != torch.bool or power.dtype != torch.bool):
        raise ValueError('LP logits/mask schema mismatch')
    if not bool(torch.isfinite(frequency).all() and torch.isfinite(lp).all()):
        raise FloatingPointError('nonfinite LP policy logits')
    if not bool(frequency_slots.any((1, 2)).all() and power.any(1).all()):
        raise ValueError('active LP mask has no feasible action')
    masked_lp = lp.masked_fill(~power[:, None, :], -torch.inf)
    mass = torch.logsumexp(masked_lp, dim=-1)
    fl_logits = (frequency[:, :, None] + mass[:, None, :]).masked_fill(~frequency_slots, -torch.inf)
    log_fl = fl_logits.flatten(1).log_softmax(-1).reshape_as(fl_logits)
    log_p = masked_lp - mass[:, :, None]
    if not bool(torch.isfinite(log_fl[frequency_slots]).all()
                and torch.isfinite(log_p[power[:, None, :].expand_as(log_p)]).all()):
        raise FloatingPointError('nonfinite LP normalization')
    fl, conditional_power = log_fl.exp(), log_p.exp()
    safe_fl = torch.where(frequency_slots, log_fl, torch.zeros_like(log_fl))
    safe_p = torch.where(power[:, None, :], log_p, torch.zeros_like(log_p))
    entropy_fl = -(fl * safe_fl).sum((1, 2))
    conditional_entropy = -(conditional_power * safe_p).sum(2)
    entropy_power = (fl.sum(1) * conditional_entropy).sum(1)
    marginal_frequency = fl.sum(2)
    log_frequency = torch.where(marginal_frequency > 0,
                                marginal_frequency.clamp_min(torch.finfo(fl.dtype).tiny).log(),
                                torch.zeros_like(marginal_frequency))
    entropy_frequency = -(marginal_frequency * log_frequency).sum(1)
    return fl, conditional_power, safe_fl, safe_p, entropy_fl, entropy_power, entropy_frequency, entropy_fl - entropy_frequency


def lp_expectation(parts, q_table):
    return torch.einsum('bfl,blp,bflp->b', parts[0], parts[1], q_table)


def lp_soft_value(parts, q1, q2, alpha):
    return lp_expectation(parts, torch.minimum(q1, q2)) + alpha * (parts[4] + parts[5])


def global_rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state().clone(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def restore_global_rng(value):
    random.setstate(value['python'])
    np.random.set_state(value['numpy'])
    torch.set_rng_state(value['torch'].cpu())
    if value['cuda'] is not None:
        if not torch.cuda.is_available():
            raise ValueError('CUDA RNG checkpoint requires CUDA runtime')
        torch.cuda.set_rng_state_all([state.cpu() for state in value['cuda']])


def rng_sha():
    result = dict(cpu=hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest())
    if torch.cuda.is_available():
        result['cuda_all'] = [hashlib.sha256(s.cpu().numpy().tobytes()).hexdigest() for s in torch.cuda.get_rng_state_all()]
    return result


class SearchAgent(WorkbenchAgent):
    def __init__(self, config, device=None):
        variant = config['variant']
        actual_device = device or config['training']['device']
        expected = build_config(variant, actual_device, config.get('phase', 'scientific'))
        for section in ('network', 'state', 'reward', 'environment'):
            for key, value in expected[section].items():
                if config[section].get(key) != value:
                    raise ValueError('W27 contract differs: ' + section + '.' + key)
        relaxed = {'batch_size', 'warmup_steps', 'replay_capacity', 'checkpoint_interval', 'total_steps'} if config.get('phase') == 'correctness' else set()
        for key, value in expected['training'].items():
            if key not in relaxed and config['training'].get(key) != value:
                raise ValueError('W27 training contract differs: ' + key)
        if config.get('group') != 'S10' or config.get('algorithm') != 'sac':
            raise ValueError('W27 requires SAC and S10 environment state')
        internal = deepcopy(config)
        internal['network'].update(sac_policy_structure='factorized', sac_critic_structure='additive')
        super().__init__(internal, 207, (100, 10, 10), device=actual_device)
        self.variant, self.config, self.train = variant, config, config['training']
        before = rng_sha()
        for index, name in enumerate(('actor', 'critic1', 'critic2')):
            network = getattr(self, name)
            network.encoder = WidthEncoder(network.encoder, 'S10', self.train['seed'] + 100003 * (index + 1)).to(self.device)
            if variant == 'LP':
                setattr(self, name, SlotPowerNetwork(network).to(self.device))
        self.target1, self.target2 = deepcopy(self.critic1).eval(), deepcopy(self.critic2).eval()
        after = rng_sha()
        if before != after:
            raise RuntimeError('W27 extension changed CPU/CUDA sampling RNG')
        self.initialization_rng = dict(before=before, after=after, unchanged=True)
        self.extension_rng = dict(cpu_unchanged=True, cuda_unchanged=True if 'cuda_all' in after else None,
                                  cpu_SHA=after['cpu'], cuda_all_SHA=after.get('cuda_all'),
                                  head_extension=self.initialization_rng)
        self.sac_policy_structure = config['network']['sac_policy_structure']
        self.sac_critic_structure = config['network']['sac_critic_structure']
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.train['actor_lr'])
        self.critic1_opt = torch.optim.Adam(self.critic1.parameters(), lr=self.train['critic_lr'])
        self.critic2_opt = torch.optim.Adam(self.critic2.parameters(), lr=self.train['critic_lr'])
        self.schema = schema_identity(variant)

    def act(self, state, deterministic=False, env_step=None, action_mask=None, beam_entity_count=None):
        if self.variant != 'LP':
            return super().act(state, deterministic, env_step, action_mask, beam_entity_count)
        mask = copy_action_mask(action_mask, self.action_dims)
        was_training = self.actor.training
        self.actor.eval()
        try:
            with torch.no_grad():
                states = torch.as_tensor(np.asarray(state, dtype=np.float32), device=self.device).reshape(1, -1)
                fl_mask, power_mask = mask_batch([mask], self.action_dims, self.device)
                parts = lp_masked_policy(self.actor(states), fl_mask, power_mask)
                fl, conditional = parts[:2]
                if deterministic:
                    index = (fl.unsqueeze(-1) * conditional[:, None, :, :]).flatten(1).argmax(1)
                    f, l, p = int(index.item() // 100), int(index.item() // 10 % 10), int(index.item() % 10)
                else:
                    index = torch.distributions.Categorical(probs=fl.flatten(1)).sample()
                    f, l = int(index.item() // 10), int(index.item() % 10)
                    p = int(torch.distributions.Categorical(probs=conditional[:, l, :]).sample().item())
                action = f, l, p
                self.last_action_info = dict(log_prob=float((parts[2][0, f, l] + parts[3][0, l, p]).item()),
                                             entropy=float((parts[4] + parts[5]).item()),
                                             entropy_frequency=float(parts[6].item()), entropy_slots=float(parts[7].item()),
                                             entropy_power=float(parts[5].item()),
                                             entropy_decomposition='H(F)+H(L|F)+H(P|F,L)',
                                             alpha=float(self.log_alpha.exp().item()))
                return action
        finally:
            self.actor.train(was_training)

    def _update_masked_sac(self):
        if self.variant != 'LP':
            return super()._update_masked_sac()
        s, actions, reward, ns, done, lengths, next_lengths, masks, next_masks = self._sample_masked_replay()
        alpha = self.log_alpha.exp().detach()
        with torch.no_grad():
            target = reward.clone()
            active = done < .5
            if active.any():
                parts = lp_masked_policy(self._sac_actor_logits(ns[active]), next_masks[0][active], next_masks[1][active])
                value = lp_soft_value(parts, lp_joint_q(self.target1(ns[active])), lp_joint_q(self.target2(ns[active])), alpha)
                target[active] += self.train['gamma'] * value
        q1, q2 = lp_gather(self.critic1(s), actions), lp_gather(self.critic2(s), actions)
        loss1, loss2 = F.mse_loss(q1, target), F.mse_loss(q2, target)
        _finite_loss(loss1 + loss2)
        self.critic1_opt.zero_grad(set_to_none=True)
        loss1.backward()
        grad1 = _grad_norm(self.critic1, self.train['gradient_clip'])
        self.critic1_opt.step()
        self.critic2_opt.zero_grad(set_to_none=True)
        loss2.backward()
        grad2 = _grad_norm(self.critic2, self.train['gradient_clip'])
        self.critic2_opt.step()
        parts = lp_masked_policy(self._sac_actor_logits(s), *masks)
        with torch.no_grad():
            min_q = torch.minimum(lp_joint_q(self.critic1(s)), lp_joint_q(self.critic2(s)))
        expected_q = lp_expectation(parts, min_q)
        entropy = parts[4] + parts[5]
        actor_loss = -(expected_q + alpha * entropy).mean()
        _finite_loss(actor_loss)
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        actor_grad = _grad_norm(self.actor, self.train['gradient_clip'])
        self.actor_opt.step()
        self.update_step += 1
        with torch.no_grad():
            for target_net, source in ((self.target1, self.critic1), (self.target2, self.critic2)):
                for tp, sp in zip(target_net.parameters(), source.parameters()):
                    tp.lerp_(sp, self.train['tau'])
        return _diagnostics(dict(actor_loss=actor_loss, critic_1_loss=loss1, critic_2_loss=loss2,
                                alpha_loss=torch.zeros_like(alpha), alpha=self.log_alpha.exp(), alpha_update_applied=0,
                                entropy=entropy.mean(), entropy_frequency=parts[6].mean(), entropy_slots=parts[7].mean(),
                                entropy_power=parts[5].mean(), entropy_decomposition='H(F)+H(L|F)+H(P|F,L)',
                                target_entropy=feasible_target_entropy(*masks, self.target_entropy_ratio).mean(),
                                legal_frequency_slot_count=masks[0].sum((1, 2)).float().mean(), legal_power_count=masks[1].sum(1).float().mean(),
                                q_mean=(q1.mean() + q2.mean()) / 2, q_min=torch.minimum(q1, q2).min(), q_max=torch.maximum(q1, q2).max(),
                                target_mean=target.mean(), target_std=target.std(unbiased=False),
                                td_abs_mean=((q1 - target).abs().mean() + (q2 - target).abs().mean()) / 2,
                                actor_grad_norm=actor_grad, critic_1_grad_norm=grad1, critic_2_grad_norm=grad2))

    def observe(self, state, action, reward, next_state, done, action_mask=None, next_action_mask=None, **kwargs):
        for value in (state, next_state):
            array = np.asarray(value)
            if array.shape != (208,) or not np.isfinite(array).all():
                raise ValueError('W27 replay state schema mismatch')
        return super().observe(state, action, reward, next_state, done, action_mask, next_action_mask, **kwargs)

    def state_dict(self):
        value = super().state_dict()
        value['w27_schema'] = self.schema
        value['global_rng'] = global_rng_state()
        return value

    def load_state_dict(self, value):
        if value.get('w27_schema') != self.schema:
            raise ValueError('W27 checkpoint variant/action schema mismatch')
        if 'global_rng' not in value:
            raise ValueError('W27 checkpoint missing complete RNG state')
        for row in value.get('replay', {}).get('rows', []):
            if any(np.asarray(row[index]).shape != (208,) or not np.isfinite(np.asarray(row[index])).all() for index in (0, 3)):
                raise ValueError('W27 checkpoint replay schema mismatch')
        super().load_state_dict(value)
        restore_global_rng(value['global_rng'])

    def policy_state(self):
        return dict(schema=self.schema, actor=self.actor.state_dict(), update_step=self.update_step)

    def parameter_counts(self):
        return {name: sum(p.numel() for p in getattr(self, name).parameters())
                for name in ('actor', 'critic1', 'critic2', 'target1', 'target2')}
