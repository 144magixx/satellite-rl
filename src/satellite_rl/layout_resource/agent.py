"""Relation-aware exact discrete SAC for the versioned layout interface.

This is a new architecture, not an adapter for legacy T0 checkpoints. Power is
an environment rule; the policy has only the 1001 discrete frequency actions.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from satellite_rl.agents.action_masking import conditional_masked_policy
from .schema import (TOKEN_DIM, RELATION_DIM, GLOBAL_DIM, ACTION_DIM, STATE_SCHEMA,
                     ACTION_SCHEMA, PHYSICS_SCHEMA, PHYSICS_SHA, observation_semantics)


@dataclass(frozen=True)
class AgentConfig:
    token_dim: int = 16
    relation_dim: int = 4
    global_dim: int = 12
    action_dim: int = 1001
    hidden_dim: int = 128
    heads: int = 8
    layers: int = 2
    gamma: float = 1.0
    alpha: float = .01
    tau: float = .005
    learning_rate: float = 3e-4
    physics_features: bool = True
    state_schema: str = "LayoutResourceTokensV1"
    action_schema: str = "JointStartCountNoOpV1"
    physics_schema: str = "spherical_gaussian_tx_v1"
    physics_SHA: str = PHYSICS_SHA

    def __post_init__(self):
        for name in ('token_dim', 'relation_dim', 'global_dim', 'action_dim', 'hidden_dim', 'heads', 'layers'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f'{name} must be a positive integer')
        if (self.token_dim, self.relation_dim, self.global_dim, self.action_dim) != (TOKEN_DIM, RELATION_DIM, GLOBAL_DIM, ACTION_DIM):
            raise ValueError('fixed observation/action protocol dimensions required')
        if (self.state_schema, self.action_schema, self.physics_schema, self.physics_SHA) != (STATE_SCHEMA, ACTION_SCHEMA, PHYSICS_SCHEMA, PHYSICS_SHA):
            raise ValueError('unsupported state/action/physics protocol')
        if not isinstance(self.physics_features, bool) or self.hidden_dim % self.heads:
            raise ValueError('invalid physical feature flag or attention heads')
        for name in ('gamma', 'alpha', 'tau', 'learning_rate'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f'{name} must be finite numeric')
        if not 0 <= self.gamma <= 1 or self.alpha < 0 or not 0 < self.tau <= 1 or self.learning_rate <= 0:
            raise ValueError('invalid SAC scalar configuration')


def collate_observations(observations, device="cpu", config=None):
    """Pad only inside the batch; reject missing/invalid actual beam tokens."""
    config = config or AgentConfig()
    if not observations:
        raise ValueError("empty observation batch")
    lengths = [len(o["tokens"]) for o in observations]
    if any(not 1 <= n <= 200 for n in lengths):
        raise ValueError("actual beam count must be 1..200")
    b, n = len(lengths), max(lengths)
    tokens = np.zeros((b, n, config.token_dim), np.float32)
    relations = np.zeros((b, n, n, config.relation_dim), np.float32)
    padding = np.ones((b, n), bool)
    globals_, indices, masks = [], [], []
    for k, (o, size) in enumerate(zip(observations, lengths)):
        if o.get('semantics') != observation_semantics():
            raise ValueError('observation semantic protocol mismatch or absent metadata')
        t, r, g, m = [np.asarray(o[key]) for key in ("tokens", "relations", "global_features", "action_mask")]
        if (t.shape != (size, config.token_dim) or r.shape != (size, size, config.relation_dim)
                or g.shape != (config.global_dim,) or m.shape != (config.action_dim,)
                or m.dtype != np.bool_ or not m.any()):
            raise ValueError("observation shapes or feasible action mask mismatch")
        supplied = np.asarray(o.get("padding_mask", np.zeros(size, bool)))
        if supplied.dtype != np.bool_ or supplied.shape != (size,) or supplied.any():
            raise ValueError("environment observations must contain only actual beams")
        index = o["current_index"]
        if isinstance(index, (bool, np.bool_)) or not isinstance(index, (int, np.integer)) or not 0 <= index < size:
            raise ValueError("current_index must identify an actual beam")
        if not all(np.isfinite(x).all() for x in (t, r, g)):
            raise ValueError("nonfinite observation")
        tokens[k, :size], relations[k, :size, :size] = t, r
        padding[k, :size] = False
        globals_.append(g); indices.append(index); masks.append(m)
    return {"tokens": torch.as_tensor(tokens, device=device),
            "relations": torch.as_tensor(relations, device=device),
            "padding_mask": torch.as_tensor(padding, device=device),
            "global_features": torch.as_tensor(np.asarray(globals_, np.float32), device=device),
            "current_index": torch.as_tensor(indices, device=device),
            "action_mask": torch.as_tensor(np.stack(masks), device=device)}


class RelationBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = config.heads
        self.attention = nn.MultiheadAttention(config.hidden_dim, config.heads, batch_first=True, dropout=0)
        self.bias = nn.Linear(config.relation_dim, config.heads, bias=False)
        self.norm1 = nn.LayerNorm(config.hidden_dim)
        self.norm2 = nn.LayerNorm(config.hidden_dim)
        self.ff = nn.Sequential(nn.Linear(config.hidden_dim, 4 * config.hidden_dim), nn.GELU(),
                                nn.Linear(4 * config.hidden_dim, config.hidden_dim))

    def forward(self, x, relations, padding):
        b, n, _ = x.shape
        relations = relations.masked_fill(padding[:, :, None, None] | padding[:, None, :, None], 0)
        bias = self.bias(relations).permute(0, 3, 1, 2)
        # All query rows retain at least one real key; padded queries are then zeroed.
        bias = bias.masked_fill(padding[:, None, None, :], -torch.inf)
        attended = self.attention(x, x, x, attn_mask=bias.reshape(b * self.heads, n, n),
                                  need_weights=False)[0]
        x = self.norm1(x + attended)
        x = self.norm2(x + self.ff(x))
        return x.masked_fill(padding.unsqueeze(-1), 0)


class LayoutJointNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Linear(config.token_dim, config.hidden_dim)
        self.blocks = nn.ModuleList([RelationBlock(config) for _ in range(config.layers)])
        self.head = nn.Sequential(nn.Linear(2 * config.hidden_dim + config.global_dim, config.hidden_dim),
                                  nn.GELU(), nn.Linear(config.hidden_dim, config.action_dim))

    def forward(self, batch):
        padding = batch["padding_mask"]
        x = self.embedding(batch["tokens"].masked_fill(padding.unsqueeze(-1), 0)).masked_fill(padding.unsqueeze(-1), 0)
        for block in self.blocks:
            x = block(x, batch["relations"], padding)
        pooled = x.sum(1) / (~padding).sum(1, keepdim=True)
        current = x[torch.arange(len(x), device=x.device), batch["current_index"]]
        return self.head(torch.cat((current, pooled, batch["global_features"]), -1))


def joint_distribution(logits, mask):
    """Reuse audited normalization with singleton count/power auxiliary axes."""
    singleton_power = logits.new_zeros((*logits.shape, 1, 1))
    parts = conditional_masked_policy((logits.unsqueeze(-1), singleton_power),
                                      mask.unsqueeze(-1), torch.ones((len(logits), 1),
                                                                    dtype=torch.bool, device=logits.device))
    return parts[0].squeeze(-1), parts[2].squeeze(-1), parts[4]


def exact_soft_value(probability, safe_logs, q1, q2, alpha):
    """Twin minimum precedes expectation; masked logs are finite zeros."""
    return (probability * (torch.minimum(q1, q2) - alpha * safe_logs)).sum(1)


class LayoutSAC:
    def __init__(self, config=None, device="cpu"):
        self.config, self.device = config or AgentConfig(), torch.device(device)
        c = self.config
        if (c.action_dim != 1001 or c.hidden_dim % c.heads or c.layers < 1 or
                not 0 <= c.gamma <= 1 or c.alpha < 0 or not 0 < c.tau <= 1):
            raise ValueError("invalid layout SAC configuration")
        self.actor = LayoutJointNetwork(c).to(self.device)
        self.critic1 = LayoutJointNetwork(c).to(self.device)
        self.critic2 = LayoutJointNetwork(c).to(self.device)
        self.target1, self.target2 = deepcopy(self.critic1), deepcopy(self.critic2)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=c.learning_rate)
        self.critic_optimizer = torch.optim.Adam(list(self.critic1.parameters()) + list(self.critic2.parameters()), lr=c.learning_rate)
        self.updates = 0

    def batch(self, observations):
        batch = collate_observations(observations, self.device, self.config)
        if not self.config.physics_features:
            # Input-only ablation: preserve geometry, demand, dynamic allocation
            # and the physical environment; remove width/peak/tx-gain features.
            batch["tokens"][..., 2:4] = 0
            batch["relations"][..., 1] = 0
        return batch

    @torch.no_grad()
    def act(self, observation, deterministic=False):
        # Terminal observations have no actionable current beam; reject explicitly.
        if observation['current_index'] == len(observation['tokens']):
            raise ValueError('terminal observation has no policy action')
        batch = self.batch([observation])
        probabilities, _, _ = joint_distribution(self.actor(batch), batch["action_mask"])
        return int(probabilities.argmax(-1).item() if deterministic else torch.multinomial(probabilities, 1).item())

    def update(self, transitions):
        """Exact expectation across feasible joint actions, terminal bootstrap zero."""
        if not transitions:
            raise ValueError("empty transition batch")
        for transition in transitions:
            action, reward, done_value = transition[1], transition[2], transition[4]
            if isinstance(action, (bool, np.bool_)) or not isinstance(action, (int, np.integer)):
                raise ValueError('replay actions must be integers without coercion')
            if not isinstance(done_value, (bool, np.bool_)):
                raise ValueError('replay done must be boolean')
            if isinstance(reward, (bool, np.bool_)) or not isinstance(reward, (int, float, np.integer, np.floating)) or not np.isfinite(reward):
                raise ValueError('replay reward must be finite numeric')
        s = self.batch([t[0] for t in transitions])
        actions = torch.as_tensor([t[1] for t in transitions], device=self.device, dtype=torch.long)
        rewards = torch.as_tensor([t[2] for t in transitions], device=self.device, dtype=torch.float32)
        done = torch.as_tensor([t[4] for t in transitions], device=self.device, dtype=torch.bool)
        if (not torch.isfinite(rewards).all() or ((actions < 0) | (actions >= self.config.action_dim)).any()
                or not s["action_mask"].gather(1, actions[:, None]).all()):
            raise ValueError("nonfinite reward or infeasible replay action")
        with torch.no_grad():
            target = rewards.clone()
            active = [i for i, t in enumerate(transitions) if not t[4]]
            if active:
                ns = self.batch([transitions[i][3] for i in active])
                probability, logs, _ = joint_distribution(self.actor(ns), ns["action_mask"])
                target[~done] += self.config.gamma * exact_soft_value(probability, logs, self.target1(ns), self.target2(ns), self.config.alpha)
        q1, q2 = self.critic1(s), self.critic2(s)
        critic_loss = F.mse_loss(q1.gather(1, actions[:, None]).squeeze(1), target) + F.mse_loss(q2.gather(1, actions[:, None]).squeeze(1), target)
        self.critic_optimizer.zero_grad(); critic_loss.backward()
        self._check_gradients(critic_loss, list(self.critic1.parameters()) + list(self.critic2.parameters()))
        probability, logs, entropy = joint_distribution(self.actor(s), s["action_mask"])
        with torch.no_grad():
            q = torch.minimum(self.critic1(s), self.critic2(s))
        actor_loss = (probability * (self.config.alpha * logs - q)).sum(1).mean()
        self.actor_optimizer.zero_grad(); actor_loss.backward()
        self._check_gradients(actor_loss, self.actor.parameters())
        # Neither optimizer mutates parameters until both gradient sets are valid.
        nets = (self.actor, self.critic1, self.critic2)
        optimizers = (self.actor_optimizer, self.critic_optimizer)
        previous_nets = [deepcopy(net.state_dict()) for net in nets]
        previous_optimizers = [deepcopy(optimizer.state_dict()) for optimizer in optimizers]
        try:
            self.critic_optimizer.step(); self.actor_optimizer.step()
            if any(not torch.isfinite(value).all() for net in nets for value in net.state_dict().values()):
                raise FloatingPointError('nonfinite SAC parameter after update')
            for optimizer in optimizers:
                self._validate_optimizer(optimizer)
        except Exception:
            for net, state in zip(nets, previous_nets): net.load_state_dict(state)
            for optimizer, state in zip(optimizers, previous_optimizers): optimizer.load_state_dict(state)
            raise
        with torch.no_grad():
            for target_net, net in ((self.target1, self.critic1), (self.target2, self.critic2)):
                for target_parameter, parameter in zip(target_net.parameters(), net.parameters()):
                    target_parameter.lerp_(parameter, self.config.tau)
        self.updates += 1
        return {"critic_loss": float(critic_loss.detach()), "actor_loss": float(actor_loss.detach()),
                "entropy": float(entropy.mean().detach()), "updates": self.updates}

    @staticmethod
    def _check_gradients(loss, parameters):
        if not torch.isfinite(loss) or any(p.grad is not None and not torch.isfinite(p.grad).all() for p in parameters):
            raise FloatingPointError('nonfinite SAC loss/gradient; update refused')

    def save(self, path, provenance=None):
        """Save model/optimizer continuation only, not task replay or RNG state."""
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"format": "layout_sac_joint_v1", "config": asdict(self.config), "semantics": observation_semantics(),
                    "networks": {key: getattr(self, key).state_dict() for key in ("actor", "critic1", "critic2", "target1", "target2")},
                    "actor_optimizer": self.actor_optimizer.state_dict(), "critic_optimizer": self.critic_optimizer.state_dict(),
                    "updates": self.updates, "provenance": provenance or {}}, path)

    def load(self, path, resume=False):
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        if (not isinstance(checkpoint, dict) or checkpoint.get("format") != "layout_sac_joint_v1"
                or checkpoint.get("config") != asdict(self.config)):
            raise ValueError("Incompatible state/action/physics schema or architecture: legacy 207/fixed-100 T0 weights cannot initialize this model")
        if checkpoint.get('semantics') != observation_semantics():
            raise ValueError('checkpoint semantic protocol mismatch')
        updates = checkpoint.get('updates')
        if isinstance(updates, bool) or not isinstance(updates, int) or updates < 0:
            raise ValueError('checkpoint updates must be a nonnegative integer')
        # Validate all tensors before mutating the live model.
        copies = {key: deepcopy(getattr(self, key)) for key in ("actor", "critic1", "critic2", "target1", "target2")}
        for key, net in copies.items():
            net.load_state_dict(checkpoint["networks"][key], strict=True)
            if any(not torch.isfinite(value).all() for value in net.state_dict().values()):
                raise ValueError('nonfinite checkpoint network tensor')
        if resume:
            temporary_actor_optimizer = torch.optim.Adam(copies['actor'].parameters(), lr=self.config.learning_rate)
            temporary_critic_optimizer = torch.optim.Adam(list(copies['critic1'].parameters()) + list(copies['critic2'].parameters()), lr=self.config.learning_rate)
            for optimizer, key in ((temporary_actor_optimizer, 'actor_optimizer'), (temporary_critic_optimizer, 'critic_optimizer')):
                # Compare against a fresh configured Adam before loading can fill
                # omitted defaults or silently replace actual learning settings.
                expected_groups = optimizer.state_dict()['param_groups']
                supplied_groups = checkpoint[key].get('param_groups', [])
                if len(supplied_groups) != len(expected_groups):
                    raise ValueError('checkpoint Adam groups differ from AgentConfig')
                for supplied, expected in zip(supplied_groups, expected_groups):
                    supplied_options = {k: v for k, v in supplied.items() if k != 'params'}
                    expected_options = {k: v for k, v in expected.items() if k != 'params'}
                    if (supplied_options != expected_options or
                            any(type(supplied_options[k]) is not type(v) for k, v in expected_options.items()) or
                            len(supplied.get('params', [])) != len(expected['params'])):
                        raise ValueError('checkpoint Adam hyperparameters differ from AgentConfig')
                optimizer.load_state_dict(checkpoint[key])
                self._validate_optimizer(optimizer)
        for key, net in copies.items():
            getattr(self, key).load_state_dict(net.state_dict(), strict=True)
        if resume:
            self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
            self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        self.updates = updates
        return checkpoint.get("provenance", {})

    @staticmethod
    def _validate_optimizer(optimizer):
        for group in optimizer.param_groups:
            if not isinstance(group['lr'], (int, float)) or not math.isfinite(group['lr']) or group['lr'] <= 0:
                raise ValueError('invalid checkpoint optimizer learning rate')
            if (not isinstance(group.get('eps'), (int, float)) or not math.isfinite(group['eps']) or group['eps'] <= 0
                    or not isinstance(group.get('weight_decay'), (int, float)) or not math.isfinite(group['weight_decay']) or group['weight_decay'] < 0
                    or len(group.get('betas', ())) != 2 or any(not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v < 1 for v in group['betas'])):
                raise ValueError('invalid checkpoint Adam hyperparameters')
            for parameter in group['params']:
                state = optimizer.state.get(parameter, {})
                if state and not {'step', 'exp_avg', 'exp_avg_sq'} <= set(state):
                    raise ValueError('incomplete checkpoint Adam state')
                for key, value in state.items():
                    if isinstance(value, torch.Tensor):
                        if not torch.isfinite(value).all() or (key in ('exp_avg', 'exp_avg_sq', 'max_exp_avg_sq') and value.shape != parameter.shape):
                            raise ValueError('invalid checkpoint optimizer tensor')
                        if key == 'step' and (value.numel() != 1 or value.item() < 0):
                            raise ValueError('invalid checkpoint optimizer step')
