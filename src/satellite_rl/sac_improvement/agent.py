"""Masked candidate-action discrete SAC for reversible complete allocations."""

from __future__ import annotations

from collections import deque
import copy
import math
import random

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .environment import ACTION_WIDTH, STATE_WIDTH


MAX_ACTIONS = 17  # 16 complete candidate layouts plus STOP.


class CandidateNetwork(nn.Module):
    def __init__(self, width, action_width=ACTION_WIDTH):
        super().__init__()
        self.state = nn.Sequential(nn.Linear(STATE_WIDTH, width), nn.LayerNorm(width), nn.ReLU(),
                                   nn.Linear(width, width), nn.ReLU())
        self.action = nn.Sequential(nn.Linear(action_width, width // 2), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(width + width // 2, width), nn.ReLU(), nn.Linear(width, 1))

    def forward(self, state, features):
        if state.ndim != 2 or features.ndim != 3 or features.shape[0] != state.shape[0]:
            raise ValueError("Candidate network input dimensions differ")
        common = self.state(state).unsqueeze(1).expand(-1, features.shape[1], -1)
        encoded = self.action(features)
        return self.head(torch.cat((common, encoded), dim=-1)).squeeze(-1)


def pack_actions(environment):
    """Return identity-bound features and validity; STOP is always last."""
    choices = environment.candidates()
    if len(choices) > MAX_ACTIONS - 1:
        raise ValueError("I0 candidate generator exceeded the frozen action limit")
    features = np.zeros((MAX_ACTIONS, len(environment.action_feature(None))), dtype=np.float32)
    mask = np.zeros(MAX_ACTIONS, dtype=bool)
    identifiers = []
    for index, choice in enumerate(choices):
        features[index] = environment.action_feature(choice)
        mask[index] = True
        identifiers.append(choice.id)
    features[-1] = environment.action_feature(None)
    mask[-1] = True
    identifiers.extend([None] * (MAX_ACTIONS - 1 - len(choices)))
    identifiers.append("STOP")
    return features, mask, tuple(identifiers)


def _distribution(logits, mask):
    if logits.shape != mask.shape or not bool(mask.any(-1).all()):
        raise ValueError("Every transition needs at least STOP as a valid action")
    logprob = F.log_softmax(logits.masked_fill(~mask, -torch.inf), dim=-1)
    prob = torch.where(mask, logprob.exp(), torch.zeros_like(logprob))
    safe_logprob = torch.where(mask, logprob, torch.zeros_like(logprob))
    entropy = -(prob * safe_logprob).sum(-1)
    return prob, safe_logprob, entropy


class Replay:
    def __init__(self, capacity, seed):
        self.rows = deque(maxlen=capacity)
        self.rng = random.Random(seed)

    def add(self, state, features, mask, identifiers, index, reward,
            next_state, next_features, next_mask, done):
        if (len(identifiers) != MAX_ACTIONS or not mask[index]
                or identifiers[index] is None or mask[-1] is not True and not bool(mask[-1])):
            raise ValueError("Invalid candidate identity or action")
        row = (np.asarray(state, np.float32).copy(), np.asarray(features, np.float32).copy(),
               np.asarray(mask, bool).copy(), tuple(identifiers), int(index), float(reward),
               np.asarray(next_state, np.float32).copy(), np.asarray(next_features, np.float32).copy(),
               np.asarray(next_mask, bool).copy(), bool(done))
        self.rows.append(row)

    def sample(self, count, device):
        rows = self.rng.sample(self.rows, count)
        def tensor(values, dtype):
            return torch.as_tensor(np.stack(values), dtype=dtype, device=device)
        return (tensor([r[0] for r in rows], torch.float32),
                tensor([r[1] for r in rows], torch.float32),
                tensor([r[2] for r in rows], torch.bool),
                torch.as_tensor([r[4] for r in rows], dtype=torch.long, device=device),
                torch.as_tensor([r[5] for r in rows], dtype=torch.float32, device=device),
                tensor([r[6] for r in rows], torch.float32),
                tensor([r[7] for r in rows], torch.float32),
                tensor([r[8] for r in rows], torch.bool),
                torch.as_tensor([r[9] for r in rows], dtype=torch.float32, device=device))

    def state_dict(self):
        return {"rows": list(self.rows), "rng_state": self.rng.getstate(),
                "capacity": self.rows.maxlen}

    def load_state_dict(self, state):
        if state["capacity"] != self.rows.maxlen:
            raise ValueError("Replay capacity changed across resume")
        self.rows = deque(state["rows"], maxlen=self.rows.maxlen)
        self.rng.setstate(state["rng_state"])


class ImprovementSAC:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config["device"])
        width = config["hidden_width"]
        action_width = ACTION_WIDTH + (config["schema_version"] == "s2fix_i0_v1")
        self.actor = CandidateNetwork(width, action_width).to(self.device)
        self.critic1 = CandidateNetwork(width, action_width).to(self.device)
        self.critic2 = CandidateNetwork(width, action_width).to(self.device)
        self.target1 = copy.deepcopy(self.critic1).eval()
        self.target2 = copy.deepcopy(self.critic2).eval()
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=config["actor_lr"])
        self.critic1_opt = torch.optim.Adam(self.critic1.parameters(), lr=config["critic_lr"])
        self.critic2_opt = torch.optim.Adam(self.critic2.parameters(), lr=config["critic_lr"])
        self.replay = Replay(config["replay_capacity"], config["seed"] + 931)
        self.updates = 0

    def choose(self, state, features, mask, deterministic=False):
        with torch.no_grad():
            state_t = torch.as_tensor(state[None], device=self.device, dtype=torch.float32)
            feat_t = torch.as_tensor(features[None], device=self.device, dtype=torch.float32)
            mask_t = torch.as_tensor(mask[None], device=self.device, dtype=torch.bool)
            logits = self.actor(state_t, feat_t)
            probability, _, _ = _distribution(logits, mask_t)
            if deterministic:
                return int(probability.argmax(-1).item())
            return int(torch.multinomial(probability[0], 1).item())

    def update(self):
        batch_size = self.config["batch_size"]
        if len(self.replay.rows) < batch_size:
            return None
        s, f, m, action, reward, ns, nf, nm, done = self.replay.sample(batch_size, self.device)
        alpha = self.config["fixed_alpha"]
        with torch.no_grad():
            next_p, next_logp, _ = _distribution(self.actor(ns, nf), nm)
            next_q = torch.minimum(self.target1(ns, nf), self.target2(ns, nf))
            next_value = (next_p * (next_q - alpha * next_logp)).sum(-1)
            target = reward + self.config["gamma"] * (1 - done) * next_value
        q1 = self.critic1(s, f).gather(1, action[:, None]).squeeze(1)
        q2 = self.critic2(s, f).gather(1, action[:, None]).squeeze(1)
        loss1, loss2 = F.mse_loss(q1, target), F.mse_loss(q2, target)
        for optimizer, loss, network in ((self.critic1_opt, loss1, self.critic1),
                                         (self.critic2_opt, loss2, self.critic2)):
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("I0 critic loss is not finite")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(network.parameters(), self.config["gradient_clip"])
            optimizer.step()
        p, logp, entropy = _distribution(self.actor(s, f), m)
        with torch.no_grad():
            q = torch.minimum(self.critic1(s, f), self.critic2(s, f))
        actor_loss = (p * (alpha * logp - q)).sum(-1).mean()
        if not bool(torch.isfinite(actor_loss)):
            raise RuntimeError("I0 actor loss is not finite")
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.config["gradient_clip"])
        self.actor_opt.step()
        tau = self.config["tau"]
        with torch.no_grad():
            for target_net, net in ((self.target1, self.critic1), (self.target2, self.critic2)):
                for target_param, source_param in zip(target_net.parameters(), net.parameters()):
                    target_param.lerp_(source_param, tau)
        self.updates += 1
        valid_count = m.sum(-1).float()
        target_entropy = self.config["target_entropy_ratio"] * valid_count.log()
        result = {"critic1_loss": float(loss1.item()), "critic2_loss": float(loss2.item()),
                  "actor_loss": float(actor_loss.item()), "alpha": float(alpha),
                  "entropy": float(entropy.mean().item()),
                  "target_entropy": float(target_entropy.mean().item()),
                  "q1": float(q1.mean().item()), "target_q": float(target.mean().item()),
                  "valid_actions": float(valid_count.mean().item()), "updates": self.updates}
        if not all(math.isfinite(v) for v in result.values()):
            raise RuntimeError("I0 update diagnostics are not finite")
        return result

    def state_dict(self):
        return {"actor": self.actor.state_dict(), "critic1": self.critic1.state_dict(),
                "critic2": self.critic2.state_dict(), "target1": self.target1.state_dict(),
                "target2": self.target2.state_dict(), "actor_opt": self.actor_opt.state_dict(),
                "critic1_opt": self.critic1_opt.state_dict(),
                "critic2_opt": self.critic2_opt.state_dict(),
                "replay": self.replay.state_dict(), "updates": self.updates}

    def load_state_dict(self, state):
        for name in ("actor", "critic1", "critic2", "target1", "target2"):
            getattr(self, name).load_state_dict(state[name], strict=True)
        for name in ("actor_opt", "critic1_opt", "critic2_opt"):
            getattr(self, name).load_state_dict(state[name])
        self.replay.load_state_dict(state["replay"])
        self.updates = int(state["updates"])
