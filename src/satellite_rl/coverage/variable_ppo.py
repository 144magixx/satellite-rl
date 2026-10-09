"""Pure variable-set policy and PPO updates for R08/R09.

Scene geometry and candidate construction belong to variable_observation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical



BEAM_FEATURES = 12
CANDIDATE_FEATURES = 16
GLOBAL_FEATURES = 8


def collate(observations, device):
    """Pad only for a batch; padding is excluded from set pooling and logits."""
    if not observations:
        raise ValueError("Cannot collate an empty PPO batch")
    b = len(observations)
    max_beams = max(1, max(len(item["beam"]) for item in observations))
    max_candidates = max(len(item["candidate"]) for item in observations)
    beams = np.zeros((b, max_beams, BEAM_FEATURES), dtype=np.float32)
    beam_mask = np.zeros((b, max_beams), dtype=bool)
    actions = np.zeros((b, max_candidates, CANDIDATE_FEATURES), dtype=np.float32)
    action_mask = np.zeros((b, max_candidates), dtype=bool)
    for i, item in enumerate(observations):
        n, k = len(item["beam"]), len(item["candidate"])
        beams[i, :n] = item["beam"]
        beam_mask[i, :n] = True
        actions[i, :k] = item["candidate"]
        action_mask[i, :k] = item["candidate_mask"]
    return {"beam": torch.as_tensor(beams, device=device),
            "beam_mask": torch.as_tensor(beam_mask, device=device),
            "global": torch.as_tensor(np.stack([x["global"] for x in observations]), device=device),
            "candidate": torch.as_tensor(actions, device=device),
            "candidate_mask": torch.as_tensor(action_mask, device=device)}


class VariableSetPolicy(nn.Module):
    """Shared beam encoder, permutation-invariant context and candidate scorer."""

    def __init__(self, hidden_size=128):
        super().__init__()
        if hidden_size != 128:
            raise ValueError("Frozen variable-set PPO uses hidden_size=128")
        self.beam_encoder = nn.Sequential(nn.Linear(BEAM_FEATURES, 128), nn.Tanh(),
                                          nn.Linear(128, 128), nn.Tanh())
        self.context = nn.Sequential(nn.Linear(128 + GLOBAL_FEATURES, 128), nn.Tanh())
        self.candidate_encoder = nn.Sequential(nn.Linear(CANDIDATE_FEATURES, 128), nn.Tanh(),
                                               nn.Linear(128, 128), nn.Tanh())
        self.query = nn.Linear(128, 128)
        self.score = nn.Linear(128, 1)
        self.value_head = nn.Sequential(nn.Linear(128, 128), nn.Tanh(), nn.Linear(128, 1))

    def forward(self, batch):
        beam = self.beam_encoder(batch["beam"])
        mask = batch["beam_mask"].unsqueeze(-1)
        pooled = (beam * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        context = self.context(torch.cat((pooled, batch["global"]), dim=-1))
        embedded = self.candidate_encoder(batch["candidate"])
        logits = self.score(torch.tanh(embedded + self.query(context)[:, None])).squeeze(-1)
        logits = logits.masked_fill(~batch["candidate_mask"], torch.finfo(logits.dtype).min)
        return logits, self.value_head(context).squeeze(-1)

    def act(self, batch, *, deterministic=False):
        logits, value = self(batch)
        distribution = Categorical(logits=logits)
        choice = logits.argmax(dim=-1) if deterministic else distribution.sample()
        return choice, distribution.log_prob(choice), value

    def evaluate(self, batch, choice):
        logits, value = self(batch)
        distribution = Categorical(logits=logits)
        return distribution.log_prob(choice), distribution.entropy(), value


@dataclass
class Transition:
    observation: dict
    choice: int
    old_log_prob: float
    value: float
    reward: float
    done: bool


def returns_and_advantages(transitions, bootstrap, *, num_envs, gamma, gae_lambda):
    """GAE over time×environment; neighboring streams never bootstrap each other."""
    if not transitions:
        raise ValueError("No real interactions to update from")
    if len(transitions) % num_envs:
        raise ValueError("Rollout interaction count is not divisible by environment count")
    horizon = len(transitions) // num_envs
    values = np.asarray([x.value for x in transitions], dtype=np.float64).reshape(horizon, num_envs)
    rewards = np.asarray([x.reward for x in transitions], dtype=np.float64).reshape(horizon, num_envs)
    dones = np.asarray([x.done for x in transitions], dtype=np.float64).reshape(horizon, num_envs)
    bootstrap = np.broadcast_to(np.asarray(bootstrap, dtype=np.float64), (num_envs,))
    next_values = np.vstack((values[1:], bootstrap[None, :]))
    advantages = np.zeros((horizon, num_envs), dtype=np.float64)
    gae = np.zeros(num_envs, dtype=np.float64)
    for i in range(horizon - 1, -1, -1):
        continuation = 1.0 - dones[i]
        delta = rewards[i] + gamma * next_values[i] * continuation - values[i]
        gae = delta + gamma * gae_lambda * continuation * gae
        advantages[i] = gae
    return (advantages + values).reshape(-1), advantages.reshape(-1)


def ppo_update(policy, optimizer, transitions, bootstrap, *, device,
               gamma=.99, gae_lambda=.95, clip_ratio=.2, entropy_coef=.01,
               value_coef=.5, max_grad_norm=.5, epochs=4, minibatch_size=64,
               num_envs=1):
    return_values, advantage_values = returns_and_advantages(
        transitions, bootstrap, num_envs=num_envs,
        gamma=gamma, gae_lambda=gae_lambda)
    returns = torch.as_tensor(return_values, dtype=torch.float32, device=device)
    advantage = torch.as_tensor(advantage_values, dtype=torch.float32, device=device)
    advantage = (advantage - advantage.mean()) / advantage.std(unbiased=False).clamp_min(1e-8)
    old_logp = torch.as_tensor([x.old_log_prob for x in transitions], dtype=torch.float32, device=device)
    choices = torch.as_tensor([x.choice for x in transitions], dtype=torch.int64, device=device)
    summaries = []
    for _ in range(epochs):
        for indices in torch.randperm(len(transitions), device=device).split(minibatch_size):
            selected = indices.cpu().tolist()
            batch = collate([transitions[i].observation for i in selected], device)
            logp, entropy, value = policy.evaluate(batch, choices[indices])
            ratio = torch.exp(logp - old_logp[indices])
            objective = torch.minimum(ratio * advantage[indices],
                                      ratio.clamp(1 - clip_ratio, 1 + clip_ratio) * advantage[indices])
            policy_loss = -objective.mean()
            value_loss = .5 * (value - returns[indices]).square().mean()
            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy.mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            optimizer.step()
            summaries.append((float(policy_loss.detach()), float(value_loss.detach()),
                              float(entropy.mean().detach()), float(norm.detach())))
    mean = np.mean(summaries, axis=0)
    return {"policy_loss": float(mean[0]), "value_loss": float(mean[1]),
            "entropy": float(mean[2]), "grad_norm": float(mean[3]),
            "minibatch_updates": len(summaries), "rollout_interactions": len(transitions)}
