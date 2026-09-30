"""Independent masked PPO for finite-horizon coverage planning.

Rollout tensors remain on the training device.  A true episode end has zero
continuation value; a rollout boundary uses the critic's next-state value.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.distributions import Categorical


class CoveragePolicy(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int = 9, hidden_sizes=(256, 256)):
        super().__init__()
        if action_dim != 9 or obs_dim < 1 or not hidden_sizes or any(int(n) < 1 for n in hidden_sizes):
            raise ValueError("Invalid coverage policy dimensions")

        def tower(output):
            layers = []
            previous = obs_dim
            for width in hidden_sizes:
                layers.extend((nn.Linear(previous, int(width)), nn.Tanh()))
                previous = int(width)
            layers.append(nn.Linear(previous, output))
            return nn.Sequential(*layers)

        self.actor = tower(action_dim)
        self.critic = tower(1)
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_sizes = tuple(int(n) for n in hidden_sizes)

    def distribution(self, obs: torch.Tensor, mask: torch.Tensor) -> Categorical:
        if obs.shape[:-1] != mask.shape[:-1] or mask.shape[-1] != self.action_dim:
            raise ValueError("Observation/mask shape mismatch")
        logits = self.actor(obs).masked_fill(~mask, torch.finfo(obs.dtype).min)
        return Categorical(logits=logits)

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs).squeeze(-1)

    @torch.no_grad()
    def act(self, obs: torch.Tensor, mask: torch.Tensor, *, deterministic=False):
        distribution = self.distribution(obs, mask)
        action = distribution.logits.argmax(dim=-1) if deterministic else distribution.sample()
        return action, distribution.log_prob(action), self.value(obs)


@dataclass
class Rollout:
    observations: torch.Tensor
    masks: torch.Tensor
    actions: torch.Tensor
    old_logp: torch.Tensor
    old_values: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    bootstrap_value: torch.Tensor

    def advantages(self, gamma: float, gae_lambda: float):
        if self.rewards.shape != self.dones.shape or self.rewards.shape != self.old_values.shape:
            raise ValueError("Rollout scalar shapes disagree")
        length = self.rewards.shape[0]
        advantage = torch.zeros_like(self.bootstrap_value)
        all_advantages = torch.empty_like(self.rewards)
        next_value = self.bootstrap_value
        for t in range(length - 1, -1, -1):
            continuation = (~self.dones[t]).to(self.rewards.dtype)
            delta = self.rewards[t] + gamma * next_value * continuation - self.old_values[t]
            advantage = delta + gamma * gae_lambda * continuation * advantage
            all_advantages[t] = advantage
            next_value = self.old_values[t]
        return all_advantages, all_advantages + self.old_values


def ppo_update(policy: CoveragePolicy, optimizer: torch.optim.Optimizer, rollout: Rollout,
               *, gamma: float = 1.0, gae_lambda: float = 0.95, clip_ratio: float = 0.2,
               entropy_coef: float = 0.01, value_coef: float = 0.5,
               max_grad_norm: float = 0.5, epochs: int = 4, minibatch_size: int = 1024):
    """Update from the exact mask and log probability used for sampling."""
    if not 0 <= gamma <= 1 or not 0 <= gae_lambda <= 1 or epochs < 1 or minibatch_size < 1:
        raise ValueError("Invalid PPO hyperparameters")
    advantages, returns = rollout.advantages(gamma, gae_lambda)
    observations = rollout.observations.reshape(-1, policy.obs_dim)
    masks = rollout.masks.reshape(-1, policy.action_dim)
    actions = rollout.actions.reshape(-1)
    old_logp = rollout.old_logp.reshape(-1)
    returns = returns.reshape(-1)
    advantages = advantages.reshape(-1)
    advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-8)
    n = len(actions)
    diagnostics = {key: torch.zeros((), device=observations.device) for key in
                   ("policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction", "grad_norm")}
    updates = 0
    for _ in range(epochs):
        order = torch.randperm(n, device=observations.device)
        for start in range(0, n, minibatch_size):
            index = order[start:start + minibatch_size]
            distribution = policy.distribution(observations[index], masks[index])
            logp = distribution.log_prob(actions[index])
            ratio = (logp - old_logp[index]).exp()
            unclipped = ratio * advantages[index]
            clipped = ratio.clamp(1 - clip_ratio, 1 + clip_ratio) * advantages[index]
            policy_loss = -torch.minimum(unclipped, clipped).mean()
            value_loss = 0.5 * (policy.value(observations[index]) - returns[index]).square().mean()
            entropy = distribution.entropy().mean()
            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm,
                                                 error_if_nonfinite=True)
            optimizer.step()
            with torch.no_grad():
                diagnostics["policy_loss"] += policy_loss.detach()
                diagnostics["value_loss"] += value_loss.detach()
                diagnostics["entropy"] += entropy.detach()
                diagnostics["approx_kl"] += ((ratio - 1) - ratio.log()).mean().detach()
                diagnostics["clip_fraction"] += ((ratio - 1).abs() > clip_ratio).float().mean()
                diagnostics["grad_norm"] += grad_norm.detach()
            updates += 1
    return {key: float((value / updates).item()) for key, value in diagnostics.items()} | {"updates": updates}
