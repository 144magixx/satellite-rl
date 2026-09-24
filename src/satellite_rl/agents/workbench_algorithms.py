"""Trainable, factorised discrete SAC, PPO and Double DQN workbench agents."""

from __future__ import annotations

from collections import deque
import copy
import random

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from satellite_rl.agents.workbench_networks import MultiHeadNetwork, ScalarNetwork


HEADS = ("frequency", "slots", "power")


def _state_tensor(states, device):
    return torch.as_tensor(np.asarray(states, dtype=np.float32), device=device)


def _joint_q(heads):
    return heads[0][:, :, None, None] + heads[1][:, None, :, None] + heads[2][:, None, None, :]


def _gather(heads, actions):
    return sum(q.gather(1, actions[:, i:i + 1]).squeeze(1) for i, q in enumerate(heads))


def _grad_norm(module, clip):
    for param in module.parameters():
        if param.grad is not None and not torch.isfinite(param.grad).all():
            raise FloatingPointError("Nonfinite gradient before optimizer step")
    norm = nn.utils.clip_grad_norm_(module.parameters(), clip)
    if not torch.isfinite(norm):
        raise FloatingPointError("Nonfinite gradient norm")
    return float(norm.detach())


def _finite_loss(loss):
    if not torch.isfinite(loss).all():
        raise FloatingPointError("Nonfinite learning loss before optimizer step")


def generalized_advantages(rewards, values, dones, last_value, gamma, gae_lambda):
    """GAE with bootstrap at rollout truncation and zero bootstrap at true terminal."""
    advantages = np.zeros(len(rewards), dtype=np.float32)
    gae = 0.0
    for i in reversed(range(len(rewards))):
        next_value = values[i + 1] if i + 1 < len(rewards) else last_value
        delta = rewards[i] + gamma * (1 - dones[i]) * next_value - values[i]
        gae = delta + gamma * gae_lambda * (1 - dones[i]) * gae
        advantages[i] = gae
    return advantages


class Replay:
    def __init__(self, capacity, seed):
        self.rows = deque(maxlen=capacity)
        self.rng = random.Random(seed)

    def add(self, state, action, reward, next_state, done):
        self.rows.append((np.asarray(state, dtype=np.float32).copy(),
                          tuple(int(x) for x in action), float(reward),
                          np.asarray(next_state, dtype=np.float32).copy(), bool(done)))

    def sample(self, count, device):
        rows = self.rng.sample(self.rows, count)
        states, actions, rewards, next_states, dones = zip(*rows)
        return (_state_tensor(states, device), torch.as_tensor(actions, dtype=torch.long, device=device),
                torch.as_tensor(rewards, dtype=torch.float32, device=device),
                _state_tensor(next_states, device), torch.as_tensor(dones, dtype=torch.float32, device=device))

    def state_dict(self):
        return {"rows": list(self.rows), "rng": self.rng.getstate(), "capacity": self.rows.maxlen}

    def load_state_dict(self, data):
        if data["capacity"] != self.rows.maxlen:
            raise ValueError("Replay capacity changed")
        self.rows = deque(data["rows"], maxlen=self.rows.maxlen)
        self.rng.setstate(data["rng"])


class WorkbenchAgent:
    def __init__(self, config, state_dim, action_dims, device="cpu"):
        self.config = config
        self.train = config["training"]
        self.device = torch.device(device)
        self.algorithm = config["algorithm"]
        if self.algorithm == "ppo" and config["network"]["dropout"]:
            raise ValueError("PPO does not support dropout: action and update probabilities must match")
        self.action_dims = tuple(action_dims)
        self.rng = random.Random(self.train["seed"] + 31)
        self.update_step = 0
        self.last_action_info = {}
        if self.algorithm == "sac":
            self.actor = MultiHeadNetwork(config, state_dim, action_dims).to(self.device)
            self.critic1 = MultiHeadNetwork(config, state_dim, action_dims).to(self.device)
            self.critic2 = MultiHeadNetwork(config, state_dim, action_dims).to(self.device)
            self.target1 = copy.deepcopy(self.critic1).to(self.device).eval()
            self.target2 = copy.deepcopy(self.critic2).to(self.device).eval()
            self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.train["actor_lr"])
            self.critic1_opt = torch.optim.Adam(self.critic1.parameters(), lr=self.train["critic_lr"])
            self.critic2_opt = torch.optim.Adam(self.critic2.parameters(), lr=self.train["critic_lr"])
            self.log_alpha = nn.Parameter(torch.tensor(np.log(self.train["initial_alpha"]),
                                                       dtype=torch.float32, device=self.device))
            self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=self.train["alpha_lr"])
            self.replay = Replay(self.train["replay_capacity"], self.train["seed"] + 41)
        elif self.algorithm == "ppo":
            self.actor = MultiHeadNetwork(config, state_dim, action_dims).to(self.device)
            self.value = ScalarNetwork(config, state_dim).to(self.device)
            self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.train["actor_lr"])
            self.value_opt = torch.optim.Adam(self.value.parameters(), lr=self.train["critic_lr"])
            self.rollout = []
        elif self.algorithm == "dqn":
            self.online = MultiHeadNetwork(config, state_dim, action_dims).to(self.device)
            self.target = copy.deepcopy(self.online).to(self.device).eval()
            self.optimizer = torch.optim.Adam(self.online.parameters(), lr=self.train["critic_lr"])
            self.replay = Replay(self.train["replay_capacity"], self.train["seed"] + 41)
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")

    def policy_network(self):
        return self.online if self.algorithm == "dqn" else self.actor

    def act(self, state, deterministic=False, env_step=None):
        net = self.policy_network()
        was_training = net.training
        net.eval()
        with torch.no_grad():
            x = _state_tensor(state, self.device).reshape(1, -1)
            heads = net(x)
            if self.algorithm == "dqn":
                epsilon = 0.0 if deterministic else self.train["eps_end"]
                if not deterministic and env_step is not None:
                    epsilon += (self.train["eps_start"] - self.train["eps_end"]) * max(
                        0.0, 1.0 - env_step / self.train["eps_decay_steps"])
                action = tuple(int(torch.argmax(q, dim=1).item()) if deterministic else
                               (self.rng.randrange(dim) if self.rng.random() < epsilon
                                else int(torch.argmax(q, dim=1).item()))
                               for q, dim in zip(heads, self.action_dims))
                self.last_action_info = {"epsilon": epsilon, "q_mean": float(_gather(
                    heads, torch.tensor([action], device=self.device)).item())}
            else:
                dists = [torch.distributions.Categorical(logits=logits) for logits in heads]
                action_t = [torch.argmax(logits, 1) if deterministic else dist.sample()
                            for logits, dist in zip(heads, dists)]
                action = tuple(int(a.item()) for a in action_t)
                self.last_action_info = {
                    "log_prob": float(sum(dist.log_prob(a) for dist, a in zip(dists, action_t)).item()),
                    "entropy": float(sum(dist.entropy() for dist in dists).item()),
                    **{f"entropy_{name}": float(dist.entropy().item()) for name, dist in zip(HEADS, dists)},
                }
                if self.algorithm == "ppo":
                    value_was_training = self.value.training
                    self.value.eval()
                    self.last_action_info["value"] = float(self.value(x).item())
                    self.value.train(value_was_training)
                    self.last_action_info["action"] = action
        net.train(was_training)
        return action

    def observe(self, state, action, reward, next_state, done):
        if self.algorithm == "ppo":
            if tuple(action) != tuple(self.last_action_info.get("action", ())):
                raise ValueError("PPO observed action differs from the sampled action and log-probability")
            self.rollout.append((np.asarray(state, dtype=np.float32).copy(), tuple(action),
                                 float(reward), np.asarray(next_state, dtype=np.float32).copy(), bool(done),
                                 self.last_action_info["log_prob"], self.last_action_info["value"]))
        else:
            self.replay.add(state, action, reward, next_state, done)

    def ready(self, env_step, done=False):
        if self.algorithm == "ppo":
            return len(self.rollout) >= self.train["rollout_steps"] or (done and bool(self.rollout))
        return (env_step >= self.train["warmup_steps"] and
                len(self.replay.rows) >= self.train["batch_size"] and
                env_step % self.train["update_every"] == 0)

    def update(self):
        if self.algorithm == "sac":
            return self._update_sac()
        if self.algorithm == "ppo":
            return self._update_ppo()
        return self._update_dqn()

    def _update_sac(self):
        s, a, r, ns, d = self.replay.sample(self.train["batch_size"], self.device)
        gamma = self.train["gamma"]
        alpha = self.log_alpha.exp().detach()
        with torch.no_grad():
            logits = self.actor(ns)
            probabilities = [q.softmax(-1) for q in logits]
            joint_prob = probabilities[0][:, :, None, None] * probabilities[1][:, None, :, None] * probabilities[2][:, None, None, :]
            min_q = torch.minimum(_joint_q(self.target1(ns)), _joint_q(self.target2(ns)))
            expected_q = (joint_prob * min_q).sum((1, 2, 3))
            entropy = sum(-(p * q.log_softmax(-1)).sum(1) for p, q in zip(probabilities, logits))
            target = r + gamma * (1 - d) * (expected_q + alpha * entropy)
        q1 = _gather(self.critic1(s), a)
        q2 = _gather(self.critic2(s), a)
        loss1, loss2 = F.mse_loss(q1, target), F.mse_loss(q2, target)
        _finite_loss(loss1 + loss2)
        self.critic1_opt.zero_grad(set_to_none=True)
        loss1.backward()
        grad1 = _grad_norm(self.critic1, self.train["gradient_clip"])
        self.critic1_opt.step()
        self.critic2_opt.zero_grad(set_to_none=True)
        loss2.backward()
        grad2 = _grad_norm(self.critic2, self.train["gradient_clip"])
        self.critic2_opt.step()
        logits = self.actor(s)
        probabilities = [q.softmax(-1) for q in logits]
        joint_prob = probabilities[0][:, :, None, None] * probabilities[1][:, None, :, None] * probabilities[2][:, None, None, :]
        with torch.no_grad():
            min_q_actor = torch.minimum(_joint_q(self.critic1(s)), _joint_q(self.critic2(s)))
        expected_q_actor = (joint_prob * min_q_actor).sum((1, 2, 3))
        entropies = [-(p * q.log_softmax(-1)).sum(1) for p, q in zip(probabilities, logits)]
        entropy_actor = sum(entropies)
        actor_loss = -(expected_q_actor + alpha * entropy_actor).mean()
        _finite_loss(actor_loss)
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        actor_grad = _grad_norm(self.actor, self.train["gradient_clip"])
        self.actor_opt.step()
        alpha_loss = (self.log_alpha * (entropy_actor.detach().mean() - self.train["target_entropy"]))
        _finite_loss(alpha_loss)
        self.alpha_opt.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_opt.step()
        self.update_step += 1
        tau = self.train["tau"]
        with torch.no_grad():
            for target_net, source_net in ((self.target1, self.critic1), (self.target2, self.critic2)):
                for target_p, source_p in zip(target_net.parameters(), source_net.parameters()):
                    target_p.lerp_(source_p, tau)
        return {"actor_loss": float(actor_loss), "critic_1_loss": float(loss1),
                "critic_2_loss": float(loss2), "alpha_loss": float(alpha_loss),
                "alpha": float(self.log_alpha.exp()), "entropy": float(entropy_actor.mean()),
                **{f"entropy_{name}": float(value.mean()) for name, value in zip(HEADS, entropies)},
                "q_mean": float((q1.mean() + q2.mean()) / 2), "q_min": float(torch.minimum(q1, q2).min()),
                "q_max": float(torch.maximum(q1, q2).max()), "target_mean": float(target.mean()),
                "target_std": float(target.std(unbiased=False)),
                "td_abs_mean": float(((q1 - target).abs().mean() + (q2 - target).abs().mean()) / 2),
                "actor_grad_norm": actor_grad, "critic_1_grad_norm": grad1, "critic_2_grad_norm": grad2}

    def _update_dqn(self):
        s, a, r, ns, d = self.replay.sample(self.train["batch_size"], self.device)
        current = _gather(self.online(s), a)
        with torch.no_grad():
            next_actions = torch.stack([q.argmax(1) for q in self.online(ns)], 1)
            next_q = _gather(self.target(ns), next_actions)
            target = r + self.train["gamma"] * (1 - d) * next_q
        loss = F.smooth_l1_loss(current, target)
        _finite_loss(loss)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad = _grad_norm(self.online, self.train["gradient_clip"])
        self.optimizer.step()
        self.update_step += 1
        target_updated = self.update_step % self.train["target_update_interval"] == 0
        if target_updated:
            self.target.load_state_dict(self.online.state_dict())
        return {"dqn_loss": float(loss), "q_mean": float(current.mean()),
                "q_min": float(current.min()), "q_max": float(current.max()),
                "target_mean": float(target.mean()), "target_std": float(target.std(unbiased=False)),
                "td_abs_mean": float((current - target).abs().mean()),
                "critic_grad_norm": grad, "target_updated": int(target_updated)}

    def _update_ppo(self):
        rows = self.rollout
        states, actions, rewards, next_states, dones, old_logp, old_values = zip(*rows)
        with torch.no_grad():
            last_value = 0.0 if dones[-1] else float(self.value(_state_tensor(next_states[-1], self.device).reshape(1, -1)))
        advantages = generalized_advantages(rewards, old_values, dones, last_value,
                                            self.train["gamma"], self.train["gae_lambda"])
        returns = advantages + np.asarray(old_values, dtype=np.float32)
        s = _state_tensor(states, self.device)
        a = torch.as_tensor(actions, dtype=torch.long, device=self.device)
        logp0 = torch.as_tensor(old_logp, dtype=torch.float32, device=self.device)
        adv = torch.as_tensor(advantages, device=self.device)
        adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
        ret = torch.as_tensor(returns, device=self.device)
        metrics = {}
        for _ in range(self.train["ppo_epochs"]):
            dists = [torch.distributions.Categorical(logits=logits) for logits in self.actor(s)]
            logp = sum(dist.log_prob(a[:, i]) for i, dist in enumerate(dists))
            entropy_heads = [dist.entropy() for dist in dists]
            entropy = sum(entropy_heads)
            ratio = torch.exp(logp - logp0)
            clipped = torch.clamp(ratio, 1 - self.train["clip_ratio"], 1 + self.train["clip_ratio"])
            actor_loss = -(torch.minimum(ratio * adv, clipped * adv)).mean() - self.train["entropy_coef"] * entropy.mean()
            value_pred = self.value(s)
            value_loss = F.mse_loss(value_pred, ret) * self.train["value_coef"]
            _finite_loss(actor_loss + value_loss)
            self.actor_opt.zero_grad(set_to_none=True)
            actor_loss.backward()
            actor_grad = _grad_norm(self.actor, self.train["gradient_clip"])
            self.actor_opt.step()
            self.value_opt.zero_grad(set_to_none=True)
            value_loss.backward()
            value_grad = _grad_norm(self.value, self.train["gradient_clip"])
            self.value_opt.step()
            with torch.no_grad():
                residual = ret - value_pred
                explained = 1 - torch.var(residual, unbiased=False) / (torch.var(ret, unbiased=False) + 1e-8)
                metrics = {"actor_loss": float(actor_loss), "value_loss": float(value_loss),
                           "entropy": float(entropy.mean()), "approx_kl": float((logp0 - logp).mean()),
                           "clip_fraction": float(((ratio - 1).abs() > self.train["clip_ratio"]).float().mean()),
                           "explained_variance": float(explained), "value_mean": float(value_pred.mean()),
                           "return_target_mean": float(ret.mean()), "actor_grad_norm": actor_grad,
                           "value_grad_norm": value_grad,
                           **{f"entropy_{name}": float(value.mean()) for name, value in zip(HEADS, entropy_heads)}}
            self.update_step += 1
        self.rollout = []
        return metrics

    def state_dict(self):
        names = ("actor", "critic1", "critic2", "target1", "target2", "value", "online", "target")
        optimizers = ("actor_opt", "critic1_opt", "critic2_opt", "alpha_opt", "value_opt", "optimizer")
        return {"networks": {n: getattr(self, n).state_dict() for n in names if hasattr(self, n)},
                "optimizers": {n: getattr(self, n).state_dict() for n in optimizers if hasattr(self, n)},
                "log_alpha": self.log_alpha.detach().cpu() if hasattr(self, "log_alpha") else None,
                "replay": self.replay.state_dict() if hasattr(self, "replay") else None,
                "rollout": getattr(self, "rollout", None), "rng": self.rng.getstate(),
                "update_step": self.update_step}

    def load_state_dict(self, data):
        for name, state in data["networks"].items():
            getattr(self, name).load_state_dict(state, strict=True)
        for name, state in data["optimizers"].items():
            getattr(self, name).load_state_dict(state)
        if data["log_alpha"] is not None:
            self.log_alpha.data.copy_(data["log_alpha"].to(self.device))
        if data["replay"] is not None:
            self.replay.load_state_dict(data["replay"])
        if data["rollout"] is not None:
            self.rollout = data["rollout"]
        self.rng.setstate(data["rng"])
        self.update_step = data["update_step"]


def build_agent(config, state_dim, action_dims, device="cpu"):
    return WorkbenchAgent(config, state_dim, action_dims, device)
