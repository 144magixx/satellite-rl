"""Trainable, factorised discrete SAC, PPO and Double DQN workbench agents."""

from __future__ import annotations

from collections import deque
import copy
import math
import random

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from satellite_rl.agents.action_masking import (conditional_masked_policy, copy_action_mask, entropy_ratio,
                                                feasible_target_entropy, mask_batch,
                                                masked_policy)
from satellite_rl.agents.workbench_networks import (ConditionalPolicyNetwork, JointQNetwork,
                                                    MultiHeadNetwork, ScalarNetwork,
                                                    TemporalMultiHeadNetwork)
from satellite_rl.agents.sequence_replay import SequenceReplay


HEADS = ("frequency", "slots", "power")


def _state_tensor(states, device):
    return torch.as_tensor(np.asarray(states, dtype=np.float32), device=device)


def _joint_q(heads):
    return heads[0][:, :, None, None] + heads[1][:, None, :, None] + heads[2][:, None, None, :]


def _gather(heads, actions):
    return sum(q.gather(1, actions[:, i:i + 1]).squeeze(1) for i, q in enumerate(heads))


def finite_tensor_check(tensors):
    """Check every tensor, using a fused CUDA max norm only when it is safe.

    The infinity norm of a nonempty real tensor is finite exactly when every
    element is finite. Empty and complex tensors use the original elementwise
    check: the norm of an empty tensor is undefined, and complex magnitude can
    overflow even when both components are finite.
    """
    by_device = {}
    for tensor in tensors:
        by_device.setdefault(tensor.device, []).append(tensor)
    with torch.no_grad():
        for group in by_device.values():
            checks = []
            eligible = {}
            foreach_norm = getattr(torch, "_foreach_norm", None)
            for tensor in group:
                if (foreach_norm is not None and tensor.device.type == "cuda"
                        and tensor.layout == torch.strided and tensor.is_contiguous()
                        and tensor.numel() > 0 and tensor.dtype in
                        (torch.float16, torch.bfloat16, torch.float32, torch.float64)):
                    eligible.setdefault(tensor.dtype, []).append(tensor)
                else:
                    checks.append(torch.isfinite(tensor).all())
            for same_dtype in eligible.values():
                try:
                    norms = foreach_norm(same_dtype, ord=math.inf)
                except (RuntimeError, TypeError, NotImplementedError):
                    checks.extend(torch.isfinite(tensor).all() for tensor in same_dtype)
                else:
                    checks.append(torch.isfinite(torch.stack(norms)).all())
            if not bool(torch.stack(checks).all()):
                return False
    return True


def ensure_finite_modules(modules, log_alpha=None):
    """Check all network parameters and SAC temperature before a checkpoint."""
    pairs = tuple(modules.items() if isinstance(modules, dict) else modules)
    tensors = [parameter for _, module in pairs for parameter in module.parameters()]
    if log_alpha is not None:
        tensors.append(log_alpha)
    if finite_tensor_check(tensors):
        return
    # A failure is rare; recover the original precise network/temperature error.
    for name, module in pairs:
        for parameter in module.parameters():
            if not bool(torch.isfinite(parameter).all()):
                raise FloatingPointError(f"Nonfinite {name} parameter")
    if log_alpha is not None and not bool(torch.isfinite(log_alpha).all()):
        raise FloatingPointError("Nonfinite SAC temperature")
    raise FloatingPointError("Nonfinite network parameter")


def _diagnostics(values):
    """Transfer scalar tensor diagnostics in one batch per device and dtype."""
    groups = {}
    for name, value in values.items():
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                raise ValueError(f"Diagnostic {name} must be scalar")
            groups.setdefault((value.device, value.dtype), []).append((name, value.detach().reshape(())))
    converted = {}
    for group in groups.values():
        numbers = torch.stack([value for _, value in group]).cpu().tolist()
        converted.update((name, float(number)) for (name, _), number in zip(group, numbers))
    return {name: converted[name] if name in converted else value for name, value in values.items()}


def _grad_norm(module, clip):
    grads = [param.grad for param in module.parameters() if param.grad is not None]
    if not finite_tensor_check(grads):
        raise FloatingPointError("Nonfinite gradient before optimizer step")
    norm = nn.utils.clip_grad_norm_(module.parameters(), clip)
    norm_value = float(norm.detach())
    if not math.isfinite(norm_value):
        raise FloatingPointError("Nonfinite gradient norm")
    return norm_value


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
    def __init__(self, capacity, seed, bucket_mode="none"):
        if bucket_mode not in ("none", "fixed_100", "mixed_four_buckets"):
            raise ValueError("Unknown replay bucket mode")
        self.bucket_mode = bucket_mode
        self.capacity = capacity
        self.rows = (deque(maxlen=capacity) if bucket_mode == "none" else [])
        self.rng = random.Random(seed)
        self.next_position = 0
        self.bucket_indices = ({name: [] for name in
                                (("anchor_100",) if bucket_mode == "fixed_100" else
                                 ("anchor_100", "lower", "middle", "upper"))}
                               if bucket_mode != "none" else None)
        self.bucket_positions = ({name: {} for name in self.bucket_indices}
                                 if self.bucket_indices is not None else None)
        self.slot_buckets = []
        self.sampled_by_bucket = {key: 0 for key in (self.bucket_indices or {})}
        self.replacement_draws = {key: 0 for key in (self.bucket_indices or {})}

    def __len__(self):
        return len(self.rows)

    def add(self, state, action, reward, next_state, done, action_mask=None, next_action_mask=None,
            bucket=None):
        row = (np.asarray(state, dtype=np.float32).copy(),
               tuple(int(x) for x in action), float(reward),
               np.asarray(next_state, dtype=np.float32).copy(), bool(done))
        row = row if action_mask is None else row + (action_mask, next_action_mask)
        if self.bucket_mode == "none":
            if bucket is not None:
                raise ValueError("Legacy replay does not accept a bucket")
            self.rows.append(row)
            return
        if bucket not in self.bucket_indices:
            raise ValueError("Transition bucket does not match configured sampling mode")
        index = self.next_position
        if index < len(self.rows):
            old_bucket = self.slot_buckets[index]
            members = self.bucket_indices[old_bucket]
            old_position = self.bucket_positions[old_bucket].pop(index)
            last_index = members.pop()
            if old_position < len(members):
                members[old_position] = last_index
                self.bucket_positions[old_bucket][last_index] = old_position
            self.rows[index] = row
            self.slot_buckets[index] = bucket
        else:
            self.rows.append(row)
            self.slot_buckets.append(bucket)
        self.bucket_positions[bucket][index] = len(self.bucket_indices[bucket])
        self.bucket_indices[bucket].append(index)
        self.next_position = (index + 1) % self.capacity

    def sample(self, count, device, action_dims=None):
        if self.bucket_mode == "none":
            rows = self.rng.sample(self.rows, count)
        else:
            if count % len(self.bucket_indices):
                raise ValueError("Batch size must divide evenly across replay buckets")
            per_bucket = count // len(self.bucket_indices)
            indices = []
            for bucket, members in self.bucket_indices.items():
                if not members:
                    raise ValueError("Every active replay bucket needs a real transition")
                chosen = (self.rng.sample(members, per_bucket) if len(members) >= per_bucket
                          else self.rng.choices(members, k=per_bucket))
                indices.extend(chosen)
                self.sampled_by_bucket[bucket] += per_bucket
                if len(members) < per_bucket:
                    self.replacement_draws[bucket] += per_bucket - len(set(chosen))
            self.rng.shuffle(indices)
            rows = [self.rows[index] for index in indices]
        states, actions, rewards, next_states, dones = zip(*(row[:5] for row in rows))
        result = (_state_tensor(states, device), torch.as_tensor(actions, dtype=torch.long, device=device),
                  torch.as_tensor(rewards, dtype=torch.float32, device=device),
                  _state_tensor(next_states, device), torch.as_tensor(dones, dtype=torch.float32, device=device))
        if action_dims is None:
            return result
        if any(len(row) != 7 for row in rows):
            raise ValueError("masked SAC replay row has no action masks")
        return result + (mask_batch([row[5] for row in rows], action_dims, device),
                         mask_batch([row[6] for row in rows], action_dims, device))

    def state_dict(self):
        if self.bucket_mode == "none":
            return {"rows": list(self.rows), "rng": self.rng.getstate(), "capacity": self.rows.maxlen}
        return {"rows": list(self.rows), "rng": self.rng.getstate(), "capacity": self.capacity,
                "bucket_mode": self.bucket_mode, "next_position": self.next_position,
                "slot_buckets": list(self.slot_buckets),
                "bucket_indices": {key: list(value) for key, value in self.bucket_indices.items()},
                "sampled_by_bucket": dict(self.sampled_by_bucket),
                "replacement_draws": dict(self.replacement_draws)}

    def load_state_dict(self, data):
        if data["capacity"] != self.capacity:
            raise ValueError("Replay capacity changed")
        if data.get("bucket_mode", "none") != self.bucket_mode:
            raise ValueError("Replay bucket mode changed")
        if self.bucket_mode == "none":
            self.rows = deque(data["rows"], maxlen=self.capacity)
        else:
            rows, buckets = list(data["rows"]), list(data["slot_buckets"])
            if len(rows) != len(buckets) or len(rows) > self.capacity:
                raise ValueError("Replay ring length mismatch")
            indices = {key: [] for key in self.bucket_indices}
            saved_indices = data["bucket_indices"]
            if set(saved_indices) != set(indices):
                raise ValueError("Replay checkpoint bucket names differ")
            for index, bucket in enumerate(buckets):
                if bucket not in indices:
                    raise ValueError("Unknown checkpoint replay bucket")
                indices[bucket].append(index)
            if any(sorted(value) != sorted(saved_indices[key]) for key, value in indices.items()):
                raise ValueError("Replay bucket index differs from stored rows")
            position = data["next_position"]
            if not isinstance(position, int) or position < 0 or position >= self.capacity or (len(rows) < self.capacity and position != len(rows)):
                raise ValueError("Invalid replay ring position")
            self.rows, self.slot_buckets = rows, buckets
            self.bucket_indices = {key: list(saved_indices[key]) for key in indices}
            self.bucket_positions = {key: {index: offset for offset, index in enumerate(value)}
                                     for key, value in self.bucket_indices.items()}
            self.next_position = position
            self.sampled_by_bucket = dict(data["sampled_by_bucket"])
            self.replacement_draws = dict(data["replacement_draws"])
        self.rng.setstate(data["rng"])

    @property
    def capacity(self):
        return self.rows.maxlen if self.bucket_mode == "none" else self._capacity

    @capacity.setter
    def capacity(self, value):
        self._capacity = value


class WorkbenchAgent:
    def __init__(self, config, state_dim, action_dims, device="cpu"):
        self.config = config
        self.train = config["training"]
        self.device = torch.device(device)
        self.algorithm = config["algorithm"]
        self.action_mask_mode = config.get("environment", {}).get("action_mask", "none")
        if self.action_mask_mode not in ("none", "joint_feasible"):
            raise ValueError(f"Unsupported action mask mode: {self.action_mask_mode}")
        if self.action_mask_mode != "none" and self.algorithm != "sac":
            raise ValueError("joint_feasible action mask currently supports SAC only")
        if self.algorithm == "ppo" and config["network"]["dropout"]:
            raise ValueError("PPO does not support dropout: action and update probabilities must match")
        self.action_dims = tuple(action_dims)
        self.temporal_mode = config["network"].get("temporal_mode", "none")
        self.temporal_window = int(config["network"].get("temporal_window", 16))
        self.temporal_enabled = self.temporal_mode in ("gru", "causal_transformer")
        if self.temporal_mode not in ("none", "previous_action", "gru", "causal_transformer"):
            raise ValueError("Unsupported temporal mode")
        if self.temporal_enabled and (self.algorithm != "sac" or
                                      self.action_mask_mode != "joint_feasible" or
                                      not 1 <= self.temporal_window <= 16 or state_dim != 211):
            raise ValueError("Temporal SAC requires joint_feasible, 211 fields and a 1–16 step window")
        self.online_history = deque(maxlen=self.temporal_window) if self.temporal_enabled else None
        self.alpha_mode = self.train.get("alpha_mode", "adaptive")
        if self.alpha_mode not in ("adaptive", "fixed"):
            raise ValueError("Unsupported SAC alpha mode")
        if self.alpha_mode == "fixed" and self.algorithm != "sac":
            raise ValueError("Fixed alpha requires SAC")
        self.sac_policy_structure = config["network"].get("sac_policy_structure", "factorized")
        self.sac_critic_structure = config["network"].get("sac_critic_structure", "additive")
        if self.sac_policy_structure not in ("factorized", "conditional"):
            raise ValueError("Unsupported SAC policy structure")
        if self.sac_critic_structure not in ("additive", "joint"):
            raise ValueError("Unsupported SAC critic structure")
        if (self.sac_policy_structure != "factorized" or self.sac_critic_structure != "additive"):
            if self.algorithm != "sac" or self.action_mask_mode != "joint_feasible":
                raise ValueError("New SAC structures require SAC with joint_feasible action masking")
            if self.action_dims != (100, 10, 10):
                raise ValueError("New SAC structures require action dimensions (100, 10, 10)")
        self.target_entropy_ratio = (entropy_ratio(self.train["target_entropy"], self.action_dims)
                                     if self.action_mask_mode == "joint_feasible" else None)
        self.rng = random.Random(self.train["seed"] + 31)
        self.update_step = 0
        self.last_action_info = {}
        if self.algorithm == "sac":
            actor_class = (TemporalMultiHeadNetwork if self.temporal_enabled else
                           ConditionalPolicyNetwork if self.sac_policy_structure == "conditional"
                           else MultiHeadNetwork)
            critic_class = (TemporalMultiHeadNetwork if self.temporal_enabled else
                            JointQNetwork if self.sac_critic_structure == "joint" else MultiHeadNetwork)
            self.actor = actor_class(config, state_dim, action_dims).to(self.device)
            self.critic1 = critic_class(config, state_dim, action_dims).to(self.device)
            self.critic2 = critic_class(config, state_dim, action_dims).to(self.device)
            self.target1 = copy.deepcopy(self.critic1).to(self.device).eval()
            self.target2 = copy.deepcopy(self.critic2).to(self.device).eval()
            self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.train["actor_lr"])
            self.critic1_opt = torch.optim.Adam(self.critic1.parameters(), lr=self.train["critic_lr"])
            self.critic2_opt = torch.optim.Adam(self.critic2.parameters(), lr=self.train["critic_lr"])
            alpha_value = (self.train["fixed_alpha"] if self.alpha_mode == "fixed"
                           else self.train["initial_alpha"])
            self.log_alpha = nn.Parameter(torch.tensor(np.log(alpha_value),
                                                       dtype=torch.float32, device=self.device),
                                          requires_grad=self.alpha_mode == "adaptive")
            self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=self.train["alpha_lr"])
            self.replay = (SequenceReplay(self.train["replay_capacity"], self.train["seed"] + 41,
                                          self.temporal_window) if self.temporal_enabled else
                           Replay(self.train["replay_capacity"], self.train["seed"] + 41,
                                  self.train.get("beam_count_mode", "none")))
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

    def reset_episode(self):
        if self.temporal_enabled:
            self.online_history.clear()

    def _sac_actor_logits(self, states):
        """Match action-time probabilities even when the actor has dropout."""
        if self.temporal_enabled:
            # Temporal modes require dropout=0. cuDNN GRU backward cannot run
            # after eval-mode forward; train/eval probabilities are identical.
            return self.actor(states)
        was_training = self.actor.training
        self.actor.eval()
        try:
            return self.actor(states)
        finally:
            self.actor.train(was_training)

    def act(self, state, deterministic=False, env_step=None, action_mask=None):
        if self.action_mask_mode == "joint_feasible":
            action_mask = copy_action_mask(action_mask, self.action_dims)
        elif action_mask is not None:
            raise ValueError("action_mask supplied while masking is disabled")
        net = self.policy_network()
        was_training = net.training
        net.eval()
        with torch.no_grad():
            x = _state_tensor(state, self.device).reshape(1, -1)
            if self.temporal_enabled:
                self.online_history.append(np.asarray(state, dtype=np.float32).copy())
                sequence = _state_tensor(list(self.online_history), self.device).unsqueeze(0)
                x = (sequence, torch.as_tensor([len(self.online_history)], device=self.device))
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
            elif self.action_mask_mode == "joint_feasible":
                frequency_slots, power = mask_batch([action_mask], self.action_dims, self.device)
                if self.sac_policy_structure == "conditional":
                    (fl, conditional_power, log_fl, log_p, entropy_fl, entropy_p,
                     entropy_f, entropy_slots) = conditional_masked_policy(
                        heads, frequency_slots, power)
                    joint = fl.unsqueeze(-1) * conditional_power
                    if deterministic:
                        index = joint.flatten(1).argmax(1)
                        frequency = index // (self.action_dims[1] * self.action_dims[2])
                        slots = index // self.action_dims[2] % self.action_dims[1]
                        power_index = index % self.action_dims[2]
                    else:
                        fl_index = torch.distributions.Categorical(probs=fl.flatten(1)).sample()
                        frequency = fl_index // self.action_dims[1]
                        slots = fl_index % self.action_dims[1]
                        power_index = torch.distributions.Categorical(
                            probs=conditional_power[0, frequency.item(), slots.item()].reshape(1, -1)).sample()
                    action = (int(frequency.item()), int(slots.item()), int(power_index.item()))
                    self.last_action_info = {
                        "log_prob": float((log_fl[0, action[0], action[1]] +
                                           log_p[0, action[0], action[1], action[2]]).item()),
                        "entropy": float((entropy_fl + entropy_p).item()),
                        "entropy_frequency": float(entropy_f.item()),
                        "entropy_slots": float(entropy_slots.item()),
                        "entropy_power": float(entropy_p.item()),
                        "target_entropy": float(feasible_target_entropy(
                            frequency_slots, power, self.target_entropy_ratio).item()),
                    }
                else:
                    (fl, p, log_fl, log_p, entropy_fl, entropy_p,
                     entropy_f, entropy_slots) = masked_policy(heads, frequency_slots, power)
                    flat_fl = fl.flatten(1)
                    fl_index = (flat_fl.argmax(1) if deterministic else
                                torch.distributions.Categorical(probs=flat_fl).sample())
                    power_index = (p.argmax(1) if deterministic else
                                   torch.distributions.Categorical(probs=p).sample())
                    frequency = fl_index // self.action_dims[1]
                    slots = fl_index % self.action_dims[1]
                    action = (int(frequency.item()), int(slots.item()), int(power_index.item()))
                    self.last_action_info = {
                        "log_prob": float((log_fl[0, action[0], action[1]] +
                                           log_p[0, action[2]]).item()),
                        "entropy": float((entropy_fl + entropy_p).item()),
                        "entropy_frequency": float(entropy_f.item()),
                        "entropy_slots": float(entropy_slots.item()),
                        "entropy_power": float(entropy_p.item()),
                        "target_entropy": float(feasible_target_entropy(
                            frequency_slots, power, self.target_entropy_ratio).item()),
                    }
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

    def observe(self, state, action, reward, next_state, done,
                action_mask=None, next_action_mask=None, actual_action=None, replay_bucket=None):
        if self.action_mask_mode == "joint_feasible":
            action_mask = copy_action_mask(action_mask, self.action_dims)
            if next_action_mask is None:
                if not done:
                    raise ValueError("nonterminal masked SAC transition requires next_action_mask")
            else:
                if done:
                    raise ValueError("terminal masked SAC transition must have no next_action_mask")
                next_action_mask = copy_action_mask(next_action_mask, self.action_dims)
            chosen = tuple(int(x) for x in action)
            if (len(chosen) != 3 or any(x < 0 or x >= dim for x, dim in zip(chosen, self.action_dims)) or
                    not action_mask["frequency_slots"][chosen[0], chosen[1]] or
                    not action_mask["power"][chosen[2]]):
                raise ValueError("observed action is forbidden by action_mask")
            if self.temporal_enabled:
                self.replay.add(state, chosen, reward, next_state, done,
                                action_mask, next_action_mask, actual_action)
            else:
                self.replay.add(state, chosen, reward, next_state, done, action_mask, next_action_mask,
                                bucket=replay_bucket)
            return
        if action_mask is not None or next_action_mask is not None:
            raise ValueError("action_mask supplied while masking is disabled")
        if self.algorithm == "ppo":
            if tuple(action) != tuple(self.last_action_info.get("action", ())):
                raise ValueError("PPO observed action differs from the sampled action and log-probability")
            self.rollout.append((np.asarray(state, dtype=np.float32).copy(), tuple(action),
                                 float(reward), np.asarray(next_state, dtype=np.float32).copy(), bool(done),
                                 self.last_action_info["log_prob"], self.last_action_info["value"]))
        else:
            self.replay.add(state, action, reward, next_state, done, bucket=replay_bucket)

    def ready(self, env_step, done=False):
        if self.algorithm == "ppo":
            return len(self.rollout) >= self.train["rollout_steps"] or (done and bool(self.rollout))
        buckets_ready = (not self.replay.bucket_indices or
                         all(self.replay.bucket_indices.values()))
        return (env_step >= self.train["warmup_steps"] and buckets_ready and
                len(self.replay) >= self.train["batch_size"] and
                env_step % self.train["update_every"] == 0)

    def update(self):
        if self.algorithm == "sac":
            return self._update_sac()
        if self.algorithm == "ppo":
            return self._update_ppo()
        return self._update_dqn()

    def _update_sac(self):
        if self.action_mask_mode == "joint_feasible":
            return self._update_masked_sac()
        s, a, r, ns, d = self.replay.sample(self.train["batch_size"], self.device)
        gamma = self.train["gamma"]
        alpha = self.log_alpha.exp().detach()
        with torch.no_grad():
            logits = self._sac_actor_logits(ns)
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
        logits = self._sac_actor_logits(s)
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
        if self.alpha_mode == "adaptive":
            alpha_loss = (self.log_alpha * (entropy_actor.detach().mean() - self.train["target_entropy"]))
            _finite_loss(alpha_loss)
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
        return _diagnostics({"actor_loss": actor_loss, "critic_1_loss": loss1,
                             "critic_2_loss": loss2, "alpha_loss": alpha_loss,
                             "alpha": self.log_alpha.exp(), "entropy": entropy_actor.mean(),
                             **{f"entropy_{name}": value.mean() for name, value in zip(HEADS, entropies)},
                             "q_mean": (q1.mean() + q2.mean()) / 2,
                             "q_min": torch.minimum(q1, q2).min(),
                             "q_max": torch.maximum(q1, q2).max(), "target_mean": target.mean(),
                             "target_std": target.std(unbiased=False),
                             "td_abs_mean": ((q1 - target).abs().mean() + (q2 - target).abs().mean()) / 2,
                             "actor_grad_norm": actor_grad, "critic_1_grad_norm": grad1,
                             "critic_2_grad_norm": grad2,
                             **({"alpha_update_applied": 0} if self.alpha_mode == "fixed" else {})})

    def _update_masked_sac(self):
        if self.temporal_enabled:
            return self._update_temporal_sac()
        if self.sac_policy_structure != "factorized" or self.sac_critic_structure != "additive":
            return self._update_joint_action_sac()
        s, a, r, ns, d, current_masks, next_masks = self.replay.sample(
            self.train["batch_size"], self.device, self.action_dims)
        current_fl, current_power = current_masks
        next_fl, next_power = next_masks
        alpha = self.log_alpha.exp().detach()
        with torch.no_grad():
            target = r.clone()
            active = d < 0.5
            if active.any():
                (next_prob_fl, next_prob_power, _, _, next_entropy_fl,
                 next_entropy_power, _, _) = masked_policy(
                    self._sac_actor_logits(ns[active]), next_fl[active], next_power[active])
                next_min_q = torch.minimum(_joint_q(self.target1(ns[active])),
                                           _joint_q(self.target2(ns[active])))
                next_expected_q = torch.einsum("bfl,bp,bflp->b", next_prob_fl,
                                               next_prob_power, next_min_q)
                target[active] += self.train["gamma"] * (
                    next_expected_q + alpha * (next_entropy_fl + next_entropy_power))
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
        (prob_fl, prob_power, _, _, entropy_fl, entropy_power,
         entropy_frequency, entropy_slots) = masked_policy(
             self._sac_actor_logits(s), current_fl, current_power)
        with torch.no_grad():
            min_q_actor = torch.minimum(_joint_q(self.critic1(s)), _joint_q(self.critic2(s)))
        expected_q_actor = torch.einsum("bfl,bp,bflp->b", prob_fl, prob_power, min_q_actor)
        entropy_actor = entropy_fl + entropy_power
        actor_loss = -(expected_q_actor + alpha * entropy_actor).mean()
        _finite_loss(actor_loss)
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        actor_grad = _grad_norm(self.actor, self.train["gradient_clip"])
        self.actor_opt.step()
        target_entropy = feasible_target_entropy(current_fl, current_power,
                                                  self.target_entropy_ratio)
        if self.alpha_mode == "adaptive":
            alpha_loss = self.log_alpha * (entropy_actor.detach() - target_entropy).mean()
            _finite_loss(alpha_loss)
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
        return _diagnostics({"actor_loss": actor_loss, "critic_1_loss": loss1,
                             "critic_2_loss": loss2, "alpha_loss": alpha_loss,
                             "alpha": self.log_alpha.exp(), "entropy": entropy_actor.mean(),
                             "entropy_frequency": entropy_frequency.mean(),
                             "entropy_slots": entropy_slots.mean(),
                             "entropy_power": entropy_power.mean(),
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

    def _masked_joint_distribution(self, states, frequency_slots, power):
        logits = self._sac_actor_logits(states)
        if self.sac_policy_structure == "conditional":
            parts = conditional_masked_policy(logits, frequency_slots, power)
            fl, conditional_power = parts[:2]
            joint = fl.unsqueeze(-1) * conditional_power
        else:
            parts = masked_policy(logits, frequency_slots, power)
            fl, power_probability = parts[:2]
            joint = fl.unsqueeze(-1) * power_probability[:, None, None, :]
        return joint, parts[4] + parts[5], parts[4], parts[5], parts[6], parts[7]

    def _critic_table(self, critic, states):
        values = critic(states)
        return values if self.sac_critic_structure == "joint" else _joint_q(values)

    def _update_temporal_sac(self):
        return self._update_joint_action_sac()

    def _update_joint_action_sac(self):
        s, a, r, ns, d, current_masks, next_masks = self.replay.sample(
            self.train["batch_size"], self.device, self.action_dims)
        current_fl, current_power = current_masks
        next_fl, next_power = next_masks
        alpha = self.log_alpha.exp().detach()
        with torch.no_grad():
            target = r.clone()
            active = d < 0.5
            if active.any():
                active_next = ((ns[0][active], ns[1][active]) if self.temporal_enabled
                               else ns[active])
                next_prob, next_entropy, *_ = self._masked_joint_distribution(
                    active_next, next_fl[active], next_power[active])
                next_min_q = torch.minimum(self._critic_table(self.target1, active_next),
                                           self._critic_table(self.target2, active_next))
                target[active] += self.train["gamma"] * (
                    (next_prob * next_min_q).sum((1, 2, 3)) + alpha * next_entropy)
        q1_table = self._critic_table(self.critic1, s)
        q2_table = self._critic_table(self.critic2, s)
        batch_index = torch.arange(a.shape[0], device=self.device)
        q1 = q1_table[batch_index, a[:, 0], a[:, 1], a[:, 2]]
        q2 = q2_table[batch_index, a[:, 0], a[:, 1], a[:, 2]]
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
        probability, entropy, entropy_fl, entropy_power, entropy_frequency, entropy_slots = (
            self._masked_joint_distribution(s, current_fl, current_power))
        with torch.no_grad():
            min_q = torch.minimum(self._critic_table(self.critic1, s),
                                  self._critic_table(self.critic2, s))
        actor_loss = -((probability * min_q).sum((1, 2, 3)) + alpha * entropy).mean()
        _finite_loss(actor_loss)
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        actor_grad = _grad_norm(self.actor, self.train["gradient_clip"])
        self.actor_opt.step()
        target_entropy = feasible_target_entropy(current_fl, current_power,
                                                  self.target_entropy_ratio)
        if self.alpha_mode == "adaptive":
            alpha_loss = self.log_alpha * (entropy.detach() - target_entropy).mean()
            _finite_loss(alpha_loss)
            self.alpha_opt.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.alpha_opt.step()
        else:
            alpha_loss = torch.zeros_like(self.log_alpha.detach())
        self.update_step += 1
        with torch.no_grad():
            for target_net, source_net in ((self.target1, self.critic1), (self.target2, self.critic2)):
                for target_p, source_p in zip(target_net.parameters(), source_net.parameters()):
                    target_p.lerp_(source_p, self.train["tau"])
        temporal_diagnostics = ({"effective_td_targets": a.shape[0],
                                 "context_steps_mean": s[1].float().mean()}
                                if self.temporal_enabled else {})
        return _diagnostics({"actor_loss": actor_loss, "critic_1_loss": loss1,
                             "critic_2_loss": loss2, "alpha_loss": alpha_loss,
                             "alpha": self.log_alpha.exp(), "entropy": entropy.mean(),
                             "entropy_frequency": entropy_frequency.mean(),
                             "entropy_slots": entropy_slots.mean(),
                             "entropy_power": entropy_power.mean(),
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
                             **temporal_diagnostics,
                             **({"alpha_update_applied": 0} if self.alpha_mode == "fixed" else {})})

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
        return _diagnostics({"dqn_loss": loss, "q_mean": current.mean(),
                             "q_min": current.min(), "q_max": current.max(),
                             "target_mean": target.mean(), "target_std": target.std(unbiased=False),
                             "td_abs_mean": (current - target).abs().mean(),
                             "critic_grad_norm": grad, "target_updated": int(target_updated)})

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
                metrics = _diagnostics({"actor_loss": actor_loss, "value_loss": value_loss,
                                        "entropy": entropy.mean(), "approx_kl": (logp0 - logp).mean(),
                                        "clip_fraction": ((ratio - 1).abs() > self.train["clip_ratio"]).float().mean(),
                                        "explained_variance": explained, "value_mean": value_pred.mean(),
                                        "return_target_mean": ret.mean(), "actor_grad_norm": actor_grad,
                                        "value_grad_norm": value_grad,
                                        **{f"entropy_{name}": value.mean() for name, value in zip(HEADS, entropy_heads)}})
            self.update_step += 1
        self.rollout = []
        return metrics

    def state_dict(self):
        names = ("actor", "critic1", "critic2", "target1", "target2", "value", "online", "target")
        optimizers = ("actor_opt", "critic1_opt", "critic2_opt", "alpha_opt", "value_opt", "optimizer")
        result = {"networks": {n: getattr(self, n).state_dict() for n in names if hasattr(self, n)},
                  "optimizers": {n: getattr(self, n).state_dict() for n in optimizers if hasattr(self, n)},
                  "log_alpha": self.log_alpha.detach().cpu() if hasattr(self, "log_alpha") else None,
                  "replay": self.replay.state_dict() if hasattr(self, "replay") else None,
                  "rollout": getattr(self, "rollout", None), "rng": self.rng.getstate(),
                  "update_step": self.update_step}
        if self.algorithm == "sac":
            result["policy_probability_mode"] = "eval_v1"
            result["alpha_mode"] = self.alpha_mode
            result["sac_policy_structure"] = self.sac_policy_structure
            result["sac_critic_structure"] = self.sac_critic_structure
            if self.temporal_mode != "none":
                result["temporal_mode"] = self.temporal_mode
                result["temporal_window"] = self.temporal_window
            if self.temporal_enabled:
                result["online_history"] = [np.asarray(row, dtype=np.float32).copy()
                                            for row in self.online_history]
        return result

    def load_state_dict(self, data):
        if self.algorithm == "sac":
            if data.get("alpha_mode", "adaptive") != self.alpha_mode:
                raise ValueError("SAC checkpoint alpha mode differs from configured mode")
            if (data.get("temporal_mode", "none") != self.temporal_mode or
                    (self.temporal_enabled and data.get("temporal_window") != self.temporal_window)):
                raise ValueError("SAC checkpoint temporal mode/window differs")
            if data.get("sac_policy_structure", "factorized") != self.sac_policy_structure or \
                    data.get("sac_critic_structure", "additive") != self.sac_critic_structure:
                raise ValueError("SAC checkpoint action structures differ from configured structures")
            mode = data.get("policy_probability_mode")
            if mode is None:
                if self.action_mask_mode == "joint_feasible":
                    raise ValueError("Masked SAC checkpoint lacks policy_probability_mode; cannot resume")
                if self.config["network"]["dropout"] > 0:
                    raise ValueError("Historical SAC dropout checkpoint uses different policy probabilities; "
                                     "resume it with its original frozen source")
            elif mode != "eval_v1":
                raise ValueError(f"Unsupported SAC policy_probability_mode: {mode}")
            if self.action_mask_mode == "joint_feasible":
                replay = data.get("replay")
                if not isinstance(replay, dict) or not isinstance(replay.get("rows"), list):
                    raise ValueError("Masked SAC checkpoint lacks replay with action masks")
                for row in replay["rows"]:
                    if (len(row) != (10 if self.temporal_enabled else 7) or row[5] is None or
                            (row[6] is None) != bool(row[4])):
                        raise ValueError("Masked SAC checkpoint replay row lacks current/next action masks")
            if self.alpha_mode == "fixed" and not math.isclose(
                    float(data["log_alpha"].exp()), float(self.train["fixed_alpha"]),
                    rel_tol=1e-6, abs_tol=1e-8):
                raise ValueError("Fixed alpha checkpoint differs from configured value")
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
        if self.temporal_enabled:
            history = data.get("online_history")
            if not isinstance(history, list) or len(history) > self.temporal_window:
                raise ValueError("Temporal checkpoint online history missing or too long")
            self.online_history = deque((np.asarray(row, dtype=np.float32).copy()
                                         for row in history), maxlen=self.temporal_window)


def build_agent(config, state_dim, action_dims, device="cpu"):
    return WorkbenchAgent(config, state_dim, action_dims, device)
