"""Masked PPO and Double DQN on the unchanged 85f T0 representation.

SAC is the original WorkbenchAgent, with no replacement update formula. PPO
uses the original T0 masked factorized distribution. DQN retains three additive
heads; masking does not introduce a learned joint action table.
"""
from copy import deepcopy
import random
import numpy as np
import torch
from torch.nn import functional as F

from satellite_rl.agents.action_masking import copy_action_mask, mask_batch, masked_policy
from satellite_rl.agents.workbench_algorithms import (
    WorkbenchAgent, build_agent as t0_build_agent, _state_tensor, _gather,
    _finite_loss, _grad_norm, _diagnostics, generalized_advantages,
)


def masked_additive_argmax(heads, frequency_slots, power):
    """Exact maximizer on the feasible product set, with lowest-index ties."""
    f, length, p = heads
    if not frequency_slots.any((1, 2)).all() or not power.any(1).all():
        raise ValueError("Active DQN successor has no legal action")
    index = (f[:, :, None] + length[:, None, :]).masked_fill(
        ~frequency_slots, -torch.inf).flatten(1).argmax(1)
    return torch.stack((index // length.shape[1], index % length.shape[1],
                        p.masked_fill(~power, -torch.inf).argmax(1)), 1)


def validate_contract(config, state_dim, action_dims):
    if state_dim != 207 or tuple(action_dims) != (100, 10, 10):
        raise ValueError("T0 comparison requires 207 state fields and (100,10,10) heads")
    n = config["network"]
    expected = {"kind": "transformer", "transformer_layout": "tsac_natural_context",
                "embedding_dim": 128, "num_heads": 8, "num_layers": 2,
                "feedforward_dim": 512, "dropout": 0.,
                "sac_policy_structure": "factorized", "sac_critic_structure": "additive"}
    if any(n.get(k) != v for k, v in expected.items()):
        raise ValueError("Network differs from frozen T0")
    e, t = config["environment"], config["training"]
    if (e["profile"] != "europe_v1" or e["action_mask"] != "joint_feasible"
            or e["power_levels_w"] != list(range(5, 51, 5)) or e["power_budget_w"] != 3000
            or config["reward"]["preset"] != "legacy_repaired" or t["gamma"] != .99):
        raise ValueError("Environment/reward differs from frozen T0")
    if config["algorithm"] == "sac" and (t["alpha_mode"] != "fixed" or t["fixed_alpha"] != .01):
        raise ValueError("T0 SAC requires fixed alpha .01")


class MaskedT0Agent(WorkbenchAgent):
    """Reuse original network constructors and checkpoint layout for new arms."""
    def __init__(self, config, state_dim, action_dims, device="cpu"):
        # The original guard restricts masks/fixed alpha to SAC. Construct only
        # the original PPO/DQN networks, then enable our independently implemented
        # mask semantics. No original module or scientific SAC code is modified.
        constructor = deepcopy(config)
        constructor["environment"]["action_mask"] = "none"
        constructor["training"]["alpha_mode"] = "adaptive"
        super().__init__(constructor, state_dim, action_dims, device)
        self.config, self.train = deepcopy(config), deepcopy(config["training"])
        self.action_mask_mode = "joint_feasible"
        self.interactions = 0
        self.minibatch_rng = random.Random(self.train["seed"] + 51)

    def act(self, state, deterministic=False, env_step=None, action_mask=None):
        mask = copy_action_mask(action_mask, self.action_dims)
        fl_mask, p_mask = mask_batch([mask], self.action_dims, self.device)
        net = self.policy_network()
        previous = net.training
        net.eval()
        try:
            with torch.no_grad():
                x = _state_tensor(state, self.device).reshape(1, -1)
                heads = net(x)
                if self.algorithm == "dqn":
                    epsilon = 0. if deterministic else self.train["eps_end"]
                    if not deterministic and env_step is not None:
                        epsilon += (self.train["eps_start"] - self.train["eps_end"]) * max(
                            0., 1. - env_step / self.train["eps_decay_steps"])
                    action = tuple(masked_additive_argmax(heads, fl_mask, p_mask)[0].tolist())
                    if not deterministic and self.rng.random() < epsilon:
                        legal_fl = np.flatnonzero(mask["frequency_slots"])
                        index = int(self.rng.choice(legal_fl.tolist()))
                        action = (index // 10, index % 10,
                                  int(self.rng.choice(np.flatnonzero(mask["power"]).tolist())))
                    self.last_action_info = {"epsilon": epsilon, "q_mean": float(_gather(
                        heads, torch.tensor([action], device=self.device)).item()), "action": action}
                else:
                    fl, p, lf, lp, ef, ep, _, _ = masked_policy(heads, fl_mask, p_mask)
                    fi = int(fl.flatten(1).argmax(1).item()) if deterministic else int(
                        torch.distributions.Categorical(probs=fl.flatten(1)).sample().item())
                    pi = int(p.argmax(1).item()) if deterministic else int(
                        torch.distributions.Categorical(probs=p).sample().item())
                    action = (fi // 10, fi % 10, pi)
                    self.last_action_info = {"action": action,
                        "log_prob": float((lf[0, action[0], action[1]] + lp[0, pi]).item()),
                        "entropy": float((ef + ep).item()), "value": float(self.value(x).item())}
                return action
        finally:
            net.train(previous)

    def observe(self, state, action, reward, next_state, done, action_mask=None,
                next_action_mask=None, actual_action=None):
        mask = copy_action_mask(action_mask, self.action_dims)
        if (next_action_mask is None) != bool(done):
            raise ValueError("Only real terminal transitions may have no successor mask")
        nxt = None if done else copy_action_mask(next_action_mask, self.action_dims)
        a = tuple(int(v) for v in action)
        if (len(a) != 3 or any(v < 0 or v >= d for v, d in zip(a, self.action_dims))
                or not mask["frequency_slots"][a[0], a[1]] or not mask["power"][a[2]]
                or not np.isfinite(reward)):
            raise ValueError("Invalid action/reward in observed transition")
        if self.algorithm == "ppo":
            if a != tuple(self.last_action_info.get("action", ())):
                raise ValueError("PPO action differs from sampled action")
            self.rollout.append((np.asarray(state, np.float32).copy(), a, float(reward),
                np.asarray(next_state, np.float32).copy(), bool(done),
                self.last_action_info["log_prob"], self.last_action_info["value"], mask, nxt))
        else:
            self.replay.add(state, a, reward, next_state, done, mask, nxt)
        self.interactions += 1

    def ready(self, env_step, done=False):
        if self.algorithm == "ppo":
            return len(self.rollout) >= self.train["rollout_steps"] or (done and bool(self.rollout))
        return super().ready(env_step, done)

    def flush(self):
        """Update a final partial PPO rollout without any extra interaction."""
        return self.update() if self.algorithm == "ppo" and self.rollout else {}

    def _update_dqn(self):
        s, a, r, ns, d, _, next_masks = self.replay.sample(
            self.train["batch_size"], self.device, self.action_dims)
        current = _gather(self.online(s), a)
        with torch.no_grad():
            target = r.clone()
            active = d < .5
            if active.any():
                fl, p = next_masks
                selected = masked_additive_argmax(self.online(ns[active]), fl[active], p[active])
                target[active] += self.train["gamma"] * _gather(self.target(ns[active]), selected)
        loss = F.smooth_l1_loss(current, target)
        _finite_loss(loss)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad = _grad_norm(self.online, self.train["gradient_clip"])
        self.optimizer.step()
        self.update_step += 1
        copied = self.update_step % self.train["target_update_interval"] == 0
        if copied:
            self.target.load_state_dict(self.online.state_dict())
        return _diagnostics({"dqn_loss": loss, "q_mean": current.mean(), "target_mean": target.mean(),
            "td_abs_mean": (current-target).abs().mean(), "critic_grad_norm": grad,
            "target_updated": int(copied)})

    def _update_ppo(self):
        rows = self.rollout
        if not rows:
            raise ValueError("No PPO rollout")
        states, actions, rewards, next_states, dones, old_logp, old_values, masks, _ = zip(*rows)
        with torch.no_grad():
            last = 0. if dones[-1] else float(self.value(
                _state_tensor(next_states[-1], self.device).reshape(1, -1)).item())
        advantages = generalized_advantages(rewards, old_values, dones, last,
                                            self.train["gamma"], self.train["gae_lambda"])
        returns = advantages + np.asarray(old_values, np.float32)
        s, a = _state_tensor(states, self.device), torch.tensor(actions, device=self.device)
        logp0 = torch.tensor(old_logp, dtype=torch.float32, device=self.device)
        adv = torch.tensor(advantages, device=self.device)
        adv = (adv-adv.mean()) / (adv.std(unbiased=False)+1e-8)
        ret = torch.tensor(returns, device=self.device)
        fl_mask, p_mask = mask_batch(masks, self.action_dims, self.device)
        metrics = {}
        for _ in range(self.train["ppo_epochs"]):
            indices = list(range(len(rows)))
            self.minibatch_rng.shuffle(indices)
            for start in range(0, len(rows), self.train["batch_size"]):
                i = torch.tensor(indices[start:start+self.train["batch_size"]], device=self.device)
                _, _, lf, lp, ef, ep, _, _ = masked_policy(self.actor(s[i]), fl_mask[i], p_mask[i])
                batch_a = a[i]
                local = torch.arange(len(i), device=self.device)
                logp = lf[local, batch_a[:, 0], batch_a[:, 1]] + lp[local, batch_a[:, 2]]
                ratio = (logp-logp0[i]).exp()
                clipped = ratio.clamp(1-self.train["clip_ratio"], 1+self.train["clip_ratio"])
                entropy = (ef+ep).mean()
                actor_loss = -torch.minimum(ratio*adv[i], clipped*adv[i]).mean() - self.train["entropy_coef"]*entropy
                value_loss = self.train["value_coef"]*F.mse_loss(self.value(s[i]), ret[i])
                _finite_loss(actor_loss+value_loss)
                self.actor_opt.zero_grad(set_to_none=True)
                actor_loss.backward()
                actor_grad = _grad_norm(self.actor, self.train["gradient_clip"])
                self.actor_opt.step()
                self.value_opt.zero_grad(set_to_none=True)
                value_loss.backward()
                value_grad = _grad_norm(self.value, self.train["gradient_clip"])
                self.value_opt.step()
                self.update_step += 1
                metrics = _diagnostics({"actor_loss": actor_loss, "value_loss": value_loss,
                    "entropy": entropy, "approx_kl": (logp0[i]-logp).mean(),
                    "actor_grad_norm": actor_grad, "value_grad_norm": value_grad})
        self.rollout = []
        return metrics

    def state_dict(self):
        result = super().state_dict()
        result.update(t0_comparison_version=1, interactions=self.interactions,
                      minibatch_rng=self.minibatch_rng.getstate())
        return result

    def load_state_dict(self, data):
        if data.get("t0_comparison_version") != 1:
            raise ValueError("Not a T0 comparison checkpoint")
        super().load_state_dict(data)
        self.interactions = int(data["interactions"])
        self.minibatch_rng.setstate(data["minibatch_rng"])


def build_agent(config, state_dim, action_dims=(100, 10, 10), device="cpu"):
    validate_contract(config, state_dim, action_dims)
    if config["algorithm"] == "sac":
        return t0_build_agent(config, state_dim, action_dims, device)
    if config["algorithm"] not in ("ppo", "dqn"):
        raise ValueError("Only SAC/PPO/Double DQN are supported")
    return MaskedT0Agent(config, state_dim, action_dims, device)
