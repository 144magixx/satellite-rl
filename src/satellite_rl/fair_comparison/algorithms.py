"""Masked discrete SAC, PPO-Clip and Transformer Double DQN for F20.

Checkpoint state includes all learning data and independent exploration/replay
streams; no global random generator is used for action or replay sampling.
"""
from copy import deepcopy
import pickle
import zlib
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .networks import TransformerNetwork, masked_distribution, parameter_sha, tensor_observations

REPLAY_FORMAT = "pickle5_zlib1_lossless_v1"


def pack_transition(transition):
    """Lossless immutable host storage: float32 bits and mask semantics retained."""
    return zlib.compress(pickle.dumps(transition, protocol=5), level=1)


def unpack_transition(payload):
    return pickle.loads(zlib.decompress(payload))


def bellman_target(reward, continuation, terminated):
    """Gamma=1: budget/rollout truncation bootstraps, true terminal does not."""
    return reward + torch.where(terminated, torch.zeros_like(continuation), continuation)


def double_dqn_continuation(online_q, target_q, mask):
    selected = online_q.masked_fill(~mask, -torch.inf).argmax(-1, keepdim=True)
    return target_q.gather(1, selected).squeeze(1)


def gae_advantages(batch, values, next_values):
    advantages = np.zeros(len(batch))
    running = 0.
    for i in range(len(batch)-1, -1, -1):
        terminal = batch[i]["terminated"]
        delta = batch[i]["reward"] + (0. if terminal else next_values[i])-values[i]
        continuation = not (terminal or batch[i]["truncated"])
        running = delta + .95*running*continuation
        advantages[i] = running
    return advantages


def _network(seed, output=1000):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return TransformerNetwork(output)


class Fair20Agent:
    def __init__(self, algorithm, candidate="A", seed=51001, device="cuda"):
        self.algorithm = algorithm.lower().replace("double_", "")
        if self.algorithm not in ("sac", "ppo", "dqn") or candidate not in ("A", "B"):
            raise ValueError("Unknown F20 arm/candidate")
        self.candidate, self.seed, self.device = candidate, int(seed), torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; no CPU fallback")
        self.exploration_rng = np.random.default_rng(seed+100)
        self.replay_rng = np.random.default_rng(seed+200)
        self.rollout_rng = np.random.default_rng(seed+300)
        self.policy = _network(seed).to(self.device)
        self.models = {"policy": self.policy}
        self.optimizers = {}
        lr = 1e-4 if candidate == "A" else 3e-4
        self.alpha = .01 if candidate == "A" else .05
        self.initialization_metadata = {"policy": {"seed": seed, "sha256": parameter_sha(self.policy)},
                                        "architecture": "preLN128_2L_4H_FFN512_GELU_dropout0", "gamma": 1.0}
        if self.algorithm == "sac":
            self.q1, self.q2 = _network(seed+1).to(self.device), _network(seed+2).to(self.device)
            self.target1, self.target2 = deepcopy(self.q1), deepcopy(self.q2)
            self.models.update(q1=self.q1, q2=self.q2, target1=self.target1, target2=self.target2)
            self.optimizers["policy"] = torch.optim.Adam(self.policy.parameters(), lr=1e-4)
            self.optimizers["q"] = torch.optim.Adam(list(self.q1.parameters())+list(self.q2.parameters()), lr=3e-4)
        elif self.algorithm == "ppo":
            self.value = _network(seed+3, 1).to(self.device)
            self.models["value"] = self.value
            self.optimizers["policy"] = torch.optim.Adam(self.policy.parameters(), lr=lr)
            self.optimizers["value"] = torch.optim.Adam(self.value.parameters(), lr=lr)
        else:
            self.target = deepcopy(self.policy)
            self.models["target"] = self.target
            self.optimizers["policy"] = torch.optim.Adam(self.policy.parameters(), lr=lr)
        for role, model in self.models.items():
            if role.startswith("target"):
                model.requires_grad_(False)
            elif role != "policy":
                self.initialization_metadata[role] = {"seed": seed+{"q1": 1, "q2": 2, "value": 3}[role],
                                                       "sha256": parameter_sha(model)}
        self.interactions = self.update_count = self.replay_cursor = 0
        self.replay, self.rollout = [], []
        self.last_update = {}

    def act(self, obs, mask, deterministic=False):
        mask = np.asarray(mask, dtype=bool)
        legal = np.flatnonzero(mask)
        if not len(legal):
            raise ValueError("Empty legal action mask")
        with torch.no_grad():
            output = self.policy(tensor_observations(obs, self.device))
            if deterministic:
                # Deployment uses precisely one matched action network for all
                # arms. PPO's value network belongs to training GAE only.
                action = int(legal[np.argmax(output[0, legal].cpu().numpy())])
                return action, {"epsilon": 0., "log_prob": 0., "value": 0.}
            if self.algorithm == "dqn":
                epsilon = max(.05, 1-.95*self.interactions/80000)
                action = int(legal[np.argmax(output[0, legal].cpu().numpy())])
                if not deterministic and self.exploration_rng.random() < epsilon:
                    action = int(self.exploration_rng.choice(legal))
                return action, {"epsilon": epsilon, "log_prob": 0., "value": 0.}
            probability, logp = masked_distribution(output, torch.as_tensor(mask[None], device=self.device))
            p = probability[0].cpu().numpy().astype(np.float64)
            if self.algorithm == "sac" and self.interactions < 4096:
                action = int(self.exploration_rng.choice(legal))
            else:
                action = int(self.exploration_rng.choice(1000, p=p/p.sum()))
            value = float(self.value(tensor_observations(obs, self.device))[0, 0]) if self.algorithm == "ppo" else 0.
            return action, {"log_prob": float(logp[0, action]), "value": value, "epsilon": 0.}

    def observe(self, obs, mask, action, reward, next_obs, next_mask, terminated, truncated=False, metadata=None):
        if terminated and truncated:
            raise ValueError("Transition cannot be both terminal and truncated")
        if not np.asarray(mask, bool)[action] or not np.isfinite(reward):
            raise ValueError("Invalid transition")
        transition = deepcopy(dict(obs=obs, mask=np.asarray(mask, bool), action=int(action), reward=float(reward),
                                   next_obs=next_obs, next_mask=np.asarray(next_mask, bool), terminated=bool(terminated),
                                   truncated=bool(truncated), metadata=metadata or {}))
        self.interactions += 1
        if self.algorithm == "ppo":
            if "log_prob" not in transition["metadata"] or "value" not in transition["metadata"]:
                raise ValueError("PPO requires stored sampling log_prob and value")
            self.rollout.append(transition)
            if len(self.rollout) == 2048:
                self.update_ppo()
        else:
            packed = pack_transition(transition)
            if len(self.replay) < 100000:
                self.replay.append(packed)
            else:
                self.replay[self.replay_cursor] = packed
            self.replay_cursor = (self.replay_cursor+1) % 100000
            if self.interactions >= 4096 and self.interactions % 4 == 0:
                indices = self.replay_rng.choice(len(self.replay), 256, replace=False)
                self.update_batch([unpack_transition(self.replay[int(i)]) for i in indices])
        return self.last_update

    def _optimize(self, name, loss, parameters):
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite F20 loss")
        optimizer = self.optimizers[name]
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
        optimizer.step()
        return float(norm)

    def update_batch(self, batch):
        """One genuine SAC/Double-DQN update; also used by bounded engineering checks."""
        if self.algorithm == "ppo":
            raise ValueError("Use update_ppo for stored PPO trajectories")
        obs = tensor_observations([x["obs"] for x in batch], self.device)
        nxt = tensor_observations([x["next_obs"] for x in batch], self.device)
        mask = torch.as_tensor(np.stack([x["mask"] for x in batch]), device=self.device)
        next_mask = torch.as_tensor(np.stack([x["next_mask"] for x in batch]), device=self.device)
        terminal = torch.as_tensor([x["terminated"] for x in batch], device=self.device, dtype=torch.bool)
        # Terminal masks may be empty; a dummy valid index only prevents undefined
        # normalization. Terminal continuation is explicitly zeroed below.
        safe_mask = next_mask.clone()
        safe_mask[terminal, 0] = True
        if not safe_mask.any(-1).all():
            raise ValueError("Nonterminal transition lacks successor legal set")
        actions = torch.as_tensor([x["action"] for x in batch], device=self.device)[:, None]
        reward = torch.as_tensor([x["reward"] for x in batch], device=self.device)
        with torch.no_grad():
            if self.algorithm == "sac":
                p, lp = masked_distribution(self.policy(nxt), safe_mask)
                q = torch.minimum(self.target1(nxt), self.target2(nxt))
                continuation = (p*(q-self.alpha*lp)).sum(-1)
            else:
                continuation = double_dqn_continuation(self.policy(nxt), self.target(nxt), safe_mask)
            target = bellman_target(reward, continuation, terminal)
        if self.algorithm == "sac":
            qloss = F.smooth_l1_loss(self.q1(obs).gather(1, actions).squeeze(1), target)+F.smooth_l1_loss(self.q2(obs).gather(1, actions).squeeze(1), target)
            self._optimize("q", qloss, list(self.q1.parameters())+list(self.q2.parameters()))
            with torch.no_grad():
                q = torch.minimum(self.q1(obs), self.q2(obs))
            p, lp = masked_distribution(self.policy(obs), mask)
            actor_loss = (p*(self.alpha*lp-q)).sum(-1).mean()
            self._optimize("policy", actor_loss, self.policy.parameters())
            with torch.no_grad():
                for source, target_net in ((self.q1, self.target1), (self.q2, self.target2)):
                    for source_param, target_param in zip(source.parameters(), target_net.parameters()):
                        target_param.lerp_(source_param, .005)
            self.last_update = {"critic_loss": float(qloss), "actor_loss": float(actor_loss)}
        else:
            loss = F.smooth_l1_loss(self.policy(obs).gather(1, actions).squeeze(1), target)
            self._optimize("policy", loss, self.policy.parameters())
            self.last_update = {"q_loss": float(loss)}
        self.update_count += 1
        if self.algorithm == "dqn" and self.update_count % 1000 == 0:
            self.target.load_state_dict(self.policy.state_dict())
        return self.last_update

    def update_ppo(self):
        if not self.rollout:
            return {}
        batch = self.rollout
        with torch.no_grad():
            nxt = np.concatenate([self.value(tensor_observations(
                [x["next_obs"] for x in batch[start:start+256]], self.device)).squeeze(1).cpu().numpy()
                for start in range(0, len(batch), 256)])
        values = np.asarray([x["metadata"]["value"] for x in batch])
        advantage = gae_advantages(batch, values, nxt)
        returns = advantage+values
        advantage = (advantage-advantage.mean())/(advantage.std()+1e-8)
        for _ in range(4):
            order = self.rollout_rng.permutation(len(batch))
            for start in range(0, len(batch), 256):
                ids = order[start:start+256]
                rows = [batch[int(i)] for i in ids]
                obs = tensor_observations([x["obs"] for x in rows], self.device)
                masks = torch.as_tensor(np.stack([x["mask"] for x in rows]), device=self.device)
                p, lp = masked_distribution(self.policy(obs), masks)
                a = torch.as_tensor([x["action"] for x in rows], device=self.device)[:, None]
                oldlp = torch.as_tensor([x["metadata"]["log_prob"] for x in rows], device=self.device)
                adv = torch.as_tensor(advantage[ids], device=self.device, dtype=torch.float32)
                ratio = (lp.gather(1, a).squeeze(1)-oldlp).exp()
                entropy = -(p*lp).sum(1).mean()
                actor_loss = -torch.minimum(ratio*adv, ratio.clamp(.8, 1.2)*adv).mean()-.01*entropy
                self._optimize("policy", actor_loss, self.policy.parameters())
                targets = torch.as_tensor(returns[ids], device=self.device, dtype=torch.float32)
                value_loss = .5*F.smooth_l1_loss(self.value(obs).squeeze(1), targets)
                self._optimize("value", value_loss, self.value.parameters())
                self.update_count += 1
                self.last_update = {"actor_loss": float(actor_loss), "value_loss": float(value_loss), "entropy": float(entropy)}
        self.rollout = []
        return self.last_update

    def flush(self):
        return self.update_ppo() if self.algorithm == "ppo" else self.last_update

    def state_dict(self):
        """Capture for immediate synchronous serialization with training paused.

        Model/optimizer tensors are views, as with PyTorch state_dict. Replay
        bytes are immutable; shallow list snapshots avoid doubling host memory.
        Do not enqueue this mapping for serialization concurrent with updates.
        """
        return {"schema": "f20_learning_v2", "replay_format": REPLAY_FORMAT,
                         "algorithm": self.algorithm, "candidate": self.candidate,
                         "seed": self.seed, "models": {k: v.state_dict() for k, v in self.models.items()},
                         "optimizers": {k: v.state_dict() for k, v in self.optimizers.items()},
                         "interactions": self.interactions, "update_count": self.update_count,
                         "replay_cursor": self.replay_cursor, "replay": list(self.replay), "rollout": list(self.rollout),
                         "rng": {k: deepcopy(getattr(self, k).bit_generator.state) for k in
                                 ("exploration_rng", "replay_rng", "rollout_rng")},
                         "initialization_metadata": deepcopy(self.initialization_metadata), "last_update": deepcopy(self.last_update)}

    def load_state_dict(self, state):
        for key in ("algorithm", "candidate", "seed"):
            if state[key] != getattr(self, key):
                raise ValueError(f"Checkpoint {key} mismatch")
        if (state["schema"] != "f20_learning_v2" or state.get("replay_format") != REPLAY_FORMAT
                or set(state["models"]) != set(self.models)):
            raise ValueError("Checkpoint scientific schema mismatch")
        if (len(state["replay"]) > 100000 or any(not isinstance(x, bytes) for x in state["replay"])
                or not 0 <= state["replay_cursor"] < 100000 or len(state["rollout"]) >= 2048):
            raise ValueError("Checkpoint replay/rollout contract mismatch")
        for key, model in self.models.items():
            model.load_state_dict(state["models"][key], strict=True)
        for key, optimizer in self.optimizers.items():
            optimizer.load_state_dict(deepcopy(state["optimizers"][key]))
        for key in ("interactions", "update_count", "replay_cursor", "rollout", "initialization_metadata", "last_update"):
            setattr(self, key, deepcopy(state[key]))
        self.replay = list(state["replay"])
        for key, value in state["rng"].items():
            getattr(self, key).bit_generator.state = deepcopy(value)


def make_agent(algorithm, candidate="A", seed=51001, device="cuda"):
    return Fair20Agent(algorithm, candidate, seed, device)
