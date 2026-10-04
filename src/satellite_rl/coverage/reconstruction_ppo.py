"""Relation graph PPO for first-stage region reconstruction.

The actor chooses (operation, i, j, k, template) with five conditional
distributions. Candidate identities only define cheap prefix masks; no
candidate is reconstructed or scored by its outcome before selection.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical


OPERATIONS = ("stop", "merge_split", "repartition2", "repartition3", "assign")
BEAM_DIM = 16
RELATION_DIM = 5
GLOBAL_DIM = 12
HEADS = 5


def action_identity(action):
    """Stable identity, including the unused head sentinels of each operation."""
    if action.operation not in OPERATIONS:
        raise ValueError("Unknown reconstruction operation")
    beams = tuple(int(x) + 1 for x in action.beams)
    if len(beams) > 3:
        raise ValueError("At most three beams per macro action")
    return (OPERATIONS.index(action.operation), *(beams + (0,) * (3 - len(beams))),
            int(action.template))


def identity_table(candidates):
    table = np.asarray([action_identity(action) for action in candidates], dtype="<i2")
    if table.ndim != 2 or table.shape[1] != HEADS:
        raise ValueError("Empty or malformed reconstruction candidate menu")
    if np.any(np.all(table[1:] == table[:-1], axis=1)):
        raise ValueError("Duplicate full macro-action identity")
    return table


def identity_digest(table):
    return hashlib.sha256(np.asarray(table, dtype="<i2").tobytes()).hexdigest()


def prefix_mask(candidates, prefix, *, width=101):
    """Choices at one conditional head, conditioned on an exact prefix."""
    head = len(prefix)
    if head >= HEADS:
        raise ValueError("A complete action has no next head")
    table = candidates if isinstance(candidates, np.ndarray) else identity_table(candidates)
    matching = np.all(table[:, :head] == np.asarray(prefix, dtype=np.int16), axis=1)
    choices = table[matching, head]
    if np.any((choices < 0) | (choices >= width)):
        raise ValueError("Candidate identity exceeds frozen action width")
    mask = np.zeros(width, dtype=np.bool_)
    mask[choices] = True
    if not mask.any():
        raise ValueError("Candidate prefix has no continuation")
    return mask


def selected_action(candidates, identity):
    matches = [action for action in candidates if action_identity(action) == tuple(identity)]
    if len(matches) != 1:
        raise ValueError("Selected candidate identity is absent or ambiguous")
    return matches[0]


def graph_observation(state, config):
    """Current-only beam/user aggregates and beam relations, in beam ID order.

    All user contributions are aggregated; exclusive and boundary users are
    retained explicitly. The final geometric validity check remains in the
    reconstruction core, using the original individual users.
    """
    from .geometry import satellite_rays_numpy, squared_ray_limit

    centers = np.asarray(state.centers, dtype=np.float64)
    positions = np.asarray(state.scene.positions, dtype=np.float64)
    demand = np.asarray(state.scene.demands, dtype=np.float64)
    assigned = np.asarray(state.assignment, dtype=np.int64)
    n = len(centers)
    if n != 100 or len(assigned) != len(demand):
        raise ValueError("Reconstruction graph expects 100 beams and all users")
    rays = satellite_rays_numpy(centers)
    user_rays = satellite_rays_numpy(positions)
    distance = np.maximum(0.0, 2.0 - 2.0 * (user_rays @ rays.T))
    limit = float(squared_ray_limit(0.25))
    reachable = distance <= limit + 1e-12
    counts = reachable.sum(axis=1)
    membership = assigned[:, None] == np.arange(n)[None, :]
    positive = demand > 0
    valid = membership & positive[:, None]
    loads = np.bincount(assigned[(assigned >= 0) & positive],
                        weights=demand[(assigned >= 0) & positive], minlength=n)[:n]
    mean = float(np.mean(loads))
    scale = max(mean, 1e-9)
    total = max(float(demand.sum()), 1e-9)
    owner_count = valid.sum(axis=0)
    exclusive = (valid & (counts == 1)[:, None])
    overlap = (valid & (counts > 1)[:, None])
    boundary = (valid & (distance >= 0.75 * limit))
    candidate_demand = reachable.T @ demand
    owned_position = np.zeros((n, 2), dtype=np.float64)
    np.add.at(owned_position, assigned[(assigned >= 0) & positive],
              positions[(assigned >= 0) & positive] * demand[(assigned >= 0) & positive, None])
    owned_position /= np.maximum(loads[:, None], 1e-9)
    margin = np.where(valid, (limit - distance) / max(limit, 1e-12), 0.0)
    margin_sum = margin.sum(axis=0) / np.maximum(owner_count, 1)
    beam = np.stack((
        (centers[:, 0] - 50.0) / 40.0, centers[:, 1] / 40.0,
        rays[:, 0], rays[:, 1], rays[:, 2],
        loads / scale, np.log1p(loads / scale), owner_count / max(len(demand), 1),
        (exclusive * demand[:, None]).sum(axis=0) / scale,
        (overlap * demand[:, None]).sum(axis=0) / scale,
        (boundary * demand[:, None]).sum(axis=0) / scale,
        candidate_demand / scale,
        margin_sum,
        (owned_position[:, 0] - centers[:, 0]) / 10.0,
        (owned_position[:, 1] - centers[:, 1]) / 10.0,
        np.asarray([beam_id in state.edge_beam_ids for beam_id in range(n)],
                   dtype=np.float64),
    ), axis=1).astype(np.float32)
    chord = np.maximum(0.0, 2.0 - 2.0 * (rays @ rays.T))
    shared = (reachable.astype(np.float32) * demand[:, None].astype(np.float32)).T @ reachable.astype(np.float32)
    shared /= scale
    center_delta = centers[:, None, :] - centers[None, :, :]
    relation = np.stack((
        np.sqrt(chord / max(limit, 1e-12)),
        shared,
        center_delta[..., 0] / 10.0,
        center_delta[..., 1] / 10.0,
        (chord <= 4.0 * limit).astype(np.float64),
    ), axis=-1).astype(np.float32)
    covered = (assigned >= 0) & positive
    coverage_users = float(np.mean(assigned >= 0))
    coverage_demand = float(demand[covered].sum() / total)
    max_steps = max(int(config.max_macro_steps), 1)
    max_calls = max(int(config.max_geometry_calls_total), 1)
    global_features = np.asarray((
        coverage_users, coverage_demand, float(len(state.edge_regions)) / 10.0,
        float(loads.std() / scale), float(loads.max() / scale),
        float((loads > 0).sum()) / n, float(np.mean(np.abs(loads / scale - 1.0) <= .2)),
        float(state.steps) / max_steps, (max_steps - float(state.steps)) / max_steps,
        float(state.geometry_calls) / max_calls,
        (max_calls - float(state.geometry_calls)) / max_calls,
        math.log1p(total) / 32.0,
    ), dtype=np.float32)
    return {"beam": beam, "relation": relation, "global": global_features}


def stack_observations(observations, device):
    return {key: torch.as_tensor(np.stack([item[key] for item in observations]),
                                  device=device, dtype=torch.float32)
            for key in ("beam", "relation", "global")}


class RelationPolicy(nn.Module):
    """Two 128-wide relation message passes with autoregressive action heads."""

    def __init__(self, hidden_size=128, message_layers=2, beam_count=100, templates=4):
        super().__init__()
        if hidden_size != 128 or message_layers != 2 or beam_count != 100 or templates != 4:
            raise ValueError("Frozen S1-03 network is 2x128, 100 beams, four templates")
        self.hidden_size = hidden_size
        self.message_layers = message_layers
        self.beam_count = beam_count
        self.action_width = beam_count + 1
        self.node_in = nn.Sequential(nn.Linear(BEAM_DIM, hidden_size), nn.Tanh())
        self.edge_gates = nn.ModuleList(nn.Linear(RELATION_DIM, 1) for _ in range(2))
        self.self_updates = nn.ModuleList(nn.Linear(hidden_size, hidden_size) for _ in range(2))
        self.neighbor_updates = nn.ModuleList(nn.Linear(hidden_size, hidden_size) for _ in range(2))
        self.global_in = nn.Sequential(nn.Linear(GLOBAL_DIM + hidden_size, hidden_size), nn.Tanh())
        self.operation_head = nn.Linear(hidden_size, len(OPERATIONS))
        self.operation_embedding = nn.Embedding(len(OPERATIONS), hidden_size)
        self.queries = nn.ModuleList(nn.Linear(hidden_size, hidden_size) for _ in range(3))
        self.prefix = nn.ModuleList(nn.Linear(hidden_size, hidden_size) for _ in range(3))
        self.selected_beam = nn.ModuleList(nn.Linear(hidden_size, hidden_size)
                                           for _ in range(3))
        self.template_head = nn.Linear(hidden_size, templates)
        self.value_head = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.Tanh(),
                                        nn.Linear(hidden_size, 1))
        self._identity_cache = {}

    def encode(self, observation):
        nodes = self.node_in(observation["beam"])
        edges = observation["relation"]
        for gate, own, neighbor in zip(self.edge_gates, self.self_updates,
                                       self.neighbor_updates):
            weights = torch.softmax(gate(edges).squeeze(-1), dim=-1)
            messages = torch.bmm(weights, nodes)
            nodes = torch.tanh(own(nodes) + neighbor(messages))
        context = self.global_in(torch.cat((nodes.mean(dim=1), observation["global"]), dim=-1))
        return nodes, context

    def value(self, observation):
        _, context = self.encode(observation)
        return self.value_head(context).squeeze(-1)

    def _head_logits(self, nodes, context, prefix, head):
        batch = nodes.shape[0]
        if head == 0:
            raw = self.operation_head(context)
        elif head < 4:
            selected = self.operation_embedding(prefix[:, 0])
            for earlier in range(1, head):
                index = prefix[:, earlier]
                valid = index > 0
                picked = nodes[torch.arange(batch, device=nodes.device),
                               (index - 1).clamp_min(0)]
                selected = selected + self.selected_beam[earlier - 1](picked) * valid[:, None]
            query = torch.tanh(self.queries[head - 1](context) +
                               self.prefix[head - 1](selected))
            beam_logits = torch.einsum("bh,bnh->bn", query, nodes) / math.sqrt(self.hidden_size)
            raw = torch.cat((torch.zeros((batch, 1), device=nodes.device), beam_logits), dim=1)
        else:
            selected = self.operation_embedding(prefix[:, 0])
            for earlier in range(1, 4):
                index = prefix[:, earlier]
                valid = index > 0
                picked = nodes[torch.arange(batch, device=nodes.device),
                               (index - 1).clamp_min(0)]
                selected = selected + self.selected_beam[earlier - 1](picked) * valid[:, None]
            raw = self.template_head(torch.tanh(context + selected))
        return raw

    def evaluate_choices(self, observation, choices, masks):
        """Joint log probability against the stored prefix masks."""
        nodes, context = self.encode(observation)
        logp = torch.zeros(len(choices), device=context.device)
        entropy = torch.zeros_like(logp)
        for head in range(HEADS):
            logits = self._head_logits(nodes, context, choices, head)
            mask = masks[:, head, :logits.shape[-1]]
            if not torch.all(mask.any(dim=-1)):
                raise ValueError("Stored prefix mask has empty head")
            distribution = Categorical(logits=logits.masked_fill(~mask, -1e9))
            logp = logp + distribution.log_prob(choices[:, head])
            entropy = entropy + distribution.entropy()
        return logp, entropy, self.value_head(context).squeeze(-1)

    @torch.no_grad()
    def act(self, observation, candidate_lists, *, deterministic=False):
        nodes, context = self.encode(observation)
        batch = len(candidate_lists)
        choices = torch.zeros((batch, HEADS), dtype=torch.long, device=context.device)
        masks = torch.zeros((batch, HEADS, self.action_width), dtype=torch.bool,
                            device=context.device)
        logp = torch.zeros(batch, device=context.device)
        previous = self._identity_cache
        current = {}
        tables = []
        digests = []
        for candidates in candidate_lists:
            key = id(candidates)
            entry = previous.get(key)
            if entry is None or entry[0] is not candidates:
                table = identity_table(candidates)
                entry = (candidates, table, identity_digest(table))
            current[key] = entry
            tables.append(entry[1])
            digests.append(entry[2])
        self._identity_cache = current
        prefixes = np.zeros((batch, HEADS), dtype=np.int16)
        for head in range(HEADS):
            head_masks = np.stack([prefix_mask(tables[row], prefixes[row, :head],
                                               width=self.action_width)
                                   for row in range(batch)])
            masks[:, head] = torch.as_tensor(head_masks, device=context.device)
            logits = self._head_logits(nodes, context, choices, head)
            distribution = Categorical(logits=logits.masked_fill(
                ~masks[:, head, :logits.shape[-1]], -1e9))
            chosen = distribution.logits.argmax(-1) if deterministic else distribution.sample()
            choices[:, head] = chosen
            prefixes[:, head] = chosen.detach().cpu().numpy()
            logp += distribution.log_prob(chosen)
        actions = []
        for row, candidates in enumerate(candidate_lists):
            matches = np.flatnonzero(np.all(tables[row] ==
                                            prefixes[row], axis=1))
            if len(matches) != 1:
                raise ValueError("Selected candidate identity is absent or ambiguous")
            actions.append(candidates[int(matches[0])])
        return actions, choices, masks, logp, self.value_head(context).squeeze(-1), digests


@dataclass
class GraphRollout:
    observation: dict
    choices: torch.Tensor
    masks: torch.Tensor
    old_logp: torch.Tensor
    old_values: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    bootstrap_value: torch.Tensor

    def advantages(self, gamma=1.0, gae_lambda=.95):
        if gamma != 1.0:
            raise ValueError("Frozen finite-horizon reconstruction task uses gamma=1")
        adv = torch.zeros_like(self.bootstrap_value)
        next_value = self.bootstrap_value
        result = torch.empty_like(self.rewards)
        for step in range(len(self.rewards) - 1, -1, -1):
            continuation = (~self.dones[step]).to(self.rewards.dtype)
            delta = self.rewards[step] + next_value * continuation - self.old_values[step]
            adv = delta + gae_lambda * continuation * adv
            result[step] = adv
            next_value = self.old_values[step]
        return result, result + self.old_values


def ppo_update(policy, optimizer, rollout, *, gamma=1.0, gae_lambda=.95,
               clip_ratio=.2, entropy_coef=.01, value_coef=.5,
               max_grad_norm=.5, epochs=4, minibatch_size=256):
    advantages, returns = rollout.advantages(gamma, gae_lambda)
    time_count, env_count = rollout.rewards.shape
    total = time_count * env_count
    flat_obs = {key: value.reshape(total, *value.shape[2:])
                for key, value in rollout.observation.items()}
    choices = rollout.choices.reshape(total, HEADS)
    masks = rollout.masks.reshape(total, HEADS, policy.action_width)
    old_logp = rollout.old_logp.reshape(total)
    returns = returns.reshape(total)
    advantages = advantages.reshape(total)
    advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-8)
    sums = {key: 0.0 for key in ("policy_loss", "value_loss", "entropy", "approx_kl",
                                     "clip_fraction", "grad_norm")}
    updates = 0
    for _ in range(epochs):
        order = torch.randperm(total, device=choices.device)
        for start in range(0, total, minibatch_size):
            index = order[start:start + minibatch_size]
            observation = {key: value[index] for key, value in flat_obs.items()}
            logp, entropy, value = policy.evaluate_choices(
                observation, choices[index], masks[index])
            ratio = torch.exp(logp - old_logp[index])
            unclipped = ratio * advantages[index]
            clipped = ratio.clamp(1.0 - clip_ratio, 1.0 + clip_ratio) * advantages[index]
            policy_loss = -torch.minimum(unclipped, clipped).mean()
            value_loss = .5 * (value - returns[index]).square().mean()
            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy.mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm,
                                                 error_if_nonfinite=True)
            optimizer.step()
            with torch.no_grad():
                sums["policy_loss"] += float(policy_loss)
                sums["value_loss"] += float(value_loss)
                sums["entropy"] += float(entropy.mean())
                sums["approx_kl"] += float(((ratio - 1) - (logp - old_logp[index])).mean())
                sums["clip_fraction"] += float(((ratio - 1).abs() > clip_ratio).float().mean())
                sums["grad_norm"] += float(grad_norm)
            updates += 1
    return {key: value / updates for key, value in sums.items()} | {"updates": updates}
