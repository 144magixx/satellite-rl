"""Five supervised objectives; no environment or training-module dependencies.

Set encoders deliberately have no position/identity embeddings. Padding and
illegal edges are masked in both attention and objectives. Teacher outcomes
are never part of the feature tensors.
"""
from __future__ import annotations

import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

MODEL_FORMAT = "s1_nondrl_supervised_v1"
ROUTES = ("01", "02", "03", "04", "05")


def canonical_route(route):
    value = str(route).removeprefix("route_").removeprefix("R").zfill(2)
    if value not in ROUTES:
        raise ValueError(f"unknown supervised route: {route}")
    return value


class SetEncoder(nn.Module):
    def __init__(self, inputs, hidden=128, layers=3, heads=4):
        super().__init__()
        self.project = nn.Linear(inputs, hidden)
        block = nn.TransformerEncoderLayer(hidden, heads, hidden * 2,
                                           dropout=0., batch_first=True,
                                           activation="gelu", norm_first=True)
        self.encoder = nn.TransformerEncoder(block, layers, enable_nested_tensor=False)

    def forward(self, values, mask):
        return self.encoder(self.project(values), src_key_padding_mask=~mask)


def pooled(values, mask):
    return (values * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)


class MaskedAssignmentModel(nn.Module):
    """01: masked aircraft-to-candidate classification, visible context only."""
    def __init__(self, hidden=128, layers=4, heads=4):
        super().__init__()
        self.aircraft = SetEncoder(8, hidden, layers, heads)
        self.candidates = SetEncoder(8, hidden, layers, heads)
        self.visible_project = nn.Linear(hidden, hidden, bias=False)
        self.cross = nn.MultiheadAttention(hidden, heads, dropout=0., batch_first=True)
        self.hidden_token = nn.Parameter(torch.zeros(hidden))
        self.scale = math.sqrt(hidden)

    def forward(self, batch):
        aircraft = self.aircraft(batch["aircraft"], batch["aircraft_mask"])
        candidate = self.candidates(batch["candidates"], batch["candidate_mask"])
        visible = batch["visible"]
        context = candidate.gather(1, visible.clamp_min(0).unsqueeze(-1).expand(-1, -1, candidate.shape[-1]))
        context = torch.where((visible >= 0).unsqueeze(-1), context, self.hidden_token)
        aircraft = aircraft + self.visible_project(context)
        cross, _ = self.cross(aircraft, candidate, candidate,
                              key_padding_mask=~batch["candidate_mask"], need_weights=False)
        logits = torch.bmm(aircraft + cross, candidate.transpose(1, 2)) / self.scale
        return {"logits": logits.masked_fill(~batch["legal"], -1e9)}


class ActionBCModel(nn.Module):
    """02: set-probability imitation of next executable actions and STOP."""
    def __init__(self, hidden=128, layers=3, heads=4):
        super().__init__()
        self.aircraft = SetEncoder(8, hidden, layers, heads)
        self.beams = SetEncoder(8, hidden, layers, heads)
        self.actions = SetEncoder(16, hidden, layers, heads)
        self.score = nn.Sequential(nn.Linear(hidden * 3, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.stop = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, batch):
        air = pooled(self.aircraft(batch["aircraft"], batch["aircraft_mask"]), batch["aircraft_mask"])
        beams = pooled(self.beams(batch["beams"], batch["beam_mask"]), batch["beam_mask"])
        context = torch.cat((air, beams), -1)
        actions = self.actions(batch["actions"], batch["action_mask"])
        scores = self.score(torch.cat((actions, context[:, None].expand(-1, actions.shape[1], -1)), -1)).squeeze(-1)
        scores = scores.masked_fill(~batch["action_mask"], -1e9)
        return {"logits": torch.cat((scores, self.stop(context)), 1)}


class ConstructModel(nn.Module):
    """03: aircraft-only set to N continuous centers and complete assignment."""
    def __init__(self, hidden=128, layers=4, heads=4):
        super().__init__()
        self.aircraft = SetEncoder(8, hidden, layers, heads)
        self.queries = nn.Parameter(torch.randn(200, hidden) * .02)
        self.cross = nn.MultiheadAttention(hidden, heads, dropout=0., batch_first=True)
        self.assignment_query = nn.Linear(hidden, hidden)
        self.center_residual = nn.Linear(hidden, 2)
        nn.init.zeros_(self.center_residual.weight)
        nn.init.zeros_(self.center_residual.bias)

    def forward(self, batch):
        aircraft = self.aircraft(batch["aircraft"], batch["aircraft_mask"])
        count = batch["slot_mask"].shape[1]
        query = self.queries[:count][None].expand(aircraft.shape[0], -1, -1)
        slots, attention = self.cross(query, aircraft, aircraft,
                                     key_padding_mask=~batch["aircraft_mask"])
        slots = slots + query
        logits = torch.bmm(aircraft, self.assignment_query(slots).transpose(1, 2)) / math.sqrt(slots.shape[-1])
        logits = logits.masked_fill(~batch["slot_mask"][:, None], -1e9)
        # Centers start as learned scene-only mixtures; no I0/teacher centers.
        centers = torch.bmm(attention, batch["positions"]) + self.center_residual(slots).tanh()
        bounds = centers.new_tensor([[25., -40.], [80., 50.]])
        centers = torch.maximum(torch.minimum(centers, bounds[1]), bounds[0])
        rays = center_rays(centers)
        return {"logits": logits, "centers": centers, "rays": rays}


def center_rays(centers):
    """Frozen europe_geo13_v1 geometry, independent of environments."""
    lat, lon = torch.deg2rad(centers[..., 0]), torch.deg2rad(centers[..., 1])
    surface = 6371. * torch.stack((lat.cos() * lon.cos(), lat.cos() * lon.sin(), lat.sin()), -1)
    slon = math.radians(13.)
    satellite = centers.new_tensor([42164. * math.cos(slon), 42164. * math.sin(slon), 0.])
    return F.normalize(surface - satellite, dim=-1)


class CandidateRankModel(nn.Module):
    """04: true-trial pairwise action ranking, with no STOP probability."""
    def __init__(self, hidden=128, layers=3, heads=4):
        super().__init__()
        self.beams = SetEncoder(8, hidden, layers, heads)
        self.actions = SetEncoder(16, hidden, layers, heads)
        self.score = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, batch):
        beam = self.beams(batch["beams"], batch["beam_mask"])
        context = pooled(beam, batch["beam_mask"])
        action = self.actions(batch["actions"], batch["action_mask"])
        return {"scores": self.score(torch.cat((action, context[:, None].expand(-1, action.shape[1], -1)), -1)).squeeze(-1)}


class RegionChoiceModel(nn.Module):
    """05: supervised region utility contrast, shared rule reconstruction."""
    def __init__(self, hidden=128, layers=3, heads=4):
        super().__init__()
        self.beams = SetEncoder(8, hidden, layers, heads)
        self.regions = SetEncoder(16, hidden, layers, heads)
        self.cross = nn.MultiheadAttention(hidden, heads, dropout=0., batch_first=True)
        self.score = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, batch):
        beam = self.beams(batch["beams"], batch["beam_mask"])
        region = self.regions(batch["actions"], batch["action_mask"])
        context, _ = self.cross(region, beam, beam, key_padding_mask=~batch["beam_mask"], need_weights=False)
        return {"scores": self.score(torch.cat((region, context), -1)).squeeze(-1)}


def make_model(route, network=None):
    route = canonical_route(route)
    network = network or {}
    options = {"hidden": int(network.get("hidden", 128)),
               "layers": int(network.get("layers", 4 if route in ("01", "03") else 3)),
               "heads": int(network.get("heads", 4))}
    if options["hidden"] % options["heads"] or min(options.values()) < 1:
        raise ValueError("invalid network dimensions")
    return dict(zip(ROUTES, (MaskedAssignmentModel, ActionBCModel, ConstructModel,
                            CandidateRankModel, RegionChoiceModel)))[route](**options)


def linear_assignment(cost):
    """Exact square Hungarian matching in O(N^3), including N=200."""
    cost = np.asarray(cost, dtype=np.float64)
    if cost.ndim != 2 or cost.shape[0] != cost.shape[1] or not np.isfinite(cost).all():
        raise ValueError("matching requires finite square costs")
    n = len(cost)
    u, v = np.zeros(n + 1), np.zeros(n + 1)
    p, way = np.zeros(n + 1, dtype=np.int64), np.zeros(n + 1, dtype=np.int64)
    for i in range(1, n + 1):
        p[0], j0 = i, 0
        minimum, used = np.full(n + 1, np.inf), np.zeros(n + 1, dtype=bool)
        while True:
            used[j0] = True
            i0, delta, j1 = p[j0], np.inf, 0
            for j in range(1, n + 1):
                if not used[j]:
                    current = cost[i0 - 1, j - 1] - u[i0] - v[j]
                    if current < minimum[j]:
                        minimum[j], way[j] = current, j0
                    if minimum[j] < delta:
                        delta, j1 = minimum[j], j
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minimum[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0], j0 = p[j1], j1
            if j0 == 0:
                break
    result = np.empty(n, dtype=np.int64)
    for j in range(1, n + 1):
        result[p[j] - 1] = j - 1
    return result


def construction_loss(output, samples):
    losses = []
    for index, sample in enumerate(samples):
        n, a = sample["target_n"], len(sample["aircraft"])
        logits = output["logits"][index, :a, :n]
        logp = logits.log_softmax(-1)
        alternatives = []
        for reference in sample["references"]:
            labels = torch.as_tensor(reference["assignment"], device=logits.device, dtype=torch.long)
            centers = torch.as_tensor(reference["centers"], device=logits.device, dtype=logits.dtype)
            rays = center_rays(centers)
            chord = torch.linalg.vector_norm(output["rays"][index, :n, None] - rays[None], dim=-1)
            distance = (2 * torch.asin((chord / 2).clamp(max=1 - 1e-7))).square() / math.radians(.5) ** 2
            # Each column is a whole teacher slot; CE cost uses ALL aircraft denominator.
            membership = F.one_hot(labels, n).to(logits.dtype)
            cost = -logp.transpose(0, 1) @ membership / a + distance / n
            row_to_teacher = linear_assignment(cost.detach().cpu().numpy())
            inverse = np.argsort(row_to_teacher)
            mapped = torch.as_tensor(inverse, device=logits.device)[labels]
            ce = F.cross_entropy(logits, mapped)
            centers_cost = distance[torch.arange(n, device=logits.device), torch.as_tensor(row_to_teacher, device=logits.device)].mean()
            alternatives.append(ce + centers_cost)
        losses.append(torch.stack(alternatives).min())
    return torch.stack(losses).mean()


def supervised_loss(route, output, batch, samples):
    route = canonical_route(route)
    if route == "03":
        return construction_loss(output, samples)
    if route == "01":
        mask = batch["hidden"] & batch["aircraft_mask"]
        if not mask.any():
            raise ValueError("MLM has no hidden labels")
        # Equal states (and mother-balanced sampling), irrespective of region size.
        return torch.stack([F.cross_entropy(output["logits"][i, mask[i]], batch["labels"][i, mask[i]])
                            for i in range(len(samples))]).mean()
    if route == "02":
        logp = output["logits"].log_softmax(-1)
        positives = batch["positives"]
        if not positives.any(1).all():
            raise ValueError("BC requires at least one real demonstration")
        return -torch.logsumexp(logp.masked_fill(~positives, -torch.inf), 1).mean()
    scores = output["scores"]
    known, utility = batch["known"] & batch["action_mask"], batch["utility"]
    difference = utility[:, :, None] - utility[:, None, :]
    pair = known[:, :, None] & known[:, None, :] & (difference > 1e-8)
    delta = scores[:, :, None] - scores[:, None, :]
    if route == "04":
        if not pair.any():
            raise ValueError("ranking dataset has no known ordered pair")
        weights = difference.clamp(.05, 1.)
        return torch.stack([(F.softplus(-delta[i])[pair[i]] * weights[i][pair[i]]).mean()
                            for i in range(len(samples))]).mean()
    # Region datasets retain all-negative states: binary contrast anchors them.
    positive = utility > 0
    if not known.any():
        raise ValueError("region sample has no completed trial")
    terms = []
    for i in range(len(samples)):
        binary = F.binary_cross_entropy_with_logits(scores[i, known[i]], positive[i, known[i]].to(scores.dtype))
        terms.append(binary + (F.softplus(-delta[i])[pair[i]].mean() if pair[i].any() else scores[i].sum() * 0))
    return torch.stack(terms).mean()
