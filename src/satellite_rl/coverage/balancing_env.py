"""Batched CUDA environment for coverage-first center adjustment.

The long C1/repair/A2 preparation is a separately archived workbench task.
This environment only consumes frozen A2 starts and updates one/two cached
distance columns per action; no CPU scene loop occurs in the training step.
"""

from __future__ import annotations

import copy
import math

import numpy as np
import torch

from satellite_rl.coverage.balancing import (ACTION_COUNTS, BalanceStart,
                                            balance_options)
from satellite_rl.coverage.environment import CoverageBatchEnv
from satellite_rl.coverage.geometry import (ray_angle_torch, satellite_rays_torch,
                                            squared_ray_limit, visible_torch)
from satellite_rl.coverage.scenes import BEAM_COUNT, CoverageDataset, CoverageScene


STOP_RUNNING = 0
STOP_ALREADY_TARGET = 1
STOP_TARGET = 2
STOP_ROUND_LIMIT = 3
STOP_BEAM_LIMIT = 4
STOP_ASSIGNMENT_LIMIT = 5
STOP_TIME_LIMIT = 6
STOP_CANDIDATE_LIMIT = 7


class BalanceBatchEnv(CoverageBatchEnv):
    """Same sampling interface as CoverageBatchEnv, with per-row episode ends."""

    candidate_dim = 8
    obs_dim = 1120

    def __init__(self, dataset: CoverageDataset, config: dict, starts: list,
                 num_envs: int = 128, device: str = "cuda", seed: int = 51001,
                 auto_reset: bool = True,
                 eligible_indices: list[int] | None = None):
        if len(starts) != len(dataset):
            raise ValueError("one frozen A2 start required per scene")
        if eligible_indices is None:
            eligible_indices = list(range(len(dataset)))
        eligible_indices = [int(index) for index in eligible_indices]
        if (not eligible_indices or len(eligible_indices) != len(set(eligible_indices)) or
                any(index < 0 or index >= len(dataset) for index in eligible_indices)):
            raise ValueError("explicit eligible A2 scene indices must be unique, in range and nonempty")
        self.balance = balance_options(config)
        self.menu_name = self.balance["action_menu"]
        prepared = []
        for original_index in eligible_indices:
            scene, start = dataset.scenes[original_index], starts[original_index]
            if isinstance(start, BalanceStart):
                checked = BalanceStart.from_record(scene, start.to_record())
            elif isinstance(start, dict) and start.get("status") != "initialization_failed":
                checked = BalanceStart.from_record(scene, start)
            else:
                raise ValueError("failed/unfrozen A2 start cannot enter PPO environment")
            if checked.status != "completed" or not checked.metrics.get("coverage_99_9", False):
                raise ValueError("PPO requires a frozen A2 start meeting both 99.9% coverage floors")
            prepared.append(checked)
        self._original_dataset = dataset
        self._eligible_indices = eligible_indices
        self._original_scene_ids = [scene.scene_id for scene in dataset.scenes]
        self._prepared_starts = prepared
        self._original_center_array = np.stack([np.asarray(scene.centers, dtype=np.float64)
                                                for scene in
                                                (dataset.scenes[index] for index in eligible_indices)])
        from dataclasses import replace
        ready_scenes = [replace(dataset.scenes[index], centers=start.centers.copy())
                        for index, start in zip(eligible_indices, prepared)]
        ready_dataset = CoverageDataset(ready_scenes, dict(dataset.metadata))
        self._balance_ready = False
        super().__init__(ready_dataset, config, num_envs=num_envs, device=device,
                         seed=seed, auto_reset=auto_reset)
        self._original_index_map = torch.as_tensor(eligible_indices,
                                                   device=self.device, dtype=torch.long)
        self.action_dim = ACTION_COUNTS[self.menu_name]
        self.horizon = min(BEAM_COUNT * self.balance["max_rounds"],
                           self.balance["max_beam_decisions"])
        scene_count = len(eligible_indices)
        max_users = self.max_users
        start_assoc = torch.full((scene_count, max_users), -1, device=self.device,
                                 dtype=torch.long)
        region_count = max((len(item.regions) for item in prepared), default=0)
        region_mask = torch.zeros((scene_count, region_count, max_users),
                                  device=self.device, dtype=torch.bool)
        region_first = torch.zeros((scene_count, region_count), device=self.device,
                                   dtype=torch.long)
        region_sizes = torch.zeros((scene_count, region_count), device=self.device,
                                   dtype=torch.long)
        region_user_lock = torch.zeros((scene_count, max_users), device=self.device,
                                        dtype=torch.bool)
        for index, item in enumerate(prepared):
            start_assoc[index, :len(item.associations)] = torch.as_tensor(
                item.associations, device=self.device, dtype=torch.long)
            for region_index, region in enumerate(item.regions):
                members = torch.as_tensor(region["aircraft_indices"], device=self.device,
                                          dtype=torch.long)
                region_mask[index, region_index, members] = True
                region_user_lock[index, members] = True
                region_first[index, region_index] = members[0]
                region_sizes[index, region_index] = len(members)
        self._start_assoc_pool = start_assoc
        self._assignment_prior_pool = torch.as_tensor(
            [int(item.stages.get("assignment", {}).get("counts", {}).get("attempts", 0))
             for item in prepared], device=self.device, dtype=torch.long)
        self._movement_reference_pool = torch.as_tensor(self._original_center_array,
                                                        device=self.device, dtype=torch.float64)
        self._region_mask_pool = region_mask
        self._region_first_pool = region_first
        self._region_size_pool = region_sizes
        self._region_user_lock_pool = region_user_lock
        self._protected_pool = start_assoc >= 0
        self._user_cell_pool = self._grid_cells_torch(self._positions.reshape(-1, 2)).reshape(
            scene_count, max_users)
        self._initial_total = (self._demands * self._user_mask).sum(dim=1)
        self.beam_processes = torch.zeros(num_envs, device=self.device, dtype=torch.long)
        self.candidate_evaluations = torch.zeros(num_envs, device=self.device, dtype=torch.long)
        self.assignment_attempts = torch.zeros(num_envs, device=self.device, dtype=torch.long)
        self.done_mask = torch.zeros(num_envs, device=self.device, dtype=torch.bool)
        self.initial_target = torch.zeros_like(self.done_mask)
        self.stop_reason_code = torch.zeros(num_envs, device=self.device, dtype=torch.long)
        self.best_centers = self.centers.clone()
        self.best_associations = self.assignment.clone()
        self.best_loads = self.loads.clone()
        self.best_quality = torch.full((num_envs, 5), -torch.inf, device=self.device,
                                       dtype=torch.float64)
        self.best_decision_steps = torch.zeros(num_envs, device=self.device, dtype=torch.long)
        self.best_beam_processes = torch.zeros_like(self.best_decision_steps)
        self.best_assignment_attempts = torch.zeros_like(self.best_decision_steps)
        self.best_candidate_evaluations = torch.zeros_like(self.best_decision_steps)
        self.best_metrics = {}
        self._menu_cache = None
        self._balance_ready = True
        self._balance_reset_rows(torch.arange(num_envs, device=self.device),
                                 self.scene_index.clone())

    def reset(self, scene_indices=None):
        if not getattr(self, "_balance_ready", False):
            return super().reset(scene_indices)
        indices = self._select_indices(scene_indices)
        rows = torch.arange(self.num_envs, device=self.device)
        self._balance_reset_rows(rows, indices)
        return self._observation(), self.action_mask()

    def _balance_reset_rows(self, rows: torch.Tensor, indices: torch.Tensor) -> None:
        if rows.numel() == 0:
            return
        super()._reset_rows(rows, indices)
        self.initial_centers[rows] = self._movement_reference_pool[indices]
        self.assignment[rows] = self._start_assoc_pool[indices]
        self._rebuild_loads()
        self.beam_processes[rows] = 0
        self.candidate_evaluations[rows] = 0
        self.assignment_attempts[rows] = self._assignment_prior_pool[indices]
        self.done_mask[rows] = False
        self.stop_reason_code[rows] = STOP_RUNNING
        self._balance_metrics()
        self.initial_target[rows] = self._metrics["balanced_target_met"][rows]
        self.best_centers[rows] = self.centers[rows]
        self.best_associations[rows] = self.assignment[rows]
        self.best_loads[rows] = self.loads[rows]
        self.best_quality[rows] = self._quality()[rows]
        self.best_decision_steps[rows] = self.step_count[rows]
        self.best_beam_processes[rows] = self.beam_processes[rows]
        self.best_assignment_attempts[rows] = self.assignment_attempts[rows]
        self.best_candidate_evaluations[rows] = self.candidate_evaluations[rows]
        if not self.best_metrics:
            self.best_metrics = {key: value.clone() for key, value in self._metrics.items()}
        else:
            for key, value in self._metrics.items():
                self.best_metrics[key][rows] = value[rows]
        self._menu_cache = None

    def _rebuild_loads(self) -> None:
        rows = torch.arange(self.num_envs, device=self.device)[:, None]
        demand = self._demands[self.scene_index]
        valid = self._user_mask[self.scene_index] & (self.assignment >= 0)
        self.loads = torch.zeros((self.num_envs, BEAM_COUNT), device=self.device,
                                  dtype=torch.float64)
        self.loads.scatter_add_(1, self.assignment.clamp_min(0),
                                torch.where(valid, demand, 0.0))

    def _region_exempt_mask(self) -> torch.Tensor:
        result = torch.zeros((self.num_envs, BEAM_COUNT), device=self.device,
                             dtype=torch.bool)
        if self._region_mask_pool.shape[1] == 0:
            return result
        mask = self._region_mask_pool[self.scene_index]
        sizes = self._region_size_pool[self.scene_index]
        first = self._region_first_pool[self.scene_index]
        owner = self.assignment.gather(1, first)
        same = self.assignment[:, None, :] == owner[:, :, None]
        members_same = (same & mask).sum(-1) == sizes
        beam_size = same.sum(-1)
        qualified = (sizes > 0) & (owner >= 0) & members_same & (beam_size == sizes)
        result.scatter_(1, owner.clamp_min(0), qualified)
        return result

    def _balance_metrics(self) -> dict:
        rows = torch.arange(self.num_envs, device=self.device)
        user_mask = self._user_mask[self.scene_index]
        demand = self._demands[self.scene_index]
        min_distance = self._distance_cache.min(dim=-1).values
        covered = user_mask & (min_distance <= squared_ray_limit(0.25) + 1e-14)
        total = self._initial_total[self.scene_index]
        covered_demand = torch.where(covered, demand, 0.0).sum(-1)
        counts = user_mask.sum(-1)
        covered_counts = covered.sum(-1)
        coverage_demand = torch.where(total > 0, covered_demand / total, 0.0)
        coverage_users = covered_counts / counts.clamp_min(1)
        all_mean = self.loads.mean(-1)
        all_std = self.loads.std(-1, unbiased=False)
        exempt = self._region_exempt_mask()
        subject = ~exempt
        subject_count = subject.sum(-1)
        subject_mean = self.loads.masked_fill(~subject, 0.0).sum(-1) / subject_count.clamp_min(1)
        subject_variance = (((self.loads - subject_mean[:, None]) ** 2) * subject).sum(-1) / subject_count.clamp_min(1)
        subject_cv = subject_variance.sqrt() / subject_mean.clamp_min(1e-30)
        subject_std = subject_variance.sqrt()
        subject_valid = subject_mean > 0
        tolerance = self.balance["subject_range_tolerance"]
        in_band = (subject & (self.loads >= (1-tolerance)*subject_mean[:, None]) &
                   (self.loads <= (1+tolerance)*subject_mean[:, None])).sum(-1)
        movement = ray_angle_torch(satellite_rays_torch(self.centers),
                                   satellite_rays_torch(self.initial_centers))
        empty = (self.loads <= 0).sum(-1)
        strict = covered_counts == counts
        near = (coverage_demand >= 0.999) & (coverage_users >= 0.999)
        edge_count = exempt.sum(-1)
        certified_count = (self._region_size_pool[self.scene_index] > 0).sum(-1)
        feasible = near & (empty == 0)
        coverage_target = strict if self.balance["coverage_goal"] == "strict_100" else near
        target = (coverage_target & (empty == 0) & subject_valid &
                  (certified_count <= self.balance["max_edge_beams"]) &
                  (edge_count <= self.balance["max_edge_beams"]) &
                  (subject_cv <= self.balance["subject_cv_target"]) &
                  (in_band >= self.balance["min_subject_beams_in_range"]))
        peak_ratio = self.loads.masked_fill(~subject, 0.0).max(-1).values / subject_mean.clamp_min(1e-30)
        score = (200.0 * (coverage_demand + coverage_users) + 0.2 * in_band -
                 2.0 * subject_cv - 0.1 * peak_ratio - 0.1 * edge_count -
                 0.01 * movement.mean(-1) / self.balance["max_movement_deg"])
        score = torch.where(subject_valid, score, -1000.0)
        self._metrics = {
            "coverage_demand": coverage_demand, "coverage_users": coverage_users,
            "uncovered_demand_bps": total - covered_demand,
            "uncovered_users": counts - covered_counts,
            "load_cv": torch.where(all_mean > 0, all_std / all_mean.clamp_min(1e-30), torch.nan),
            "load_std_bps": all_std, "max_load_bps": self.loads.max(-1).values,
            "empty_beams": empty, "active_beams": torch.full_like(empty, BEAM_COUNT),
            "subject_cv": torch.where(subject_valid, subject_cv, torch.nan),
            "subject_std_bps": subject_std,
            "subject_mean_bps": subject_mean,
            "subject_beams": subject_count, "subject_in_band_beams": in_band,
            "subject_in_band_fraction_all_100": in_band / BEAM_COUNT,
            "subject_max_mean_ratio": torch.where(subject_valid, peak_ratio, torch.nan),
            "edge_beam_count": edge_count, "certified_edge_region_count": certified_count,
            "edge_demand_fraction": ((self.loads * exempt).sum(-1) /
                                     total.clamp_min(1.0)),
            "edge_limit_met": (certified_count <= self.balance["max_edge_beams"]) &
                              (edge_count <= self.balance["max_edge_beams"]),
            "movement_mean_deg": movement.mean(-1), "movement_max_deg": movement.max(-1).values,
            "coverage_100": strict, "coverage_99_9": near,
            "feasible": feasible, "balanced_target_met": target,
            "score": score,
        }
        return self._metrics

    def _quality(self) -> torch.Tensor:
        m = self._metrics
        # Exactly the same lexicographic order as the non-learning route.
        return torch.stack((m["coverage_100"].double(),
                            m["coverage_99_9"].double(),
                            m["subject_in_band_beams"].double(),
                            -torch.nan_to_num(m["subject_cv"], nan=torch.inf),
                            -m["max_load_bps"]), dim=1)

    @staticmethod
    def _lexicographically_better(actual: torch.Tensor, best: torch.Tensor):
        better = torch.zeros(actual.shape[0], device=actual.device, dtype=torch.bool)
        equal = torch.ones_like(better)
        for column in range(actual.shape[1]):
            tolerance = 1e-9 if column < 4 else 1e-4
            better |= equal & (actual[:, column] > best[:, column] + tolerance)
            equal &= (actual[:, column] - best[:, column]).abs() <= tolerance
        return better

    @staticmethod
    def _move_toward(center: torch.Tensor, target: torch.Tensor,
                     angle_deg: float) -> torch.Tensor:
        start = satellite_rays_torch(center)
        total_angle = ray_angle_torch(start, satellite_rays_torch(target))
        low = torch.zeros_like(total_angle)
        high = torch.ones_like(total_angle)
        for _ in range(28):
            fraction = (low + high) / 2
            candidate = center + fraction[..., None] * (target - center)
            actual = ray_angle_torch(start, satellite_rays_torch(candidate))
            low = torch.where(actual < angle_deg, fraction, low)
            high = torch.where(actual < angle_deg, high, fraction)
        bounded = center + ((low + high) / 2)[..., None] * (target - center)
        return torch.where((total_angle <= angle_deg)[..., None], target, bounded)

    def _directional(self, center: torch.Tensor, direction: tuple,
                     multiple: float = 1.0) -> torch.Tensor:
        vector = center.new_tensor(direction)
        vector = vector / torch.linalg.vector_norm(vector)
        return self._move_toward(center, center + vector * 10.0,
                                 multiple * self.balance["step_angle_deg"])

    def _business_candidates(self, beam: torch.Tensor, center: torch.Tensor) -> tuple:
        rows = torch.arange(self.num_envs, device=self.device)
        demand = self._demands[self.scene_index]
        positions = self._positions[self.scene_index]
        valid_users = self._user_mask[self.scene_index]
        missing = valid_users & (self.assignment < 0)
        selected_ray = satellite_rays_torch(center)
        distance = ((self._user_rays[self.scene_index] - selected_ray[:, None, :]) ** 2).sum(-1)
        missed_distance = distance.masked_fill(~missing, torch.inf)
        nearest_missed = torch.argsort(missed_distance, dim=1, stable=True)[:, :2]
        all_targets = []
        all_valid = []
        for k in range(2):
            idx = nearest_missed[:, k]
            all_targets.append(positions[rows, idx])
            all_valid.append(missing[rows, idx])
        owner = self.assignment == beam[:, None]
        weights = torch.where(owner, demand, 0.0)
        total = weights.sum(-1)
        center_of_mass = (positions * weights[..., None]).sum(1) / total.clamp_min(1e-30)[:, None]
        all_targets.append(center_of_mass)
        all_valid.append(total > 0)
        heavy = torch.argsort(
            -self.loads.masked_fill(torch.nn.functional.one_hot(beam, BEAM_COUNT).bool(),
                                    -torch.inf), dim=1, stable=True)[:, :2]
        for h in range(2):
            other = heavy[:, h]
            other_owner = self.assignment == other[:, None]
            weighted = torch.where(other_owner, demand, 0.0)
            other_total = weighted.sum(-1)
            top_users = torch.argsort(-weighted, dim=1, stable=True)[:, :2]
            for k in range(2):
                idx = top_users[:, k]
                all_targets.append(positions[rows, idx])
                all_valid.append(other_owner[rows, idx] & (demand[rows, idx] > 0))
            if h == 0:
                all_targets.append((positions * weighted[..., None]).sum(1) /
                                   other_total.clamp_min(1e-30)[:, None])
                all_valid.append(other_total > 0)
        # Two missing + own centroid + two per heavy beam + one heavy centroid = 8.
        targets = torch.stack(all_targets, dim=1)
        exists = torch.stack(all_valid, dim=1)
        bounded = self._move_toward(center[:, None, :].expand_as(targets), targets,
                                    4 * self.balance["step_angle_deg"])
        return bounded, exists

    def _build_menu(self):
        if self._menu_cache is not None:
            return self._menu_cache
        rows = torch.arange(self.num_envs, device=self.device)
        beam = self.step_count.remainder(BEAM_COUNT)
        center = self.centers[rows, beam]
        size = self.action_dim
        primary = center[:, None, :].expand(-1, size, -1).clone()
        secondary = primary.clone()
        pair = torch.full((self.num_envs, size), -1, device=self.device,
                          dtype=torch.long)
        exists = torch.zeros((self.num_envs, size), device=self.device,
                             dtype=torch.bool)
        cost = torch.ones((self.num_envs, size), device=self.device,
                           dtype=torch.long)
        exists[:, 0] = True
        index = 1
        directions = ((-1., 0.), (1., 0.), (0., -1.), (0., 1.))
        if self.menu_name == "X1":
            directions += ((-1., -1.), (-1., 1.), (1., -1.), (1., 1.))
        scales = (1, 2, 4) if self.menu_name in ("X2", "X3") else (1,)
        for scale in scales:
            for direction in directions:
                primary[:, index] = self._directional(center, direction, scale)
                exists[:, index] = True
                index += 1
        if self.menu_name == "X3":
            candidates, available = self._business_candidates(beam, center)
            count = min(size - index, candidates.shape[1])
            primary[:, index:index + count] = candidates[:, :count]
            exists[:, index:index + count] = available[:, :count]
            index += count
            # Later identical centers have no additional physical meaning;
            # retain their slot IDs but mark them unavailable.
            for slot in range(13, index):
                duplicate = ((primary[:, :slot] - primary[:, slot:slot + 1])
                             .abs().amax(-1) <= 1e-10) & exists[:, :slot]
                exists[:, slot] &= ~duplicate.any(-1)
        if self.menu_name == "X4":
            rays = satellite_rays_torch(self.centers)
            selected = rays[rows, beam]
            distance = ((rays - selected[:, None, :]) ** 2).sum(-1)
            distance[rows, beam] = torch.inf
            neighbors = torch.topk(distance, k=4, largest=False).indices
            for neighbor_index in range(4):
                other_beam = neighbors[:, neighbor_index]
                other_center = self.centers[rows, other_beam]
                for direction in directions:
                    primary[:, index] = self._directional(center, direction)
                    secondary[:, index] = self._directional(other_center, direction)
                    pair[:, index] = other_beam
                    exists[:, index] = True
                    cost[:, index] = 2
                    index += 1
                for direction in directions:
                    primary[:, index] = self._directional(center, direction)
                    opposite = (-direction[0], -direction[1])
                    secondary[:, index] = self._directional(other_center, opposite)
                    pair[:, index] = other_beam
                    exists[:, index] = True
                    cost[:, index] = 2
                    index += 1
        if index > size:
            raise AssertionError("balance action dimension overflow")
        self._menu_cache = primary, secondary, pair, exists, cost
        return self._menu_cache

    def action_mask(self) -> torch.Tensor:
        primary, secondary, pair, exists, cost = self._build_menu()
        rows = torch.arange(self.num_envs, device=self.device)
        beam = self.step_count.remainder(BEAM_COUNT)
        original = self.initial_centers[rows, beam]
        first_move = ray_angle_torch(satellite_rays_torch(primary),
                                     satellite_rays_torch(original[:, None, :]))
        first_ok = visible_torch(primary) & (first_move <= self.balance["max_movement_deg"] + 1e-9)
        other_original = self.initial_centers[rows[:, None], pair.clamp_min(0)]
        second_move = ray_angle_torch(satellite_rays_torch(secondary),
                                      satellite_rays_torch(other_original))
        second_ok = (pair < 0) | (visible_torch(secondary) &
                                 (second_move <= self.balance["max_movement_deg"] + 1e-9))
        budget = self.beam_processes[:, None] + cost <= self.balance["max_beam_decisions"]
        assignment_budget = self.assignment_attempts[:, None] < self.balance["max_assignment_attempts"]
        candidate_budget = self.candidate_evaluations[:, None] < self.balance["max_candidate_evaluations"]
        mask = (exists & first_ok & second_ok & budget & assignment_budget &
                candidate_budget & ~self.done_mask[:, None])
        mask[:, 0] = True  # finished rows and initially solved scenes keep safely.
        mask[self.initial_target, 1:] = False
        return mask

    def candidate_features(self) -> torch.Tensor:
        primary, secondary, pair, exists, cost = self._build_menu()
        rows = torch.arange(self.num_envs, device=self.device)
        beam = self.step_count.remainder(BEAM_COUNT)
        center = self.centers[rows, beam]
        normalized = ((primary - primary.new_tensor([52.5, 5.0])) /
                      primary.new_tensor([27.5, 45.0]))
        grid = self._business_grid_pool[self.scene_index].gather(
            1, self._grid_cells_torch(primary))
        mean = self.loads.mean(-1).clamp_min(1.0)
        own = (self.loads[rows, beam] / mean)[:, None].expand(-1, self.action_dim)
        other = self.loads[rows[:, None], pair.clamp_min(0)] / mean[:, None]
        displacement = (ray_angle_torch(satellite_rays_torch(primary),
                                        satellite_rays_torch(center[:, None, :])) /
                        self.balance["max_movement_deg"])
        rays = satellite_rays_torch(self.centers)
        proposed = satellite_rays_torch(primary)
        distances = ((proposed[:, :, None, :] - rays[:, None, :, :]) ** 2).sum(-1)
        distances.scatter_(2, beam[:, None, None].expand(-1, self.action_dim, 1),
                           torch.inf)
        nearest = distances.min(-1).values.sqrt()
        # Chord / full-width is a bounded geometric overlap proxy.
        overlap = (1.0 - nearest / math.sqrt(squared_ray_limit(0.5))).clamp(0, 1)
        features = torch.stack((normalized[..., 0], normalized[..., 1], grid,
                                own, other, displacement, (pair >= 0).double(),
                                overlap), dim=-1)
        return torch.where(self.action_mask()[..., None], features, 0.0).float()

    def _observation(self):
        if not getattr(self, "_balance_ready", False):
            return super()._observation()
        center = ((self.centers - self.centers.new_tensor([52.5, 5.0])) /
                  self.centers.new_tensor([27.5, 45.0]))
        initial = ((self.initial_centers - self.initial_centers.new_tensor([52.5, 5.0])) /
                   self.initial_centers.new_tensor([27.5, 45.0]))
        loads = self.loads / self.loads.mean(-1, keepdim=True).clamp_min(1.0)
        edge = self._region_exempt_mask().double()
        mask = self._user_mask[self.scene_index] & (self.assignment < 0)
        missing_grid = torch.zeros((self.num_envs, 256), device=self.device,
                                   dtype=torch.float64)
        missing_grid.scatter_add_(1, self._user_cell_pool[self.scene_index],
                                  torch.where(mask, self._demands[self.scene_index], 0.0))
        missing_grid /= self._initial_total[self.scene_index].clamp_min(1.0)[:, None]
        m = self._metrics
        globals_ = torch.stack((self.step_count.double() / max(self.horizon, 1),
                                self.beam_processes.double() / max(self.balance["max_beam_decisions"], 1),
                                m["coverage_demand"], m["coverage_users"],
                                torch.nan_to_num(m["subject_cv"], nan=0.0),
                                m["subject_in_band_fraction_all_100"],
                                m["empty_beams"].double() / BEAM_COUNT,
                                m["edge_beam_count"].double() / BEAM_COUNT), dim=1)
        result = torch.cat((center.flatten(1), initial.flatten(1), loads, edge,
                            self._business_grid_pool[self.scene_index],
                            missing_grid, globals_), dim=1)
        if result.shape[1] != self.obs_dim:
            raise AssertionError(f"observation size {result.shape[1]} != {self.obs_dim}")
        return result.float()

    def step(self, actions):
        actions = torch.as_tensor(actions, device=self.device, dtype=torch.long).flatten()
        if actions.numel() != self.num_envs:
            raise ValueError("one balance action per environment required")
        rows = torch.arange(self.num_envs, device=self.device)
        legal = self.action_mask()
        if bool(((actions < 0) | (actions >= self.action_dim)).any().item()):
            raise ValueError("balance action outside menu")
        if bool((~legal[rows, actions]).any().item()):
            raise ValueError("balance action masked as illegal")
        was_done = self.done_mask.clone()
        working = ~was_done
        center_candidate_checks = working & (actions != 0)
        self.candidate_evaluations += center_candidate_checks.long()
        score_before = self._metrics["score"].clone()
        primary, secondary, pair_menu, _, costs = self._build_menu()
        beam = self.step_count.remainder(BEAM_COUNT)
        pair = pair_menu[rows, actions]
        pair_rows = working & (pair >= 0)
        selected_primary = primary[rows, actions]
        selected_secondary = secondary[rows, actions]
        cost = costs[rows, actions]
        old_center = self.centers[rows, beam].clone()
        old_pair_center = self.centers[rows, pair.clamp_min(0)].clone()
        old_main_col = self._distance_cache[rows, :, beam].clone()
        old_pair_col = self._distance_cache[rows, :, pair.clamp_min(0)].clone()
        old_assignment = self.assignment.clone()
        old_loads = self.loads.clone()
        update_rows = rows[working]
        self.centers[update_rows, beam[working]] = selected_primary[working]
        self.centers[rows[pair_rows], pair[pair_rows]] = selected_secondary[pair_rows]
        rays = self._user_rays[self.scene_index]
        main_rays = satellite_rays_torch(self.centers[rows, beam])
        main_dist = ((rays - main_rays[:, None, :]) ** 2).sum(-1)
        self._distance_cache[update_rows, :, beam[working]] = main_dist[working]
        if bool(pair_rows.any().item()):
            pair_rays = satellite_rays_torch(self.centers[rows, pair.clamp_min(0)])
            pair_dist = ((rays - pair_rays[:, None, :]) ** 2).sum(-1)
            self._distance_cache[rows[pair_rows], :, pair[pair_rows]] = pair_dist[pair_rows]
        minimum, nearest = self._distance_cache.min(dim=-1)
        valid_users = self._user_mask[self.scene_index]
        covered = valid_users & (minimum <= squared_ray_limit(0.25) + 1e-14)
        previous_eligible = self._distance_cache.gather(
            2, old_assignment.clamp_min(0)[..., None]).squeeze(-1) <= squared_ray_limit(0.25) + 1e-14
        preserve = (old_assignment >= 0) & previous_eligible & covered
        proposed_assignment = torch.where(preserve, old_assignment,
                                          torch.where(covered, nearest, -1))
        self.assignment[working] = proposed_assignment[working]
        self._rebuild_loads()
        protected = self._protected_pool[self.scene_index]
        still_covered = (~protected | covered).all(-1)
        nonempty = (self.loads > 0).all(-1)
        accepted = working & (actions != 0) & still_covered & nonempty
        rejection_reason_code = torch.where(
            was_done, 3, torch.where(
                (actions == 0) &
                (self.assignment_attempts >= self.balance["max_assignment_attempts"]),
                5, torch.where(
                    (actions == 0) &
                    (self.candidate_evaluations >= self.balance["max_candidate_evaluations"]),
                    6, torch.where(actions == 0, 4,
                                     torch.where(~still_covered, 1,
                                                 torch.where(~nonempty, 2, 0))))))
        rejected = working & ~accepted
        if bool(rejected.any().item()):
            self.centers[rows[rejected], beam[rejected]] = old_center[rejected]
            self._distance_cache[rows[rejected], :, beam[rejected]] = old_main_col[rejected]
            rejected_pair = rejected & pair_rows
            if bool(rejected_pair.any().item()):
                self.centers[rows[rejected_pair], pair[rejected_pair]] = old_pair_center[rejected_pair]
                self._distance_cache[rows[rejected_pair], :, pair[rejected_pair]] = old_pair_col[rejected_pair]
            self.assignment[rejected] = old_assignment[rejected]
            self.loads[rejected] = old_loads[rejected]
        # Exactly one best complete-user transfer after an accepted decision.
        demand = self._demands[self.scene_index]
        source = self.assignment.clamp_min(0)
        source_load = self.loads.gather(1, source)
        destination_load = self.loads[:, None, :]
        delta = 2 * demand[..., None] * (destination_load - source_load[..., None] + demand[..., None])
        eligible = self._distance_cache <= squared_ray_limit(0.25) + 1e-14
        locked = self._region_user_lock_pool[self.scene_index]
        allowed = (eligible & valid_users[..., None] & (self.assignment >= 0)[..., None] &
                   ~locked[..., None] & (demand > 0)[..., None] &
                   (source_load > demand + 1e-8)[..., None] &
                   (torch.arange(BEAM_COUNT, device=self.device)[None, None, :] != source[..., None]) &
                   accepted[:, None, None])
        allowed_flat = allowed.flatten(1)
        remaining = (self.balance["max_assignment_attempts"] -
                     self.assignment_attempts).clamp_min(0)
        prefix = allowed_flat.long().cumsum(1)
        examined_flat = allowed_flat & (prefix <= remaining[:, None])
        post_move_attempts = examined_flat.sum(-1)
        self.assignment_attempts += post_move_attempts
        delta = delta.masked_fill(~examined_flat.reshape_as(allowed), torch.inf)
        minimum_delta, flat_index = delta.flatten(1).min(-1)
        transfer = accepted & (minimum_delta < -1e-7)
        transfer_user = flat_index.div(BEAM_COUNT, rounding_mode="floor")
        transfer_beam = flat_index.remainder(BEAM_COUNT)
        self.assignment[rows[transfer], transfer_user[transfer]] = transfer_beam[transfer]
        if bool(transfer.any().item()):
            self._rebuild_loads()
        self.step_count[working] += 1
        self.beam_processes[working] += torch.where(accepted[working], cost[working], 1)
        self._balance_metrics()
        quality = self._quality()
        better = working & self._lexicographically_better(quality, self.best_quality)
        self.best_quality[better] = quality[better]
        self.best_centers[better] = self.centers[better]
        self.best_associations[better] = self.assignment[better]
        self.best_loads[better] = self.loads[better]
        self.best_decision_steps[better] = self.step_count[better]
        self.best_beam_processes[better] = self.beam_processes[better]
        self.best_assignment_attempts[better] = self.assignment_attempts[better]
        self.best_candidate_evaluations[better] = self.candidate_evaluations[better]
        for key, value in self._metrics.items():
            self.best_metrics[key][better] = value[better]
        target = self._metrics["balanced_target_met"]
        max_steps = self.step_count >= self.horizon
        max_beam = self.beam_processes >= self.balance["max_beam_decisions"]
        max_assignment = self.assignment_attempts >= self.balance["max_assignment_attempts"]
        max_candidates = self.candidate_evaluations >= self.balance["max_candidate_evaluations"]
        newly_done = working & (self.initial_target | target | max_steps |
                                max_beam | max_assignment | max_candidates)
        self.stop_reason_code = torch.where(
            newly_done & self.initial_target, STOP_ALREADY_TARGET,
            torch.where(newly_done & target, STOP_TARGET,
                        torch.where(newly_done & max_assignment, STOP_ASSIGNMENT_LIMIT,
                                    torch.where(newly_done & max_candidates, STOP_CANDIDATE_LIMIT,
                                                torch.where(newly_done & max_beam, STOP_BEAM_LIMIT,
                                                            torch.where(newly_done & max_steps, STOP_ROUND_LIMIT,
                                                                        self.stop_reason_code))))))
        self.done_mask |= newly_done
        terminal_metrics = {key: value.clone() for key, value in self._metrics.items()}
        terminal_metrics["scene_index"] = self._original_index_map[self.scene_index].clone()
        terminal_metrics["stop_reason_code"] = self.stop_reason_code.clone()
        terminal_metrics["beam_processes"] = self.beam_processes.clone()
        terminal_metrics["assignment_attempts"] = self.assignment_attempts.clone()
        terminal_metrics["candidate_evaluations"] = self.candidate_evaluations.clone()
        terminal_metrics["decision_steps"] = self.step_count.clone()
        terminal_metrics["done"] = self.done_mask.clone()
        best_metrics = {key: value.clone() for key, value in self.best_metrics.items()}
        best_metrics["scene_index"] = self._original_index_map[self.scene_index].clone()
        best_metrics["stop_reason_code"] = self.stop_reason_code.clone()
        best_metrics["beam_processes"] = self.best_beam_processes.clone()
        best_metrics["assignment_attempts"] = self.best_assignment_attempts.clone()
        best_metrics["candidate_evaluations"] = self.best_candidate_evaluations.clone()
        best_metrics["decision_steps"] = self.best_decision_steps.clone()
        terminal = {
            "finished_rows": newly_done.clone(),
            "episode_metrics": terminal_metrics,
            "best_metrics": best_metrics,
            "terminal_layouts": self.centers.clone(),
            "terminal_associations": self.assignment.clone(),
            "terminal_beam_loads_bps": self.loads.clone(),
            "terminal_aircraft_mask": valid_users.clone(),
            "best_layouts": self.best_centers.clone(),
            "best_associations": self.best_associations.clone(),
            "best_beam_loads_bps": self.best_loads.clone(),
            "requested_action": actions.clone(),
            "actual_action": torch.where(accepted, actions, 0),
            "action_accepted": accepted.clone(),
            "post_move_transfer": transfer.clone(),
            "post_move_attempts": post_move_attempts.clone(),
            "center_candidate_checks": center_candidate_checks.long(),
            "rejection_reason_code": rejection_reason_code.clone(),
            "stop_reason_code": self.stop_reason_code.clone(),
        }
        reward = (self._metrics["score"] - score_before).float()
        reward = torch.where(working, reward, 0.0)
        done_return = self.done_mask.clone()
        if self.auto_reset and bool(newly_done.any().item()):
            new_indices = torch.as_tensor(self.rng.integers(len(self.dataset),
                                                             size=int(newly_done.sum().item())),
                                          device=self.device, dtype=torch.long)
            self._balance_reset_rows(rows[newly_done], new_indices)
        self._menu_cache = None
        return self._observation(), reward, done_return, terminal

    def terminal_snapshot(self, stop_reason: str = "time_limit") -> dict:
        """End unfinished evaluation rows without creating an environment action."""
        if self.auto_reset:
            raise ValueError("terminal_snapshot is reserved for non-resetting evaluation")
        reason = {"time_limit": STOP_TIME_LIMIT,
                  "already_target": STOP_ALREADY_TARGET}.get(stop_reason)
        if reason is None:
            raise ValueError("unsupported terminal snapshot reason")
        finished = ~self.done_mask
        self.done_mask[finished] = True
        self.stop_reason_code[finished] = reason
        current = {key: value.clone() for key, value in self._metrics.items()}
        best = {key: value.clone() for key, value in self.best_metrics.items()}
        for result in (current, best):
            result["scene_index"] = self._original_index_map[self.scene_index].clone()
            result["stop_reason_code"] = self.stop_reason_code.clone()
            is_best = result is best
            result["beam_processes"] = (self.best_beam_processes if is_best else self.beam_processes).clone()
            result["assignment_attempts"] = (self.best_assignment_attempts if is_best else self.assignment_attempts).clone()
            result["candidate_evaluations"] = (self.best_candidate_evaluations if is_best else self.candidate_evaluations).clone()
            result["decision_steps"] = (self.best_decision_steps if is_best else self.step_count).clone()
            result["done"] = self.done_mask.clone()
        self._menu_cache = None
        return {
            "finished_rows": finished.clone(), "episode_metrics": current,
            "best_metrics": best, "terminal_layouts": self.centers.clone(),
            "terminal_associations": self.assignment.clone(),
            "terminal_beam_loads_bps": self.loads.clone(),
            "terminal_aircraft_mask": self._user_mask[self.scene_index].clone(),
            "best_layouts": self.best_centers.clone(),
            "best_associations": self.best_associations.clone(),
            "best_beam_loads_bps": self.best_loads.clone(),
            "requested_action": torch.full((self.num_envs,), -1,
                                           device=self.device, dtype=torch.long),
            "actual_action": torch.full((self.num_envs,), -1,
                                        device=self.device, dtype=torch.long),
            "action_accepted": torch.zeros(self.num_envs, device=self.device,
                                            dtype=torch.bool),
            "post_move_transfer": torch.zeros(self.num_envs, device=self.device,
                                               dtype=torch.bool),
            "post_move_attempts": torch.zeros(self.num_envs, device=self.device,
                                               dtype=torch.long),
            "center_candidate_checks": torch.zeros(self.num_envs, device=self.device,
                                                     dtype=torch.long),
            "rejection_reason_code": torch.full((self.num_envs,), 7,
                                                device=self.device, dtype=torch.long),
            "stop_reason_code": self.stop_reason_code.clone(),
        }

    def state_dict(self):
        return {
            "balance_options": dict(self.balance),
            "start_sha256": [item.start_sha256 for item in self._prepared_starts],
            "dataset_scene_ids": [scene.scene_id for scene in self.dataset.scenes],
            "original_scene_ids": list(self._original_scene_ids),
            "eligible_indices": list(self._eligible_indices),
            "num_envs": self.num_envs, "auto_reset": self.auto_reset,
            "rng_state": copy.deepcopy(self.rng.bit_generator.state),
            "scene_index": self.scene_index.clone(), "step_count": self.step_count.clone(),
            "beam_processes": self.beam_processes.clone(),
            "assignment_attempts": self.assignment_attempts.clone(),
            "candidate_evaluations": self.candidate_evaluations.clone(),
            "done_mask": self.done_mask.clone(), "initial_target": self.initial_target.clone(),
            "stop_reason_code": self.stop_reason_code.clone(),
            "centers": self.centers.clone(), "assignment": self.assignment.clone(),
            "loads": self.loads.clone(), "best_centers": self.best_centers.clone(),
            "best_associations": self.best_associations.clone(),
            "best_loads": self.best_loads.clone(), "best_quality": self.best_quality.clone(),
            "best_decision_steps": self.best_decision_steps.clone(),
            "best_beam_processes": self.best_beam_processes.clone(),
            "best_assignment_attempts": self.best_assignment_attempts.clone(),
            "best_candidate_evaluations": self.best_candidate_evaluations.clone(),
            "best_metrics": {key: value.clone() for key, value in self.best_metrics.items()},
        }

    def load_state_dict(self, state):
        if (state["balance_options"] != self.balance or
                state["start_sha256"] != [item.start_sha256 for item in self._prepared_starts] or
                state["dataset_scene_ids"] != [scene.scene_id for scene in self.dataset.scenes] or
                state.get("original_scene_ids") != self._original_scene_ids or
                state.get("eligible_indices") != self._eligible_indices or
                state["num_envs"] != self.num_envs or state["auto_reset"] != self.auto_reset):
            raise ValueError("balance checkpoint configuration/start mismatch")
        self.rng.bit_generator.state = copy.deepcopy(state["rng_state"])
        for key in ("scene_index", "step_count", "beam_processes", "assignment_attempts",
                    "candidate_evaluations",
                    "done_mask",
                    "initial_target", "stop_reason_code", "centers", "assignment",
                    "loads", "best_centers", "best_associations", "best_loads",
                    "best_quality", "best_decision_steps", "best_beam_processes",
                    "best_assignment_attempts", "best_candidate_evaluations"):
            target = getattr(self, key)
            value = torch.as_tensor(state[key], device=self.device, dtype=target.dtype)
            if value.shape != target.shape:
                raise ValueError(f"balance checkpoint {key} shape mismatch")
            setattr(self, key, value.clone())
        if set(state["best_metrics"]) != set(self.best_metrics):
            raise ValueError("balance checkpoint best metric keys mismatch")
        for key, target in self.best_metrics.items():
            value = torch.as_tensor(state["best_metrics"][key], device=self.device,
                                    dtype=target.dtype)
            if value.shape != target.shape:
                raise ValueError(f"balance checkpoint best metric {key} shape mismatch")
            self.best_metrics[key] = value.clone()
        self.initial_centers = self._movement_reference_pool[self.scene_index].clone()
        self._build_distance_cache(torch.arange(self.num_envs, device=self.device))
        saved_loads = self.loads.clone()
        self._rebuild_loads()
        if not torch.allclose(self.loads, saved_loads, atol=1e-6, rtol=1e-12):
            raise ValueError("balance checkpoint demand ledger mismatch")
        self._balance_metrics()
        self._menu_cache = None
