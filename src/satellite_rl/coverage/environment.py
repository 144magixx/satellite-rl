"""Torch-batched first-stage 100-beam coverage planning environment."""

from __future__ import annotations

import copy

import numpy as np
import torch

from satellite_rl.coverage.geometry import (
    ray_angle_torch, satellite_rays_torch, squared_ray_limit, visible_torch)
from satellite_rl.coverage.scenes import (
    BEAM_COUNT, EMPTY_PARKING_LATLON, candidate_positions_sha256,
    positive_position_candidates)


ACTION_OFFSETS = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1),
                  (-1, -1), (-1, 1), (1, -1), (1, 1))
GRID_SIDE = 16
OBS_DIM = 964  # original 508 + initial/current active masks (200) + demand grid (256)


def coverage_options(config):
    """Read coverage-specific options from either shared or direct config."""
    if not isinstance(config, dict):
        raise TypeError("coverage config must be a mapping")
    options = config.get("coverage", config)
    if not isinstance(options, dict):
        raise TypeError("coverage options must be a mapping")
    out = {
        "initialization": options.get("initialization", "rule"),
        "candidate_capacity": int(options.get("candidate_capacity", 2048)),
        "coordinate_step_deg": float(options.get("coordinate_step_deg", 0.1)),
        "max_movement_deg": float(options.get("max_movement_deg", 0.25)),
        "beam_width_deg": float(options.get("beam_width_deg", 0.5)),
        "sweeps": int(options.get("sweeps", 2)),
        "min_coverage_demand": float(options.get("min_coverage_demand", 0.99)),
        "min_coverage_users": float(options.get("min_coverage_users", 0.99)),
        "score_coverage_demand_weight": float(options.get("score_coverage_demand_weight", 100.0)),
        "score_coverage_users_weight": float(options.get("score_coverage_users_weight", 1.0)),
        "score_load_cv_weight": float(options.get("score_load_cv_weight", 1.0)),
        "score_movement_weight": float(options.get("score_movement_weight", 0.1)),
    }
    if options.get("environment_profile", "europe_v1") != "europe_v1":
        raise ValueError("first-stage coverage geometry requires europe_v1")
    if out["initialization"] not in ("rule", "cluster", "empty"):
        raise ValueError("coverage initialization must be rule, cluster or empty")
    if not 100 <= out["candidate_capacity"] <= 65536:
        raise ValueError("candidate_capacity must be in [100,65536]")
    if out["coordinate_step_deg"] <= 0 or out["max_movement_deg"] <= 0:
        raise ValueError("step and maximum movement must be positive")
    if out["beam_width_deg"] != 0.5 or out["sweeps"] < 1:
        raise ValueError("first-stage version fixes beam width at 0.5 degrees; sweeps must be positive")
    if any(not 0 <= out[k] <= 1 for k in ("min_coverage_demand", "min_coverage_users")):
        raise ValueError("coverage floors must be in [0,1]")
    return out


class CoverageBatchEnv:
    """Every action changes one stable beam ID, then reassigns every aircraft.

    Reward equals score(after)-score(before). With PPO gamma=1, episode reward
    telescopes to terminal score minus initial score. The score is
    ``100*demand_coverage + user_coverage - load_cv -
    0.1*mean_movement/max_movement`` by default. No stage-two rate, SI or SGM
    is used. Feasibility is reported separately and is never fabricated.
    """

    obs_dim = OBS_DIM
    # The persistent cache is [parallel environments, padded users, 100].
    # Build it in small float64 blocks; a 2048-env/1887-user batch otherwise
    # materializes a much larger [N,U,100,3] temporary on the GPU.
    _distance_env_chunk = 64
    _distance_user_chunk = 256
    _column_env_chunk = 128

    def __init__(self, dataset, config, num_envs=32, device="cuda", seed=51001,
                 auto_reset=True):
        self.dataset = dataset
        self.options = coverage_options(config)
        self.initialization = self.options["initialization"]
        if dataset.metadata.get("initialization") not in (None, self.initialization):
            raise ValueError("dataset initializer differs from environment configuration")
        self.action_dim = (self.options["candidate_capacity"] if self.initialization == "empty"
                           else len(ACTION_OFFSETS))
        self.num_envs = int(num_envs)
        if self.num_envs < 1 or len(dataset) < 1:
            raise ValueError("environment and dataset must be nonempty")
        self.device = torch.device(device)
        self.auto_reset = bool(auto_reset)
        self.rng = np.random.default_rng(seed)
        self.horizon = BEAM_COUNT if self.initialization == "empty" else BEAM_COUNT * self.options["sweeps"]
        self._step_in_episode = 0
        self.max_users = max(len(s.demands) for s in dataset.scenes)
        count = len(dataset)
        positions = np.zeros((count, self.max_users, 2), dtype=np.float64)
        demands = np.zeros((count, self.max_users), dtype=np.float64)
        user_mask = np.zeros((count, self.max_users), dtype=bool)
        initial = np.zeros((count, BEAM_COUNT, 2), dtype=np.float64)
        initial_active = np.ones((count, BEAM_COUNT), dtype=bool)
        grids = np.zeros((count, GRID_SIDE * GRID_SIDE), dtype=np.float64)
        candidate_pool = np.zeros((count, self.action_dim, 2), dtype=np.float64)
        candidate_mask = np.zeros((count, self.action_dim), dtype=bool)
        candidate_representative = np.zeros((count, self.action_dim), dtype=np.int64)
        candidate_grid_share = np.zeros((count, self.action_dim), dtype=np.float64)
        candidate_hashes = []
        scene_valid = []
        for i, scene in enumerate(dataset.scenes):
            length = len(scene.demands)
            if scene.positions.shape != (length, 2) or scene.centers.shape != (BEAM_COUNT, 2):
                raise ValueError(f"scene {i} has incompatible aircraft or 100-beam arrays")
            positions[i, :length] = scene.positions
            demands[i, :length] = scene.demands
            user_mask[i, :length] = True
            initial[i] = scene.centers
            grid_cell = self._grid_cells_numpy(scene.positions)
            np.add.at(grids[i], grid_cell, scene.demands)
            total_demand = float(scene.demands.sum())
            if total_demand > 0:
                grids[i] /= total_demand
            candidates = positive_position_candidates(scene.positions, scene.demands)
            candidate_hash = candidate_positions_sha256(candidates)
            candidate_hashes.append(candidate_hash)
            scene_valid.append(scene.metadata.get("initialization_valid") is not False)
            if self.initialization == "empty":
                candidate_count = len(candidates)
                if candidate_count > self.action_dim:
                    raise ValueError(f"scene {i} has {candidate_count} unique positive-demand candidates; "
                                     f"increase candidate_capacity above {self.action_dim}, never truncate")
                if scene.metadata.get("candidate_sha256") not in (None, candidate_hash):
                    raise ValueError(f"scene {i} candidate hash differs from aircraft input")
                if scene.metadata.get("candidate_count") not in (None, candidate_count):
                    raise ValueError(f"scene {i} candidate count differs from aircraft input")
                if not np.array_equal(scene.centers,
                                      np.tile(EMPTY_PARKING_LATLON, (BEAM_COUNT, 1))):
                    raise ValueError(f"scene {i} empty-start centers must be inactive parking coordinates")
                initial_active[i] = False
                candidate_pool[i, :candidate_count] = candidates
                candidate_mask[i, :candidate_count] = True
                positive_indices = np.flatnonzero(scene.demands > 0)
                _, first = np.unique(scene.positions[positive_indices], axis=0, return_index=True)
                candidate_representative[i, :candidate_count] = positive_indices[first]
                candidate_grid_share[i, :candidate_count] = grids[i, self._grid_cells_numpy(candidates)]
                if candidate_count < BEAM_COUNT:
                    scene_valid[-1] = False
            elif scene.metadata.get("initial_active_count", BEAM_COUNT) not in (0, BEAM_COUNT):
                raise ValueError(f"scene {i} invalid initial active-beam count")
        self._positions = torch.as_tensor(positions, device=self.device)
        self._demands = torch.as_tensor(demands, device=self.device)
        self._user_mask = torch.as_tensor(user_mask, device=self.device)
        self._initial_pool = torch.as_tensor(initial, device=self.device)
        self._initial_active_pool = torch.as_tensor(initial_active, device=self.device)
        self._business_grid_pool = torch.as_tensor(grids, device=self.device)
        self._candidate_pool = torch.as_tensor(candidate_pool, device=self.device)
        self._candidate_pool_mask = torch.as_tensor(candidate_mask, device=self.device)
        self._candidate_representative = torch.as_tensor(candidate_representative, device=self.device)
        self._candidate_grid_share = torch.as_tensor(candidate_grid_share, device=self.device)
        self._candidate_hashes = candidate_hashes
        self._scene_valid = torch.as_tensor(scene_valid, device=self.device)
        self._user_rays = satellite_rays_torch(self._positions)
        self._offsets = torch.as_tensor(ACTION_OFFSETS, device=self.device, dtype=torch.float64)
        self.scene_index = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.step_count = torch.zeros_like(self.scene_index)
        self.initial_centers = torch.zeros((self.num_envs, BEAM_COUNT, 2),
                                           device=self.device, dtype=torch.float64)
        self.centers = self.initial_centers.clone()
        self.initial_active_mask = torch.zeros((self.num_envs, BEAM_COUNT),
                                               device=self.device, dtype=torch.bool)
        self.active_mask = self.initial_active_mask.clone()
        self.candidate_used = torch.zeros((self.num_envs, self.action_dim),
                                          device=self.device, dtype=torch.bool)
        self.assignment = torch.full((self.num_envs, self.max_users), -1,
                                     device=self.device, dtype=torch.long)
        self.loads = torch.zeros((self.num_envs, BEAM_COUNT), device=self.device,
                                 dtype=torch.float64)
        self._distance_cache = torch.empty((self.num_envs, self.max_users, BEAM_COUNT),
                                            device=self.device, dtype=torch.float64)
        self._metrics = {}
        # Construction must permit an evaluation collection containing failed
        # initializers; its caller records each failure explicitly. A later
        # training reset still samples the whole collection and rejects any
        # unrepresentable layout instead of silently removing that scene.
        first_valid = next((i for i, scene in enumerate(dataset.scenes)
                            if np.isfinite(scene.centers).all() and
                            (self.initialization != "empty" or scene_valid[i])), None)
        if first_valid is None:
            raise ValueError("dataset contains no representable 100-beam layout")
        self.reset(scene_indices=[first_valid])

    @staticmethod
    def _grid_cells_numpy(positions):
        lat = np.floor((np.asarray(positions)[:, 0] - 25.0) * GRID_SIDE / 55.0).astype(np.int64)
        lon = np.floor((np.asarray(positions)[:, 1] + 40.0) * GRID_SIDE / 90.0).astype(np.int64)
        return np.clip(lat, 0, GRID_SIDE - 1) * GRID_SIDE + np.clip(lon, 0, GRID_SIDE - 1)

    @staticmethod
    def _grid_cells_torch(positions):
        lat = ((positions[..., 0] - 25.0) * GRID_SIDE / 55.0).floor().long().clamp(0, GRID_SIDE - 1)
        lon = ((positions[..., 1] + 40.0) * GRID_SIDE / 90.0).floor().long().clamp(0, GRID_SIDE - 1)
        return lat * GRID_SIDE + lon

    def _select_indices(self, scene_indices):
        if scene_indices is None:
            return torch.as_tensor(self.rng.integers(len(self.dataset), size=self.num_envs),
                                   device=self.device, dtype=torch.long)
        indices = torch.as_tensor(scene_indices, device=self.device, dtype=torch.long).flatten()
        if indices.numel() == 1:
            indices = indices.expand(self.num_envs).clone()
        if indices.numel() != self.num_envs or (indices < 0).any() or (indices >= len(self.dataset)).any():
            raise ValueError("scene_indices must have one index per environment")
        return indices

    def reset(self, scene_indices=None):
        indices = self._select_indices(scene_indices)
        self._reset_rows(torch.arange(self.num_envs, device=self.device), indices)
        self._step_in_episode = 0
        return self._observation(), self._action_mask()

    def _reset_rows(self, rows, indices):
        if rows.numel() == 0:
            return
        if self.initialization == "empty" and not self._scene_valid[indices].all():
            bad = indices[~self._scene_valid[indices]].tolist()
            raise ValueError(f"empty-start scenes {bad} have fewer than 100 unique positive-demand candidates")
        initial = self._initial_pool[indices]
        if not torch.isfinite(initial).all():
            bad = indices[~torch.isfinite(initial).all(dim=(1, 2))].tolist()
            raise ValueError(f"initialization failed for scenes {bad}; retain as failure, do not train")
        self.scene_index[rows] = indices
        self.step_count[rows] = 0
        self.initial_centers[rows] = initial
        self.centers[rows] = initial
        self.initial_active_mask[rows] = self._initial_active_pool[indices]
        self.active_mask[rows] = self.initial_active_mask[rows]
        self.candidate_used[rows] = False
        self._build_distance_cache(rows)
        self._recompute_from_cache()

    def _build_distance_cache(self, rows):
        """Rebuild all columns for reset/restore in bounded float64 blocks."""
        for offset in range(0, rows.numel(), self._distance_env_chunk):
            batch_rows = rows[offset:offset + self._distance_env_chunk]
            # Empty-start active beams are a prefix in their placement order;
            # parked centers are never evaluated as a physical beam.
            active_count = (int(self.active_mask[batch_rows[0]].sum().item())
                            if self.initialization == "empty" else BEAM_COUNT)
            if self.initialization == "empty":
                self._distance_cache[batch_rows] = torch.inf
            if active_count == 0:
                continue
            rays = satellite_rays_torch(self.centers[batch_rows, :active_count])
            for start in range(0, self.max_users, self._distance_user_chunk):
                users_slice = slice(start, min(start + self._distance_user_chunk, self.max_users))
                users = self._user_rays[self.scene_index[batch_rows], users_slice, :]
                # Preserve the original subtraction, square and reduction
                # order for each user/beam pair; only the outer batch is tiled.
                squared = (users[:, :, None, :] - rays[:, None, :, :]).square().sum(-1)
                self._distance_cache[batch_rows, users_slice, :active_count] = squared

    def _refresh_distance_column(self, beam):
        """A step changes one center per environment; update only those rays."""
        for start in range(0, self.num_envs, self._column_env_chunk):
            rows = torch.arange(start, min(start + self._column_env_chunk, self.num_envs),
                                device=self.device)
            selected_beam = beam[rows]
            rays = satellite_rays_torch(self.centers[rows, selected_beam])
            users = self._user_rays[self.scene_index[rows]]
            squared = (users[:, :, None, :] - rays[:, None, None, :]).square().sum(-1).squeeze(-1)
            self._distance_cache[rows, :, selected_beam] = squared

    def _recompute(self):
        """Full rebuild for explicit center edits and exact checkpoint restore."""
        self._build_distance_cache(torch.arange(self.num_envs, device=self.device))
        self._recompute_from_cache()

    def _recompute_from_cache(self):
        # Nearest beam, every user's unique association, every beam's load and
        # uncovered demand are still recomputed globally after each action.
        nearest_distance, nearest = self._distance_cache.min(dim=-1)
        valid_users = self._user_mask[self.scene_index]
        covered = valid_users & (nearest_distance <= squared_ray_limit(0.25) + 1e-14)
        self.assignment = torch.where(covered, nearest, -1)
        demand = self._demands[self.scene_index]
        self.loads = torch.zeros((self.num_envs, BEAM_COUNT), device=self.device, dtype=torch.float64)
        self.loads.scatter_add_(1, nearest, torch.where(covered, demand, 0.0))
        total = torch.where(valid_users, demand, 0.0).sum(dim=1)
        covered_demand = torch.where(covered, demand, 0.0).sum(dim=1)
        uncovered_demand = torch.where(valid_users & ~covered, demand, 0.0).sum(dim=1)
        counts = valid_users.sum(dim=1)
        covered_counts = covered.sum(dim=1)
        coverage_demand = torch.where(total > 0, covered_demand / total, torch.zeros_like(total))
        coverage_users = covered_counts / counts.clamp_min(1)
        mean = self.loads.mean(dim=1)
        std = self.loads.std(dim=1, unbiased=False)
        cv = torch.where(mean > 0, std / mean, torch.zeros_like(mean))
        if self.initialization == "empty":
            movement = torch.zeros_like(self.loads)
            movement_applicable = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        else:
            movement = ray_angle_torch(satellite_rays_torch(self.centers),
                                       satellite_rays_torch(self.initial_centers))
            movement_applicable = torch.ones(self.num_envs, device=self.device, dtype=torch.bool)
        score = (self.options["score_coverage_demand_weight"] * coverage_demand +
                 self.options["score_coverage_users_weight"] * coverage_users -
                 self.options["score_load_cv_weight"] * cv -
                 self.options["score_movement_weight"] *
                 movement.mean(dim=1) / self.options["max_movement_deg"])
        empty = (self.loads <= 0).sum(dim=1)
        active_count = self.active_mask.sum(dim=1)
        feasible = ((coverage_demand >= self.options["min_coverage_demand"]) &
                    (coverage_users >= self.options["min_coverage_users"]) &
                    (empty == 0) & (active_count == BEAM_COUNT))
        self._nearest_active_squared = nearest_distance
        self._metrics = {
            "coverage_demand": coverage_demand,
            "coverage_users": coverage_users,
            "uncovered_demand_bps": uncovered_demand,
            "uncovered_users": counts - covered_counts,
            "load_cv": cv,
            "load_std_bps": std,
            "max_load_bps": self.loads.max(dim=1).values,
            "empty_beams": empty,
            "active_beams": active_count,
            "movement_mean_deg": movement.mean(dim=1),
            "movement_max_deg": movement.max(dim=1).values,
            "movement_applicable": movement_applicable,
            "feasible": feasible,
            "score": score,
        }

    def _action_mask(self):
        if self.initialization == "empty":
            return self._candidate_pool_mask[self.scene_index] & ~self.candidate_used
        beam = self.step_count.remainder(BEAM_COUNT)
        rows = torch.arange(self.num_envs, device=self.device)
        selected = self.centers[rows, beam]
        original = self.initial_centers[rows, beam]
        candidates = selected[:, None, :] + self.options["coordinate_step_deg"] * self._offsets[None]
        movement = ray_angle_torch(satellite_rays_torch(candidates),
                                   satellite_rays_torch(original[:, None, :]))
        mask = visible_torch(candidates) & (movement <= self.options["max_movement_deg"] + 1e-12)
        mask[:, 0] = True
        return mask

    def action_mask(self):
        return self._action_mask().clone()

    @staticmethod
    def _disk_overlap_fraction(nearest_squared):
        """Equal-angle disk area overlap with the nearest other active beam.

        This is an observation proxy for geometric overlap, not interference,
        signal quality or capacity. At no active beam, +inf maps exactly to 0.
        """
        chord = nearest_squared.clamp_min(0.0).sqrt()
        separation = torch.rad2deg(2.0 * torch.asin((chord / 2.0).clamp(max=1.0)))
        ratio = (separation / 0.5).clamp(0.0, 1.0)
        return (2.0 / np.pi) * (torch.acos(ratio) - ratio * (1.0 - ratio.square()).clamp_min(0).sqrt())

    def candidate_features(self):
        """[N,A,4] candidate coordinate, fixed-grid share, overlap proxy."""
        rows = torch.arange(self.num_envs, device=self.device)
        if self.initialization == "empty":
            candidates = self._candidate_pool[self.scene_index]
            representative = self._candidate_representative[self.scene_index]
            nearest = self._nearest_active_squared.gather(1, representative)
            grid_share = self._candidate_grid_share[self.scene_index]
        else:
            beam = self.step_count.remainder(BEAM_COUNT)
            selected = self.centers[rows, beam]
            candidates = selected[:, None, :] + self.options["coordinate_step_deg"] * self._offsets[None]
            grid_share = self._business_grid_pool[self.scene_index].gather(
                1, self._grid_cells_torch(candidates))
            candidate_rays = satellite_rays_torch(candidates)
            active_rays = satellite_rays_torch(self.centers)
            squared = (candidate_rays[:, :, None, :] - active_rays[:, None, :, :]).square().sum(-1)
            other_active = self.active_mask.clone()
            other_active[rows, beam] = False
            nearest = squared.masked_fill(~other_active[:, None, :], torch.inf).min(dim=-1).values
        center = candidates.new_tensor([52.5, 5.0])
        scale = candidates.new_tensor([27.5, 45.0])
        normalized = (candidates - center) / scale
        overlap = self._disk_overlap_fraction(nearest)
        features = torch.cat((normalized, grid_share[..., None], overlap[..., None]), dim=-1)
        return torch.where(self._action_mask()[..., None], features, 0.0).to(torch.float32)

    def _observation(self):
        # All 100 stable beam IDs remain in the observation; loads are scaled
        # relative to mean demand. The raw aircraft list is not observed by
        # the policy and this compression must be stated in results.
        lat_center, lon_center = 52.5, 5.0
        scale = self.centers.new_tensor([27.5, 45.0])
        current = ((self.centers - self.centers.new_tensor([lat_center, lon_center])) / scale)
        initial = ((self.initial_centers - self.centers.new_tensor([lat_center, lon_center])) / scale)
        current = torch.where(self.active_mask[..., None], current, 0.0).flatten(1)
        initial = torch.where(self.initial_active_mask[..., None], initial, 0.0).flatten(1)
        mean = self.loads.mean(dim=1, keepdim=True)
        loads = self.loads / mean.clamp_min(1.0)
        beam = self.step_count.remainder(BEAM_COUNT).to(torch.float64) / (BEAM_COUNT - 1)
        rest = (self.horizon - self.step_count).to(torch.float64) / self.horizon
        fields = torch.stack((beam, rest, self._metrics["coverage_demand"],
                              self._metrics["coverage_users"], self._metrics["load_cv"],
                              self._metrics["movement_mean_deg"] / self.options["max_movement_deg"],
                              self._metrics["movement_max_deg"] / self.options["max_movement_deg"],
                              self._metrics["empty_beams"].to(torch.float64) / BEAM_COUNT), dim=1)
        return torch.cat((current, initial, loads, fields,
                          self.initial_active_mask.to(torch.float64),
                          self.active_mask.to(torch.float64),
                          self._business_grid_pool[self.scene_index]), dim=1).to(torch.float32)

    def step(self, actions):
        if not self.auto_reset and self._step_in_episode >= self.horizon:
            raise RuntimeError("terminal coverage episode requires reset before another step")
        actions = torch.as_tensor(actions, device=self.device, dtype=torch.long).flatten()
        if actions.numel() != self.num_envs:
            raise ValueError("one action per environment required")
        in_range = (actions >= 0).all() & (actions < self.action_dim).all()
        if self.device.type == "cuda" and hasattr(torch, "_assert_async"):
            torch._assert_async(in_range, "coverage action outside configured action space")
        elif not bool(in_range):
            raise ValueError(f"coverage action must be in [0,{self.action_dim - 1}]")
        rows = torch.arange(self.num_envs, device=self.device)
        mask = self._action_mask()
        valid = mask[rows, actions].all()
        if self.device.type == "cuda" and hasattr(torch, "_assert_async"):
            torch._assert_async(valid, "illegal coverage action")
        elif not bool(valid):
            raise ValueError("illegal coverage action; caller must respect action mask")
        previous_score = self._metrics["score"].clone()
        beam = self.step_count.remainder(BEAM_COUNT)
        if self.initialization == "empty":
            self.centers[rows, beam] = self._candidate_pool[self.scene_index, actions]
            self.active_mask[rows, beam] = True
            self.candidate_used[rows, actions] = True
        else:
            self.centers[rows, beam] += self.options["coordinate_step_deg"] * self._offsets[actions]
        self.step_count += 1
        self._step_in_episode += 1
        self._refresh_distance_column(beam)
        self._recompute_from_cache()
        reward = (self._metrics["score"] - previous_score).to(torch.float32)
        done = self.step_count >= self.horizon
        terminal = {key: value.clone() for key, value in self._metrics.items()}
        terminal["scene_index"] = self.scene_index.clone()
        terminal["done"] = done.clone()
        infos = {"episode_metrics": terminal,
                 "episodes_finished": self._step_in_episode >= self.horizon,
                 "terminal_layouts": self.centers.clone(),
                 "terminal_active_masks": self.active_mask.clone(),
                 "terminal_associations": self.assignment.clone(),
                 "terminal_aircraft_mask": self._user_mask[self.scene_index].clone()}
        if self.auto_reset and self._step_in_episode >= self.horizon:
            new_indices = torch.as_tensor(self.rng.integers(len(self.dataset), size=self.num_envs),
                                          device=self.device, dtype=torch.long)
            self._reset_rows(rows, new_indices)
            self._step_in_episode = 0
        return self._observation(), reward, done, infos

    def metrics(self):
        return {key: value.clone() for key, value in self._metrics.items()}

    def layouts(self):
        return self.centers.clone()

    def active_masks(self):
        return self.active_mask.clone()

    def associations(self):
        return self.assignment.clone(), self._user_mask[self.scene_index].clone()

    def state_dict(self):
        return {"scene_index": self.scene_index.clone(),
                "step_count": self.step_count.clone(),
                "positions": self._positions.clone(),
                "demands": self._demands.clone(),
                "user_mask": self._user_mask.clone(),
                "initial_pool": self._initial_pool.clone(),
                "initial_centers": self.initial_centers.clone(),
                "centers": self.centers.clone(),
                "initial_active_mask": self.initial_active_mask.clone(),
                "active_mask": self.active_mask.clone(),
                "candidate_used": self.candidate_used.clone(),
                "candidate_hashes": list(self._candidate_hashes),
                "assignment": self.assignment.clone(), "loads": self.loads.clone(),
                "rng_state": copy.deepcopy(self.rng.bit_generator.state),
                "dataset_scene_ids": [s.scene_id for s in self.dataset.scenes],
                "options": dict(self.options), "num_envs": self.num_envs,
                "action_dim": self.action_dim, "obs_dim": self.obs_dim,
                "auto_reset": self.auto_reset,
                "step_in_episode": self._step_in_episode}

    def load_state_dict(self, state):
        if (state["dataset_scene_ids"] != [s.scene_id for s in self.dataset.scenes] or
                state["options"] != self.options or state["num_envs"] != self.num_envs or
                state["auto_reset"] != self.auto_reset or
                state["action_dim"] != self.action_dim or state["obs_dim"] != self.obs_dim or
                state["candidate_hashes"] != self._candidate_hashes):
            raise ValueError("coverage environment checkpoint configuration/scene mismatch")
        for key, actual in (("positions", self._positions), ("demands", self._demands),
                            ("user_mask", self._user_mask), ("initial_pool", self._initial_pool)):
            saved = torch.as_tensor(state[key], device=self.device, dtype=actual.dtype)
            if saved.shape != actual.shape or not torch.equal(saved, actual):
                raise ValueError(f"checkpoint aircraft/layout data mismatch: {key}")
        for key in ("scene_index", "step_count", "initial_centers", "centers",
                    "initial_active_mask", "active_mask", "candidate_used", "assignment", "loads"):
            target = getattr(self, key)
            source = torch.as_tensor(state[key], device=self.device, dtype=target.dtype)
            if source.shape != target.shape:
                raise ValueError(f"checkpoint {key} shape mismatch")
            setattr(self, key, source.clone())
        self.rng.bit_generator.state = copy.deepcopy(state["rng_state"])
        self._step_in_episode = int(state["step_in_episode"])
        if not 0 <= self._step_in_episode <= self.horizon:
            raise ValueError("invalid checkpoint episode step")
        if self.initialization == "empty":
            expected = (torch.arange(BEAM_COUNT, device=self.device)[None, :] <
                        self.step_count[:, None])
            if (not torch.equal(self.active_mask, expected) or
                    not torch.equal(self.initial_active_mask,
                                    torch.zeros_like(self.initial_active_mask)) or
                    not torch.equal(self.candidate_used.sum(dim=1), self.step_count)):
                raise ValueError("empty-start checkpoint placement masks or candidate usage differ")
        elif (not self.active_mask.all() or not self.initial_active_mask.all() or
              self.candidate_used.any()):
            raise ValueError("microadjustment checkpoint has invalid active/candidate masks")
        # Derived metrics are checked from the restored physical arrays.
        self._recompute()
