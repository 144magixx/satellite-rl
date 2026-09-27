"""Composable observations and rewards around the unchanged satellite environment."""

from contextlib import contextmanager
from copy import deepcopy
from decimal import Decimal
from threading import RLock
from types import SimpleNamespace
import math

import numpy as np

from satellite_rl.envs import satellite as legacy
from satellite_rl.envs.regions import get_profile, profile_metadata, geo_distances_km, validate_locations
from satellite_rl.workbench.config import FEATURES, config_summary, validate_config


SCENE_KEYS = ("beam_location", "beam_G_Tx_max", "UE_G_Rx_max", "distances", "rate", "noise_temperature")
_BUDGET_LOCK = RLock()


@contextmanager
def _budget_scope(budget):
    """Legacy code reads its module global during reset and step; restore it always."""
    with _BUDGET_LOCK:
        previous = legacy.power_threshold
        legacy.power_threshold = float(budget)
        try:
            yield
        finally:
            legacy.power_threshold = previous


@contextmanager
def _seed_scope(seed):
    with _BUDGET_LOCK:
        previous = np.random.get_state()
        np.random.seed(int(seed))
        try:
            yield
        finally:
            np.random.set_state(previous)


def validate_scene(scene, profile="china_legacy"):
    """Copy and check a full 100-beam scene in processing order."""
    if not isinstance(scene, dict):
        raise ValueError("场景必须是数组映射")
    if set(scene) != set(SCENE_KEYS) | {"original_indices"}:
        raise ValueError("场景字段不完整或含未知字段")
    out = {}
    shapes = {"beam_location": (100, 2), "beam_G_Tx_max": (100, 1),
              "UE_G_Rx_max": (100, 1), "distances": (100,),
              "rate": (100, 1), "noise_temperature": (100, 1),
              "original_indices": (100,)}
    for key, shape in shapes.items():
        arr = np.asarray(scene[key])
        if arr.shape != shape or not (np.issubdtype(arr.dtype, np.integer) or np.issubdtype(arr.dtype, np.floating)) or not np.isfinite(arr).all():
            raise ValueError(f"场景字段 {key} 形状或数值无效，应为 {shape}")
        out[key] = np.array(arr, dtype=np.int64 if key == "original_indices" else np.float64, copy=True)
    indices = np.asarray(scene["original_indices"])
    if not np.array_equal(indices, out["original_indices"]) or set(out["original_indices"].tolist()) != set(range(100)):
        raise ValueError("original_indices 必须是0至99的完整排列")
    if (out["rate"] <= 0).any() or (out["noise_temperature"] <= 0).any() or (out["distances"] <= 0).any():
        raise ValueError("业务需求、噪声温度与距离必须为正")
    if ((out["beam_location"][:, 0] < -90) | (out["beam_location"][:, 0] > 90)).any() or ((out["beam_location"][:, 1] < -180) | (out["beam_location"][:, 1] > 180)).any():
        raise ValueError("波束位置超出经纬度范围")
    region = get_profile(profile)
    validate_locations(out["beam_location"], region)
    expected_distance = geo_distances_km(out["beam_location"], region)
    if not np.allclose(out["distances"], expected_distance, rtol=1e-8, atol=1e-6):
        raise ValueError("场景距离与波束位置不一致")
    return out


def _float(value):
    return float(np.asarray(value).reshape(-1)[0])


def _json(value):
    if isinstance(value, np.ndarray):
        return [_json(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return _json(value.item())
    if isinstance(value, dict):
        return {k: _json(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _jain(values):
    arr = np.asarray(values, dtype=np.float64)
    sum_sq = float(np.square(arr).sum())
    return float(arr.sum() ** 2 / (len(arr) * sum_sq)) if len(arr) and sum_sq > 0 else 0.0


def _sgm(rates, demands, beta):
    rates, demands = np.asarray(rates, dtype=float), np.asarray(demands, dtype=float)
    result = np.zeros_like(rates)
    positive = rates > 0
    ratio = rates[positive] / demands[positive]
    real = np.where(ratio >= 1, ratio - 1, 1 - 1 / ratio)
    imag = (rates[positive] - demands[positive]) / beta
    result[positive] = 1 - (1 - np.exp(-np.abs(real + 1j * imag))) ** 3
    return result


class WorkbenchEnv:
    """Strict wrapper. The raw environment remains the source of physical outcomes."""

    def __init__(self, config):
        self.config = validate_config(config)
        self.summary = config_summary(self.config)
        self.region_profile = get_profile(self.config["environment"]["profile"])
        self.environment_metadata = profile_metadata(self.region_profile)
        self.scene_metadata = None
        self.raw_env = None
        self._scene = None
        self._trace = []
        self._last_legacy_state = None
        self._prior = None
        self._overload_seen = False
        self._mask_used_power = Decimal(0)

    def reset(self, seed=None, scene=None):
        if scene is None:
            scenario_seed = np.random.SeedSequence().generate_state(1)[0] if seed is None else seed
            with _seed_scope(scenario_seed):
                beam = legacy.BeamInfo(profile=self.region_profile)
            arrays = {key: np.array(beam.beam_info[key], copy=True) for key in SCENE_KEYS}
            arrays["original_indices"] = np.array(beam.shuffle_indices, copy=True)
            arrays = validate_scene(arrays, self.region_profile)
            scene_metadata = deepcopy(beam.generation_metadata) or dict(
                kind=self.region_profile.default_scene_kind, measured_traffic=False)
        else:
            arrays = validate_scene(scene, self.region_profile)
            scene_metadata = dict(kind="imported_scene", measured_traffic=None)
        self._scene = arrays
        self.scene_metadata = scene_metadata
        beam = SimpleNamespace(beam_info={key: np.array(arrays[key], copy=True) for key in SCENE_KEYS})
        frequency = legacy.SatelliteFreqState()
        current = legacy.CurrentBeamState(beam.beam_info, profile=self.region_profile)
        preset = self.config["reward"]["preset"]
        raw_variant = preset if preset in ("legacy_original", "legacy_repaired", "fused") else "legacy_repaired"
        budget = self.config["environment"]["power_budget_w"]
        with _budget_scope(budget):
            self.raw_env = legacy.Env(frequency, beam, current, reward_variant=raw_variant,
                                      basic_fraction=self.config["environment"]["basic_fraction"],
                                      profile=self.region_profile)
            self._last_legacy_state = self.raw_env.reset(beam)
        self._trace = []
        self._overload_seen = False
        self._mask_used_power = Decimal(0)
        self._prior = self._snapshot()
        return self._state()

    def scene(self):
        self._require_reset()
        return {key: np.array(value, copy=True) for key, value in self._scene.items()}

    def _require_reset(self):
        if self.raw_env is None:
            raise RuntimeError("请先 reset 环境")

    def _snapshot(self):
        env = self.raw_env
        count = env.beam_idx
        rates = np.maximum(np.asarray(env.beam_bps, dtype=np.float64), 0)
        demands = self._scene["rate"].reshape(-1)[:count]
        ratios = np.minimum(rates / demands, 1) if count else np.zeros(0)
        beta = float(self._scene["rate"].mean())
        all_ratios = np.zeros(env.beam_num, dtype=np.float64)
        all_ratios[:count] = ratios
        return dict(count=count, rates=rates.copy(), demands=demands.copy(), ratios=ratios,
                    si=float(ratios.sum()), basic=float(np.minimum(rates / demands / env.basic_fraction, 1).sum()) if count else 0.0,
                    sgm=float(_sgm(rates, demands, beta).sum()), throughput=float(rates.sum()) / 1e9,
                    useful_throughput=float(np.minimum(rates, demands).sum()) / 1e9,
                    jain=_jain(all_ratios), min_si=float(all_ratios.min()),
                    shortfall=float(self.raw_env.beam_num - ratios.sum()))

    def _state(self):
        env, legacy_state = self.raw_env, self._last_legacy_state
        idx = env.beam_idx
        count = env.beam_num
        parts = []
        base = {"beam": legacy_state[:3], "power": legacy_state[3:4],
                "interference": legacy_state[4:104], "occupancy": legacy_state[104:204]}
        ratios = self._snapshot()["ratios"]
        occupancy = env.satellite_freq_state.freq_pool_real
        group = idx % 8
        free = occupancy[group] == 0
        longest = max((len(x) for x in np.split(free, np.where(np.diff(free.astype(int)) != 0)[0] + 1) if len(x) and x[0]), default=0)
        definitions = {
            "progress": [idx / count], "remaining_count": [(count - idx) / count],
            "group": [group / 7],
            "power_per_remaining": [(self.config["environment"]["power_budget_w"] - env.total_power) / max(1, count - idx) / 30],
            "served_shortfall": [float((1 - ratios).mean()) if len(ratios) else 0],
            "served_met": [float(np.mean(ratios >= 1)) if len(ratios) else 0],
            "served_min": [float(ratios.min()) if len(ratios) else 0],
            "other_occupancy": occupancy[[g for g in range(8) if g != group]].reshape(-1),
            "longest_free": [longest / 100],
        }
        if self.config["state"]["demand_visibility"] == "known":
            demands = self._scene["rate"].reshape(-1)
            definitions.update(beta=[float(demands.mean()) / 6e8],
                               remaining_demand=[float(demands[idx:].sum()) / (count * 6e8)],
                               remaining_demand_mean=[float(demands[idx:].mean()) / 6e8 if idx < count else 0])
        for fid in self.config["state"]["features"]:
            value = np.asarray(base[fid] if fid in base else definitions[fid], dtype=np.float32).reshape(-1)
            parts.append(value)
        state = np.concatenate(parts)
        if state.size != self.summary["state_dim"] or not np.isfinite(state).all():
            raise ValueError("状态维度或数值无效")
        return state

    def _project(self, requested):
        """Choose the first jointly feasible contiguous allocation from requested start onward."""
        env = self.raw_env
        budget_left = self.config["environment"]["power_budget_w"] - env.total_power
        levels = self.config["environment"]["power_levels_w"]
        group = env.beam_idx % 8
        occupied = env.satellite_freq_state.freq_pool_real[group]
        freq, slots, power_idx = requested
        candidate_powers = [i for i in range(power_idx, -1, -1) if levels[i] <= budget_left]
        for f in list(range(freq, 100)) + list(range(0, freq)):
            free_run = 0
            for k in range(f, min(100, f + slots)):
                if occupied[k] > 0:
                    break
                free_run += 1
            if free_run and candidate_powers:
                return f, free_run, candidate_powers[0]
        # No positive-slot feasible action exists. The wrapper records an explicit
        # no-service outcome without passing a zero requested length to legacy step.
        return freq, 0, power_idx

    def action_mask(self):
        """Exact legal pairs and power levels from the current resource ledger.

        This is independent of the configurable observation layout. Slot-head
        indices 0..9 represent actual lengths 1..10. Future beams are reserved
        one slot in their own group and one minimum configured power level.
        """
        self._require_reset()
        if self.config["environment"].get("action_mask", "none") == "none":
            return None
        env = self.raw_env
        if env.beam_idx >= env.beam_num or self._trace and self._trace[-1]["done"]:
            return None
        remaining = env.beam_num - env.beam_idx
        remaining_group = len(range(env.beam_idx, env.beam_num, 8))
        occupied = np.asarray(env.satellite_freq_state.freq_pool_real[env.beam_idx % 8]) > 0
        max_slots = int(np.count_nonzero(~occupied)) - (remaining_group - 1)
        free_run = np.zeros(100, dtype=np.int64)
        run = 0
        for position in range(99, -1, -1):
            run = 0 if occupied[position] else run + 1
            free_run[position] = run
        lengths = np.arange(1, 11)
        frequency_slots = lengths[None, :] <= np.minimum(free_run, max_slots)[:, None]
        levels = self.config["environment"]["power_levels_w"]
        budget = float(self.config["environment"]["power_budget_w"])
        # Exact nominal watt ledger: a relative/absolute tolerance could admit a
        # genuinely unaffordable level for small/custom power grids. Actual dBW
        # conversion is conservatively below the selected nominal level at step.
        power_left = Decimal(str(budget)) - self._mask_used_power
        reserve = Decimal(str(levels[0])) * (remaining - 1)
        power = np.array([Decimal(str(level)) + reserve <= power_left for level in levels], dtype=bool)
        if not frequency_slots.any() or not power.any():
            raise RuntimeError(
                "无合法动作：不能在剩余资源下为每束保留一槽和最低功率；"
                f"beam={env.beam_idx}, free_slots={int((~occupied).sum())}, "
                f"remaining_group={remaining_group}, power_left={power_left:g}, "
                f"power_reserve={reserve:g}")
        return {"frequency_slots": frequency_slots, "power": power}

    def step(self, action_indices):
        self._require_reset()
        if self.raw_env.beam_idx >= self.raw_env.beam_num or self._trace and self._trace[-1]["done"]:
            raise RuntimeError("回合已结束，请 reset")
        if not isinstance(action_indices, (tuple, list, np.ndarray)) or len(action_indices) != 3:
            raise ValueError("动作必须有频率起点、槽数索引、功率索引三个整数")
        dims = self.summary["action_dims"]
        values = []
        for i, (value, dim) in enumerate(zip(action_indices, dims)):
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not 0 <= value < dim:
                raise ValueError(f"动作索引 {i} 超界")
            values.append(int(value))
        freq_idx, slot_idx, power_idx = values
        mask = self.action_mask()
        if mask is not None and (not mask["frequency_slots"][freq_idx, slot_idx]
                                 or not mask["power"][power_idx]):
            raise ValueError("动作不满足联合无效动作掩码；必须在策略采样前屏蔽非法组合")
        levels = self.config["environment"]["power_levels_w"]
        request = dict(group=self.raw_env.beam_idx % 8, freq=freq_idx, slots=slot_idx + 1,
                       power_w=float(levels[power_idx]), indices=values)
        before = self._prior
        mode = self.config["environment"]["action_mode"]
        projected = self._project((freq_idx, slot_idx + 1, power_idx)) if mode == "feasible_projection" else (freq_idx, slot_idx + 1, power_idx)
        p_freq, p_slots, p_power_idx = projected
        power_w = levels[p_power_idx]
        no_feasible = mode == "feasible_projection" and p_slots == 0
        power_db = 10 * np.log10(power_w)
        if mode == "feasible_projection" or mask is not None:
            power_db -= 1e-12  # Avoid floating-point round-up of an exactly feasible watt budget.
        action = [request["group"], p_freq, 1 if no_feasible else p_slots, power_db]
        with _budget_scope(self.config["environment"]["power_budget_w"]):
            if no_feasible:
                occupied = self.raw_env.satellite_freq_state.freq_pool_real
                previous_occupancy = float(occupied[request["group"], p_freq])
                previous_variant = self.raw_env.reward_variant
                occupied[request["group"], p_freq] = max(previous_occupancy, 1)
                self.raw_env.reward_variant = "legacy_repaired" if previous_variant == "legacy_original" else previous_variant
                try:
                    legacy_state, legacy_reward, done, raw_info = self.raw_env.step(action)
                finally:
                    occupied[request["group"], p_freq] = previous_occupancy
                    self.raw_env.reward_variant = previous_variant
                self.raw_env.occupy -= 1
                raw_info["occupy"] = 0
                if self.config["reward"]["preset"] in ("legacy_original", "legacy_repaired"):
                    legacy_reward += 1
            else:
                legacy_state, legacy_reward, done, raw_info = self.raw_env.step(action)
        if mask is not None:
            if (int(action[1]) != freq_idx or int(action[2]) != slot_idx + 1
                    or int(raw_info["out"]) or int(raw_info["occupy"]) or self.raw_env.overload):
                raise RuntimeError("合法动作执行后出现裁剪或超功率；停止并保留现场")
            if not math.isclose(float(raw_info["actual_power"]), float(levels[power_idx]),
                                rel_tol=1e-9, abs_tol=1e-9):
                raise RuntimeError("掩码动作的实际功率与所选功率档位不一致")
            self._mask_used_power += Decimal(str(levels[power_idx]))
        self._last_legacy_state = legacy_state
        after = self._snapshot()
        self._prior = after
        actual = dict(group=int(action[0]), freq=int(action[1]), slots=int(action[2]),
                      power_w=float(raw_info["actual_power"]), charged_power_w=float(raw_info["power"]),
                      selected_power_w=float(power_w), indices=[p_freq, (p_slots - 1) if p_slots else None, p_power_idx])
        result = self._reward(before, after, raw_info, legacy_reward, request, actual, done)
        state = self._state()
        info = _json(raw_info)
        info.update(reward_terms=result[1], reward_preset=self.config["reward"]["preset"],
                    requested_action=request, actual_action=actual, legacy_reward=legacy_reward,
                    projection_applied=mode == "feasible_projection" and projected != (freq_idx, slot_idx + 1, power_idx),
                    reward_total=result[0])
        info["action_mask"] = self.config["environment"].get("action_mask", "none")
        if mask is not None:
            info["legal_frequency_slot_pairs"] = int(mask["frequency_slots"].sum())
            info["legal_power_levels"] = int(mask["power"].sum())
            info["legal_action_count"] = info["legal_frequency_slot_pairs"] * info["legal_power_levels"]
        self._trace.append(dict(beam_id=int(raw_info["beam_id"]), requested_action=request,
                                actual_action=actual, reward=result[0], done=bool(done),
                                out=int(raw_info["out"]), occupy=int(raw_info["occupy"])))
        return state, result[0], bool(done), info

    def _reward(self, before, after, info, legacy_reward, requested, actual, done):
        preset = self.config["reward"]["preset"]
        if preset != "custom":
            if preset == "fused":
                components = {"basic_delta": after["basic"] - before["basic"],
                              "si_delta": after["si"] - before["si"],
                              "sgm_delta": after["sgm"] - before["sgm"],
                              "power_cost": -_float(info["actual_power"]) / self.config["environment"]["power_budget_w"]}
                weights = {"basic_delta": 2.0, "si_delta": 1.0,
                           "sgm_delta": 0.2, "power_cost": 1.0}
            else:
                increment = _float(info["satisfaction_increment"])
                sat = 1 if increment >= 1 else -1
                current_sgm = _float(info["sgm_rate"])
                current_rate = max(0.0, _float(self.raw_env.beam_bps[-1]))
                requirement = _float(info["required_rate"])
                gap = abs(current_rate - requirement)
                sgm_penalty = max(0.0, gap - 5e7) / 1e9 if gap > 5e7 else 0.0
                components = {"legacy_threshold": float(sat), "current_sgm": current_sgm,
                              "legacy_sgm_penalty": -sgm_penalty,
                              "occupy_penalty": -_float(info["occupy"]) / max(1, _float(info["origin_slots"]))}
                weights = {key: 1.0 for key in components}
            total = float(legacy_reward)
            # Preserve the raw environment's exact reward, including any numerical residual.
            residual = total - sum(value * weights[key] for key, value in components.items())
            if abs(residual) > 1e-10:
                components["legacy_residual"] = residual
                weights["legacy_residual"] = 1.0
            return total, {key: dict(raw=value, scaled=value, weighted=value * weights[key], timing="step")
                           for key, value in components.items()}
        budget = self.config["environment"]["power_budget_w"]
        this_rate = max(0.0, _float(self.raw_env.beam_bps[-1]))
        demand = _float(info["required_rate"])
        prior_rates = before["rates"]
        changed_prior = after["rates"][:len(prior_rates)]
        interference_loss = max(0.0, float((prior_rates - changed_prior).sum())) / 1e9
        excess = max(0.0, this_rate - demand) / 1e9
        raw = {
            "si_delta": after["si"] - before["si"],
            "basic_delta": after["basic"] - before["basic"],
            "sgm_delta": after["sgm"] - before["sgm"],
            "current_si": min(this_rate / demand, 1),
            "current_sgm": float(_sgm([this_rate], [demand], float(self._scene["rate"].mean()))[0]),
            "throughput_delta": after["throughput"] - before["throughput"],
            "useful_throughput_delta": after["useful_throughput"] - before["useful_throughput"],
            "jain_delta": after["jain"] - before["jain"],
            "min_si_delta": after["min_si"] - before["min_si"],
            "shortfall_delta": before["shortfall"] - after["shortfall"],
            "excess_supply": -excess,
            "power_cost": -_float(info["actual_power"]) / budget,
            "slot_cost": -actual["slots"] / 10,
            "interference_loss": -interference_loss,
            "out_penalty": -_float(info["out"]) / requested["slots"],
            "occupy_penalty": -_float(info["occupy"]) / requested["slots"],
            "clip_penalty": -(requested["slots"] - actual["slots"]) / requested["slots"],
            "overload_penalty": -max(0, self.raw_env.total_power - budget) / budget if self.raw_env.overload and not self._overload_seen else 0,
            "zero_slot_penalty": -1 if actual["slots"] == 0 else 0,
            "terminal_si": after["si"] / self.raw_env.beam_num if done else 0,
            "terminal_basic": after["basic"] / self.raw_env.beam_num if done else 0,
            "terminal_sgm": after["sgm"] / self.raw_env.beam_num if done else 0,
            "terminal_throughput": after["throughput"] if done else 0,
            "terminal_jain": _jain(self._all_ratios()) if done else 0,
            "terminal_min_si": float(self._all_ratios().min()) if done else 0,
            "incomplete_penalty": -(self.raw_env.beam_num - self.raw_env.beam_idx) / self.raw_env.beam_num if done else 0,
        }
        self._overload_seen = self.raw_env.overload
        terms, total = {}, 0.0
        for entry in self.config["reward"]["terms"]:
            value = float(raw[entry["id"]])
            scaled = value / entry["scale"]
            weighted = entry["weight"] * scaled
            terms[entry["id"]] = dict(raw=value, scaled=scaled, weighted=weighted, timing=entry["timing"])
            total += weighted
        return float(total), terms

    def _all_ratios(self):
        rates = np.zeros(self.raw_env.beam_num, dtype=np.float64)
        rates[:len(self.raw_env.beam_bps)] = np.maximum(self.raw_env.beam_bps, 0)
        return np.minimum(rates / self._scene["rate"].reshape(-1), 1)

    def beam_results(self):
        self._require_reset()
        env = self.raw_env
        rates = np.zeros(env.beam_num)
        rates[:len(env.beam_bps)] = np.maximum(env.beam_bps, 0)
        demands = self._scene["rate"].reshape(-1)
        beta = float(demands.mean())
        sgm = _sgm(rates, demands, beta)
        actions = {int(row[0]): row for row in env.history_beam}
        result = []
        for i in range(env.beam_num):
            row = actions.get(i)
            result.append(dict(beam_id=i, original_index=int(self._scene["original_indices"][i]),
                               lat=float(self._scene["beam_location"][i, 0]),
                               lon=float(self._scene["beam_location"][i, 1]),
                               required_bps=float(demands[i]), rate_bps=float(rates[i]),
                               si=float(min(rates[i] / demands[i], 1)), sgm=float(sgm[i]),
                               group=int(row[1]) if row else i % 8,
                               freq=int(row[2]) if row else None,
                               slots=int(row[3]) if row else 0,
                               power_w=float(10 ** (row[4] / 10)) if row and row[3] > 0 else 0.0,
                               basic_met=bool(rates[i] >= env.basic_fraction * demands[i])))
        return result

    def metrics(self):
        self._require_reset()
        env = self.raw_env
        beams = self.beam_results()
        rates = np.array([b["rate_bps"] for b in beams])
        ratios = np.array([b["si"] for b in beams])
        sgm = np.array([b["sgm"] for b in beams])
        requested_slots = sum(t["requested_action"]["slots"] for t in self._trace)
        actual_slots = sum(t["actual_action"]["slots"] for t in self._trace)
        zero_slots = sum(t["actual_action"]["slots"] == 0 for t in self._trace)
        clipped = sum(t["requested_action"]["slots"] != t["actual_action"]["slots"]
                      or t["requested_action"]["freq"] != t["actual_action"]["freq"]
                      or not math.isclose(t["requested_action"]["power_w"], t["actual_action"]["power_w"], rel_tol=1e-9)
                      for t in self._trace)
        throughput = float(rates.sum())
        return dict(beam_count=env.beam_num, processed_count=env.beam_idx,
                    complete=env.beam_idx == env.beam_num, overload=bool(env.overload),
                    final_si_total=float(ratios.sum()), final_si_mean=float(ratios.mean()),
                    final_sgm_total=float(sgm.sum()), final_sgm_mean=float(sgm.mean()),
                    basic_met_count=sum(b["basic_met"] for b in beams),
                    zero_capacity_count=int(np.sum(rates <= 0)),
                    jain=float(_jain(ratios)), throughput_bps=throughput,
                    power_w=float(env.total_power), power_accounted_w=float(env.total_power),
                    radiated_power_w=sum(t["actual_action"]["power_w"] for t in self._trace),
                    power_budget_w=float(self.config["environment"]["power_budget_w"]),
                    requested_slots=requested_slots, actual_slots=actual_slots, allocated_slots=actual_slots,
                    occupancy_fraction=float(np.mean(env.satellite_freq_state.freq_pool_real)),
                    spectral_efficiency=throughput / (actual_slots * 2.5e7) if actual_slots else 0.0,
                    interference_beams=int(env.interfere_num),
                    in_range_interference_beams=int(env.interfere_in_range_num),
                    zero_slot_count=zero_slots,
                    zero_slot_rate=zero_slots / env.beam_idx if env.beam_idx else 0.0,
                    out_slots=sum(t["out"] for t in self._trace),
                    occupied_slots=sum(t["occupy"] for t in self._trace),
                    clipped_count=clipped,
                    termination_reason="overload" if env.overload else "complete" if env.beam_idx == env.beam_num else "in_progress")
