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
from satellite_rl.state_layouts import natural_shapes
from satellite_rl.workbench.config import (FEATURES, config_summary, natural_state_width,
                                           validate_config)


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


def validate_scene(scene, profile="china_legacy", max_beams=200):
    """Copy and check a scene in processing order, without padding beams."""
    if not isinstance(scene, dict):
        raise ValueError("场景必须是数组映射")
    if set(scene) != set(SCENE_KEYS) | {"original_indices"}:
        raise ValueError("场景字段不完整或含未知字段")
    rate_shape = np.asarray(scene["rate"]).shape
    if not rate_shape:
        raise ValueError("场景rate必须是逐束数组")
    count = rate_shape[0]
    if not 0 <= count <= max_beams <= 200:
        raise ValueError("目标波束数量须在0至配置上限之间，且不得超过200")
    out = {}
    shapes = {"beam_location": (count, 2), "beam_G_Tx_max": (count, 1),
              "UE_G_Rx_max": (count, 1), "distances": (count,),
              "rate": (count, 1), "noise_temperature": (count, 1),
              "original_indices": (count,)}
    for key, shape in shapes.items():
        arr = np.asarray(scene[key])
        if arr.shape != shape or not (np.issubdtype(arr.dtype, np.integer) or np.issubdtype(arr.dtype, np.floating)) or not np.isfinite(arr).all():
            raise ValueError(f"场景字段 {key} 形状或数值无效，应为 {shape}")
        out[key] = np.array(arr, dtype=np.int64 if key == "original_indices" else np.float64, copy=True)
    if count == 0:
        return out  # No business: there is no allocation, SI or SGM denominator.
    indices = np.asarray(scene["original_indices"])
    if not np.array_equal(indices, out["original_indices"]) or set(out["original_indices"].tolist()) != set(range(count)):
        raise ValueError(f"original_indices 必须是0至{count - 1}的完整排列")
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


def beam_service_fields(rate_bps, required_bps, sgm):
    """One beam's physical service, retaining supply above demand.

    ``rate_bps`` is the raw environment's allocated capacity after the current
    interference update. SI is capped; the capacity/demand ratio is not.
    """
    raw_capacity = float(rate_bps)
    demand = float(required_bps)
    matching = float(sgm)
    if (not math.isfinite(raw_capacity) or not math.isfinite(demand)
            or not math.isfinite(matching) or demand <= 0 or not 0 <= matching <= 1):
        raise ValueError("Beam capacity, demand, or SGM is invalid")
    capacity = max(0.0, raw_capacity)
    ratio = capacity / demand
    return {"required_bps": demand, "capacity_bps": capacity,
            "rate_bps": capacity,  # Existing result consumers use this name.
            "effective_served_bps": min(capacity, demand),
            "capacity_demand_ratio": ratio, "si": min(ratio, 1.0),
            "sgm": matching}


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
        self._previous_actual_action = np.zeros(4, dtype=np.float32)
        self._no_business = False

    def reset(self, seed=None, scene=None):
        max_beams = self.config["environment"]["beam_count_max"]
        if scene is None:
            scenario_seed = np.random.SeedSequence().generate_state(1)[0] if seed is None else seed
            with _seed_scope(scenario_seed):
                beam = legacy.BeamInfo(profile=self.region_profile, count=100)
            arrays = {key: np.array(beam.beam_info[key], copy=True) for key in SCENE_KEYS}
            arrays["original_indices"] = np.array(beam.shuffle_indices, copy=True)
            arrays = validate_scene(arrays, self.region_profile, max_beams=max_beams)
            scene_metadata = deepcopy(beam.generation_metadata) or dict(
                kind=self.region_profile.default_scene_kind, measured_traffic=False)
        else:
            arrays = validate_scene(scene, self.region_profile, max_beams=max_beams)
            scene_metadata = dict(kind="imported_scene", measured_traffic=None)
        if len(arrays["rate"]) == 0:
            self._scene = arrays
            self.scene_metadata = dict(scene_metadata, no_business=True)
            self.raw_env = None
            self._no_business = True
            self._trace = []
            self._prior = None
            return np.zeros(self.summary["state_dim"], dtype=np.float32)
        self._no_business = False
        if self.config["network"].get("graph_mode", "none") == "interference_bias":
            from satellite_rl.agents.graph_encoder import validate_physical_contract
            validate_physical_contract(arrays, self.config, {
                "beam_width_deg": legacy.bw,
                "carrier_start_hz": legacy.f_min,
                # The raw PHY uses this literal for every slot bandwidth term.
                "slot_bandwidth_hz": 2.5e7,
                "light_speed_m_s": legacy.c,
                "transmit_gain_dbi": float(arrays["beam_G_Tx_max"][0, 0]),
                "receive_gain_dbi": float(arrays["UE_G_Rx_max"][0, 0]),
                "latitude_bounds_deg": self.region_profile.latitude_bounds_deg,
                "longitude_bounds_deg": self.region_profile.longitude_bounds_deg,
                "satellite_longitude_deg": self.region_profile.satellite_longitude_deg,
                "earth_radius_km": self.region_profile.earth_radius_km,
                "orbit_radius_km": self.region_profile.orbit_radius_km,
            })
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
        self._previous_actual_action = np.zeros(4, dtype=np.float32)
        self._overload_seen = False
        self._mask_used_power = Decimal(0)
        self._prior = self._snapshot()
        return self._state()

    def scene(self):
        self._require_reset()
        return {key: np.array(value, copy=True) for key, value in self._scene.items()}

    def _require_reset(self):
        if self.raw_env is None and not self._no_business:
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
                    si=float(ratios.sum()), mean_si=float(ratios.mean()) if count else 0.0,
                    basic=float(np.minimum(rates / demands / env.basic_fraction, 1).sum()) if count else 0.0,
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
        power_budget = self.config["environment"]["power_budget_w"]
        # S2-P18 修正2：原来的 30 W 是 3000W/100束写死的“每束参考功率”，N≠100 时
        # 系统性低估大束数的资源紧张程度（方案18 第 2.4 节）。这里改成随束数变化的
        # 公平份额 budget/N；N=100 时 3000/100 与原来的 30 是同一个浮点数，因此
        # 100 束逐位不变。这是纠正参考量，不是新增信息。
        power_reference_w = power_budget / count
        # 当前组剩余预留义务：与掩码 action_mask 的 remaining_group 同一口径，
        # 即包含当前束在内、本复用组还需要处理多少束。
        count_remaining = count - idx
        group_remaining = len(range(idx, count, 8))
        # 剩余合法槽上限：与掩码 max_slots 同一口径，先数本组未被占用的位置，
        # 再扣掉为同组其余波束各预留的 1 槽；负值表示当前束已无槽可分。
        group_free_slots = float(np.count_nonzero(~(np.asarray(occupancy[group]) > 0)))
        slots_available = max(0.0, group_free_slots - (group_remaining - 1))
        definitions = {
            "progress": [idx / count], "remaining_count": [count_remaining / count],
            "group": [group / 7],
            "power_per_remaining": [(power_budget - env.total_power) / max(1, count_remaining) / 30],
            "served_shortfall": [float((1 - ratios).mean()) if len(ratios) else 0],
            "served_met": [float(np.mean(ratios >= 1)) if len(ratios) else 0],
            "served_min": [float(ratios.min()) if len(ratios) else 0],
            "other_occupancy": occupancy[[g for g in range(8) if g != group]].reshape(-1),
            "longest_free": [longest / 100],
            "previous_actual_action": self._previous_actual_action,
            "group_remaining_obligation": [group_remaining / count],
            "remaining_slot_limit": [slots_available / 100],
        }
        power_left = power_budget - env.total_power
        if power_left < -1e-7 and "power_pressure" in self.config["state"]["features"]:
            raise RuntimeError("剩余功率为负，资源账错误")
        remaining = count_remaining
        definitions.update(
            beam_count_normalized=[count / 200],
            remaining_beams_normalized=[remaining / 200],
            power_pressure=[power_left / (power_left + power_reference_w * remaining) if remaining else 0],
        )
        if self.config["state"]["demand_visibility"] == "known":
            demands = self._scene["rate"].reshape(-1)
            definitions.update(beta=[float(demands.mean()) / 6e8],
                               remaining_demand=[float(demands[idx:].sum()) / (count * 6e8)],
                               remaining_demand_mean=[float(demands[idx:].mean()) / 6e8 if idx < count else 0])
        features = self.config["state"]["features"]
        if "global_context" in features:
            definitions.update(self._structured_context(definitions, ratios))
        elif "natural_global_context" in features:
            context = self._structured_context(definitions, ratios)
            level = self.config["state"]["context_level"]
            definitions["natural_global_context"] = context["global_context"][:3 if level == "compact" else 10]
            if "natural_beam_context" in features:
                beam_rows = context["beam_context"]
                beam_width = context["beam_width"]
                definitions["natural_beam_context"] = (
                    beam_rows[:, :, :beam_width] if beam_rows.ndim == 3
                    else beam_rows[:, :beam_width])
            if "natural_resource_context" in features:
                definitions["natural_resource_context"] = context["resource_context"][:, :204 if level == "full" else 4]
        for fid in self.config["state"]["features"]:
            value = np.asarray(base[fid] if fid in base else definitions[fid], dtype=np.float32).reshape(-1)
            parts.append(value)
        state = np.concatenate(parts)
        if state.size != self.state_width(count) or not np.isfinite(state).all():
            raise ValueError("状态维度或数值无效")
        return state

    def state_width(self, count):
        """本场景实际观测宽度：仅“按实际束数构造”的对照臂随 N 变化。"""
        return natural_state_width(self.config, count)

    def _structured_context(self, definitions, ratios):
        """Build four visibility levels from information available before the action."""
        env = self.raw_env
        idx, count = env.beam_idx, env.beam_num
        level = self.config["state"]["context_level"]
        # S2-P18 手段G：可选按场景实际束数构造逐束字段（context_beam_count="actual"）。
        # 默认档位仍用固定 (10,10,11) 的 100 束布局，所以既有对照逐位不变。
        actual_count = self.config["network"].get("context_beam_count", "fixed_100") == "actual"
        beam_width = 5 if level == "layout" else 11
        # 逐束编码档位按实际束数构造；默认档位保持历史 (10,10,11) 的 100 束布局。
        beam_rows = count if actual_count else 100
        global_context = np.zeros(10, dtype=np.float32)
        # 注意：这里始终按 11 通道分配，实际写入观测时再按档位截取到 beam_width 列，
        # 与历史实现（layout 档取前 5 列）保持一致。
        beam_context = np.zeros((beam_rows, 11), dtype=np.float32)
        resource_context = np.zeros((8, 204), dtype=np.float32)

        global_context[:3] = (definitions["beta"][0], definitions["progress"][0],
                              definitions["remaining_demand"][0])
        if level in ("summary", "layout", "full"):
            global_context[3:] = (
                definitions["remaining_count"][0], definitions["group"][0],
                definitions["power_per_remaining"][0], definitions["served_shortfall"][0],
                definitions["served_met"][0], definitions["served_min"][0],
                definitions["longest_free"][0],
            )
            if idx == count:
                # There is no next beam at the terminal state. Keep aggregate
                # service/resource records, but zero current-beam-only fields.
                global_context[[4, 5, 9]] = 0
            occupancy = np.asarray(env.satellite_freq_state.freq_pool_real)
            for group in range(8):
                occupied = occupancy[group] > 0
                free = ~occupied
                longest = max((len(x) for x in np.split(free, np.flatnonzero(np.diff(free.astype(int))) + 1)
                               if len(x) and x[0]), default=0)
                historical = [item for item in env.history_beam if item[1] == group and item[3] > 0]
                group_ids = np.arange(group, idx, 8)
                resource_context[group, :4] = (
                    float(occupied.mean()), longest / 100,
                    sum(10 ** (entry[4] / 10) for entry in historical)
                    / self.config["environment"]["power_budget_w"],
                    float(ratios[group_ids].mean()) if len(group_ids) else 0,
                )

        if level in ("layout", "full"):
            locations = self._scene["beam_location"]
            lat_min, lat_max = self.region_profile.latitude_bounds_deg
            lon_min, lon_max = self.region_profile.longitude_bounds_deg
            rows = beam_context.reshape(beam_rows, 11)[:, :beam_width]
            rows[:count, 0] = 2 * (locations[:, 0] - (lat_max + lat_min) / 2) / (lat_max - lat_min)
            rows[:count, 1] = 2 * (locations[:, 1] - (lon_max + lon_min) / 2) / (lon_max - lon_min)
            rate_mean = (legacy.rate_min + legacy.rate_max - legacy.rate_step) / 2 * 1e6
            demands = self._scene["rate"].reshape(-1)
            rows[:count, 2] = demands / rate_mean
            rows[:count, 3] = np.arange(count) % 8 / 7
            rows[:count, 4] = np.arange(count) / (count - 1)

        if level == "full":
            rows = beam_context.reshape(beam_rows, 11)[:, :beam_width]
            rows[:idx, 5] = 1
            rates = np.maximum(np.asarray(env.beam_bps, dtype=np.float64), 0)
            rows[:idx, 9] = rates / 6e8
            rows[:idx, 10] = ratios
            for beam_id, group, frequency, slots, power_db in env.history_beam:
                if not 0 <= beam_id < idx or group != beam_id % 8:
                    raise ValueError("历史波束记录与处理顺序不一致")
                rows[beam_id, 6:9] = (
                    frequency / 99, slots / 10,
                    (10 ** (power_db / 10) if slots > 0 else 0)
                    / max(self.config["environment"]["power_levels_w"]),
                )
            occupancy = np.asarray(env.satellite_freq_state.freq_pool_real, dtype=np.float32)
            resource_context[:, 4:104] = occupancy
            if idx < count:
                # Interference depends on the prospective group's polarization,
                # so the eight groups share two exact historical calculations.
                for polarization in (0, 1):
                    watts = env._calc_group_slot_interference(polarization)
                    db = 10 * np.log10(watts + 1e-30)
                    normalized = np.clip((db - legacy.P_MIN_DB) /
                                         (legacy.P_MAX_DB - legacy.P_MIN_DB), 0, 1)
                    resource_context[polarization::2, 104:204] = normalized
        return {"global_context": global_context,
                "beam_context": beam_context,
                "beam_width": beam_width,
                "resource_context": resource_context}

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
        if self._no_business:
            return None
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
        if self._no_business:
            raise RuntimeError("无业务场景没有可执行动作")
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
        self._previous_actual_action = np.asarray(
            [actual["freq"] / 99, actual["slots"] / 10, actual["power_w"] / 50, 1],
            dtype=np.float32)
        result = self._reward(before, after, raw_info, legacy_reward, request, actual, done)
        state = self._state()
        info = _json(raw_info)
        # raw_info.current_satisfaction is capped and raw_info.sgm_rate is a
        # reward term. Neither can reconstruct physical capacity or service SGM.
        current_rate = self.raw_env.beam_bps[-1]
        current_demand = self._scene["rate"].reshape(-1)[int(raw_info["beam_id"])]
        current_sgm = _sgm([current_rate], [current_demand],
                           float(self._scene["rate"].mean()))[0]
        info["beam_service_step"] = beam_service_fields(current_rate, current_demand, current_sgm)
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

    def _throughput_reference_gbps(self):
        """ΔΣmin(C,D) 的归一化参考量（方案18 第 3 节手段A）。

        默认取“全场景平均每束需求”除以 1e9。选择理由：该参考量就是逐束满足率的
        分母自身的平均，于是归一化后的项恰好等于“本步新增的有效承载相当于多少束
        的平均需求”，与 N 无关、与场景负载同尺度，权重可以直接读成"每多做满一束
        业务值多少分"。也可切换为场景总需求（总量口径）或固定数值 Gbps。
        """
        shaping = self.config["reward"].get("shaping", {})
        mode = shaping.get("throughput_reference", "mean_demand")
        if isinstance(mode, (int, float)) and not isinstance(mode, bool):
            if mode <= 0:
                raise ValueError("吞吐归一化参考量必须是正数")
            return float(mode)
        demands = self._scene["rate"].reshape(-1)
        if mode == "mean_demand":
            return float(demands.mean()) / 1e9
        if mode == "total_demand":
            return float(demands.sum()) / 1e9
        raise ValueError(f"未知吞吐归一化参考量: {mode!r}")

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
            if preset == "legacy_repaired_throughput":
                self._add_throughput_terms(components, weights, before, after, info, actual)
                total += sum(value * weights[key] for key, value in components.items()
                             if key in ("throughput_delta", "spectrum_occupancy"))
            total += self._shape_reward(components, weights, before, after)
            return total, self._components(components, weights, preset)
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
        total += self._shape_reward(components=None, weights=None, before=before, after=after,
                                    extra=terms)
        return float(total), terms

    @staticmethod
    def _components(components, weights, preset):
        return {key: dict(raw=value, scaled=value, weighted=value * weights[key], timing="step",
                          preset=preset, enable="legacy_repaired")
                for key, value in components.items()}

    def _add_throughput_terms(self, components, weights, before, after, info, actual):
        """S2-P18 手段A：在保留原四项的前提下追加“用起频谱/多做业务量”的正项。

        默认权重为 0，只有把 preset 设为 ``legacy_repaired_throughput`` 并显式给出
        正权重时才生效，因此 ``legacy_repaired`` 的返回奖励逐位不变。两项都只读取
        本步已经发生的结果，不改变物理、动作可行性或终态统计：

        - ``throughput_delta``：本步有效承载增量 ΔΣmin(C,D)，除以参考量。参考量默认
          取“全场景平均每束需求”，见 :meth:`_throughput_reference_gbps` 的说明。
        - ``spectrum_occupancy``：本步实际占用槽数 / 当前组可用槽数。分母取
          ``该组100个位置 − 同组其余波束各预留1槽``，与动作掩码的可行槽口径一致；
          这样“把空槽用起来”与“给后续束留出义务”在同一个尺度上比较。
        """
        shaping = self.config["reward"].get("shaping", {})
        w_throughput = float(shaping.get("w_throughput", 0))
        w_spectrum = float(shaping.get("w_spectrum", 0))
        if w_throughput:
            reference = self._throughput_reference_gbps()
            components["throughput_delta"] = (
                after["useful_throughput"] - before["useful_throughput"]) / reference
            weights["throughput_delta"] = w_throughput
        if w_spectrum:
            group = int(info["group"])
            remaining_group = len(range(self.raw_env.beam_idx, self.raw_env.beam_num, 8))
            available = 100.0 - (remaining_group - 1)
            if available <= 0:
                raise RuntimeError("当前组可用槽上限非正，资源账错误")
            components["spectrum_occupancy"] = float(actual["slots"]) / available
            weights["spectrum_occupancy"] = w_spectrum

    def _shape_reward(self, components, weights, before, after, extra=None):
        """S2-P18 手段F：跨束势能型奖励整形（potential-based shaping）。

        势函数（只依赖状态，不依赖动作）：

            Φ(s) = mean_i min(C_i / D_i, 1)

        其中 i 取该场景全部目标波束，未服务束的 C_i 记 0；C_i 取每步干扰更新后的
        逐束链路容量 ``raw_env.beam_bps``，D_i 为场景需求。均值的分母固定为场景束数
        N，所以 Φ 的量纲是“平均需求满足率”，与 N 无关。

        整形项为 ``w_shape × (γ·Φ(s') − Φ(s))``，γ 取算法实际使用的折扣。这是标准的
        势能型整形（Ng、Harada 与 Russell，1999）：对任意折扣与任意 MDP，回报中该
        差分沿轨迹求和后只剩端点势函数 ``γ^(T-1)·Φ(s_T) − Φ(s_0)``，与策略无关，
        因此最优策略集合不变。实现层面该性质由“只读状态、不读动作、只按相邻状态
        差分相加”保证，并由 ``tests/test_s2p18_arms.py`` 的确定性小例核对。
        """
        weight = float(self.config["reward"].get("shaping", {}).get("w_shape", 0))
        if not weight:
            return 0.0
        gamma = float(self.train_gamma())
        term = gamma * float(after["mean_si"]) - float(before["mean_si"])
        if extra is not None:
            extra["cross_beam_potential"] = dict(raw=term, scaled=term, weighted=weight * term,
                                                 timing="step")
        else:
            components["cross_beam_potential"] = term
            weights["cross_beam_potential"] = weight
        return weight * term

    def train_gamma(self):
        """当前回合实际使用的折扣；由 runner 按束数日程写入。"""
        return float(getattr(self, "_effective_gamma", self.config["training"]["gamma"]))

    def _all_ratios(self):
        rates = np.zeros(self.raw_env.beam_num, dtype=np.float64)
        rates[:len(self.raw_env.beam_bps)] = np.maximum(self.raw_env.beam_bps, 0)
        return np.minimum(rates / self._scene["rate"].reshape(-1), 1)

    def beam_results(self):
        self._require_reset()
        if self._no_business:
            return []
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
                               **beam_service_fields(rates[i], demands[i], sgm[i]),
                               group=int(row[1]) if row else i % 8,
                               freq=int(row[2]) if row else None,
                               slots=int(row[3]) if row else 0,
                               power_w=float(10 ** (row[4] / 10)) if row and row[3] > 0 else 0.0,
                               basic_met=bool(rates[i] >= env.basic_fraction * demands[i])))
        return result

    def metrics(self, beams=None):
        self._require_reset()
        if self._no_business:
            return dict(beam_count=0, processed_count=0, complete=True,
                        termination_reason="no_business", final_si_total=None,
                        final_si_mean=None, final_sgm_total=None,
                        final_sgm_mean=None, basic_met_count=None,
                        zero_capacity_count=None, power_w=0.0,
                        power_budget_w=float(self.config["environment"]["power_budget_w"]))
        env = self.raw_env
        beams = self.beam_results() if beams is None else beams
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
