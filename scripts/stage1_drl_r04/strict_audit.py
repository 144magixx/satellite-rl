"""Read-only geometry and service-algebra replay for D12A archived scenes.

GEO footprint verification reuses the project's pure ray geometry. RF link
capacity is checked algebraically against logged values, not recomputed from
the off-axis interference model; that scope is explicit in every result.
"""

from __future__ import annotations

import io
import json
import math
from types import SimpleNamespace

import numpy as np

from satellite_rl.coverage.drl_r04_core import r04_objective
from satellite_rl.coverage.variable_config import VariableConfig
from satellite_rl.coverage.variable_design import Layout, metrics


def _close(a, b, *, atol=1e-6, rtol=1e-8):
    return (isinstance(a, (int, float, np.number)) and
            isinstance(b, (int, float, np.number)) and
            np.isfinite(a) and np.isfinite(b) and
            abs(float(a) - float(b)) <= atol + rtol * max(abs(float(a)), abs(float(b))))


def scenes_from_npz(data):
    """Decode frozen variable_aircraft_v1 NPZ bytes without extracting ZIP."""
    with np.load(io.BytesIO(data), allow_pickle=False) as archive:
        if str(archive["format_version"]) != "variable_aircraft_v1":
            raise ValueError("strict audit requires frozen variable_aircraft_v1 input")
        lengths = archive["lengths"].copy()
        positions = archive["positions"].copy()
        demands = archive["demands"].copy()
        ids = json.loads(str(archive["aircraft_ids_json"]))
        scene_ids = archive["scene_ids"].tolist()
        hashes = archive["source_hashes"].tolist()
    if (positions.shape[:2] != demands.shape or positions.shape[2:] != (2,) or
            len(lengths) != len(scene_ids) or len(ids) != len(scene_ids) or
            len(hashes) != len(scene_ids)):
        raise ValueError("frozen NPZ scene array shape differs")
    scenes = []
    for i, length in enumerate(lengths):
        count = int(length)
        if count < 0 or count > demands.shape[1] or len(ids[i]) != count or len(set(ids[i])) != count:
            raise ValueError("frozen NPZ aircraft count or identity differs")
        scenes.append(SimpleNamespace(scene_id=str(scene_ids[i]),
                                      source_hash=str(hashes[i]),
                                      aircraft_ids=list(ids[i]),
                                      positions=positions[i, :count].copy(),
                                      demands=demands[i, :count].copy()))
    return scenes


def _sgm(capacity, demand, beta):
    if capacity <= 0:
        return 0.0
    ratio = capacity / demand
    real = ratio - 1 if ratio >= 1 else 1 - 1 / ratio
    imag = (capacity - demand) / beta
    return 1 - (1 - math.exp(-abs(complex(real, imag)))) ** 3


def audit_scene_row(scene, row):
    errors = []
    facts = {"geometry_version": "shared_exact_satellite_ray_europe_v1",
             "rf_scope": "logged_capacity_service_algebra_only_no_off_axis_recompute"}
    cfg = VariableConfig(max_beams=200)
    scaled = SimpleNamespace(scene_id=scene.scene_id, source_hash=scene.source_hash,
                             aircraft_ids=scene.aircraft_ids, positions=scene.positions,
                             demands=np.asarray(scene.demands, dtype=float) * row.get("demand_multiplier", 1))
    if row.get("scene_id") != scene.scene_id or row.get("scene_sha256") != scene.source_hash:
        errors.append("frozen_scene_identity_differs")
    if len(scaled.demands) == 0 or not np.isfinite(scaled.demands).all() or np.any(scaled.demands < 0):
        errors.append("frozen_scene_demand_invalid")
        return sorted(set(errors)), facts
    layouts = {}
    for name, field in (("initial", "initial_layout"), ("current", "raw_terminal_layout"),
                        ("best", "best_layout")):
        try:
            layout = Layout.from_record(row[field])
            if len(layout.centers) > 200 or not np.allclose(layout.widths, .5, atol=1e-12):
                errors.append(f"{name}_N_or_fixed_width_invalid")
            if len(layout.assignment) != len(scaled.demands) or np.any(layout.assignment < 0):
                errors.append(f"{name}_unique_assignment_incomplete")
            result = metrics(scaled, layout, cfg)
            if not result["coverage_100"] or not result["demand_conserved"] or result["empty_beams"]:
                errors.append(f"{name}_exact_geometry_or_demand_invalid")
            score = r04_objective(layout, scaled.demands, cfg.reference_capacity_bps, 200)
            target = row.get({"initial": "initial_j", "current": "current_j", "best": "best_j"}[name])
            if not _close(score, target, atol=1e-10):
                errors.append(f"{name}_J_replay_differs")
            layouts[name] = (layout, result)
            facts[name] = {"N": len(layout.centers), "load_cv": result["load_cv"],
                           "J_replayed": score, "N_lt_50": len(layout.centers) < 50,
                           "N_eq_200": len(layout.centers) == 200}
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            errors.append(f"{name}_layout_replay_failed:{type(exc).__name__}")
    if "best" not in layouts:
        return sorted(set(errors)), facts
    best_layout, best_metrics = layouts["best"]
    physical = row.get("terminal_physics", {})
    digest = physical.get("layout_aliases", {}).get("best")
    measured = physical.get("layouts_by_sha256", {}).get(digest, {})
    beams = measured.get("beams", [])
    if len(beams) != len(best_layout.centers):
        errors.append("best_physical_beam_count_differs")
        return sorted(set(errors)), facts
    if measured.get("metrics", {}).get("complete_allocation") is not True:
        errors.append("best_downstream_allocation_incomplete")
    demand = np.asarray(best_metrics["beam_demand_bps"], dtype=float)
    beta = float(np.mean(demand[demand > 0])) if np.any(demand > 0) else None
    # Load balance is relative beam demand, not the capacity/demand ratio.
    demand_band = int(np.count_nonzero((demand > 0) & (demand >= .8 * beta) &
                                       (demand <= 1.2 * beta))) if beta else 0
    capacity_ratio_band = 0
    slots_used = set()
    total_power = 0.0
    for b, beam in enumerate(beams):
        d, c, carried = beam.get("demand_bps"), beam.get("capacity_bps"), beam.get("carried_bps")
        if not all(isinstance(x, (int, float)) and math.isfinite(x) and x >= 0 for x in (d, c, carried)):
            errors.append("best_beam_D_C_carried_invalid")
            continue
        if not _close(d, demand[b], atol=.01) or not _close(carried, min(c, d), atol=.01):
            errors.append("best_beam_demand_or_min_capacity_differs")
        if d > 0:
            ratio = c / d
            capacity_ratio_band += .8 <= ratio <= 1.2
            if not _close(beam.get("unclipped_capacity_demand_ratio"), ratio, atol=1e-8) or not _close(beam.get("si"), min(ratio, 1), atol=1e-8):
                errors.append("best_beam_ratio_or_SI_differs")
            if beta is None or not _close(beam.get("sgm"), _sgm(c, d, beta), atol=1e-8):
                errors.append("best_beam_SGM_formula_differs")
        elif any(beam.get(key) is not None for key in ("unclipped_capacity_demand_ratio", "si", "sgm")):
            errors.append("zero_demand_beam_service_not_null")
        slots, power, freq, group = (beam.get(key) for key in ("slots", "power_w", "freq", "group"))
        if d > 0:
            if (not isinstance(slots, int) or not 1 <= slots <= 10 or
                    not isinstance(freq, int) or not 0 <= freq < 100 or freq + slots > 100 or
                    not isinstance(group, int) or not 0 <= group < 8 or
                    not isinstance(power, (int, float)) or not any(_close(power, x, atol=1e-6)
                                                                    for x in range(5, 51, 5))):
                errors.append("best_beam_resource_action_invalid")
            else:
                for slot in range(freq, freq + slots):
                    if (group, slot) in slots_used:
                        errors.append("same_group_slot_conflict")
                    slots_used.add((group, slot))
                total_power += power
        elif slots != 0 or not _close(power, 0):
            errors.append("zero_demand_beam_uses_resource")
    if total_power > 3000 + 1e-6:
        errors.append("total_power_exceeds_3000W")
    if not _close(total_power, measured.get("metrics", {}).get("total_power_w"), atol=1e-6):
        errors.append("total_power_summary_differs")
    facts["best"].update(positive_demand_beams=int(np.count_nonzero(demand > 0)),
                         positive_band_0p8_1p2=demand_band,
                         capacity_demand_ratio_band_0p8_1p2=capacity_ratio_band,
                         positive_demand_mean_bps=beta, total_power_w=total_power,
                         occupied_group_slots=len(slots_used))
    service = measured.get("aircraft_service", {})
    capacities, rates, shares, carried = (service.get(key) for key in
                                         ("capacity_bps", "full_time_rate_bps",
                                          "airtime_fraction", "carried_bps"))
    ratios, si = service.get("unclipped_capacity_demand_ratio"), service.get("si")
    n = len(scaled.demands)
    if any(not isinstance(x, list) or len(x) != n for x in
           (capacities, rates, shares, carried, ratios, si)):
        errors.append("aircraft_TDMA_arrays_missing")
    else:
        total_carried = 0.0
        for u in range(n):
            d = float(scaled.demands[u])
            c, rate, share, take = capacities[u], rates[u], shares[u], carried[u]
            if not all(isinstance(x, (int, float)) and math.isfinite(x) and x >= 0
                       for x in (c, rate, share, take)):
                errors.append("aircraft_TDMA_value_invalid")
                continue
            if not _close(c, rate * share, atol=.01) or not _close(take, min(c, d), atol=.01):
                errors.append("aircraft_TDMA_capacity_or_carried_differs")
            if d > 0:
                if not _close(ratios[u], c / d, atol=1e-8) or not _close(si[u], min(c / d, 1), atol=1e-8):
                    errors.append("aircraft_TDMA_ratio_or_SI_differs")
            elif ratios[u] is not None or si[u] is not None:
                errors.append("zero_demand_aircraft_service_not_null")
            total_carried += take
        for b in range(len(beams)):
            members = np.flatnonzero(best_layout.assignment == b)
            if demand[b] > 0 and not _close(sum(shares[u] for u in members), 1, atol=1e-10):
                errors.append("aircraft_TDMA_share_not_one")
            for u in members:
                expected = float(scaled.demands[u] / demand[b]) if demand[b] else 0.0
                if not _close(shares[u], expected, atol=1e-10):
                    errors.append("aircraft_TDMA_demand_share_differs")
        total_d = float(np.sum(scaled.demands))
        fraction = total_carried / total_d if total_d else None
        if not _close(fraction, service.get("demand_weighted_carried_fraction"), atol=1e-10):
            errors.append("aircraft_TDMA_all_demand_fraction_differs")
        valid = scaled.demands > 0
        if all(isinstance(c, (int, float)) and math.isfinite(c) for c in capacities):
            met = sum(capacities[u] >= scaled.demands[u] for u in range(n) if valid[u])
            if (service.get("positive_demand_users") != int(np.count_nonzero(valid)) or
                    service.get("positive_demand_users_met") != met):
                errors.append("aircraft_TDMA_met_count_differs")
        facts["best"]["aircraft_TDMA_carried_fraction"] = fraction
    timings = measured.get("timing", {})
    segments = [timings.get(key) for key in ("interface_seconds", "allocation_seconds",
                                             "verification_seconds")]
    if (not all(isinstance(x, (int, float)) and math.isfinite(x) and x >= 0 for x in segments) or
            not _close(timings.get("total_seconds"), sum(segments), atol=1e-7)):
        errors.append("physical_segment_time_sum_differs")
    unique = physical.get("layouts_by_sha256", {})
    parts = [x.get("timing", {}).get("total_seconds") for x in unique.values()]
    if (not all(isinstance(x, (int, float)) and math.isfinite(x) and x >= 0 for x in parts) or
            not isinstance(row.get("algorithm_seconds"), (int, float)) or
            not _close(row.get("total_seconds"), row["algorithm_seconds"] + sum(parts), atol=1e-7)):
        errors.append("algorithm_plus_physical_total_time_differs")
    return sorted(set(errors)), facts
