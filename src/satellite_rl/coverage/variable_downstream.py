"""Versioned layout-to-resource interface and independent user-level checks.

The original workbench greedy allocator and joint feasibility mask are reused.
Variable transmit widths use an explicitly idealized constant beam-solid-angle
gain product.  This module never mutates the legacy global beamwidth.  The
fixed 0.5 degree case is independently checked against the original PHY.
"""
from __future__ import annotations

from time import perf_counter

import numpy as np

from satellite_rl.envs.regions import geo_distances_km
from satellite_rl.envs.satellite import AntennaPattern
from satellite_rl.power_refinement.physics import FixedAllocationPhysics
from satellite_rl.workbench.baselines import BaselinePolicy
from satellite_rl.workbench.config import default_config
from satellite_rl.workbench.environment import WorkbenchEnv
from .geometry import satellite_rays_numpy, ray_angle_numpy
from .variable_config import VariableConfig
from .variable_design import (apply_action, candidates, geometry_audit,
                              initial_layout, metrics)


PHYSICS_VERSION = "s1_v10_scaled_transmit_width_shannon_v1"
USER_SERVICE_VERSION = "aircraft_off_axis_demand_proportional_tdma_v1"
C = 2.99792458e8
K = 1.380649e-23
SLOT_BW = 25e6
F0 = 17.7e9
LOSS = 10 ** (-5 / 10)


def transmit_pattern(gain_db, angle_deg, width_deg):
    """Exactly the existing main-lobe/continued-tail formula, varying width."""
    width = np.asarray(width_deg, dtype=float)
    angle = np.abs(np.asarray(angle_deg, dtype=float))
    if np.any(width <= 0) or not np.isfinite(width).all():
        raise ValueError("Transmit full width must be finite and positive")
    sigma = width / (2 * np.sqrt(2 * np.log10(2)))
    edge = width / 2
    return np.where(angle <= edge, np.asarray(gain_db) - 3 * (angle / edge) ** 2,
                    np.asarray(gain_db) - 3 - 10 / np.log(10)
                    * (angle ** 2 - edge ** 2) / (2 * sigma ** 2))


def peak_gain(width_deg):
    return 50.0 + 20 * np.log10(0.5 / np.asarray(width_deg, dtype=float))


class WidthPattern(AntennaPattern):
    """Width is uniquely encoded in this version's declared transmit peak."""
    def transmitting_antenna_pattern(self, G_max, phi):
        width = 0.5 * 10 ** ((50.0 - float(G_max)) / 20.0)
        return float(transmit_pattern(float(G_max), float(phi), width))

    def receiving_antenna_pattern(self, G_max, phi):
        # Receiver antenna does not change with the satellite's beam width.
        return float(transmit_pattern(float(G_max), float(phi), 0.5))


def allocator_config():
    config = default_config()
    config["name"] = "S1-V10 common frozen greedy allocation"
    config["environment"].update(profile="europe_v1", beam_count_max=200,
                                  power_budget_w=3000, action_mask="joint_feasible",
                                  action_mode="legacy_clip")
    return config


def _physical_scene(scene, layout, cfg):
    audit = metrics(scene, layout, cfg)
    loads = np.asarray(audit["beam_demand_bps"], dtype=float)
    # Zero-demand geometry beams stay in layout output but require no RF action.
    positive = np.flatnonzero(loads > 0)
    order = positive[np.argsort(-loads[positive], kind="stable")]
    n = len(order)
    widths = layout.widths[order]
    arrays = {"beam_location": layout.centers[order].copy(),
              "beam_G_Tx_max": peak_gain(widths).reshape(n, 1),
              "UE_G_Rx_max": np.full((n, 1), 40.0),
              "distances": geo_distances_km(layout.centers[order], cfg.profile),
              "rate": loads[order].reshape(n, 1),
              "noise_temperature": np.full((n, 1), 290.0),
              "original_indices": order.astype(np.int64)}
    return arrays, order, widths, audit


def width_physics(arrays, config, rows, widths):
    """Independent matrix construction including transmitter-specific widths."""
    obj = FixedAllocationPhysics.from_scene(arrays, config, rows)
    direction = satellite_rays_numpy(arrays["beam_location"])
    angles = np.rad2deg(np.arccos(np.clip(direction @ direction.T, -1, 1)))
    owner = obj.slot_owner
    occupied = np.concatenate([np.arange(obj.starts[i], obj.starts[i] + obj.slots[i])
                               for i in range(obj.beam_count)])
    slot_frequency = F0 + (occupied + 0.5) * SLOT_BW
    distance = arrays["distances"][owner] * 1000
    path = (C / (4 * np.pi * distance * slot_frequency)) ** 2
    tx = arrays["beam_G_Tx_max"].reshape(-1)
    rx = arrays["UE_G_Rx_max"].reshape(-1)
    gains = 10 ** ((transmit_pattern(tx[None, :], angles[owner], widths[None, :])
                    + rx[owner, None]) / 10)
    overlap = ((obj.groups[owner, None] % 2 == obj.groups[None, :] % 2)
               & (obj.groups[owner, None] != obj.groups[None, :])
               & (occupied[:, None] >= obj.starts[None, :])
               & (occupied[:, None] < obj.starts[None, :] + obj.slots[None, :]))
    cross = np.zeros_like(gains)
    cross[:, obj.active] = gains[:, obj.active] * path[:, None] / obj.slots[None, obj.active]
    obj.interference = cross * overlap
    return obj


def _user_service(scene, layout, arrays, order, allocations):
    """Each beam gives demand-proportional airtime, summing to one.

    Full-time user rates use actual off-axis geometry and all actual RF actions.
    Multiplying those rates by disjoint airtime fractions avoids duplicating the
    whole beam bandwidth for every user. This is a declared scheduling model,
    not observed traffic or an optimal user scheduler.
    """
    n_users = len(scene.demands)
    capacities = np.zeros(n_users)
    full_time = np.zeros(n_users)
    shares = np.zeros(n_users)
    reverse = {int(old): new for new, old in enumerate(order)}
    user_rays = satellite_rays_numpy(scene.positions)
    beam_rays = satellite_rays_numpy(arrays["beam_location"])
    distances = geo_distances_km(scene.positions, "europe_v1") * 1000
    powers = np.asarray([a["power_w"] for a in allocations])
    starts = np.asarray([a["freq"] if a["slots"] else 0 for a in allocations])
    lengths = np.asarray([a["slots"] for a in allocations])
    groups = np.asarray([a["group"] for a in allocations])
    tx = arrays["beam_G_Tx_max"].reshape(-1)
    widths = layout.widths[order]
    link_checks = 0
    for u, old in enumerate(layout.assignment):
        if int(old) not in reverse or scene.demands[u] <= 0:
            continue
        b = reverse[int(old)]
        if lengths[b] <= 0:
            continue
        link_checks += 1
        members = layout.assignment == old
        shares[u] = float(scene.demands[u] / np.asarray(scene.demands)[members].sum())
        angles = ray_angle_numpy(user_rays[u], beam_rays)
        gain = 10 ** ((transmit_pattern(tx, angles, widths) + 40.0) / 10)
        center = F0 + (starts[b] + lengths[b] / 2) * SLOT_BW
        desired = LOSS * powers[b] / lengths[b] * gain[b] * (C / (4 * np.pi * distances[u] * center)) ** 2
        rate = 0.0
        for slot in range(starts[b], starts[b] + lengths[b]):
            interfering = ((groups % 2 == groups[b] % 2) & (groups != groups[b])
                           & (starts <= slot) & (slot < starts + lengths) & (lengths > 0))
            freq = F0 + (slot + 0.5) * SLOT_BW
            cross = np.sum(powers[interfering] / lengths[interfering] * gain[interfering])
            cross *= (C / (4 * np.pi * distances[u] * freq)) ** 2
            rate += SLOT_BW * np.log2(1 + desired / (K * 290 * SLOT_BW + cross))
        full_time[u] = float(rate)
        capacities[u] = rate * shares[u]
    demand = np.asarray(scene.demands, dtype=float)
    valid = demand > 0
    ratios = np.divide(capacities, demand, out=np.zeros_like(capacities), where=valid)
    total = float(demand.sum())
    for old in order:
        if not np.isclose(shares[layout.assignment == old].sum(), 1.0, atol=1e-10):
            raise ValueError("User airtime allocation does not sum to one")
    return {"model": USER_SERVICE_VERSION, "aircraft_link_checks": link_checks,
            "capacity_bps": capacities.tolist(), "full_time_rate_bps": full_time.tolist(),
            "airtime_fraction": shares.tolist(),
            "carried_bps": np.minimum(capacities, demand).tolist(),
            "unclipped_capacity_demand_ratio": [float(r) if v else None for r, v in zip(ratios, valid)],
            "si": [float(min(r, 1)) if v else None for r, v in zip(ratios, valid)],
            "demand_weighted_carried_fraction": float(np.minimum(capacities, demand).sum() / total) if total else None,
            "positive_demand_users_met": int(np.count_nonzero(valid & (capacities >= demand))),
            "positive_demand_users": int(valid.sum()),
            "all_positive_demand_users_met": bool(np.all(capacities[valid] >= demand[valid])) if np.any(valid) else None,
            "limitation": "Declared Shannon user-link and proportional-demand TDMA model; not observed traffic."}


def evaluate_layout(scene, layout, config, *, route="R01"):
    started = perf_counter()
    controls = config.get("coverage", {}).get("variable", {})
    cfg = VariableConfig(**{k: (tuple(v) if k == "width_choices_deg" else v)
                            for k, v in controls.items() if k in VariableConfig.__dataclass_fields__})
    arrays, order, widths, geometry = _physical_scene(scene, layout, cfg)
    prepared = perf_counter()
    common_config = allocator_config()
    env = WorkbenchEnv(common_config)
    state = env.reset(seed=51001, scene=arrays)
    actions = []
    if len(order):
        env.raw_env.antenna_pattern = WidthPattern()
        policy = BaselinePolicy("greedy", env)
        done = False
        while not done:
            requested = policy.act(state, deterministic=True, action_mask=env.action_mask())
            state, reward, done, info = env.step(requested)
            actions.append({"requested_indices": list(map(int, requested)),
                            "source_preference": dict(policy.last_action_info),
                            "info": info})
        allocated = env.beam_results()
        allocation_done = perf_counter()
        phy = width_physics(arrays, common_config, allocated, widths)
        powers = np.asarray([row["power_w"] for row in allocated])
        if powers.sum() > 3000 + 1e-7:
            raise ValueError("Downstream total power exceeded")
        independent = phy.evaluate(powers)
        measured = np.asarray([row["capacity_bps"] for row in allocated])
        if not np.allclose(measured, independent.capacity_bps, rtol=2e-8, atol=2e-3):
            raise ValueError("Original sequential and independent terminal PHY disagree")
        user_service = _user_service(scene, layout, arrays, order, allocated)
    else:
        allocation_done = perf_counter()
        allocated = []
        powers = np.empty(0)
        user_service = {"model": USER_SERVICE_VERSION, "aircraft_link_checks": 0,
                        "capacity_bps": [0.0] * len(scene.demands),
                        "airtime_fraction": [0.0] * len(scene.demands),
                        "demand_weighted_carried_fraction": 0.0 if np.sum(scene.demands) else None,
                        "all_positive_demand_users_met": False if np.sum(scene.demands) else None}
    final_done = perf_counter()
    loads = geometry["beam_demand_bps"]
    rows = [{"beam_id": b, "stable_beam_id": layout.beam_ids[b],
             "lat": float(layout.centers[b, 0]), "lon": float(layout.centers[b, 1]),
             "demand_bps": float(d), "capacity_bps": 0.0, "carried_bps": 0.0,
             "unclipped_capacity_demand_ratio": None if d == 0 else 0.0,
             "si": None if d == 0 else 0.0, "sgm": None if d == 0 else 0.0,
             "slots": 0, "power_w": 0.0, "width_deg": float(layout.widths[b]),
             "capacity_source": PHYSICS_VERSION, "timing_scope": "terminal_after_all_actions",
             "units": {"demand_bps": "bit/s", "capacity_bps": "bit/s", "power_w": "W"},
             "status": "geometry_only_zero_demand" if d == 0 else "unserved"}
            for b, d in enumerate(loads)]
    for i, old in enumerate(order):
        row = allocated[i]
        cap, demand = float(row["capacity_bps"]), float(loads[old])
        rows[int(old)].update(capacity_bps=cap, carried_bps=min(cap, demand),
                             unclipped_capacity_demand_ratio=cap / demand,
                             si=min(cap / demand, 1.0), sgm=float(row["sgm"]),
                             processing_index=i, group=row["group"], freq=row["freq"],
                             slots=row["slots"], power_w=row["power_w"], status="allocated")
    total_demand = float(np.sum(scene.demands))
    carried = float(sum(row["carried_bps"] for row in rows))
    return {"beams": rows, "metrics": {"demand_weighted_carried_fraction": carried / total_demand if total_demand else None,
            "beam_mean_si": float(np.mean([r["si"] for r in rows if r["si"] is not None])) if len(order) else None,
            "beam_mean_sgm": float(np.mean([r["sgm"] for r in rows if r["sgm"] is not None])) if len(order) else None,
            "total_power_w": float(powers.sum()), "complete_allocation": bool(env.metrics()["complete"]),
            "processed_count": int(env.metrics()["processed_count"]),
            "total_demand_bps": total_demand, "carried_bps": carried,
            "uncovered_demand_bps": geometry["uncovered_demand_bps"]},
            "physics_version": PHYSICS_VERSION, "allocator": "workbench_feasible_greedy_v2",
            "processing_to_layout": order.tolist(), "physical_scene": {k: v.tolist() for k, v in arrays.items()},
            "actual_allocations": allocated, "actions": actions, "aircraft_service": user_service,
            "timing": {"interface_seconds": prepared - started,
                       "allocation_seconds": allocation_done - prepared,
                       "verification_seconds": final_done - allocation_done,
                       "total_seconds": final_done - started},
            "calls": {"layout_allocation_calls": 1, "allocation_decisions": len(actions),
                      "independent_physics_checks": int(bool(len(order))),
                      "aircraft_link_checks": user_service["aircraft_link_checks"]}}


def _solve_feedback_impl(scene, cfg, *, config, route="R10"):
    if route != "R10":
        raise ValueError("Feedback route identity must be R10")
    start = perf_counter()
    layout, initialization = initial_layout(scene, cfg, "R04")
    base = evaluate_layout(scene, layout, config, route=route)
    score = base["metrics"]["demand_weighted_carried_fraction"] or 0.0
    calls = 1
    ledger = [{"operation": "initial_allocation", "carried_fraction": score}]
    budget = {"geometry_calls": int(initialization.get("geometry_calls", 0)),
              "max_geometry_calls": cfg.max_geometry_calls}
    # All candidate physical probes are charged, including rejections.
    for _ in range(cfg.max_actions):
        improved = False
        for action, _features in candidates(scene, layout, cfg, mode="local", budget=budget):
            if action.operation == "stop" or calls >= 16:
                continue
            proposal, info = apply_action(scene, layout, cfg, action, budget=budget)
            if not info.get("accepted"):
                ledger.append({"action": action.to_record(), "accepted": False, "reason": info.get("reason")})
                continue
            trial = evaluate_layout(scene, proposal, config, route=route)
            calls += 1
            trial_score = trial["metrics"]["demand_weighted_carried_fraction"] or 0.0
            accepted = trial_score > score + 1e-10
            ledger.append({"action": action.to_record(), "accepted": accepted,
                           "carried_fraction": trial_score, "physical_probe": calls,
                           "probe_seconds": trial["timing"]["total_seconds"]})
            if accepted:
                layout, score = proposal, trial_score
                improved = True
                break
        if not improved or calls >= 16:
            break
    return layout, {"route": route, "initialization": initialization, "actions": ledger,
                    "geometry_budget_charges": budget["geometry_calls"], "physical_feedback_calls": calls,
                    "physical_feedback_limit": 16, "total_seconds": perf_counter() - start,
                    "final_independent_allocation_calls": 1,
                    "total_allocation_call_limit_including_final": 17,
                    "objective": "full-aircraft-demand-normalized centre-link carried capacity"}


def solve_feedback(scene, cfg, *, config, route="R10"):
    with geometry_audit() as audit:
        layout, info = _solve_feedback_impl(scene, cfg, config=config, route=route)
    info["geometry_calls"] = audit["matrix_calls"]
    info["actual_geometry_calls"] = audit["matrix_calls"]
    info["actual_ray_pair_checks"] = audit["ray_pair_checks"]
    return layout, info
