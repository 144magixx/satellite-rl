"""Fixed greedy allocation and SCA through the existing shared link physics."""
from time import perf_counter

import numpy as np

from satellite_rl.shared_interface.geometry import directions_and_distance
from satellite_rl.shared_interface.link import build_link_coefficients, capacity_bps
from satellite_rl.shared_interface.spherical_gaussian import PHYSICS, linear_gain
from satellite_rl.power_refinement.physics import FixedAllocationPhysics
from satellite_rl.power_refinement.optimizer import optimize_power

REQUEST_BPS = 21_600_000


def greedy_allocation(layout):
    """Assign eight groups, nonoverlapping intervals, and proportional power."""
    n = len(layout.widths)
    demand = np.bincount(layout.owner, minlength=n).astype(np.int64) * REQUEST_BPS
    directions, _ = directions_and_distance(layout.centers_latlon)
    angle = 2 * np.arcsin(np.minimum(np.linalg.norm(
        directions[:, None] - directions[None, :], axis=-1) / 2, 1))
    gain, _ = linear_gain(angle, np.deg2rad(layout.widths)[None, :])
    slots = np.clip(np.ceil(demand / 100_000_000), 1, 10).astype(int)
    order = np.argsort(-demand, kind="stable")
    groups = np.full(n, -1, int)
    used = np.zeros(8, int)
    overflow_penalty = gain.max() * n
    for i in order:
        cost = [gain[i, (groups >= 0) & (groups % 2 == group % 2) & (groups != group)].sum()
                + overflow_penalty * max(0, used[group] + slots[i] - 100)
                for group in range(8)]
        groups[i] = group = int(np.argmin(cost))
        used[group] += slots[i]
    for group in range(8):
        members = np.flatnonzero(groups == group)
        if slots[members].sum() > 100:
            # One slot per beam, then demand-proportional residual via largest remainder.
            quota = (100 - len(members)) * demand[members] / demand[members].sum()
            count = np.minimum(slots[members], 1 + np.floor(quota).astype(int))
            while count.sum() < 100 and np.any(count < slots[members]):
                eligible = np.where(count < slots[members], quota - (count - 1), -np.inf)
                count[np.argmax(eligible)] += 1
            slots[members] = count
    starts = np.zeros(n, int)
    cursor = np.zeros(8, int)
    remaining = np.bincount(groups, weights=slots, minlength=8).astype(int)
    assigned = np.zeros(n, bool)
    for i in order:
        group = groups[i]
        other = np.flatnonzero(assigned & (groups % 2 == group % 2) & (groups != group))
        choices = np.arange(cursor[group], 101 - remaining[group])
        overlap = np.maximum(0, np.minimum(choices[:, None] + slots[i], starts[other] + slots[other])
                             - np.maximum(choices[:, None], starts[other]))
        costs = overlap @ gain[i, other]
        starts[i] = choices[np.argmin(costs)]
        cursor[group] = starts[i] + slots[i]
        remaining[group] -= slots[i]
        assigned[i] = True
    power = 3000. * demand / demand.sum()
    # Keep floating-point sum inside the exact shared-link budget comparison.
    power *= 3000. / max(3000., np.nextafter(power.sum(), np.inf))
    ids = list(map(int, layout.beam_ids))
    allocation = dict(beam_ids=ids, group_by_beam_id=dict(zip(ids, map(int, groups))),
                      actual_start_slot=starts, actual_slot_count=slots,
                      actual_whole_beam_power_w=power, content_sha256="s1fix")
    link_layout = dict(beam_ids=ids, centres_lat_lon_deg=layout.centers_latlon,
                       full_width_deg=layout.widths,
                       request_bps_by_beam_int64=demand, content_sha256="s1fix")
    return link_layout, allocation, groups


def build_problem(layout):
    """Use plain dictionaries and construct FixedAllocationPhysics directly."""
    link_layout, allocation, groups = greedy_allocation(layout)
    coefficients = build_link_coefficients(link_layout, allocation, PHYSICS)
    problem = FixedAllocationPhysics(coefficients.demand_bps, groups,
                                     allocation["actual_start_slot"],
                                     allocation["actual_slot_count"],
                                     coefficients.signal, coefficients.interference,
                                     coefficients.noise, coefficients.slot_owner, 3000.)
    return problem, coefficients.powers_initial.copy(), coefficients


def service_metrics(problem, evaluation):
    """Weight beam service by its aircraft count for aircraft-level summaries."""
    counts = np.rint(problem.demand_bps / REQUEST_BPS).astype(int)
    service = np.repeat(evaluation.beam_si, counts)
    worst = int(np.ceil(.1 * len(service)))
    return dict(aircraft_mean=float(np.dot(counts, evaluation.beam_si) / counts.sum()),
                aircraft_worst10=float(np.partition(service, worst - 1)[:worst].mean()),
                aircraft_ge90=float(np.mean(service >= .9)), SI=evaluation.si, SGM=evaluation.sgm)


def evaluate_layout(layout, sca_options=None, arrays_path=None):
    """Return greedy/SCA metrics and times; shared-link parity is tested separately."""
    started = perf_counter()
    problem, power, coefficients = build_problem(layout)
    initial = problem.evaluate(power)
    t_second = perf_counter() - started
    options = dict(maxiter=30, max_evaluations=500, time_limit_seconds=30.)
    options.update(sca_options or {})
    started = perf_counter()
    result = optimize_power(problem, power, "sca", options)
    t_sca = perf_counter() - started
    final = problem.evaluate(result["power_w"])
    if arrays_path is not None:
        from pathlib import Path
        path = Path(arrays_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, demand_bps=problem.demand_bps, beam_ids=layout.beam_ids,
                            groups=problem.groups, starts=problem.starts, slots=problem.slots,
                            power_initial_w=power, power_final_w=result["power_w"],
                            capacity_initial_bps=initial.capacity_bps,
                            capacity_final_bps=final.capacity_bps)
    row = {f"greedy_{key}": value for key, value in service_metrics(problem, initial).items()}
    row.update({f"sca_{key}": value for key, value in service_metrics(problem, final).items()})
    row.update(t_second=t_second, t_sca=t_sca, sca_status=result["termination_reason"])
    return row
