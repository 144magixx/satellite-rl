"""Budgeted DE/rand/1/bin and particle swarm search for fixed-slot power.

Both algorithms evaluate the unmodified physical model.  Population
initialization and every repaired trial consume one full-scene capacity call;
there is no gradient solver or post-search polishing step.
"""

from time import perf_counter

import numpy as np

from .physics import FixedAllocationPhysics


class _SearchLimit(Exception):
    pass


def _project_simplex(power, cap):
    """Euclidean projection onto nonnegative power with sum at most cap."""
    power = np.maximum(np.asarray(power, dtype=np.float64), 0.0)
    if power.sum() <= cap:
        return power
    sorted_power = np.sort(power)[::-1]
    threshold = (np.cumsum(sorted_power) - cap) / np.arange(1, len(power) + 1)
    positive = np.flatnonzero(sorted_power > threshold)
    level = threshold[positive[-1]]
    return np.maximum(power - level, 0.0)


def _metrics(problem, power, evaluation):
    capacity = evaluation.capacity_bps
    return {
        "final_si_mean": evaluation.si,
        "final_sgm_mean": evaluation.sgm,
        "basic_met_count": int(np.count_nonzero(capacity >= 0.1 * problem.demand_bps)),
        "power_w": float(power.sum()),
        "required_bps": problem.demand_bps.tolist(),
        "capacity_bps": capacity.tolist(),
        "effective_served_bps": np.minimum(capacity, problem.demand_bps).tolist(),
        "capacity_demand_ratio": (capacity / problem.demand_bps).tolist(),
        "beam_si": evaluation.beam_si.tolist(),
        "beam_sgm": evaluation.beam_sgm.tolist(),
    }


def optimize_population(problem: FixedAllocationPhysics, initial_power, method,
                        options=None, callback=None, stop_requested=None):
    """Search the active-beam simplex, retaining only verified feasible gains.

    ``optimizer_seed`` is mandatory and must be saved with the task. DE uses
    DE/rand/1/bin with greedy target selection. PSO uses inertia and personal/
    swarm best attraction. The first member of each population is exactly p0.
    Feasible candidates outrank infeasible candidates; infeasible candidates
    are ordered by normalized basic-service, SGM and (phase two) SI shortfall.
    """
    if not isinstance(problem, FixedAllocationPhysics):
        raise TypeError("problem must be FixedAllocationPhysics")
    if method not in {"de", "pso"}:
        raise ValueError(f"Unknown population method: {method}")
    opts = dict(options or {})
    allowed = {"power_cap_w", "maxiter", "max_evaluations", "maxeval",
               "time_limit_seconds", "max_time_seconds", "si_tolerance",
               "sgm_tolerance", "constraint_tolerance", "protect_basic_service",
               "optimizer_seed", "population_size", "de_mutation", "de_crossover",
               "pso_inertia", "pso_cognitive", "pso_social"}
    if set(opts) - allowed:
        raise ValueError(f"Unknown population options: {sorted(set(opts) - allowed)}")
    seed = opts.get("optimizer_seed")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("optimizer_seed must be an explicit uint32 integer")
    cap = float(opts.get("power_cap_w", problem.power_budget_w))
    maxiter = int(opts.get("maxiter", 300))
    maxeval = int(opts.get("max_evaluations", opts.get("maxeval", 2000)))
    max_time = float(opts.get("time_limit_seconds", opts.get("max_time_seconds", 30.0)))
    si_tol = float(opts.get("si_tolerance", 1e-8))
    sgm_tol = float(opts.get("sgm_tolerance", 1e-8))
    basic_tol = float(opts.get("constraint_tolerance", 1e-8))
    size = int(opts.get("population_size", 24))
    mutation = float(opts.get("de_mutation", 0.7))
    crossover = float(opts.get("de_crossover", 0.8))
    inertia = float(opts.get("pso_inertia", 0.7))
    cognitive = float(opts.get("pso_cognitive", 1.4))
    social = float(opts.get("pso_social", 1.4))
    if (opts.get("protect_basic_service", True) is not True
            or not np.isfinite([cap, max_time, si_tol, sgm_tol, basic_tol,
                                mutation, crossover, inertia, cognitive, social]).all()
            or not 0 < cap <= min(3000.0, problem.power_budget_w)
            or maxiter < 1 or maxeval < 1 or max_time <= 0
            or si_tol < 0 or sgm_tol < 0 or basic_tol < 0
            or not 0 < mutation <= 2 or not 0 <= crossover <= 1
            or not 0 <= inertia < 1 or cognitive < 0 or social < 0
            or not 4 <= size <= 256):
        raise ValueError("Invalid population search controls")
    initial = np.asarray(initial_power, dtype=np.float64)
    initial_ev = problem.evaluate(initial)
    if initial.sum() > cap + 1e-7:
        raise ValueError("Initial power exceeds selected budget")
    rng = np.random.default_rng(seed)
    active = np.flatnonzero(problem.active)
    n = len(active)
    start = perf_counter()
    count = 1
    best_power = initial.copy()
    best_ev = initial_ev
    basic_floor = np.minimum(initial_ev.capacity_bps / problem.demand_bps, 0.1)
    baseline_sgm = initial_ev.sgm
    history = [{"phase": "initial", "evaluation": 1,
                "final_si_mean": initial_ev.si, "final_sgm_mean": initial_ev.sgm,
                "power_w": float(initial.sum()), "elapsed_seconds": 0.0}]
    phase = "si"
    phase2_si_floor = None
    generation = 0
    termination = "completed"
    phase_budget = max(1, int(0.65 * maxeval))

    def check_limit():
        if stop_requested is not None and stop_requested():
            raise _SearchLimit("stop_requested")
        if perf_counter() - start >= max_time:
            raise _SearchLimit("time_limit")
        if count >= maxeval:
            raise _SearchLimit("evaluation_limit")

    def full_power(x):
        p = np.zeros(problem.beam_count, dtype=np.float64)
        if n:
            p[active] = _project_simplex(x, cap)
        return p

    def evaluate(x):
        nonlocal count
        check_limit()
        p = full_power(x)
        ev = problem.evaluate(p)
        count += 1
        retain(p, ev)
        return p, ev

    def rank(p, ev):
        ratios = ev.capacity_bps / problem.demand_bps
        basic_gap = float(np.maximum(basic_floor - ratios - basic_tol, 0).sum())
        sgm_gap = max(0.0, baseline_sgm - ev.sgm - sgm_tol)
        si_gap = (max(0.0, phase2_si_floor - ev.si)
                  if phase2_si_floor is not None else 0.0)
        violation = basic_gap + sgm_gap + si_gap
        if violation > 1e-12:
            return (0, -violation, ev.si, ev.sgm)
        if phase == "si":
            return (1, ev.si, ev.sgm, -float(p.sum()))
        return (1, ev.sgm, ev.si, -float(p.sum()))

    def retain(p, ev):
        nonlocal best_power, best_ev
        if (not np.isfinite(p).all() or np.any(p < 0)
                or np.any(p[~problem.active] != 0) or p.sum() > cap + 1e-7
                or ev.sgm < baseline_sgm - sgm_tol - 1e-10
                or np.any(ev.capacity_bps / problem.demand_bps
                          < basic_floor - basic_tol)
                or ev.si < initial_ev.si - si_tol - 1e-10
                or (phase2_si_floor is not None
                    and ev.si < phase2_si_floor - 1e-10)):
            return
        if phase == "si":
            improved = (ev.si > best_ev.si + 1e-10
                        or abs(ev.si - best_ev.si) <= 1e-10
                        and ev.sgm > best_ev.sgm + 1e-10)
        else:
            improved = (ev.sgm > best_ev.sgm + 1e-10
                        or abs(ev.sgm - best_ev.sgm) <= 1e-10
                        and ev.si > best_ev.si + 1e-10)
        if improved:
            best_power, best_ev = p.copy(), ev
            entry = {"phase": phase, "evaluation": count,
                     "final_si_mean": ev.si, "final_sgm_mean": ev.sgm,
                     "power_w": float(p.sum()),
                     "elapsed_seconds": perf_counter() - start}
            history.append(entry)
            if callback is not None:
                callback(dict(entry))

    def initial_member(index):
        base = initial[active].copy()
        if index == 0 or not n:
            return base
        scale = max(1.0, cap / max(n, 1), float(base.mean()))
        if index % 4 == 0:
            # Global multiplicative spread, including below 5 and above 50 W.
            return base * rng.lognormal(0.0, 0.65, n)
        if index % 4 == 1:
            # Single-beam moves allow tight basic-service guards to survive.
            j = int(rng.integers(n))
            base[j] += rng.choice([-1.0, 1.0]) * rng.uniform(0.1, 1.0) * scale
            return base
        if index % 4 == 2 and n > 1:
            i, j = rng.choice(n, 2, replace=False)
            amount = rng.uniform(0.1, 1.0) * scale
            base[i] -= amount
            base[j] += amount
            return base
        mask = rng.random(n) < min(0.15, 3 / n)
        base[mask] += rng.normal(0.0, scale * 0.4, int(mask.sum()))
        return base

    population = [(initial.copy(), initial_ev)]
    velocities = [np.zeros(n)]
    try:
        for index in range(1, size):
            p, ev = evaluate(initial_member(index))
            population.append((p, ev))
            velocities.append(rng.normal(0, max(1.0, cap / max(n, 1)) * 0.2, n))
        personal = [(p.copy(), ev) for p, ev in population]
        for generation in range(1, maxiter + 1):
            check_limit()
            if phase == "si" and count >= phase_budget:
                phase = "sgm"
                phase2_si_floor = best_ev.si - si_tol
                # Phase two always has a truly feasible guide, even if all
                # other members violate the newly fixed SI floor.
                population[0] = (best_power.copy(), best_ev)
                personal[0] = (best_power.copy(), best_ev)
            for index in range(size):
                check_limit()
                current, current_ev = population[index]
                if method == "de":
                    others = np.delete(np.arange(size), index)
                    a, b, c = rng.choice(others, 3, replace=False)
                    mutant = (population[a][0][active] + mutation
                              * (population[b][0][active] - population[c][0][active]))
                    mask = rng.random(n) < crossover
                    if n:
                        mask[int(rng.integers(n))] = True
                    trial = np.where(mask, mutant, current[active])
                else:
                    guide = max(personal, key=lambda item: rank(*item))[0][active]
                    velocities[index] = (inertia * velocities[index]
                                         + cognitive * rng.random(n)
                                         * (personal[index][0][active] - current[active])
                                         + social * rng.random(n)
                                         * (guide - current[active]))
                    velocity_cap = max(1.0, cap / max(n, 1)) * 3
                    velocities[index] = np.clip(velocities[index],
                                                -velocity_cap, velocity_cap)
                    trial = current[active] + velocities[index]
                candidate, candidate_ev = evaluate(trial)
                if method == "de":
                    if rank(candidate, candidate_ev) > rank(current, current_ev):
                        population[index] = (candidate, candidate_ev)
                else:
                    population[index] = (candidate, candidate_ev)
                    if rank(candidate, candidate_ev) > rank(*personal[index]):
                        personal[index] = (candidate.copy(), candidate_ev)
    except _SearchLimit as exc:
        termination = str(exc)
    if termination == "completed":
        termination = "iteration_limit"
    if len(history) == 1 and termination == "iteration_limit":
        termination = "no_feasible_improvement"
    error = {
        "budget_excess_w": max(0.0, float(best_power.sum() - cap)),
        "negative_power_w": max(0.0, float(-best_power.min())),
        "empty_slot_power_w": float(np.max(np.abs(best_power[~problem.active])))
        if np.any(~problem.active) else 0.0,
        "sgm_shortfall": max(0.0, baseline_sgm - best_ev.sgm),
        "basic_shortfall_ratio": float(np.maximum(
            basic_floor - best_ev.capacity_bps / problem.demand_bps, 0).max()),
        "si_shortfall": max(0.0, initial_ev.si - best_ev.si),
        "phase2_si_shortfall": (max(0.0, phase2_si_floor - best_ev.si)
                                if phase2_si_floor is not None else 0.0),
    }
    return {
        "method": method, "power_w": best_power.tolist(),
        "initial_metrics": _metrics(problem, initial, initial_ev),
        "metrics": _metrics(problem, best_power, best_ev),
        "history": history, "evaluations": count,
        "full_scene_evaluations": count,
        "scalar_capacity_evaluations": 0,
        "elapsed_seconds": perf_counter() - start,
        "constraint_error": error,
        "termination_reason": termination,
        "converged": False,
        "fallback_used": bool(np.array_equal(best_power, initial)),
        "solver_status": [{"phase": phase, "success": False,
                           "message": termination, "iterations": generation}],
        "solver_power_scale_w": None,
        "warm_start_status": None,
        "algorithm_components": ["de_rand_1_bin"] if method == "de"
        else ["pso_inertia_cognitive_social"],
        "power_cap_w": cap,
        "optimizer_seed": seed,
        "population_size": size,
        "iterations": generation,
    }
