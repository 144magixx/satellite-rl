"""Finite-budget, local fixed-slot power refinement with feasible fallback."""

from time import perf_counter

import numpy as np

from .physics import FixedAllocationPhysics


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


class _LimitReached(Exception):
    pass


class _WarmBudget(Exception):
    pass


def optimize_power(problem: FixedAllocationPhysics, initial_power, method="none",
                   options=None, callback=None, stop_requested=None):
    """Improve SI first, then SGM, retaining only true feasible candidates.

    ``method`` is ``none``, ``discrete`` (original 5 W moves), or ``slsqp``
    with a bounded demand-matching warm start between its two local phases.
    Neither solver status nor output implies a global optimum.
    ``callback(entry)`` runs for each accepted improvement.
    """
    if method in {"sca", "wmmse"}:
        from .model_based import optimize_model_based
        return optimize_model_based(problem, initial_power, method, options,
                                    callback=callback, stop_requested=stop_requested)
    if method in {"de", "pso"}:
        from .population import optimize_population
        return optimize_population(problem, initial_power, method, options,
                                   callback=callback, stop_requested=stop_requested)
    if not isinstance(problem, FixedAllocationPhysics):
        raise TypeError("problem must be FixedAllocationPhysics")
    if method not in {"none", "discrete", "slsqp"}:
        raise ValueError(f"Unknown method: {method}")
    opts = dict(options or {})
    allowed = {"power_cap_w", "maxiter", "max_evaluations", "maxeval",
               "time_limit_seconds", "max_time_seconds", "ftol", "si_tolerance",
               "sgm_tolerance", "constraint_tolerance", "protect_basic_service",
               "discrete_step_w"}
    if set(opts) - allowed:
        raise ValueError(f"Unknown optimization options: {sorted(set(opts) - allowed)}")
    cap = float(opts.get("power_cap_w", problem.power_budget_w))
    maxiter = int(opts.get("maxiter", 30))
    max_evaluations = int(opts.get("max_evaluations", opts.get("maxeval", 5000)))
    time_limit = float(opts.get("time_limit_seconds", opts.get("max_time_seconds", 60.0)))
    ftol = float(opts.get("ftol", 1e-8))
    si_tolerance = float(opts.get("si_tolerance", 0.0))
    sgm_tolerance = float(opts.get("sgm_tolerance", 0.0))
    feasibility_tolerance = float(opts.get("constraint_tolerance", 1e-8))
    if (opts.get("protect_basic_service", True) is not True
            or float(opts.get("discrete_step_w", 5.0)) != 5.0):
        raise ValueError("Basic service protection and original 5 W grid are fixed")
    if (not 0 < cap <= min(3000.0, problem.power_budget_w)
            or maxiter < 1 or max_evaluations < 1 or time_limit <= 0
            or ftol <= 0 or si_tolerance < 0 or sgm_tolerance < 0
            or feasibility_tolerance < 0
            or not np.isfinite([cap, time_limit, ftol, si_tolerance,
                                  sgm_tolerance, feasibility_tolerance]).all()):
        raise ValueError("Invalid optimization limits or power cap")
    initial = np.asarray(initial_power, dtype=np.float64)
    initial_eval = problem.evaluate(initial, with_jacobian=False)
    if initial.sum() > cap + 1e-7:
        raise ValueError("Initial power exceeds selected budget")
    start_time = perf_counter()
    floor_ratio = np.minimum(initial_eval.capacity_bps / problem.demand_bps, 0.1)
    baseline_sgm = initial_eval.sgm
    best_power = initial.copy()
    best_eval = initial_eval
    count = 1
    full_evaluations = 1
    scalar_evaluations = 0
    history = [{"phase": "initial", "evaluation": count,
                "final_si_mean": initial_eval.si, "final_sgm_mean": initial_eval.sgm,
                "power_w": float(initial.sum()), "elapsed_seconds": 0.0}]
    solver_status = []
    termination = "unchanged" if method == "none" else "completed"
    phase_limit = None
    phase_time_limit = None
    discrete_completed = False
    discrete_exhausted = False
    power_scale = None
    phase2_si_floor = None
    warm_start_status = None

    def check_limit():
        if stop_requested is not None and stop_requested():
            raise _LimitReached("stop_requested")
        if perf_counter() - start_time >= time_limit:
            raise _LimitReached("time_limit")
        if count >= max_evaluations:
            raise _LimitReached("evaluation_limit")
        if phase_limit is not None and count >= phase_limit:
            raise _LimitReached("phase_evaluation_limit")
        if phase_time_limit is not None and perf_counter() - start_time >= phase_time_limit:
            raise _LimitReached("phase_time_limit")

    def feasible(power, ev, si_floor=None):
        return (np.isfinite(power).all() and np.all(power >= 0)
                and np.all(power[~problem.active] == 0)
                and power.sum() <= cap + 1e-7
                and ev.sgm >= baseline_sgm - sgm_tolerance - 1e-10
                and np.all(ev.capacity_bps / problem.demand_bps
                           >= floor_ratio - feasibility_tolerance)
                and (si_floor is None or ev.si >= si_floor - 1e-10))

    def evaluate(power, jac=False):
        nonlocal count, full_evaluations
        check_limit()
        # SLSQP can present tiny negative bound-rounding errors. Its search
        # point remains unmodified; only the physical evaluation is projected.
        safe = np.maximum(np.asarray(power, dtype=np.float64), 0.0)
        safe[~problem.active] = 0.0
        ev = problem.evaluate(safe, with_jacobian=jac)
        count += 1
        full_evaluations += 1
        return ev

    def retain(power, ev, phase, si_floor=None):
        nonlocal best_power, best_eval
        if not feasible(power, ev, si_floor):
            return
        if phase == "si":
            improved = (ev.si > best_eval.si + 1e-10
                        or (abs(ev.si - best_eval.si) <= 1e-10
                            and ev.sgm > best_eval.sgm + 1e-10))
        else:
            improved = (ev.sgm > best_eval.sgm + 1e-10
                        or (abs(ev.sgm - best_eval.sgm) <= 1e-10
                            and ev.si > best_eval.si + 1e-10))
        if improved:
            best_power, best_eval = np.array(power, copy=True), ev
            entry = {"phase": phase, "evaluation": count,
                     "final_si_mean": ev.si, "final_sgm_mean": ev.sgm,
                     "power_w": float(power.sum()),
                     "elapsed_seconds": perf_counter() - start_time}
            history.append(entry)
            if callback is not None:
                callback(dict(entry))

    def demand_matching_warm_start():
        """Reduce oversupplied beam power with frozen current interference.

        Each scalar capacity computation consumes the same one-call budget as
        a full evaluation. Intermediate points may be infeasible for SGM; only
        independently checked full-scene points are retained. The final SLSQP
        starts from the best retained feasible point, never an unchecked one.
        """
        nonlocal count, scalar_evaluations, warm_start_status
        started = perf_counter()
        phase_floor = best_eval.si - si_tolerance
        working = best_power.copy()
        scalar_start = scalar_evaluations
        full_start = full_evaluations
        retained_start = len(history)
        warm_budget = min(800, int(max_evaluations * 0.4))
        reserve = max(2, int(max_evaluations * 0.3))
        ceiling = min(max_evaluations - reserve, count + warm_budget)
        changed = 0
        sweeps = 0

        def scalar_capacity(owner, watts):
            nonlocal count, scalar_evaluations
            if (count >= ceiling or perf_counter() - start_time >= 0.7 * time_limit):
                raise _WarmBudget
            check_limit()
            slot_rows = problem.slot_owner == owner
            denominator = (problem.noise[slot_rows]
                           + problem.interference[slot_rows] @ working)
            signal = problem.signal[slot_rows] * watts
            value = float((2.5e7 / np.log(2)
                           * np.log1p(signal / denominator)).sum())
            count += 1
            scalar_evaluations += 1
            return value

        for sweep in range(2):
            sweep_changed = 0
            for owner in np.flatnonzero(problem.active):
                if count + 10 >= ceiling or perf_counter() - start_time >= 0.7 * time_limit:
                    break
                current = working[owner]
                if current <= 0:
                    continue
                try:
                    present = scalar_capacity(owner, current)
                    demand = problem.demand_bps[owner]
                    if present <= demand * (1 + 1e-5):
                        continue
                    low, high = 0.0, current
                    for _ in range(8):
                        middle = (low + high) / 2
                        if scalar_capacity(owner, middle) >= demand:
                            high = middle
                        else:
                            low = middle
                    if current - high <= 1e-7:
                        continue
                    working[owner] = high
                    ev = evaluate(working)
                    retain(working, ev, "demand_matching_warm_start", phase_floor)
                    changed += 1
                    sweep_changed += 1
                except _WarmBudget:
                    break
            sweeps = sweep + 1
            if not sweep_changed or count + 10 >= ceiling:
                break
        warm_start_status = {
            "component": "demand_matching_warm_start",
            "scalar_capacity_evaluations": scalar_evaluations - scalar_start,
            "full_scene_evaluations": full_evaluations - full_start,
            "changed_beams": changed, "sweeps": sweeps,
            "accepted_candidates": len(history) - retained_start,
            "elapsed_seconds": perf_counter() - started,
            "stopped_by_reservation": count + 10 >= ceiling,
        }

    try:
        if method == "discrete":
            # Input model actions are 5 W multiples. Do not silently quantize.
            active = np.flatnonzero(problem.active)
            if not np.allclose(initial[active] / 5, np.rint(initial[active] / 5),
                               atol=1e-7, rtol=0):
                raise ValueError("Discrete baseline requires original 5 W power levels")
            if np.any(initial[active] < 5 - 1e-7) or np.any(initial[active] > 50 + 1e-7):
                raise ValueError("Discrete baseline requires active beams in 5–50 W")
            for phase in ("si", "sgm"):
                si_floor = None if phase == "si" else best_eval.si - si_tolerance
                if phase == "sgm":
                    phase2_si_floor = si_floor
                for _ in range(maxiter):
                    check_limit()
                    old = best_power.copy()
                    # Single-beam increments/decrements, then 5 W transfers.
                    moves = []
                    for i in active:
                        if best_power[i] >= 10 - 1e-8:
                            moves.append((i, -1, 0))
                        if best_power[i] <= 45 + 1e-8:
                            moves.append((i, -1, 1))
                    for i in active:
                        if best_power[i] < 10 - 1e-8:
                            continue
                        for j in active:
                            if i != j and best_power[j] <= 45 + 1e-8:
                                moves.append((i, j, 2))
                    selected = None
                    selected_ev = best_eval
                    try:
                        for i, j, kind in moves:
                            candidate = old.copy()
                            if kind == 0:
                                candidate[i] -= 5
                            elif kind == 1:
                                candidate[i] += 5
                            else:
                                candidate[i] -= 5
                                candidate[j] += 5
                            if candidate.sum() > cap + 1e-7:
                                continue
                            ev = evaluate(candidate)
                            if not feasible(candidate, ev, si_floor):
                                continue
                            if phase == "si":
                                key, current = (ev.si, ev.sgm, -candidate.sum()), (
                                    selected_ev.si, selected_ev.sgm,
                                    -selected.sum() if selected is not None else -old.sum())
                            else:
                                key, current = (ev.sgm, ev.si, -candidate.sum()), (
                                    selected_ev.sgm, selected_ev.si,
                                    -selected.sum() if selected is not None else -old.sum())
                            if key > current and (key[0] > current[0] + 1e-10
                                                  or key[1] > current[1] + 1e-10):
                                selected, selected_ev = candidate, ev
                    except _LimitReached:
                        if selected is not None:
                            retain(selected, selected_ev, phase, si_floor)
                        raise
                    if selected is None:
                        break
                    retain(selected, selected_ev, phase, si_floor)
                    if np.array_equal(old, best_power):
                        break
                else:
                    discrete_exhausted = True
            discrete_completed = not discrete_exhausted
        elif method == "slsqp":
            try:
                from scipy.optimize import minimize
            except ImportError as exc:
                raise RuntimeError("SLSQP requires scipy") from exc
            active = np.flatnonzero(problem.active)
            if len(active):
                demand = problem.demand_bps
                n_active = len(active)
                # Dimensionless p/scale makes power and auxiliary SI variables
                # comparable for SLSQP's line search and stopping tests.
                power_scale = cap / n_active
                for phase in ("si", "sgm"):
                    phase_limit = (max(2, min(max_evaluations - 1,
                                              int(max_evaluations * 0.65)))
                                   if phase == "si" and max_evaluations > 2 else None)
                    phase_time_limit = 0.65 * time_limit if phase == "si" else None
                    check_limit()
                    phase_start = best_power.copy()
                    phase_si_floor = best_eval.si - si_tolerance if phase == "sgm" else None
                    if phase == "sgm":
                        phase2_si_floor = phase_si_floor
                    cache_x = None
                    cache_ev = None

                    def evaluated_x(x):
                        nonlocal cache_x, cache_ev
                        p = np.zeros(problem.beam_count)
                        p[active] = x[:n_active] * power_scale
                        if cache_x is None or not np.array_equal(x, cache_x):
                            cache_ev = evaluate(p, jac=True)
                            cache_x = np.array(x, copy=True)
                            retain(p, cache_ev, phase, phase_si_floor)
                        return cache_ev

                    def capacity_constraint(x):
                        ev = evaluated_x(x)
                        return ev.capacity_bps / demand - x[n_active:]

                    def capacity_jac(x):
                        ev = evaluated_x(x)
                        return np.column_stack((ev.capacity_jacobian[:, active] * power_scale
                                                / demand[:, None],
                                                -np.eye(problem.beam_count)))

                    def floor_constraint(x):
                        return evaluated_x(x).capacity_bps / demand - floor_ratio

                    def floor_jac(x):
                        return (evaluated_x(x).capacity_jacobian[:, active] * power_scale
                                / demand[:, None])

                    def sgm_constraint(x):
                        return evaluated_x(x).sgm - baseline_sgm

                    def sgm_jac(x):
                        return evaluated_x(x).sgm_jacobian[active] * power_scale

                    if phase == "si":
                        u0 = np.minimum(best_eval.beam_si, 1)
                        x0 = np.concatenate((phase_start[active] / power_scale, u0))
                        constraints = [
                            {"type": "ineq", "fun": lambda x: cap / power_scale - x[:n_active].sum(),
                             "jac": lambda x: np.r_[-np.ones(n_active),
                                                        np.zeros(problem.beam_count)]},
                            {"type": "ineq", "fun": capacity_constraint,
                             "jac": capacity_jac},
                            {"type": "ineq", "fun": floor_constraint,
                             "jac": lambda x: np.column_stack((floor_jac(x),
                                                              np.zeros((problem.beam_count,
                                                                        problem.beam_count))))},
                            {"type": "ineq", "fun": sgm_constraint,
                             "jac": lambda x: np.r_[sgm_jac(x),
                                                      np.zeros(problem.beam_count)]},
                        ]
                        objective = lambda x: -float(np.mean(x[n_active:]))
                        objective_jac = lambda x: np.r_[np.zeros(n_active),
                                                         -np.ones(problem.beam_count) / problem.beam_count]
                        bounds = [(0, None)] * n_active + [(0, 1)] * problem.beam_count
                    else:
                        # Keep the SI epigraph in phase two as well. Direct
                        # min(C/D, 1) has a kink at demand and a zero gradient
                        # once every beam is capped, obscuring feasible SGM
                        # directions along the SI=1 boundary.
                        u0 = np.minimum(best_eval.beam_si, 1)
                        x0 = np.concatenate((phase_start[active] / power_scale, u0))
                        constraints = [
                            {"type": "ineq", "fun": lambda x: cap / power_scale - x[:n_active].sum(),
                             "jac": lambda x: np.r_[-np.ones(n_active),
                                                        np.zeros(problem.beam_count)]},
                            {"type": "ineq", "fun": capacity_constraint,
                             "jac": capacity_jac},
                            {"type": "ineq", "fun": floor_constraint,
                             "jac": lambda x: np.column_stack((floor_jac(x),
                                                              np.zeros((problem.beam_count,
                                                                        problem.beam_count))))},
                            {"type": "ineq", "fun": sgm_constraint,
                             "jac": lambda x: np.r_[sgm_jac(x),
                                                      np.zeros(problem.beam_count)]},
                            {"type": "ineq", "fun": lambda x: np.mean(x[n_active:]) - phase_si_floor,
                             "jac": lambda x: np.r_[np.zeros(n_active),
                                                      np.ones(problem.beam_count) / problem.beam_count]},
                        ]
                        objective = lambda x: -evaluated_x(x).sgm
                        objective_jac = lambda x: np.r_[-sgm_jac(x),
                                                          np.zeros(problem.beam_count)]
                        bounds = [(0, None)] * n_active + [(0, 1)] * problem.beam_count
                    try:
                        result = minimize(objective, x0, method="SLSQP", jac=objective_jac,
                                          bounds=bounds, constraints=constraints,
                                          options={"maxiter": maxiter, "ftol": ftol, "disp": False})
                    except _LimitReached as exc:
                        solver_status.append({"phase": phase, "success": False,
                                              "message": str(exc), "iterations": None})
                        if phase == "si" and str(exc) in {
                                "phase_evaluation_limit", "phase_time_limit"}:
                            phase_limit = None
                            phase_time_limit = None
                            demand_matching_warm_start()
                            continue
                        raise
                    solver_status.append({"phase": phase, "success": bool(result.success),
                                          "message": str(result.message),
                                          "iterations": int(result.nit)})
                    phase_limit = None
                    phase_time_limit = None
                    if phase == "si":
                        demand_matching_warm_start()
    except _LimitReached as exc:
        termination = str(exc)
    except Exception:
        # Programmer/data errors are not solver failures and must be visible.
        raise

    if method == "slsqp" and any(not item["success"] for item in solver_status):
        if termination == "completed":
            termination = "solver_nonconvergence"
    if method == "discrete" and not discrete_completed and termination == "completed":
        termination = "iteration_limit"
    if method != "none" and len(history) == 1 and termination == "completed":
        termination = "no_feasible_improvement"
    converged = (method == "none" or
                 (method == "discrete" and discrete_completed and termination in {
                     "completed", "no_feasible_improvement"}) or
                 (method == "slsqp" and
                  (not np.any(problem.active) or
                   len(solver_status) == 2 and all(row["success"] for row in solver_status))
                  and termination in {"completed", "no_feasible_improvement"}))
    error = {
        "budget_excess_w": max(0.0, float(best_power.sum() - cap)),
        "negative_power_w": max(0.0, float(-best_power.min())),
        "empty_slot_power_w": float(np.max(np.abs(best_power[~problem.active])))
        if np.any(~problem.active) else 0.0,
        "sgm_shortfall": max(0.0, baseline_sgm - best_eval.sgm),
        "basic_shortfall_ratio": float(np.maximum(
            floor_ratio - best_eval.capacity_bps / problem.demand_bps, 0).max()),
        "si_shortfall": max(0.0, initial_eval.si - best_eval.si),
        "phase2_si_shortfall": (max(0.0, phase2_si_floor - best_eval.si)
                                if phase2_si_floor is not None else 0.0),
    }
    return {
        "method": method, "power_w": best_power.tolist(),
        "initial_metrics": _metrics(problem, initial, initial_eval),
        "metrics": _metrics(problem, best_power, best_eval),
        "history": history, "evaluations": count,
        "full_scene_evaluations": full_evaluations,
        "scalar_capacity_evaluations": scalar_evaluations,
        "elapsed_seconds": perf_counter() - start_time,
        "constraint_error": error,
        "termination_reason": termination,
        "converged": bool(converged),
        "fallback_used": bool(method != "none" and np.array_equal(best_power, initial)),
        "solver_status": solver_status,
        "solver_power_scale_w": power_scale,
        "warm_start_status": warm_start_status,
        "algorithm_components": (
            ["initial_allocation"] if method == "none" else
            ["discrete_5w_coordinate_and_exchange"] if method == "discrete" else
            ["slsqp_si_epigraph", "demand_matching_warm_start",
             "slsqp_sgm_epigraph"]),
        "power_cap_w": cap,
    }
