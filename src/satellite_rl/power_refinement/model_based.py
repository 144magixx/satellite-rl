"""Model-based fixed-slot power proposals with exact physical acceptance.

SCA solves a trust-region linear convex approximation of the *actual* SI
epigraph and then of SGM. WMMSE solves a weighted sum-rate surrogate on the
slot links. Neither surrogate is asserted to equal the original lexicographic
SI/SGM problem; every retained point is checked with the original physics.
"""

from time import perf_counter

import numpy as np

from .physics import FixedAllocationPhysics, _sgm


def _metrics(problem, power, ev):
    capacity = ev.capacity_bps
    return {
        "final_si_mean": ev.si,
        "final_sgm_mean": ev.sgm,
        "basic_met_count": int(np.count_nonzero(capacity >= 0.1 * problem.demand_bps)),
        "power_w": float(power.sum()),
        "required_bps": problem.demand_bps.tolist(),
        "capacity_bps": capacity.tolist(),
        "effective_served_bps": np.minimum(capacity, problem.demand_bps).tolist(),
        "capacity_demand_ratio": (capacity / problem.demand_bps).tolist(),
        "beam_si": ev.beam_si.tolist(),
        "beam_sgm": ev.beam_sgm.tolist(),
    }


def _wmmse_proposal(problem, power, ev, cap, phase):
    """One exact weighted-MSE block update for a local rate surrogate.

    Each occupied slot is a scalar Gaussian interference link, while all
    slots of a beam share x_i=sqrt(p_i). The receiver and MSE weights have
    closed forms. With those fixed, the x step is a diagonal convex quadratic
    under sum(x_i**2)<=cap, solved through a scalar multiplier.
    """
    active = problem.active
    if phase == "si":
        weights = np.where(ev.capacity_bps < problem.demand_bps * (1 - 1e-8),
                           1 / problem.demand_bps, 0.0)
    else:
        _, derivative = _sgm(ev.capacity_bps, problem.demand_bps)
        weights = np.maximum(derivative, 0)
    if not np.any(weights > 0):
        return None
    weights = weights / weights.max()
    owner = problem.slot_owner
    # Row normalization removes tiny physical channel/noise units without
    # changing each slot's SINR, MMSE receiver or power minimizer.
    signal = problem.signal / problem.noise
    interference = problem.interference / problem.noise[:, None]
    x = np.sqrt(power)
    denominator = 1.0 + interference @ power
    desired = signal * power[owner]
    total = denominator + desired
    receiver = np.sqrt(signal) * x[owner] / total
    mse_weight = total / denominator
    row_weight = weights[owner] * mse_weight
    linear = np.bincount(owner, weights=row_weight * receiver * np.sqrt(signal),
                         minlength=problem.beam_count)
    quadratic = interference.T @ (row_weight * receiver**2)
    quadratic += np.bincount(owner, weights=row_weight * receiver**2 * signal,
                             minlength=problem.beam_count)
    linear[~active] = 0.0
    quadratic[~active] = 0.0
    if not np.any(linear > 0):
        return None

    def candidate(multiplier):
        result = np.divide(linear, quadratic + multiplier,
                           out=np.zeros_like(linear), where=quadratic + multiplier > 0)
        result[~active] = 0
        return result

    trial = candidate(0.0)
    if float(trial @ trial) > cap:
        low, high = 0.0, 1.0
        while float(candidate(high) @ candidate(high)) > cap:
            high *= 2
        for _ in range(50):
            middle = (low + high) / 2
            if float(candidate(middle) @ candidate(middle)) > cap:
                low = middle
            else:
                high = middle
        trial = candidate(high)
    return trial**2


def _sca_proposal(problem, power, ev, cap, phase, si_floor, trust_w,
                  floor_ratio, baseline_sgm, sgm_tolerance, remaining_time):
    """Solve an LP formed by first-order capacity/SGM models in a trust box."""
    from scipy.optimize import linprog

    active = np.flatnonzero(problem.active)
    m, n = len(active), problem.beam_count
    scale = cap / m
    x0 = power[active] / scale
    ratio = ev.capacity_bps / problem.demand_bps
    jac = ev.capacity_jacobian[:, active] * scale / problem.demand_bps[:, None]
    # Variables are scaled whole-beam powers x and epigraph service u.
    # u_i <= C_i^linear/D_i, 0<=u_i<=1.
    capacity_rows = np.column_stack((-jac, np.eye(n)))
    capacity_rhs = ratio - jac @ x0
    floor_rows = np.column_stack((-jac, np.zeros((n, n))))
    floor_rhs = ratio - jac @ x0 - floor_ratio
    grad_sgm = ev.sgm_jacobian[active] * scale
    sgm_row = np.r_[-grad_sgm, np.zeros(n)][None, :]
    sgm_rhs = np.array([ev.sgm - baseline_sgm + sgm_tolerance - grad_sgm @ x0])
    budget_row = np.r_[np.ones(m), np.zeros(n)][None, :]
    rows = [capacity_rows, floor_rows, sgm_row, budget_row]
    rhs = [capacity_rhs, floor_rhs, sgm_rhs, np.array([cap / scale])]
    if phase == "sgm":
        rows.append(np.r_[np.zeros(m), -np.ones(n)][None, :])
        rhs.append(np.array([-n * si_floor]))
        objective = np.r_[-grad_sgm, np.zeros(n)]
    else:
        objective = np.r_[np.zeros(m), -np.ones(n) / n]
    trust = trust_w / scale
    bounds = [(max(0, value - trust), value + trust) for value in x0]
    bounds.extend([(0, 1)] * n)
    result = linprog(objective, A_ub=np.vstack(rows), b_ub=np.concatenate(rhs),
                     bounds=bounds, method="highs",
                     options={"time_limit": max(0.001, remaining_time)})
    if not result.success:
        return None, str(result.message)
    proposed = np.zeros(problem.beam_count)
    proposed[active] = np.maximum(result.x[:m], 0) * scale
    return proposed, str(result.message)


def optimize_model_based(problem: FixedAllocationPhysics, initial_power,
                         method="sca", options=None, callback=None,
                         stop_requested=None):
    """Use model-based proposals within the original two-phase acceptance rule.

    Budget counts one complete physical capacity evaluation per candidate.
    LP and WMMSE surrogate operations are recorded separately and do not count
    as physical calls. A solver result is never returned without true checks.
    """
    if not isinstance(problem, FixedAllocationPhysics):
        raise TypeError("problem must be FixedAllocationPhysics")
    if method not in {"sca", "wmmse"}:
        raise ValueError(f"Unknown model-based method: {method}")
    opts = dict(options or {})
    allowed = {"power_cap_w", "maxiter", "max_evaluations", "maxeval",
               "time_limit_seconds", "max_time_seconds", "ftol", "si_tolerance",
               "sgm_tolerance", "constraint_tolerance", "protect_basic_service",
               "discrete_step_w"}
    if set(opts) - allowed:
        raise ValueError(f"Unknown optimization options: {sorted(set(opts) - allowed)}")
    cap = float(opts.get("power_cap_w", problem.power_budget_w))
    maxiter = int(opts.get("maxiter", 300))
    maxeval = int(opts.get("max_evaluations", opts.get("maxeval", 2000)))
    limit_seconds = float(opts.get("time_limit_seconds", opts.get("max_time_seconds", 30.0)))
    si_tolerance = float(opts.get("si_tolerance", 0.0))
    sgm_tolerance = float(opts.get("sgm_tolerance", 0.0))
    constraint_tolerance = float(opts.get("constraint_tolerance", 1e-8))
    ftol = float(opts.get("ftol", 1e-8))
    if (opts.get("protect_basic_service", True) is not True
            or float(opts.get("discrete_step_w", 5.0)) != 5.0):
        raise ValueError("Basic service protection and original grid control are fixed")
    if (not np.isfinite([cap, limit_seconds, si_tolerance, sgm_tolerance,
                         constraint_tolerance, ftol]).all()
            or not 0 < cap <= min(3000, problem.power_budget_w)
            or maxiter < 1 or maxeval < 1 or limit_seconds <= 0 or ftol <= 0
            or min(si_tolerance, sgm_tolerance, constraint_tolerance) < 0):
        raise ValueError("Invalid optimization limits")
    initial = np.asarray(initial_power, dtype=np.float64)
    original = problem.evaluate(initial, with_jacobian=True)
    if initial.sum() > cap + 1e-7:
        raise ValueError("Initial power exceeds selected budget")
    floor_ratio = np.minimum(original.capacity_bps / problem.demand_bps, 0.1)
    best_power, best_eval = initial.copy(), original
    start = perf_counter()
    evaluations = 1
    proposals = 0
    lp_subproblems = 0
    lp_elapsed_seconds = 0.0
    history = [{"phase": "initial", "evaluation": 1, "final_si_mean": original.si,
                "final_sgm_mean": original.sgm, "power_w": float(initial.sum()),
                "elapsed_seconds": 0.0}]
    solver_status = []
    termination = "completed"
    phase2_si_floor = None

    def timed_out():
        return perf_counter() - start >= limit_seconds

    def check_budget():
        if stop_requested is not None and stop_requested():
            return "stop_requested"
        if timed_out():
            return "time_limit"
        if evaluations >= maxeval:
            return "evaluation_limit"
        return None

    def valid(power, ev, si_floor):
        return (np.isfinite(power).all() and np.all(power >= 0)
                and np.all(power[~problem.active] == 0)
                and power.sum() <= cap + 1e-7
                and ev.sgm >= original.sgm - sgm_tolerance - 1e-10
                and np.all(ev.capacity_bps / problem.demand_bps
                           >= floor_ratio - constraint_tolerance)
                and (si_floor is None or ev.si >= si_floor - 1e-10))

    for phase in ("si", "sgm"):
        phase_start = perf_counter()
        phase_iterations = 0
        accepted = 0
        phase_stop_reason = None
        message = "no proposal"
        if phase == "sgm":
            phase2_si_floor = best_eval.si - si_tolerance
        trust = max(cap / max(1, np.count_nonzero(problem.active)), 1.0)
        trust *= 0.5
        for _ in range(maxiter):
            reason = check_budget()
            if reason:
                termination = reason
                phase_stop_reason = reason
                break
            if phase == "si" and (evaluations >= max(2, int(0.65 * maxeval))
                                  or perf_counter() - start >= 0.65 * limit_seconds):
                phase_stop_reason = "phase_budget"
                break
            if not np.any(problem.active):
                phase_stop_reason = "no_active_beams"
                break
            phase_iterations += 1
            if method == "sca":
                lp_started = perf_counter()
                proposal, message = _sca_proposal(
                    problem, best_power, best_eval, cap, phase, phase2_si_floor,
                    trust, floor_ratio, original.sgm, sgm_tolerance,
                    limit_seconds - (perf_counter() - start))
                lp_subproblems += 1
                lp_elapsed_seconds += perf_counter() - lp_started
            else:
                proposal = _wmmse_proposal(problem, best_power, best_eval, cap, phase)
                message = "weighted-MMSE rate proposal"
            proposals += 1
            if proposal is None:
                if method == "sca" and trust > 1e-5:
                    trust *= 0.5
                    continue
                if method == "sca":
                    phase_stop_reason = "trust_region_small"
                elif phase == "si":
                    phase_stop_reason = ("surrogate_zero_weights" if not np.any(
                        best_eval.capacity_bps < problem.demand_bps * (1 - 1e-8))
                        else "surrogate_zero_gain")
                else:
                    _, derivative = _sgm(best_eval.capacity_bps, problem.demand_bps)
                    phase_stop_reason = ("surrogate_zero_weights" if not np.any(
                        derivative > 0) else "surrogate_zero_gain")
                break
            direction = proposal - best_power
            if np.max(np.abs(direction)) < 1e-9:
                phase_stop_reason = "surrogate_stationary_direction"
                break
            improvement = False
            for backtrack in range(9):
                reason = check_budget()
                if reason:
                    termination = reason
                    phase_stop_reason = reason
                    break
                step = 2.0 ** -backtrack
                candidate = best_power + step * direction
                candidate[~problem.active] = 0
                candidate = np.maximum(candidate, 0)
                ev = problem.evaluate(candidate, with_jacobian=True)
                evaluations += 1
                if not valid(candidate, ev, phase2_si_floor):
                    continue
                if phase == "si":
                    improvement = (ev.si > best_eval.si + 1e-10
                                   or (abs(ev.si - best_eval.si) <= 1e-10
                                       and ev.sgm > best_eval.sgm + 1e-10))
                else:
                    improvement = (ev.sgm > best_eval.sgm + 1e-10
                                   or (abs(ev.sgm - best_eval.sgm) <= 1e-10
                                       and ev.si > best_eval.si + 1e-10))
                if improvement:
                    best_power, best_eval = candidate, ev
                    accepted += 1
                    entry = {"phase": phase, "evaluation": evaluations,
                             "final_si_mean": ev.si, "final_sgm_mean": ev.sgm,
                             "power_w": float(candidate.sum()),
                             "elapsed_seconds": perf_counter() - start,
                             "step_fraction": step}
                    history.append(entry)
                    if callback is not None:
                        callback(dict(entry))
                    break
            if termination != "completed":
                break
            if method == "sca":
                trust = min(cap, trust * 1.4) if improvement else trust * 0.5
                if trust < 1e-5:
                    phase_stop_reason = "trust_region_small"
                    break
            elif not improvement:
                phase_stop_reason = "no_acceptable_step"
                break
        if phase_stop_reason is None:
            phase_stop_reason = "iteration_limit"
        solver_status.append({"phase": phase, "success": accepted > 0,
                              "success_scope": "accepted true feasible improvement only",
                              "stop_reason": phase_stop_reason,
                              "message": message if phase_iterations else "no proposal",
                              "iterations": phase_iterations, "accepted": accepted,
                              "elapsed_seconds": perf_counter() - phase_start})
        if termination != "completed":
            break
        termination = phase_stop_reason
        if phase == "si":
            # A local phase stop still permits the SGM phase; only global
            # stop/time/evaluation limits prevent it.
            termination = "completed"
    if termination == "completed":
        termination = solver_status[-1]["stop_reason"]
    error = {
        "budget_excess_w": max(0.0, float(best_power.sum() - cap)),
        "negative_power_w": max(0.0, float(-best_power.min())),
        "empty_slot_power_w": (float(np.max(np.abs(best_power[~problem.active])))
                               if np.any(~problem.active) else 0.0),
        "sgm_shortfall": max(0.0, original.sgm - best_eval.sgm),
        "basic_shortfall_ratio": float(np.maximum(
            floor_ratio - best_eval.capacity_bps / problem.demand_bps, 0).max()),
        "si_shortfall": max(0.0, original.si - best_eval.si),
        "phase2_si_shortfall": (max(0.0, phase2_si_floor - best_eval.si)
                                if phase2_si_floor is not None else 0.0),
    }
    return {
        "method": method, "power_w": best_power.tolist(),
        "initial_metrics": _metrics(problem, initial, original),
        "metrics": _metrics(problem, best_power, best_eval),
        "history": history, "evaluations": evaluations,
        "full_scene_evaluations": evaluations, "scalar_capacity_evaluations": 0,
        "elapsed_seconds": perf_counter() - start, "constraint_error": error,
        "termination_reason": termination, "converged": False,
        "convergence_scope": "original SI/SGM stationary point not certified",
        "fallback_used": bool(np.array_equal(best_power, initial)),
        "solver_status": solver_status,
        "solver_power_scale_w": (cap / np.count_nonzero(problem.active)
                                  if np.any(problem.active) else None),
        "warm_start_status": None,
        "algorithm_components": (["sca_trust_region_lp", "true_metric_backtracking"]
                                 if method == "sca" else
                                 ["slot_weighted_mmse_rate_proposal", "true_metric_backtracking"]),
        "surrogate_proposals": proposals, "power_cap_w": cap,
        "lp_subproblems": lp_subproblems,
        "lp_elapsed_seconds": lp_elapsed_seconds,
        "surrogate_weight_rule": (
            None if method == "sca" else
            "SI: 1/D for beams below demand, zero otherwise; "
            "SGM: positive part of dSGM/dC; weights normalized per step; "
            "weighted sum-rate is a proposal only"),
    }
