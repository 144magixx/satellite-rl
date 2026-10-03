"""SI-first refinement from a frozen, already evaluated SCA allocation.

Both variants use the same trust box and true-physics acceptance.  The LP uses
the tangent capacity, while the LB variant retains the concave log(total)
term and only linearizes log(interference).  Neither claims global optimality.
"""

from time import perf_counter

import numpy as np

from .model_based import _metrics
from .physics import FixedAllocationPhysics


_BW_OVER_LN2 = 2.5e7 / np.log(2.0)
_SI_TIE = 1e-10
_SGM_TIE = 1e-10


def _capacity_lower_bound(problem, centre, trial):
    """Beam capacities and gradient of the tight global concave lower bound.

    This NumPy version is used for numerical checks.  The optimization model
    below uses the same expression in CVXPY, with each row divided by noise.
    """
    h = problem.interference / problem.noise[:, None]
    a = problem.signal / problem.noise
    owner = problem.slot_owner
    q0 = 1.0 + h @ centre
    q = 1.0 + h @ trial
    total = q + a * trial[owner]
    lower_slot = _BW_OVER_LN2 * (np.log(total) - np.log(q0) - (q - q0) / q0)
    slot_grad = _BW_OVER_LN2 * (h / total[:, None] - h / q0[:, None])
    slot_grad[np.arange(len(owner)), owner] += _BW_OVER_LN2 * a / total
    cap = np.bincount(owner, weights=lower_slot, minlength=problem.beam_count)
    jac = np.zeros((problem.beam_count, problem.beam_count))
    np.add.at(jac, owner, slot_grad)
    return cap, jac


def _proposal_lp(problem, power, ev, cap, phase, anchor_si, floor_ratio,
                 trust_w, remaining_time):
    from scipy.optimize import linprog

    active = np.flatnonzero(problem.active)
    n, m = problem.beam_count, len(active)
    scale = cap / m
    x0 = power[active] / scale
    ratio = ev.capacity_bps / problem.demand_bps
    jac = ev.capacity_jacobian[:, active] * scale / problem.demand_bps[:, None]
    rows = [np.column_stack((-jac, np.eye(n))),
            np.column_stack((-jac, np.zeros((n, n)))),
            np.r_[np.ones(m), np.zeros(n)][None, :]]
    rhs = [ratio - jac @ x0,
           ratio - jac @ x0 - floor_ratio,
           np.array([cap / scale])]
    if phase == "sgm":
        rows.append(np.r_[np.zeros(m), -np.ones(n)][None, :])
        rhs.append(np.array([-n * (anchor_si - 1e-8)]))
        objective = np.r_[-ev.sgm_jacobian[active] * scale, np.zeros(n)]
    else:
        objective = np.r_[np.zeros(m), -np.ones(n) / n]
    trust = trust_w / scale
    bounds = [(max(0.0, value - trust), value + trust) for value in x0]
    bounds.extend([(0.0, 1.0)] * n)
    result = linprog(objective, A_ub=np.vstack(rows), b_ub=np.concatenate(rhs),
                     bounds=bounds, method="highs",
                     options={"time_limit": max(0.001, remaining_time)})
    if not result.success:
        return None, str(result.message)
    candidate = np.zeros(n)
    candidate[active] = np.maximum(result.x[:m], 0.0) * scale
    return candidate, str(result.message)


def _proposal_lb(problem, power, ev, cap, phase, anchor_si, floor_ratio,
                 trust_w, remaining_time):
    """Solve the log(total)-tangent(log(interference)) concave subproblem."""
    try:
        import cvxpy as cp
    except ImportError as exc:
        raise RuntimeError("sca_si_lb requires cvxpy with Clarabel") from exc

    deadline = perf_counter() + remaining_time
    active = np.flatnonzero(problem.active)
    n, m = problem.beam_count, len(active)
    scale = cap / m
    x0 = power[active] / scale
    h = problem.interference[:, active] * scale / problem.noise[:, None]
    a = problem.signal * scale / problem.noise
    owner = problem.slot_owner
    q0 = 1.0 + h @ x0
    x = cp.Variable(m, nonneg=True)
    u = cp.Variable(n)
    q = 1.0 + h @ x
    total = q + cp.multiply(a, x[np.searchsorted(active, owner)])
    lower_slot = _BW_OVER_LN2 * (cp.log(total) - np.log(q0) -
                                  cp.multiply(1.0 / q0, q - q0))
    # Explicit incidence avoids a Python loop of scalar CVXPY expressions.
    incidence = np.zeros((n, len(owner)))
    incidence[owner, np.arange(len(owner))] = 1.0
    lower_ratio = cp.multiply(1.0 / problem.demand_bps,
                              incidence @ lower_slot)
    trust = trust_w / scale
    constraints = [x <= x0 + trust, x >= np.maximum(0.0, x0 - trust),
                   cp.sum(x) <= cap / scale, u >= 0, u <= 1,
                   u <= lower_ratio, lower_ratio >= floor_ratio]
    if phase == "sgm":
        constraints.append(cp.sum(u) / n >= anchor_si - 1e-8)
        objective = cp.Maximize(ev.sgm_jacobian[active] @ ((x - x0) * scale))
    else:
        objective = cp.Maximize(cp.sum(u) / n)
    task = cp.Problem(objective, constraints)
    try:
        # CVXPY's solver time limit excludes canonicalization.  Compile first,
        # then give Clarabel only the time still left for this entire call.
        task.get_problem_data(cp.CLARABEL)
        solver_seconds = deadline - perf_counter()
        if solver_seconds <= 0:
            return None, "time_limit_during_canonicalization"
        task.solve(solver="CLARABEL", max_iter=150,
                   time_limit=solver_seconds, verbose=False)
    except cp.SolverError as exc:
        return None, str(exc)
    if task.status not in {"optimal", "optimal_inaccurate"} or x.value is None:
        return None, str(task.status)
    candidate = np.zeros(n)
    candidate[active] = np.maximum(np.asarray(x.value).reshape(-1), 0.0) * scale
    return candidate, (f"{task.status}; compilation_seconds="
                       f"{task.compilation_time}; solver_seconds="
                       f"{task.solver_stats.solve_time}")


def optimize_priority(problem: FixedAllocationPhysics, reference_power,
                      method="sca_si_lp", options=None, callback=None,
                      stop_requested=None):
    """Improve SI first, then SGM under a nondecreasing best-SI anchor.

    The starting A allocation counts as the first physical evaluation.  A
    rejected proposal also consumes a full-scene evaluation.  The exact A
    vector is retained for every solver or validity failure.
    """
    if not isinstance(problem, FixedAllocationPhysics):
        raise TypeError("problem must be FixedAllocationPhysics")
    if method not in {"sca_si_lp", "sca_si_lb"}:
        raise ValueError(f"Unknown priority method: {method}")
    opts = dict(options or {})
    allowed = {"power_cap_w", "maxiter", "max_evaluations", "maxeval",
               "time_limit_seconds", "max_time_seconds", "si_stall_window",
               "si_stall_tolerance", "si_min_evaluations", "si_max_fraction",
               "constraint_tolerance", "safety_margin_ratio",
               "service_margin_ratio", "si_tolerance", "sgm_tolerance",
               "protect_basic_service", "baseline_service_floor_bps"}
    if set(opts) - allowed:
        raise ValueError(f"Unknown optimization options: {sorted(set(opts) - allowed)}")
    cap = float(opts.get("power_cap_w", problem.power_budget_w))
    maxiter = int(opts.get("maxiter", 300))
    maxeval = int(opts.get("max_evaluations", opts.get("maxeval", 2000)))
    limit = float(opts.get("time_limit_seconds", opts.get("max_time_seconds", 30.0)))
    stall_window = int(opts.get("si_stall_window", 200))
    stall_tolerance = float(opts.get("si_stall_tolerance", 1e-6))
    min_evals = int(opts.get("si_min_evaluations", 200))
    max_fraction = float(opts.get("si_max_fraction", 0.85))
    tolerance = float(opts.get("constraint_tolerance", 1e-8))
    margin = float(opts.get("service_margin_ratio",
                            opts.get("safety_margin_ratio", 5e-9)))
    if (opts.get("protect_basic_service", True) is not True or
            not np.isfinite([cap, limit, stall_tolerance, max_fraction,
                             tolerance, margin]).all() or
            not 0 < cap <= min(3000.0, problem.power_budget_w) or
            maxiter < 1 or maxeval < 1 or limit <= 0 or
            stall_window < 1 or min_evals < 1 or not 0 < max_fraction < 1 or
            stall_tolerance < 0 or tolerance != 1e-8 or
            float(opts.get("si_tolerance", 1e-8)) != 1e-8 or
            float(opts.get("sgm_tolerance", 1e-8)) != 1e-8 or
            not 0 <= margin <= tolerance):
        raise ValueError("Invalid priority optimization limits")
    initial = np.asarray(reference_power, dtype=np.float64)
    if initial.shape != (problem.beam_count,) or initial.sum() > cap + 1e-7:
        raise ValueError("Reference power exceeds selected budget or has wrong shape")
    start = perf_counter()
    original = problem.evaluate(initial, with_jacobian=True)
    reference_ratio = original.capacity_bps / problem.demand_bps
    baseline_floor = opts.get("baseline_service_floor_bps")
    if baseline_floor is None:
        baseline_ratio = np.zeros(problem.beam_count)
    else:
        baseline_floor = np.asarray(baseline_floor, dtype=np.float64)
        if (baseline_floor.shape != (problem.beam_count,) or
                not np.isfinite(baseline_floor).all() or
                np.any(baseline_floor < 0) or
                np.any(baseline_floor > 0.1 * problem.demand_bps + 1e-5)):
            raise ValueError("Invalid baseline service floor")
        baseline_ratio = baseline_floor / problem.demand_bps
    # A beam already meeting 0.1D must never newly cross below it.  An
    # initially short beam retains the old 1e-8 ratio numerical tolerance.
    was_met = reference_ratio >= 0.1
    floor_ratio = np.where(was_met, 0.1,
                           np.maximum(reference_ratio, baseline_ratio) - tolerance)
    surrogate_floor = np.where(was_met,
                               np.minimum(reference_ratio, 0.1 + margin),
                               np.minimum(reference_ratio, floor_ratio + margin))
    best_power, best_eval = initial.copy(), original
    anchor_si = original.si
    highest_si_power, highest_si_eval = initial.copy(), original
    evaluations, proposals, subproblems = 1, 0, 0
    surrogate_seconds = 0.0
    history = [{"phase": "initial", "evaluation": 1,
                "final_si_mean": original.si, "final_sgm_mean": original.sgm,
                "power_w": float(initial.sum()), "elapsed_seconds": 0.0}]
    solver_status = []
    termination = "completed"
    phase2_floor = None
    phase1_best_si = original.si

    def global_stop():
        if stop_requested is not None and stop_requested():
            return "stop_requested"
        if perf_counter() - start >= limit:
            return "time_limit"
        if evaluations >= maxeval:
            return "evaluation_limit"
        return None

    def valid(power, ev, phase):
        if (not np.isfinite(power).all() or np.any(power < 0) or
                np.any(power[~problem.active] != 0) or
                power.sum() > cap + 1e-7):
            return False
        ratio = ev.capacity_bps / problem.demand_bps
        if np.any(ratio[was_met] < 0.1):
            return False
        if np.any(ratio[~was_met] < floor_ratio[~was_met]):
            return False
        return phase != "sgm" or ev.si >= anchor_si - 1e-8

    for phase in ("si", "sgm"):
        phase_start = perf_counter()
        phase_evals_start = evaluations
        phase_iterations = accepted = 0
        phase_reason = None
        message = "no proposal"
        trust = max(cap / max(1, np.count_nonzero(problem.active)), 1.0) * 0.5
        if phase == "sgm":
            phase1_best_si = anchor_si
            phase2_floor = anchor_si - 1e-8
        for _ in range(maxiter):
            reason = global_stop()
            if reason:
                termination = phase_reason = reason
                break
            if phase == "si":
                used = evaluations - 1
                elapsed = perf_counter() - start
                if (used >= max_fraction * (maxeval - 1) or
                        elapsed >= max_fraction * limit):
                    phase_reason = "phase_reserve"
                    break
                if used >= max(min_evals, stall_window):
                    # Accepted SI values alone establish progress; the window
                    # spans all true calls, including rejected candidates.
                    old = max((row["final_si_mean"] for row in history
                               if row["phase"] in {"initial", "si"} and
                               row["evaluation"] <= evaluations - stall_window),
                              default=original.si)
                    if anchor_si - old <= stall_tolerance:
                        phase_reason = "si_stall"
                        break
            if not np.any(problem.active):
                phase_reason = "no_active_beams"
                break
            phase_iterations += 1
            proposal_fn = _proposal_lp if method == "sca_si_lp" else _proposal_lb
            begun = perf_counter()
            proposal, message = proposal_fn(
                problem, best_power, best_eval, cap, phase, anchor_si,
                surrogate_floor, trust, limit - (perf_counter() - start))
            subproblems += 1
            surrogate_seconds += perf_counter() - begun
            proposals += 1
            if proposal is None:
                trust *= 0.5
                if trust < 1e-5:
                    phase_reason = "trust_region_small"
                    break
                continue
            direction = proposal - best_power
            if np.max(np.abs(direction)) < 1e-9:
                phase_reason = "surrogate_stationary_direction"
                break
            improved = False
            for backtrack in range(9):
                reason = global_stop()
                if reason:
                    termination = phase_reason = reason
                    break
                step = 2.0 ** -backtrack
                candidate = np.maximum(best_power + step * direction, 0.0)
                candidate[~problem.active] = 0.0
                ev = problem.evaluate(candidate, with_jacobian=True)
                evaluations += 1
                if not valid(candidate, ev, phase):
                    continue
                # The best feasible SI anchor is monotone.  In phase two a
                # higher-SI candidate may leave the old selected point inside
                # the tightened floor; preserve whichever has higher SGM.
                anchor_changed = ev.si > anchor_si + _SI_TIE
                selected_changed = False
                if anchor_changed:
                    anchor_si = ev.si
                    highest_si_power, highest_si_eval = candidate.copy(), ev
                    phase2_floor = anchor_si - 1e-8 if phase == "sgm" else None
                    if (phase == "si" or best_eval.si < phase2_floor or
                            ev.sgm > best_eval.sgm + _SGM_TIE):
                        best_power, best_eval = candidate, ev
                        selected_changed = True
                    improved = True
                elif phase == "si":
                    improved = (ev.si > best_eval.si + _SI_TIE or
                                (abs(ev.si - best_eval.si) <= _SI_TIE and
                                 ev.sgm > best_eval.sgm + _SGM_TIE))
                    if improved:
                        best_power, best_eval = candidate, ev
                        selected_changed = True
                else:
                    improved = (ev.sgm > best_eval.sgm + _SGM_TIE and
                                ev.si >= anchor_si - 1e-8)
                    if improved:
                        best_power, best_eval = candidate, ev
                        selected_changed = True
                if improved:
                    accepted += 1
                    entry = {"phase": phase, "evaluation": evaluations,
                             "final_si_mean": best_eval.si,
                             "final_sgm_mean": best_eval.sgm,
                             "best_si_mean": anchor_si,
                             "power_w": float(best_power.sum()),
                             "anchor_changed": anchor_changed,
                             "selected_changed": selected_changed,
                             "elapsed_seconds": perf_counter() - start,
                             "step_fraction": step}
                    history.append(entry)
                    if callback is not None:
                        callback(dict(entry))
                    break
            if termination != "completed":
                break
            trust = min(cap, trust * 1.4) if improved else trust * 0.5
            if trust < 1e-5:
                phase_reason = "trust_region_small"
                break
        if phase_reason is None:
            phase_reason = "iteration_limit"
        solver_status.append({"phase": phase, "success": accepted > 0,
                              "success_scope": "accepted true feasible improvement only",
                              "stop_reason": phase_reason, "message": message,
                              "iterations": phase_iterations,
                              "evaluations": evaluations - phase_evals_start,
                              "accepted": accepted,
                              "elapsed_seconds": perf_counter() - phase_start})
        if phase == "si":
            phase1_best_si = anchor_si
        if termination != "completed":
            break
        if phase == "sgm":
            termination = phase_reason
    if termination == "completed" and solver_status:
        termination = solver_status[-1]["stop_reason"]
    final_ratio = best_eval.capacity_bps / problem.demand_bps
    error = {
        "budget_excess_w": max(0.0, float(best_power.sum() - cap)),
        "negative_power_w": max(0.0, float(-best_power.min())),
        "empty_slot_power_w": (float(np.max(np.abs(best_power[~problem.active])))
                               if np.any(~problem.active) else 0.0),
        "sgm_shortfall": max(0.0, original.sgm - best_eval.sgm),
        "basic_shortfall_ratio": float(np.maximum(floor_ratio - final_ratio, 0).max()),
        "si_shortfall": max(0.0, original.si - best_eval.si),
        "phase2_si_shortfall": (max(0.0, phase2_floor - best_eval.si)
                                if phase2_floor is not None else 0.0),
    }
    return {
        "method": method, "power_w": best_power.tolist(),
        "initial_metrics": _metrics(problem, initial, original),
        "metrics": _metrics(problem, best_power, best_eval),
        "history": history, "evaluations": evaluations,
        "full_scene_evaluations": evaluations, "scalar_capacity_evaluations": 0,
        "elapsed_seconds": perf_counter() - start,
        "constraint_error": error, "termination_reason": termination,
        "converged": False,
        "convergence_scope": "original SI/SGM stationary point not certified",
        "fallback_used": bool(np.array_equal(best_power, initial)),
        "solver_status": solver_status,
        "solver_power_scale_w": (cap / np.count_nonzero(problem.active)
                                 if np.any(problem.active) else None),
        "warm_start_status": None,
        "algorithm_components": (["si_priority_lp", "true_metric_backtracking"]
                                 if method == "sca_si_lp" else
                                 ["log_capacity_concave_lower_bound",
                                  "si_priority", "true_metric_backtracking"]),
        "surrogate_proposals": proposals,
        "surrogate_subproblems": subproblems,
        "surrogate_elapsed_seconds": surrogate_seconds,
        "surrogate_timing_scope": "CVXPY canonicalization plus solver or LP solver",
        "lp_subproblems": subproblems if method == "sca_si_lp" else 0,
        "lp_elapsed_seconds": surrogate_seconds if method == "sca_si_lp" else 0.0,
        "power_cap_w": cap, "phase1_best_si_mean": phase1_best_si,
        "best_feasible_si_mean": anchor_si, "phase2_si_floor": phase2_floor,
        "highest_si_power_w": highest_si_power.tolist(),
        "highest_si_metrics": _metrics(problem, highest_si_power,
                                        highest_si_eval),
        "sgm_change_from_reference": best_eval.sgm - original.sgm,
        "si_stall_window": stall_window,
        "si_stall_tolerance": stall_tolerance,
        "si_min_evaluations": min_evals,
        "si_max_fraction": max_fraction,
        "safety_margin_ratio": margin,
    }
