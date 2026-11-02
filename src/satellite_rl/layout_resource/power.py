"""Fixed-slot continuous power adapter for the authoritative shared layout.

The shared coefficient/capacity kernel remains the only physical source. Legacy
scene construction (including its index-based groups and 0.5 degree pattern) is
never called. A failed or infeasible solver trial is reported, not projected.
"""
from copy import deepcopy
from time import perf_counter

import numpy as np

from satellite_rl.power_refinement.physics import FixedAllocationPhysics, PhysicalEvaluation, _sgm
from satellite_rl.power_refinement.priority import optimize_priority
from satellite_rl.shared_interface.content import seal
from satellite_rl.shared_interface.link import capacity_bps
from satellite_rl.shared_interface.stages import UniformStage3FixedInput


class _InfeasibleTrial(ValueError):
    pass


class LayoutPowerProblem(FixedAllocationPhysics):
    """Expose shared coefficients and their derivative to existing SCA code."""

    def __init__(self, stage3_input):
        if not isinstance(stage3_input, UniformStage3FixedInput):
            raise TypeError("published UniformStage3FixedInput required")
        stage = UniformStage3FixedInput.from_layout_and_allocation(
            stage3_input.case, stage3_input.layout, stage3_input.allocation,
            stage3_input.authority, stage3_input.physics)
        requested = stage.allocation['requested_actions_separate']
        if not isinstance(requested, dict) or requested.get('terminal') is not True:
            raise ValueError('complete terminal S2 allocation required; partial export refused')
        self.stage = stage
        fixed = stage.fixed_initial_evaluation()
        self.coefficients = fixed["coefficients"]
        coeff = self.coefficients
        ids = coeff.beam_ids
        allocation = stage.allocation
        order = [allocation["beam_ids"].index(bid) for bid in ids]
        super().__init__(coeff.demand_bps.astype(np.float64),
            np.array([allocation["group_by_beam_id"][bid] for bid in ids], dtype=np.int64),
            allocation["actual_start_slot"][order].copy(),
            allocation["actual_slot_count"][order].copy(),
            coeff.signal, coeff.interference, coeff.noise, coeff.slot_owner,
            stage.physics["power_budget_w"])
        self.beam_ids = ids
        self.initial_power_w = coeff.powers_initial.copy()

    def evaluate(self, power_w, with_jacobian=False):
        power = np.asarray(power_w, dtype=np.float64)
        capacity = np.asarray(capacity_bps(self.coefficients, power), dtype=np.float64)
        beam_si = np.minimum(capacity / self.demand_bps, 1.0)
        beam_sgm, sgm_prime = _sgm(capacity, self.demand_bps)
        values = (capacity, float(beam_si.mean()), float(beam_sgm.mean()), beam_si, beam_sgm)
        if not with_jacobian:
            return PhysicalEvaluation(*values)
        denominator = self.noise + self.interference @ power
        desired = self.signal * power[self.slot_owner]
        scale = self.stage.physics["slot_Hz"] / np.log(2)
        row_jac = (-scale * desired[:, None] * self.interference
                   / (denominator[:, None] * (denominator + desired)[:, None]))
        row_jac[np.arange(len(self.slot_owner)), self.slot_owner] += (
            scale * self.signal / (denominator + desired))
        jac = np.zeros((self.beam_count, self.beam_count), dtype=np.float64)
        np.add.at(jac, self.slot_owner, row_jac)
        si_prime = np.where(capacity < self.demand_bps, 1 / self.demand_bps, 0.0)
        return PhysicalEvaluation(*values, jac, si_prime @ jac / self.beam_count,
                                  sgm_prime @ jac / self.beam_count)

    def refined_record(self, power_w, *, method, diagnostics):
        """Preserve the actual S2 start, all slots/groups, and immutable parents."""
        result = self.evaluate(power_w)
        return seal(dict(schema="ContinuousLayoutPowerV1",
            parent_case_SHA=self.stage.case["content_sha256"],
            parent_layout_SHA=self.stage.layout["content_sha256"],
            parent_allocation_SHA=self.stage.allocation["content_sha256"],
            physics_config_SHA=self.coefficients.metadata["physics_config_SHA"],
            beam_ids=list(self.beam_ids), group_by_beam_id=dict(zip(self.beam_ids, self.groups.tolist())),
            actual_start_slot=self.starts.copy(), actual_slot_count=self.slots.copy(),
            initial_whole_beam_power_w=self.initial_power_w.copy(),
            refined_whole_beam_power_w=np.asarray(power_w, np.float64).copy(),
            request_bps_by_beam_int64=self.coefficients.demand_bps.copy(),
            capacity_bps=result.capacity_bps.copy(), SI=result.si, SGM=result.sgm,
            units=dict(power="whole_beam_watt", request="bit/s", capacity="bit/s"),
            method=method, diagnostics=deepcopy(diagnostics)))


def refine_layout_power(stage3_input, *, options):
    """Run an explicitly bounded existing SI-priority LP/SCA on the new PHY.

    This library call is not a workbench training/experiment launcher. Every
    call requires an explicit finite budget. The adapter refuses overspending
    proposals (including solver rounding above 3000 W) and reports a fallback
    to the exact S2 vector if such a trial aborts the existing solver.
    """
    required = {"maxiter", "maxeval", "max_time_seconds"}
    if not isinstance(options, dict) or not required <= options.keys():
        raise ValueError("explicit maxiter/maxeval/max_time_seconds required")
    if set(options) != required:
        raise ValueError('only maxiter/maxeval/max_time_seconds supported; no overriding aliases')
    for key in ("maxiter", "maxeval"):
        if type(options[key]) is not int or options[key] < 1:
            raise ValueError("positive integer solver budget required")
    if options['maxeval'] < 4:
        raise ValueError('maxeval must cover fixed input, solver initial, final check and record (at least 4)')
    seconds = options["max_time_seconds"]
    if type(seconds) not in (int, float) or not np.isfinite(seconds) or seconds <= 0:
        raise ValueError("finite positive solver time budget required")
    problem = LayoutPowerProblem(stage3_input)
    started = perf_counter()
    evaluations = 1  # Constructor's shared fixed-initial capacity evaluation.
    original_evaluate = problem.evaluate

    def counted(power_w, with_jacobian=False):
        nonlocal evaluations
        evaluations += 1
        power = np.asarray(power_w, dtype=np.float64)
        if (power.shape != (problem.beam_count,) or not np.isfinite(power).all()
                or (power < 0).any() or (power[~problem.active] != 0).any()
                or float(power.sum()) > problem.power_budget_w):
            raise _InfeasibleTrial('solver proposed power outside shared constraints')
        return original_evaluate(power, with_jacobian)

    problem.evaluate = counted
    try:
        solver_options = dict(options, maxeval=options['maxeval'] - 3)
        result = optimize_priority(problem, problem.initial_power_w.copy(),
                                   method="sca_si_lp", options=solver_options)
        power = np.asarray(result["power_w"], np.float64)
        counted(power)  # Strict shared constraints also apply at publication.
        diagnostics = dict(status="completed_bounded_solver", solver=result,
                           adapter_evaluation_attempts=evaluations,
                           effect_claim="no trained-policy or global-optimum claim")
    except _InfeasibleTrial as error:
        # No clipping, reassignment or substitution of an unverified candidate.
        power = problem.initial_power_w.copy()
        diagnostics = dict(status="solver_rejected_fallback", error=str(error),
                           adapter_evaluation_attempts=evaluations,
                           fallback="exact actual S2 initial power",
                           effect_claim="failed solver call, no improvement claim")
    problem.evaluate = original_evaluate
    diagnostics["options"] = deepcopy(options)
    diagnostics["record_verification_calls"] = 1
    diagnostics['total_capacity_attempts_including_record'] = evaluations + 1
    diagnostics["adapter_elapsed_seconds"] = perf_counter() - started
    return problem.refined_record(power, method="sca_si_lp", diagnostics=diagnostics)
