"""Noise-limited and frozen-interference literature power allocations."""
import numpy as np
from satellite_rl.s2fix.evaluate import fixed_problem

B = 25e6


def noise_limited(problem, alloc, interference=None):
    """Per-watt/per-slot SNR using mean received interference per beam.

    The physical signal coefficient already divides by the slot count.
    Frozen interference is in received watts per occupied receiver slot.
    """
    owner = problem.slot_owner
    def mean(rows):
        return np.bincount(owner, weights=rows, minlength=len(alloc.counts)) / alloc.counts
    # Signal (including 1/k) and noise are constant across a beam's slots
    # in the shared v2 kernel; interference varies with slot overlaps.
    signal = mean(problem.signal) * alloc.counts
    denominator = mean(problem.noise)
    if interference is not None:
        denominator = denominator + mean(interference)
    return signal / denominator * 10 ** (-problem.implementation_loss_dB / 10)


def waterfill(gamma, k, D, budget=3000, cap=50, omega=0.0):
    gamma, k, D = map(lambda a: np.asarray(a, float), (gamma, k, D))
    required = k * np.expm1(np.minimum(D / (k * B), 4.5) * np.log(2)) / gamma
    upper = np.minimum(cap, required)

    def power(level):
        if level + omega == 0:
            return upper.copy()
        return np.clip(k * (B / ((level + omega) * np.log(2)) - 1 / gamma), 0, upper)

    p = power(0.)
    if p.sum() <= budget:
        return p
    lo, hi = 0., float(np.max(B * gamma / np.log(2)))
    for _ in range(40):
        mid = (lo + hi) / 2
        if power(mid).sum() > budget:
            lo = mid
        else:
            hi = mid
    return power(hi)


def C1(stage_input, g1_alloc):
    problem = fixed_problem(stage_input, *g1_alloc)
    return waterfill(noise_limited(problem, g1_alloc), g1_alloc.counts, problem.demand_bps)


def C2(stage_input, g1_alloc, seed, probe):
    """Coarse true-objective search followed by a frozen-interference solve.

    The fine solve optimizes its approximation, and need not improve the
    actual coupled-interference objective over the coarse allocation.
    """
    from .anneal import anneal
    problem = fixed_problem(stage_input, *g1_alloc)
    omega = .001 * problem.demand_bps.sum() / 3000

    def objective(result):
        return (np.maximum(result['demand_bps'] - result['capacity_bps'], 0).sum()
                + omega * result['total_power_w'])

    probe.objective = objective
    coarse = anneal(stage_input, g1_alloc, probe, 320, seed, objective, ['power'])
    gamma = noise_limited(problem, coarse, problem.interference @ coarse.powers)
    return waterfill(gamma, coarse.counts, problem.demand_bps, omega=omega)
