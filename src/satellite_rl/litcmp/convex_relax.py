"""Continuous noise-limited bandwidth/power relaxation solved by Lagrangian dual decomposition, then slot packing.

maximize sum_b min(k_b B log2(1 + g_b p_b / k_b), 4.5 B k_b, d_b)
subject to sum k <= 800, sum p <= 3000, 1 <= k_b <= 10, 0 <= p_b <= 50.
Two multipliers (slots, power) couple the beams; given them, each beam's power is the
water-filling level capped where its capacity stops counting, and its slot count is the
best point on a fine grid. The multipliers are found by nested bisection on the two budgets.
"""
import numpy as np
from satellite_rl.s2fix.evaluate import fixed_problem
from .power_lit import B, noise_limited
from .moves import Alloc, pack

K_GRID = np.linspace(1, 10, 181)


def _beam_best(gamma, demand, lam, mu):
    k = K_GRID[None, :]
    g = gamma[:, None]
    ceiling = np.minimum(4.5 * B * k, demand[:, None])
    p_sat = k / g * np.expm1(ceiling / (k * B) * np.log(2))
    p_water = k * (B / (mu * np.log(2)) - 1 / g) if mu > 0 else np.full_like(p_sat, np.inf)
    p = np.clip(p_water, 0, np.minimum(p_sat, 50))
    served = np.minimum(k * B * np.log1p(g * p / k) / np.log(2), ceiling)
    best = np.argmax(served - lam * k - mu * p, axis=1)
    rows = np.arange(len(gamma))
    return K_GRID[best], p[rows, best]


def _slots_for(gamma, demand, mu):
    """Smallest slot price that keeps the total slot count within 800."""
    if _beam_best(gamma, demand, 0., mu)[0].sum() <= 800:
        return 0.
    low, high = 0., 1.
    while _beam_best(gamma, demand, high, mu)[0].sum() > 800:
        high *= 2
    for _ in range(50):
        mid = (low + high) / 2
        low, high = (low, mid) if _beam_best(gamma, demand, mid, mu)[0].sum() <= 800 else (mid, high)
    return high


def convex_relax(stage_input, g1_alloc):
    problem = fixed_problem(stage_input, *g1_alloc)
    gamma = noise_limited(problem, g1_alloc)
    demand = np.asarray(problem.demand_bps, float)
    power = lambda mu: _beam_best(gamma, demand, _slots_for(gamma, demand, mu), mu)[1].sum()
    mu = 0.
    if power(0.) > 3000:
        low, high = 0., 1.
        while power(high) > 3000:
            high *= 2
        for _ in range(50):
            mid = (low + high) / 2
            low, high = (low, mid) if power(mid) <= 3000 else (mid, high)
        mu = high
    k, p = _beam_best(gamma, demand, _slots_for(gamma, demand, mu), mu)
    starts, counts = pack(g1_alloc.groups, np.clip(np.rint(k), 1, 10).astype(int), demand)
    return Alloc(g1_alloc.groups.copy(), starts, counts, p.copy())
