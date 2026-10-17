"""Continuous noise-limited bandwidth/power relaxation, then slot packing."""
import numpy as np
from scipy.optimize import minimize
from satellite_rl.s2fix.evaluate import fixed_problem
from .power_lit import B, noise_limited
from .moves import Alloc, pack


def convex_relax(stage_input, g1_alloc):
    problem = fixed_problem(stage_input, *g1_alloc)
    gamma = noise_limited(problem, g1_alloc)
    demand = problem.demand_bps
    scale, n = demand.sum(), len(demand)
    eye = np.eye(n)

    def capacity(k, p):
        return k * B * np.log1p(gamma * p / k) / np.log(2) / scale

    def constraints(x):
        k, p, s = x.reshape(3, n)
        return np.r_[capacity(k, p) - s, 4.5 * B * k / scale - s,
                     (800 - k.sum()) / 800, (3000 - p.sum()) / 3000]

    def jacobian(x):
        k, p, _ = x.reshape(3, n)
        z = gamma * p / k
        dk = B / (scale * np.log(2)) * (np.log1p(z) - z / (1 + z))
        dp = B * gamma / (scale * np.log(2) * (1 + z))
        return np.vstack((np.hstack((eye * dk, eye * dp, -eye)),
                          np.hstack((eye * (4.5 * B / scale), np.zeros((n, n)), -eye)),
                          np.r_[-np.ones(n) / 800, np.zeros(2 * n)],
                          np.r_[np.zeros(n), -np.ones(n) / 3000, np.zeros(n)]))

    k, p = g1_alloc.counts.astype(float), g1_alloc.powers
    s = np.minimum.reduce((capacity(k, p), 4.5 * B * k / scale, demand / scale))
    gradient = np.r_[np.zeros(2 * n), -np.ones(n)]
    result = minimize(lambda x: -x[2*n:].sum(), np.r_[k, p, s], jac=lambda x: gradient,
                      bounds=[(1, 10)] * n + [(0, 50)] * n + [(0, d / scale) for d in demand],
                      constraints=dict(type='ineq', fun=constraints, jac=jacobian),
                      method='SLSQP', options=dict(maxiter=500, ftol=1e-9))
    if not result.success:
        raise RuntimeError(f'B3 SLSQP failed: {result.message}')
    k, p, _ = result.x.reshape(3, n)
    starts, counts = pack(g1_alloc.groups, np.clip(np.rint(k), 1, 10).astype(int), demand)
    return Alloc(g1_alloc.groups.copy(), starts, counts, p.copy())
