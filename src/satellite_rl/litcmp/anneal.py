"""Geometrically cooled simulated annealing for bandwidth and power."""
import numpy as np
from .moves import KINDS, propose
from .objective import j_gap


def anneal(stage_input, start, probe, budget, seed, objective=j_gap, kinds=KINDS):
    probe.objective = objective
    x = start
    value = objective(probe(x, count=False))
    temperature = .05 * max(value, .01)
    cooling = .02 ** (1 / budget)
    rng = np.random.default_rng(seed)
    for _ in range(budget):
        y = propose(x, rng, kinds)
        candidate = objective(probe(y))
        if candidate <= value or rng.random() < np.exp(-(candidate - value) / temperature):
            x, value = y, candidate
        temperature *= cooling
    return probe.best
