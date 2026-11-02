"""Fixed-group genetic search with demand-ordered packing."""
import numpy as np
from satellite_rl.s2fix.greedy import proportional_power
from .moves import Alloc, pack
from .objective import j_gap


def genetic(stage_input, start, probe, budget, seed, objective=j_gap):
    size, generations = {32: (8, 3), 320: (16, 19), 3200: (32, 99)}[budget]
    rng = np.random.default_rng(seed)
    n = len(start.groups)
    demand = np.asarray(stage_input.layout['request_bps_by_beam_int64'])
    probe.objective = objective

    def mutate(individual):
        counts, raw = (x.copy() for x in individual)
        for i in np.flatnonzero(rng.random(n) < 1 / n):
            if rng.random() < .5:
                counts[i] = np.clip(counts[i] + rng.choice([-1, 1]), 1, 10)
            else:
                raw[i] = np.clip(raw[i] * rng.uniform(.8, 1.25), 0, 50)
        return counts, raw

    def evaluate(individual):
        counts, raw = individual
        starts, counts = pack(start.groups, counts, demand)
        alloc = Alloc(start.groups.copy(), starts, counts, proportional_power(raw))
        return objective(probe(alloc))

    ancestor = (start.counts.copy(), start.powers.copy())
    population = [ancestor] + [mutate(ancestor) for _ in range(size - 1)]
    scores = [objective(probe(start, count=False))] + [evaluate(x) for x in population[1:]]

    def select():
        a, b = rng.choice(size, 2, replace=False)
        return population[a if scores[a] <= scores[b] else b]

    for _ in range(generations):
        children = [population[int(np.argmin(scores))]]
        while len(children) < size:
            a, b = select(), select()
            if rng.random() < .9:
                mask = rng.random(n) < .5
                pair = [tuple(np.where(mask, x, y) for x, y in zip(a, b)),
                        tuple(np.where(mask, y, x) for x, y in zip(a, b))]
            else:
                pair = [a, b]
            children.extend(mutate(child) for child in pair)
        population = children[:size]
        # Reevaluate the elite too: explicit population/generation budgets are 31/319/3199.
        scores = [evaluate(x) for x in population]
    return probe.best
