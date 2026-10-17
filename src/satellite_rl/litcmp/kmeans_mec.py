"""A2: load-weighted K-means++ with enclosing-circle repair."""
from time import perf_counter
import numpy as np
from satellite_rl.coverage.s1fix import geom
from .geom_plane import fit_budget, assemble


def kmeans_layout(frame, N_budget, seed=51001):
    started = perf_counter()
    points = geom.rays(frame.positions)
    K = min(N_budget, len(np.unique(points, axis=0)))
    rng = np.random.default_rng(seed)
    centers = [points[rng.integers(len(points))]]
    for _ in range(1, K):
        d2 = geom.angles(points, centers).min(axis=1) ** 2
        centers.append(points[rng.choice(len(points), p=d2 / d2.sum())])
    centers = np.array(centers)
    delta = np.zeros(K)
    t_init = perf_counter() - started
    started = perf_counter()
    for _ in range(30):
        previous = None
        for _ in range(50):
            distance = geom.angles(points, centers)
            owner = (distance * (1 + .5 * np.maximum(delta, 0))).argmin(axis=1)
            loads = np.bincount(owner, minlength=K)
            largest = np.flatnonzero(owner == loads.argmax())
            for j in range(K):
                if loads[j]:
                    centers[j] = points[owner == j].sum(axis=0)
                    centers[j] /= np.linalg.norm(centers[j])
                else:
                    centers[j] = points[largest[distance[largest, loads.argmax()].argmax()]]
            if previous is not None and np.array_equal(owner, previous):
                break
            previous = owner.copy()
        delta = (loads - len(points) / K) / (len(points) / K)
        if np.abs(delta).max() < .2:
            break
    clusters = [np.flatnonzero(owner == j) for j in range(K) if loads[j]]
    clusters, centers, feasible = fit_budget(points, clusters, N_budget)
    return assemble(frame, clusters, centers, N_budget, t_init, perf_counter() - started, feasible)
