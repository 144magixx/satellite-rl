"""A3: damped Voronoi-generator load balancing."""
from time import perf_counter
import numpy as np
from scipy.spatial import Delaunay
from satellite_rl.coverage.s1fix import geom
from .geom_plane import to_plane, from_plane, hex_lattice, fit_budget, assemble


def voronoi_layout(frame, N_budget):
    started = perf_counter()
    members = geom.rays(frame.positions)
    c0, radius = geom.cone(members)
    xy = to_plane(members, c0)
    d0 = 2 * radius / np.sqrt(N_budget)
    while True:
        grid = hex_lattice(radius, d0)
        grid = grid[np.linalg.norm(grid, axis=1) <= radius]
        if len(grid) >= N_budget:
            break
        d0 *= .95
    g = grid[np.argsort(np.linalg.norm(grid, axis=1), kind="stable")[:N_budget]].copy()
    v = np.zeros_like(g)
    average = len(members) / N_budget
    t_init = perf_counter() - started
    started = perf_counter()
    for _ in range(200):
        owner = np.sum((xy[:, None] - g[None]) ** 2, axis=-1).argmin(axis=1)
        loads = np.bincount(owner, minlength=N_budget)
        if np.max(np.abs(loads - average)) / average < .2:
            break
        triangles = Delaunay(g).simplices
        edges = np.unique(np.sort(np.concatenate((triangles[:, [0, 1]], triangles[:, [1, 2]],
                                                  triangles[:, [0, 2]])), axis=1), axis=0)
        i, j = edges.T
        direction = g[j] - g[i]
        force = ((loads[j] - loads[i]) / average)[:, None] * direction / np.linalg.norm(direction, axis=1)[:, None]
        F = np.zeros_like(g)
        np.add.at(F, i, force)
        np.add.at(F, j, force)
        v = .5 * v + .1 * d0 * F
        g += v
    owner = np.sum((xy[:, None] - g[None]) ** 2, axis=-1).argmin(axis=1)
    used = np.unique(owner)
    clusters = [np.flatnonzero(owner == j) for j in used]
    clusters, centers, feasible = fit_budget(members, clusters, N_budget, "keep", from_plane(g[used], c0))
    return assemble(frame, clusters, centers, N_budget, t_init, perf_counter() - started, feasible)
