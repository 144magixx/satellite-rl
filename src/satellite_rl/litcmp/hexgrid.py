"""A1: fixed-width hexagonal satellite-angle grid."""
from time import perf_counter
import numpy as np
from satellite_rl.coverage.s1fix import geom
from .geom_plane import to_plane, from_plane, hex_lattice, assemble


def hex_layout(frame, N_budget):
    started = perf_counter()
    members = geom.rays(frame.positions)
    c0, radius = geom.cone(members)
    xy = to_plane(members, c0)
    t_init = perf_counter() - started
    started = perf_counter()
    for w in np.round(np.arange(.3, 1.001, .02), 2):
        grid = hex_lattice(radius, .85 * w)
        owner = np.sum((xy[:, None] - grid[None]) ** 2, axis=-1).argmin(axis=1)
        used = np.unique(owner)
        clusters = [np.flatnonzero(owner == j) for j in used]
        if len(clusters) <= N_budget:
            break
    return assemble(frame, clusters, from_plane(grid[used], c0), N_budget, t_init,
                    perf_counter() - started, True, fixed_width=float(w))
