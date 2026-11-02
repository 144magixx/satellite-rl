"""Plane construction and spherical coverage repair for literature layouts."""
from dataclasses import dataclass
from time import perf_counter
import numpy as np
from satellite_rl.coverage.s1fix import geom


@dataclass
class Stage1Result:
    layout: geom.Layout
    feasible: bool
    t_init: float
    t_opt: float
    t_check: float
    t_total: float
    spacing_violations: int
    N_budget: int


def tangent(c0):
    e1 = np.cross(c0, np.eye(3)[np.argmin(np.abs(c0))])
    e1 /= np.linalg.norm(e1)
    return e1, np.cross(c0, e1)


def to_plane(rays, c0):
    e1, e2 = tangent(c0)
    return np.column_stack((rays @ e1, rays @ e2)) / (rays @ c0)[:, None] * (180 / np.pi)


def from_plane(xy, c0):
    e1, e2 = tangent(c0)
    r = c0 + np.deg2rad(xy) @ np.stack((e1, e2))
    return r / np.linalg.norm(r, axis=-1, keepdims=True)


def hex_lattice(R, d):
    extent = int(np.ceil((R + d) / (d * np.sqrt(3) / 2)))
    return np.array([(i * d + (j % 2) * d / 2, j * d * np.sqrt(3) / 2)
                     for j in range(-extent, extent + 1)
                     for i in range(-extent, extent + 1)
                     if np.hypot(i * d + (j % 2) * d / 2, j * d * np.sqrt(3) / 2) <= R + d])


def _split(points):
    first, second = np.unravel_index(np.argmax(geom.angles(points, points)), (len(points), len(points)))
    centers = points[[first, second]].copy()
    for _ in range(10):
        owner = geom.angles(points, centers).argmin(axis=1)
        centers = np.array([points[owner == j].sum(axis=0) for j in range(2)])
        centers /= np.linalg.norm(centers, axis=1)[:, None]
    return owner


def fit_budget(member_rays, clusters, N_budget, center_mode="cone", centers_rays=None):
    """Repair actual coverage width; retain generator centers only if unsplit.

    A displaced coincident cluster is terminal and uses its common member ray.
    """
    pending = [(np.asarray(c), None if centers_rays is None else centers_rays[i])
               for i, c in enumerate(clusters)]
    clusters, centers = [], []
    while pending:
        members, original = pending.pop(0)
        center, radius = geom.cone(member_rays[members])
        if center_mode == "keep" and original is not None:
            radius = geom.angles(original, member_rays[members]).max()
        if radius > .5 + 1e-10:
            if len(np.unique(member_rays[members], axis=0)) == 1:
                pending.append((members, None))
            else:
                owner = _split(member_rays[members])
                pending.extend((members[owner == j], None) for j in range(2))
        else:
            if center_mode == "keep" and original is not None:
                center = original
            clusters.append(members)
            centers.append(center)
    while len(clusters) > N_budget:
        distances = geom.angles(centers, centers)
        best = None
        for i, j in zip(*np.where(np.triu(distances <= 1., 1))):
            members = np.concatenate((clusters[i], clusters[j]))
            center, radius = geom.cone(member_rays[members])
            if radius <= .5 + 1e-10 and (best is None or radius < best[0]):
                best = radius, i, j, members, center
        if best is None:
            return clusters, np.array(centers), False
        _, i, j, members, center = best
        clusters[i], centers[i] = members, center
        del clusters[j], centers[j]
    return clusters, np.array(centers), True


def assemble(frame, clusters, centers_rays, N_budget, t_init, t_opt, feasible, fixed_width=None):
    started = perf_counter()
    members = geom.rays(frame.positions)
    owner = np.empty(len(members), dtype=np.int64)
    for j, cluster in enumerate(clusters):
        owner[cluster] = j
    positions = geom.ray_to_latlon(centers_rays)
    centers_rays = geom.rays(positions)
    widths = geom.envelope(centers_rays, members, owner)
    if fixed_width is not None:
        widths[:] = fixed_width
    layout = geom.Layout(positions, widths, owner, np.arange(len(clusters)))
    valid = geom.validate_layout(layout, members, N_budget, spacing=False, fixed_width=fixed_width)
    distances = geom.angles(centers_rays, centers_rays)
    np.fill_diagonal(distances, np.inf)
    nearest = distances.min(axis=1)
    violations = int(np.count_nonzero((nearest < .15) | (widths / 2 > nearest)))
    t_check = perf_counter() - started
    return Stage1Result(layout, bool(feasible and valid), t_init, t_opt, t_check,
                        t_init + t_opt + t_check, violations, N_budget)
