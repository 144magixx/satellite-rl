"""Small satellite-angle geometry tools for S1-FIX (degrees throughout)."""

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from satellite_rl.coverage.geometry import satellite_rays_numpy
from satellite_rl.envs.regions import get_profile


@dataclass
class Layout:
    centers_latlon: np.ndarray
    widths: np.ndarray
    owner: np.ndarray
    beam_ids: np.ndarray
    stack_id: np.ndarray | None = None


def rays(positions):
    return satellite_rays_numpy(positions, "europe_v1")


def angles(a, b):
    """Pairwise angular distances, with stable precision at zero distance."""
    a, b = np.atleast_2d(a), np.atleast_2d(b)
    chord = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)
    return np.rad2deg(2 * np.arcsin(np.minimum(chord / 2, 1)))


def neighbor_lists(centers, points, radius=0.5):
    chord = 2 * np.sin(np.deg2rad(radius + 1e-10) / 2)
    return cKDTree(np.asarray(centers)).query_ball_point(points, chord)


def _circle2(a, b):
    center = (a + b) / 2
    return center, np.linalg.norm(a - center)


def _circle3(a, b, c):
    matrix = 2 * np.stack((b - a, c - a))
    if abs(np.linalg.det(matrix)) < 1e-18:
        options = [_circle2(a, b), _circle2(a, c), _circle2(b, c)]
        return max(options, key=lambda item: item[1])
    center = np.linalg.solve(matrix, np.array([b @ b - a @ a, c @ c - a @ a]))
    return center, np.linalg.norm(a - center)


def _welzl(points):
    """Randomized incremental form of Welzl's minimum enclosing circle."""
    p = np.asarray(points)[np.random.default_rng(51001).permutation(len(points))]
    center, radius = p[0].copy(), 0.0
    for i, a in enumerate(p):
        if np.linalg.norm(a - center) <= radius + 1e-13:
            continue
        center, radius = a.copy(), 0.0
        for j in range(i):
            b = p[j]
            if np.linalg.norm(b - center) <= radius + 1e-13:
                continue
            center, radius = _circle2(a, b)
            for k in range(j):
                c = p[k]
                if np.linalg.norm(c - center) > radius + 1e-13:
                    center, radius = _circle3(a, b, c)
    return center, radius


def cone(member_rays):
    """Gnomonic minimum enclosing circle, mapped back to a unit ray."""
    members = np.asarray(member_rays, dtype=float)
    axis = members.mean(axis=0)
    axis /= np.linalg.norm(axis)
    basis = np.eye(3)[np.argmin(np.abs(axis))]
    u = np.cross(axis, basis)
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    plane = np.column_stack((members @ u, members @ v)) / (members @ axis)[:, None]
    center, _ = _welzl(plane)
    direction = axis + center[0] * u + center[1] * v
    direction /= np.linalg.norm(direction)
    return direction, float(angles(direction, members).max())


def ray_to_latlon(directions):
    """Intersect satellite-origin rays with the near Earth surface."""
    direction = np.asarray(directions, dtype=float)
    profile = get_profile("europe_v1")
    lon = np.deg2rad(profile.satellite_longitude_deg)
    satellite = profile.orbit_radius_km * np.array([np.cos(lon), np.sin(lon), 0.0])
    dot = direction @ satellite
    distance = -dot - np.sqrt(np.maximum(dot ** 2 - (satellite @ satellite - profile.earth_radius_km ** 2), 0.))
    point = satellite + distance[..., None] * direction
    return np.stack((np.rad2deg(np.arctan2(point[..., 2], np.linalg.norm(point[..., :2], axis=-1))),
                     np.rad2deg(np.arctan2(point[..., 1], point[..., 0]))), axis=-1)


def envelope(centers_rays, member_rays, owner):
    widths = np.full(len(centers_rays), 0.3)
    if len(owner):
        distance = np.rad2deg(2 * np.arcsin(np.minimum(
            np.linalg.norm(member_rays - centers_rays[owner], axis=1) / 2, 1)))
        maximum = np.zeros(len(centers_rays))
        np.maximum.at(maximum, owner, distance)
        widths = np.maximum(widths, 2 * maximum) + 1e-9
        # Keep numerical envelope slack inside the physical 1-degree limit.
        widths = np.where((widths > 1) & (widths <= 1 + 1e-8), 1., widths)
    return widths


def validate_layout(layout, member_rays, N_budget, spacing=True, fixed_width=None,
                    per_beam_cap=None, stack_id=None):
    """The one algorithmic legality check: coverage, widths, count and spacing."""
    centers = np.asarray(layout.centers_latlon)
    widths, owner = np.asarray(layout.widths), np.asarray(layout.owner)
    count = len(centers)
    if not 0 <= count <= min(N_budget, 200):
        return False
    if centers.shape != (count, 2) or widths.shape != (count,):
        return False
    if owner.shape != (len(member_rays),) or len(layout.beam_ids) != count:
        return False
    if not (np.isfinite(centers).all() and np.isfinite(widths).all()):
        return False
    if len(np.unique(layout.beam_ids)) != count or owner.dtype.kind not in "iu":
        return False
    if not len(member_rays):
        return count == 0
    if count == 0 or (owner < 0).any() or (owner >= count).any():
        return False
    if per_beam_cap is not None and np.bincount(owner, minlength=count).max() > per_beam_cap:
        return False
    if (np.bincount(owner, minlength=count) == 0).any():
        return False
    if (widths < 0.3 - 1e-8).any() or (widths > 1 + 1e-8).any():
        return False
    center_rays = rays(centers)
    required = envelope(center_rays, member_rays, owner)
    if fixed_width is None:
        exact = np.ones(count, dtype=bool)
        stacks = layout.stack_id if stack_id is None else stack_id
        if stacks is not None:
            exact = np.asarray(stacks) < 0
        if not np.allclose(widths[exact], required[exact], rtol=0, atol=3e-8):
            return False
    elif not np.allclose(widths, fixed_width, rtol=0, atol=1e-8):
        return False
    if (required > widths + 3e-8).any():
        return False
    if spacing and count > 1:
        distances = angles(center_rays, center_rays)
        np.fill_diagonal(distances, np.inf)
        stacks = layout.stack_id if stack_id is None else stack_id
        if stacks is not None:
            stacks = np.asarray(stacks)
            distances[(stacks[:, None] >= 0) & (stacks[:, None] == stacks[None, :])] = np.inf
        if (distances + 1e-8 < np.maximum(widths[:, None], widths[None, :]) / 2).any():
            return False
    return True


def metrics(layout):
    n_beams = len(layout.widths)
    if n_beams == 0:
        return dict(N_used=0, peak=0, top10=0., CV=0., Omega_sr=0.,
                    width_mean=0., width_min=0., width_max=0.)
    loads = np.bincount(layout.owner, minlength=n_beams)
    return dict(N_used=n_beams, peak=int(loads.max()),
                top10=float(np.sort(loads)[-int(np.ceil(n_beams * .1)):].mean()),
                CV=float(loads.std() / loads.mean()),
                Omega_sr=float(np.sum(2 * np.pi * (1 - np.cos(np.deg2rad(layout.widths / 2))))),
                width_mean=float(np.mean(layout.widths)), width_min=float(np.min(layout.widths)),
                width_max=float(np.max(layout.widths)))


def forecast(lat, lon, gs_kt, track_deg, dt_s):
    lat, lon, speed, track = np.broadcast_arrays(lat, lon, gs_kt, track_deg)
    valid = np.isfinite(speed) & np.isfinite(track)
    travel = np.where(valid, speed, 0) * 0.514444444 * dt_s / 6371000
    heading = np.deg2rad(np.where(valid, track, 0))
    phi, lam = np.deg2rad(lat), np.deg2rad(lon)
    phi2 = np.arcsin(np.sin(phi) * np.cos(travel) + np.cos(phi) * np.sin(travel) * np.cos(heading))
    lam2 = lam + np.arctan2(np.sin(heading) * np.sin(travel) * np.cos(phi),
                          np.cos(travel) - np.sin(phi) * np.sin(phi2))
    result = np.stack((np.rad2deg(phi2), (np.rad2deg(lam2) + 180) % 360 - 180), axis=-1)
    return np.where(valid[..., None], result, np.stack((lat, lon), axis=-1))
