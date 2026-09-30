"""Satellite-view geometry for first-stage fixed-width coverage.

Ground locations are latitude/longitude degrees.  Aircraft altitude is not used
in the existing aviation adapter's coverage rule and is likewise absent here.
Squared distance between unit rays avoids acos cancellation for tiny angles.
"""

from __future__ import annotations

import numpy as np
import torch

from satellite_rl.envs.regions import get_profile


def satellite_rays_numpy(locations, profile="europe_v1"):
    p = get_profile(profile)
    a = np.asarray(locations, dtype=np.float64)
    lat, lon = np.deg2rad(a[..., 0]), np.deg2rad(a[..., 1])
    surface = p.earth_radius_km * np.stack(
        (np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)), axis=-1)
    slon = np.deg2rad(p.satellite_longitude_deg)
    satellite = p.orbit_radius_km * np.array([np.cos(slon), np.sin(slon), 0.0])
    rays = surface - satellite
    return rays / np.linalg.norm(rays, axis=-1, keepdims=True)


def satellite_rays_torch(locations, profile="europe_v1"):
    p = get_profile(profile)
    lat, lon = torch.deg2rad(locations[..., 0]), torch.deg2rad(locations[..., 1])
    surface = p.earth_radius_km * torch.stack(
        (lat.cos() * lon.cos(), lat.cos() * lon.sin(), lat.sin()), dim=-1)
    slon = np.deg2rad(p.satellite_longitude_deg)
    satellite = locations.new_tensor(
        [p.orbit_radius_km * np.cos(slon), p.orbit_radius_km * np.sin(slon), 0.0])
    rays = surface - satellite
    return torch.nn.functional.normalize(rays, dim=-1)


def squared_ray_limit(angle_deg):
    return 4.0 * np.sin(np.deg2rad(float(angle_deg)) / 2.0) ** 2


def ray_angle_numpy(first, second):
    chord = np.linalg.norm(np.asarray(first) - np.asarray(second), axis=-1)
    return np.rad2deg(2.0 * np.arcsin(np.minimum(chord / 2.0, 1.0)))


def ray_angle_torch(first, second):
    chord = torch.linalg.vector_norm(first - second, dim=-1)
    return torch.rad2deg(2.0 * torch.asin((chord / 2.0).clamp(max=1.0)))


def visible_numpy(locations, profile="europe_v1"):
    p = get_profile(profile)
    a = np.asarray(locations, dtype=np.float64)
    in_bounds = ((a[..., 0] >= p.latitude_bounds_deg[0]) &
                 (a[..., 0] <= p.latitude_bounds_deg[1]) &
                 (a[..., 1] >= p.longitude_bounds_deg[0]) &
                 (a[..., 1] <= p.longitude_bounds_deg[1]))
    lat = np.deg2rad(a[..., 0])
    lon_delta = np.deg2rad(a[..., 1] - p.satellite_longitude_deg)
    cos_angle = np.cos(lat) * np.cos(lon_delta)
    elevation = np.rad2deg(np.arctan2(
        cos_angle - p.earth_radius_km / p.orbit_radius_km,
        np.sqrt(np.maximum(0.0, 1.0 - cos_angle ** 2))))
    return in_bounds & (elevation >= p.min_elevation_deg)


def visible_torch(locations, profile="europe_v1"):
    p = get_profile(profile)
    lat, lon = locations[..., 0], locations[..., 1]
    bounds = ((lat >= p.latitude_bounds_deg[0]) & (lat <= p.latitude_bounds_deg[1]) &
              (lon >= p.longitude_bounds_deg[0]) & (lon <= p.longitude_bounds_deg[1]))
    cosine = torch.deg2rad(lat).cos() * torch.deg2rad(lon - p.satellite_longitude_deg).cos()
    elevation = torch.rad2deg(torch.atan2(
        cosine - p.earth_radius_km / p.orbit_radius_km,
        (1.0 - cosine.square()).clamp_min(0.0).sqrt()))
    return bounds & (elevation >= p.min_elevation_deg)


def assign_numpy(positions, demands, centers, radius_deg=0.25, profile="europe_v1"):
    """Independent float64 assignment; tie goes to lowest stable beam ID."""
    users = satellite_rays_numpy(positions, profile)
    beams = satellite_rays_numpy(centers, profile)
    distances = np.sum((users[:, None, :] - beams[None, :, :]) ** 2, axis=-1)
    nearest = np.argmin(distances, axis=1)
    covered = distances[np.arange(len(users)), nearest] <= squared_ray_limit(radius_deg) + 1e-14
    assignment = np.where(covered, nearest, -1)
    loads = np.bincount(assignment[covered], weights=np.asarray(demands)[covered], minlength=len(beams))
    return assignment, loads
