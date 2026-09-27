"""Versioned GEO regions. Europe is a synthetic reference, not measured traffic."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RegionProfile:
    id: str
    label: str
    environment_version: str
    satellite_longitude_deg: float
    latitude_bounds_deg: tuple
    longitude_bounds_deg: tuple
    min_elevation_deg: object
    default_scene_kind: str
    earth_radius_km: int = 6371
    orbit_radius_km: int = 42164


_PROFILES = {
    "china_legacy": RegionProfile(
        "china_legacy", "中国原环境", "china_legacy_v1", 122.2,
        (4.0, 54.0), (73.0, 135.0), None, "china_city_synthetic_demand"),
    "europe_v1": RegionProfile(
        "europe_v1", "欧洲独立环境", "europe_geo13_v1", 13.0,
        (25.0, 80.0), (-40.0, 50.0), 10.0, "europe_synthetic_reference_grid"),
}


def get_profile(profile="china_legacy"):
    if isinstance(profile, RegionProfile):
        if profile not in _PROFILES.values():
            raise ValueError("地区参数必须来自已登记的环境版本")
        return profile
    if not isinstance(profile, str) or profile not in _PROFILES:
        raise ValueError(f"未知地区环境: {profile!r}")
    return _PROFILES[profile]


def profile_metadata(profile="china_legacy"):
    p = get_profile(profile)
    return dict(
        id=p.id, label=p.label, environment_version=p.environment_version,
        satellite_longitude_deg=p.satellite_longitude_deg,
        satellite_latitude_deg=0.0, earth_radius_km=p.earth_radius_km,
        orbit_radius_km=p.orbit_radius_km,
        latitude_bounds_deg=list(p.latitude_bounds_deg),
        longitude_bounds_deg=list(p.longitude_bounds_deg),
        min_elevation_deg=p.min_elevation_deg, coordinate_order=["latitude", "longitude"],
        distance_unit="km", default_scene_kind=p.default_scene_kind,
        position_normalization=dict(
            version=f"{p.id}_latlon_affine_v1", output_range=[-1, 1],
            latitude_bounds_deg=list(p.latitude_bounds_deg),
            longitude_bounds_deg=list(p.longitude_bounds_deg)),
        validation_policy="legacy_global_coordinates" if p.id == "china_legacy" else "region_and_elevation",
        beam_count=100, beam_width_deg=0.5,
        parameter_basis=("中国历史参数与城市生成器保持兼容；未新增地区或仰角过滤"
                         if p.id == "china_legacy" else
                         "13°E及25–80°N、40°W–50°E包络参考Al-Hraishawi等2020；"
                         "10°最小仰角、球形地球6371km、GEO半径42164km为本项目工程约定；"
                         "默认网格位置与需求均为合成检查样例，不是航空或实测业务"),
        reference_url="https://arxiv.org/abs/2007.07148" if p.id == "europe_v1" else None,
    )


def profile_catalog():
    return [profile_metadata(p) for p in _PROFILES]


def _locations(locations):
    a = np.asarray(locations, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != 2 or not np.isfinite(a).all():
        raise ValueError("位置应为有限的N×2数组，顺序为纬度、经度")
    if (np.abs(a[:, 0]) > 90).any() or (np.abs(a[:, 1]) > 180).any():
        raise ValueError("位置超出经纬度范围，顺序为纬度、经度")
    return a


def geo_distances_km(locations, profile="china_legacy"):
    """Slant range, using the legacy Cartesian arithmetic order."""
    p, a = get_profile(profile), _locations(locations)
    satellite_lon_rad = np.deg2rad(p.satellite_longitude_deg)
    x_sat = p.orbit_radius_km * np.cos(satellite_lon_rad)
    y_sat = p.orbit_radius_km * np.sin(satellite_lon_rad)
    lat_rad, lon_rad = np.deg2rad(a[:, 0]), np.deg2rad(a[:, 1])
    x_ground = p.earth_radius_km * np.cos(lat_rad) * np.cos(lon_rad)
    y_ground = p.earth_radius_km * np.cos(lat_rad) * np.sin(lon_rad)
    z_ground = p.earth_radius_km * np.sin(lat_rad)
    return np.sqrt((x_sat - x_ground) ** 2 + (y_sat - y_ground) ** 2 + z_ground ** 2)


def elevation_degrees(locations, profile="china_legacy"):
    p, a = get_profile(profile), _locations(locations)
    cos_angle = np.cos(np.deg2rad(a[:, 0])) * np.cos(
        np.deg2rad(a[:, 1] - p.satellite_longitude_deg))
    return np.rad2deg(np.arctan2(
        cos_angle - p.earth_radius_km / p.orbit_radius_km,
        np.sqrt(np.maximum(0.0, 1.0 - cos_angle ** 2))))


def validate_locations(locations, profile="china_legacy"):
    """Reject invalid Europe points; never silently clip or remove imported rows."""
    p, a = get_profile(profile), _locations(locations)
    if p.id == "china_legacy":
        return a
    lat_min, lat_max = p.latitude_bounds_deg
    lon_min, lon_max = p.longitude_bounds_deg
    outside = ((a[:, 0] < lat_min) | (a[:, 0] > lat_max)
               | (a[:, 1] < lon_min) | (a[:, 1] > lon_max))
    if outside.any():
        raise ValueError(f"{p.id} 地区范围外有{int(outside.sum())}束；位置顺序必须为纬度、经度")
    unavailable = elevation_degrees(a, p) < p.min_elevation_deg
    if unavailable.any():
        raise ValueError(f"{p.id} 有{int(unavailable.sum())}束低于最小仰角{p.min_elevation_deg:g}°")
    return a


def synthetic_reference_locations(profile="europe_v1", count=100):
    """Sample distinct visible grid points using the caller's NumPy RNG."""
    p = get_profile(profile)
    if p.id != "europe_v1":
        raise ValueError("此合成网格仅用于欧洲参考环境")
    lat, lon = np.meshgrid(np.linspace(*p.latitude_bounds_deg, 20),
                           np.linspace(*p.longitude_bounds_deg, 24), indexing="ij")
    candidates = np.column_stack((lat.ravel(), lon.ravel()))
    visible = elevation_degrees(candidates, p) >= p.min_elevation_deg
    valid = candidates[visible]
    if count > len(valid):
        raise ValueError("可见合成候选位置不足")
    positions = valid[np.random.choice(len(valid), size=count, replace=False)]
    return positions, dict(kind=p.default_scene_kind, candidate_count=len(candidates),
                           excluded_below_elevation_count=int((~visible).sum()),
                           selected_count=count, measured_traffic=False)
