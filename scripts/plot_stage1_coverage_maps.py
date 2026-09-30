"""Render audited first-stage aircraft/beam coverage maps from saved results.

This script only reads seven existing workbench result JSON files. A footprint
is the near intersection of a 0.25-degree GEO beam ray cone with a spherical
Earth, projected back to latitude/longitude. It is geometric coverage, not
link capacity, interference, or actual delivered aircraft service.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path

import numpy as np


GROUPS = ("R0", "R1", "R2", "C0", "C1", "C2", "P0")
SCENE_INDICES = tuple(5 + 10 * index for index in range(10))
EARTH_RADIUS_KM = 6371.0
ORBIT_RADIUS_KM = 42164.0
SATELLITE_LONGITUDE_DEG = 13.0
BEAM_FULL_WIDTH_DEG = 0.5
BEAM_HALF_ANGLE_DEG = BEAM_FULL_WIDTH_DEG / 2
MIN_ELEVATION_DEG = 10.0
LAT_BOUNDS = (25.0, 80.0)
LON_BOUNDS = (-40.0, 50.0)
BOUNDARY_SAMPLES = 360
ANGLE_QA_TOL_DEG = 1e-7
SPHERE_QA_TOL_KM = 1e-5


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def source_business_sha(aircraft_ids, positions, demands):
    """Hash result-visible aircraft business; timestamps stay in source SHA."""
    rows = sorted((str(identifier), float(point[0]), float(point[1]), float(demand))
                  for identifier, point, demand in zip(aircraft_ids, positions, demands))
    return hashlib.sha256(json.dumps(rows, ensure_ascii=False, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def satellite_position():
    longitude = np.deg2rad(SATELLITE_LONGITUDE_DEG)
    return ORBIT_RADIUS_KM * np.array([np.cos(longitude), np.sin(longitude), 0.0])


def ground_position(latlon):
    point = np.asarray(latlon, dtype=np.float64)
    lat, lon = np.deg2rad(point[..., 0]), np.deg2rad(point[..., 1])
    return EARTH_RADIUS_KM * np.stack((np.cos(lat) * np.cos(lon),
                                      np.cos(lat) * np.sin(lon), np.sin(lat)), axis=-1)


def ground_latlon(xyz):
    xyz = np.asarray(xyz, dtype=np.float64)
    radius = np.linalg.norm(xyz, axis=-1)
    lat = np.rad2deg(np.arcsin(np.clip(xyz[..., 2] / radius, -1.0, 1.0)))
    lon = np.rad2deg(np.arctan2(xyz[..., 1], xyz[..., 0]))
    return np.stack((lat, lon), axis=-1)


def elevation_degrees(latlon):
    point = np.asarray(latlon, dtype=np.float64)
    cosine = np.cos(np.deg2rad(point[..., 0])) * np.cos(
        np.deg2rad(point[..., 1] - SATELLITE_LONGITUDE_DEG))
    return np.rad2deg(np.arctan2(
        cosine - EARTH_RADIUS_KM / ORBIT_RADIUS_KM,
        np.sqrt(np.maximum(0.0, 1.0 - cosine ** 2))))


def inside_visible_region(latlon):
    point = np.asarray(latlon, dtype=np.float64)
    lat, lon = point[..., 0], point[..., 1]
    return (np.isfinite(point).all(axis=-1) &
            (lat >= LAT_BOUNDS[0]) & (lat <= LAT_BOUNDS[1]) &
            (lon >= LON_BOUNDS[0]) & (lon <= LON_BOUNDS[1]) &
            (elevation_degrees(point) >= MIN_ELEVATION_DEG))


def _beam_basis(center):
    satellite = satellite_position()
    axis = ground_position(center) - satellite
    axis /= np.linalg.norm(axis)
    reference = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(axis, reference))) > .95:
        reference = np.array([0.0, 1.0, 0.0])
    first = np.cross(axis, reference)
    first /= np.linalg.norm(first)
    second = np.cross(axis, first)
    return satellite, axis, first, second


def _cone_intersection(basis, phases):
    """Near sphere intersection of cone rays; no flat-Earth approximation."""
    satellite, axis, first, second = basis
    phase = np.asarray(phases, dtype=np.float64)
    alpha = np.deg2rad(BEAM_HALF_ANGLE_DEG)
    direction = (np.cos(alpha) * axis + np.sin(alpha) *
                 (np.cos(phase)[..., None] * first + np.sin(phase)[..., None] * second))
    projection = np.einsum("...i,i->...", direction, satellite)
    discriminant = projection ** 2 - (ORBIT_RADIUS_KM ** 2 - EARTH_RADIUS_KM ** 2)
    intersects = discriminant >= 0.0
    near = -projection - np.sqrt(np.maximum(discriminant, 0.0))
    intersects &= near > 0.0
    xyz = satellite + near[..., None] * direction
    latlon = ground_latlon(xyz)
    latlon = np.where(intersects[..., None], latlon, np.nan)
    return latlon, xyz, direction, intersects


def _boundary_valid(latlon, intersects):
    return intersects & inside_visible_region(latlon)


def _transition(basis, lo, hi, valid_lo):
    """Bisection locates a geographic/visibility clip along the cone ring."""
    for _ in range(30):
        midpoint = (lo + hi) / 2.0
        point, _, _, intersects = _cone_intersection(basis, np.array([midpoint]))
        valid = bool(_boundary_valid(point, intersects)[0])
        if valid == valid_lo:
            lo = midpoint
        else:
            hi = midpoint
    phase = lo if valid_lo else hi
    point, _, _, _ = _cone_intersection(basis, np.array([phase]))
    return point[0]


def footprint(center):
    """Return clipped lat/lon line segments plus independent geometry QA."""
    basis = _beam_basis(center)
    phases = np.linspace(0.0, 2.0 * np.pi, BOUNDARY_SAMPLES + 1)
    projected, xyz, direction, intersects = _cone_intersection(basis, phases)
    valid = _boundary_valid(projected, intersects)
    outside_roi = (intersects &
                   ((projected[:, 0] < LAT_BOUNDS[0]) | (projected[:, 0] > LAT_BOUNDS[1]) |
                    (projected[:, 1] < LON_BOUNDS[0]) | (projected[:, 1] > LON_BOUNDS[1])))
    below_elevation = intersects & (elevation_degrees(projected) < MIN_ELEVATION_DEG)
    clipped = not bool(valid.all())
    segments = []
    current = []
    for i in range(BOUNDARY_SAMPLES):
        left, right = bool(valid[i]), bool(valid[i + 1])
        if left and right:
            if not current:
                current.append(projected[i])
            current.append(projected[i + 1])
        elif left and not right:
            if not current:
                current.append(projected[i])
            current.append(_transition(basis, phases[i], phases[i + 1], True))
            segments.append(np.asarray(current))
            current = []
        elif not left and right:
            current = [_transition(basis, phases[i], phases[i + 1], False), projected[i + 1]]
    if current:
        segments.append(np.asarray(current))
    # Eight fixed ray azimuths are sampled again, independently of the plotted
    # polyline. Their off-axis angle and Earth radius must match the constants.
    qa_phases = np.deg2rad(np.arange(0.0, 360.0, 45.0))
    _, sample_xyz, _, sample_hit = _cone_intersection(basis, qa_phases)
    axis = basis[1]
    hit_rays = sample_xyz[sample_hit] - basis[0]
    hit_rays /= np.linalg.norm(hit_rays, axis=1, keepdims=True)
    chord = np.linalg.norm(hit_rays - axis, axis=1)
    angles = np.rad2deg(2.0 * np.arcsin(np.minimum(chord / 2.0, 1.0)))
    angle_error = (float(np.max(np.abs(angles - BEAM_HALF_ANGLE_DEG)))
                   if sample_hit.any() else None)
    sphere_error = (float(np.max(np.abs(np.linalg.norm(sample_xyz[sample_hit], axis=1) -
                                         EARTH_RADIUS_KM))) if sample_hit.any() else None)
    if (angle_error is None or angle_error > ANGLE_QA_TOL_DEG or
            sphere_error is None or sphere_error > SPHERE_QA_TOL_KM):
        raise ValueError(f"cone/sphere QA failed for center {center}: angle={angle_error}, "
                         f"sphere={sphere_error}")
    if not segments:
        raise ValueError(f"beam at {center} has no visible outline in the declared Europe region")
    return {"segments": segments, "complete": not clipped,
            "fill": projected if not clipped else None,
            "qa": {"azimuth_samples": 8,
                   "sphere_intersections": int(sample_hit.sum()),
                   "miss_sphere_points": int((~intersects).sum()),
                   "outside_europe_roi_points": int(outside_roi.sum()),
                   "below_10deg_elevation_points": int(below_elevation.sum()),
                   "max_off_axis_error_deg": angle_error,
                   "max_sphere_radius_error_km": sphere_error,
                   "boundary_samples": BOUNDARY_SAMPLES,
                   "valid_projected_points": int(valid.sum()),
                   "clipped_to_region_or_visibility": clipped}}


def _geojson_polygons(geometry):
    if not isinstance(geometry, dict):
        return
    kind = geometry.get("type")
    if kind == "Polygon":
        coordinates = geometry.get("coordinates") or []
        if coordinates:
            yield coordinates[0]
    elif kind == "MultiPolygon":
        for polygon in geometry.get("coordinates") or []:
            if polygon:
                yield polygon[0]
    elif kind == "GeometryCollection":
        for item in geometry.get("geometries") or []:
            yield from _geojson_polygons(item)


def load_basemap(path):
    if path is None:
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    geometries = ([item.get("geometry") for item in data.get("features", [])]
                  if data.get("type") == "FeatureCollection" else [data])
    polygons = []
    for geometry in geometries:
        for coordinates in _geojson_polygons(geometry):
            polygon = np.asarray(coordinates, dtype=np.float64)
            if polygon.ndim != 2 or polygon.shape[1] < 2:
                continue
            polygon = polygon[:, :2]
            if (np.max(polygon[:, 0]) < LON_BOUNDS[0] or
                    np.min(polygon[:, 0]) > LON_BOUNDS[1] or
                    np.max(polygon[:, 1]) < LAT_BOUNDS[0] or
                    np.min(polygon[:, 1]) > LAT_BOUNDS[1]):
                continue
            polygons.append(polygon)
    return polygons


def _row_layout(row):
    centers = np.asarray(row.get("final_centers", row.get("final_positions")), dtype=np.float64)
    assignments = np.asarray(row.get("associations", row.get("final_associations")), dtype=np.int64)
    active = np.asarray(row.get("active_beam_mask", [True] * 100), dtype=bool)
    loads = np.asarray(row["beam_loads_bps"], dtype=np.float64)
    if (centers.shape != (100, 2) or active.shape != (100,) or
            loads.shape != (100,) or not np.isfinite(centers).all() or
            not np.isfinite(loads).all() or not active.all()):
        raise ValueError("selected completed scene lacks 100 finite active centers and loads")
    return centers, assignments, active, loads


def load_results(results_dir):
    results, hashes = {}, {}
    for group in GROUPS:
        path = results_dir / f"{group}.json"
        hashes[group] = sha256(path)
        document = json.loads(path.read_text(encoding="utf-8"))
        rows = document.get("scenes", [])
        if len(rows) != 100:
            raise ValueError(f"{group}: expected all 100 original scenes")
        for index in SCENE_INDICES:
            row = rows[index]
            if row.get("scene_index") != index or row.get("status") != "completed":
                raise ValueError(f"{group} scene {index}: missing completed original result")
            input_data = row.get("input") or {}
            positions = np.asarray(input_data.get("positions"), dtype=np.float64)
            demands = np.asarray(input_data.get("demands_bps"), dtype=np.float64)
            ids = input_data.get("aircraft_ids")
            centers, associations, active, loads = _row_layout(row)
            if (positions.shape != (len(ids), 2) or demands.shape != (len(ids),) or
                    associations.shape != (len(ids),) or len(ids) == 0 or
                    not np.isfinite(positions).all() or not np.isfinite(demands).all()):
                raise ValueError(f"{group} scene {index}: raw aircraft dimensions differ")
            if (int(np.count_nonzero(associations >= 0)) !=
                    len(ids) - int(row["metrics"]["uncovered_users"])):
                raise ValueError(f"{group} scene {index}: saved associations and uncovered count differ")
            business_sha = source_business_sha(ids, positions, demands)
            row["_plot"] = {"positions": positions, "demands": demands,
                            "ids": ids, "centers": centers,
                            "associations": associations, "active": active,
                            "loads_mbps": loads / 1e6,
                            "business_input_sha256": business_sha,
                            "selected_time_utc": input_data.get("metadata", {}).get("selected_time_utc")}
        results[group] = document
    for index in SCENE_INDICES:
        reference = results["R0"]["scenes"][index]
        ref_data = reference["_plot"]
        identity = (reference["scene_id"], reference["scene_sha256"],
                    ref_data["business_input_sha256"], ref_data["selected_time_utc"])
        for group in GROUPS[1:]:
            row = results[group]["scenes"][index]
            value = row["_plot"]
            if (row["scene_id"], row["scene_sha256"],
                    value["business_input_sha256"], value["selected_time_utc"]) != identity:
                raise ValueError(f"scene {index}: seven groups use different aircraft business or time")
    return results, hashes


def build_footprints(results):
    registry = {}
    maximum_load = 0.0
    for group in GROUPS:
        for index in SCENE_INDICES:
            row = results[group]["scenes"][index]
            item = row["_plot"]
            maximum_load = max(maximum_load, float(item["loads_mbps"].max()))
            footprints = [footprint(center) for center in item["centers"]]
            item["footprints"] = footprints
            registry[(group, index)] = footprints
    return registry, maximum_load


def common_scene_extents(results):
    """Share one viewport among seven groups for each original aircraft scene."""
    extents = {}
    for index in SCENE_INDICES:
        parts = [results["R0"]["scenes"][index]["_plot"]["positions"]]
        for group in GROUPS:
            item = results[group]["scenes"][index]["_plot"]
            parts.append(item["centers"])
            parts.extend(segment for outline in item["footprints"]
                         for segment in outline["segments"])
        points = np.concatenate(parts, axis=0)
        lat_min, lat_max = float(points[:, 0].min()), float(points[:, 0].max())
        lon_min, lon_max = float(points[:, 1].min()), float(points[:, 1].max())
        lat_pad = max(1.0, (lat_max - lat_min) * .04)
        lon_pad = max(1.0, (lon_max - lon_min) * .04)
        extents[index] = (
            max(LON_BOUNDS[0], lon_min - lon_pad),
            min(LON_BOUNDS[1], lon_max + lon_pad),
            max(LAT_BOUNDS[0], lat_min - lat_pad),
            min(LAT_BOUNDS[1], lat_max + lat_pad),
        )
    return extents


def _draw_map(ax, row, *, norm, cmap, basemap, extent, compact=False):
    item = row["_plot"]
    for polygon in basemap:
        ax.fill(polygon[:, 0], polygon[:, 1], facecolor="#eef2f4",
                edgecolor="#bdc8cf", linewidth=.28 if compact else .4,
                zorder=0)
    for beam, outline in enumerate(item["footprints"]):
        color = cmap(norm(item["loads_mbps"][beam]))
        if outline["fill"] is not None:
            ring = outline["fill"]
            ax.fill(ring[:, 1], ring[:, 0], color=color, alpha=.045, zorder=1)
        for segment in outline["segments"]:
            ax.plot(segment[:, 1], segment[:, 0], color=color,
                    alpha=.58 if compact else .72,
                    linewidth=.45 if compact else .72, zorder=2)
    positions = item["positions"]
    covered = item["associations"] >= 0
    ax.scatter(positions[covered, 1], positions[covered, 0],
               s=2.8 if compact else 7, c="#2f3942", alpha=.64,
               linewidths=0, zorder=4)
    if (~covered).any():
        ax.scatter(positions[~covered, 1], positions[~covered, 0],
                   s=11 if compact else 25, c="#c62232", marker="x",
                   linewidths=.65 if compact else 1.0, zorder=5)
    ax.scatter(item["centers"][:, 1], item["centers"][:, 0],
               c=item["loads_mbps"], cmap=cmap, norm=norm, marker="+",
               s=8 if compact else 18, linewidths=.5 if compact else .8, zorder=3)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect(1.0 / np.cos(np.deg2rad((extent[2] + extent[3]) / 2.0)))
    ax.grid(color="#aab5bf", alpha=.2, linewidth=.4)
    if not compact:
        ax.set_xlabel("Longitude (°E)")
        ax.set_ylabel("Latitude (°N)")


def _title(group, row, *, compact=False):
    metric = row["metrics"]
    clock = row["_plot"]["selected_time_utc"] or "UTC time missing"
    if compact:
        return (f"{group} · {clock[11:16]} UTC\n"
                f"cover {100*metric['coverage_demand']:.2f}% · CV {metric['load_cv']:.3f}"
                f" · missed {int(metric['uncovered_users'])}")
    return (f"{group} · {clock} · demand coverage {100*metric['coverage_demand']:.3f}%"
            f" · load CV {metric['load_cv']:.3f}\n"
            f"uncovered aircraft {int(metric['uncovered_users'])} · "
            f"feasible {'yes' if metric['feasible'] else 'no'}")


def _legend_note(ax):
    ax.axis("off")
    ax.text(.05, .9, "Shared symbols", transform=ax.transAxes, fontsize=15, weight="bold")
    ax.text(.05, .78, "• dark dot: geometrically covered aircraft", transform=ax.transAxes)
    ax.text(.05, .68, "× red: uncovered aircraft", color="#c62232", transform=ax.transAxes)
    ax.text(.05, .58, "+ : beam center", transform=ax.transAxes)
    ax.text(.05, .47, "Contour: GEO ray cone ∩ spherical Earth", transform=ax.transAxes)
    ax.text(.05, .36, "Color: assigned demand (Mbps); shared across 70 maps", transform=ax.transAxes)
    ax.text(.05, .2, "Open outline: clipped by Earth limb, Europe ROI, or 10° elevation.\n"
            "Coverage is geometry, not capacity or interference.",
            transform=ax.transAxes, fontsize=10)


def render(results, output, basemap, maximum_load, extents, *, single_maps=True):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    cmap = plt.get_cmap("viridis")
    norm = Normalize(vmin=0.0, vmax=max(maximum_load, 1.0))
    scalar = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    records = []
    for group in GROUPS:
        group_dir = output / group
        group_dir.mkdir(parents=True, exist_ok=True)
        if single_maps:
            for index in SCENE_INDICES:
                row = results[group]["scenes"][index]
                fig, ax = plt.subplots(figsize=(13, 10), dpi=170)
                _draw_map(ax, row, norm=norm, cmap=cmap, basemap=basemap,
                          extent=extents[index])
                ax.set_title(_title(group, row), fontsize=13)
                bar = fig.colorbar(scalar, ax=ax, fraction=.028, pad=.025)
                bar.set_label("Assigned demand per beam (Mbps)")
                fig.text(.5, .018,
                         "Geometric footprint only · 13°E GEO · 0.5° full width · clipped at Earth limb / Europe ROI / 10° elevation",
                         ha="center", fontsize=9, color="#52606d")
                fig.tight_layout(rect=(0, .025, 1, 1))
                file = group_dir / f"scene_{index:03d}.png"
                fig.savefig(file, dpi=170)
                plt.close(fig)
                qa = [footprint_data["qa"] for footprint_data in row["_plot"]["footprints"]]
                records.append({"group": group, "scene_index": index, "scene_id": row["scene_id"],
                                "source_sha256": row["scene_sha256"],
                                "business_input_sha256": row["_plot"]["business_input_sha256"],
                                "selected_time_utc": row["_plot"]["selected_time_utc"],
                                "terminal_layout_sha256": row.get("final_layout_hash"),
                                "active_beams": int(row["_plot"]["active"].sum()),
                                "covered_aircraft": int((row["_plot"]["associations"] >= 0).sum()),
                                "uncovered_aircraft": int((row["_plot"]["associations"] < 0).sum()),
                                "demand_coverage": float(row["metrics"]["coverage_demand"]),
                                "load_cv": float(row["metrics"]["load_cv"]),
                                "feasible": bool(row["metrics"]["feasible"]),
                                "clipped_beams": sum(value["clipped_to_region_or_visibility"] for value in qa),
                                "max_off_axis_error_deg": max(value["max_off_axis_error_deg"] for value in qa),
                                "max_sphere_radius_error_km": max(value["max_sphere_radius_error_km"] for value in qa),
                                "boundary_qa_by_beam": qa,
                                "image": str(file.relative_to(output)).replace("\\", "/"),
                                "shared_scene_extent_lonlat": list(extents[index]),
                                "image_sha256": sha256(file),
                                "image_pixels": [int(round(13*170)), int(round(10*170))]})
        fig, axes = plt.subplots(2, 5, figsize=(25, 11), dpi=130)
        for ax, index in zip(axes.flat, SCENE_INDICES):
            row = results[group]["scenes"][index]
            _draw_map(ax, row, norm=norm, cmap=cmap, basemap=basemap,
                      extent=extents[index], compact=True)
            ax.set_title(_title(group, row, compact=True), fontsize=10)
        fig.suptitle(f"{group}: ten fixed test scenes (every 10th temporal stratum midpoint)", fontsize=15)
        fig.subplots_adjust(bottom=.12, top=.92, hspace=.28, wspace=.16)
        colorbar = fig.add_axes([.32, .062, .36, .018])
        fig.colorbar(scalar, cax=colorbar, orientation="horizontal").set_label("Beam demand (Mbps)")
        file = output / f"overview_{group}.png"
        fig.savefig(file, dpi=130, bbox_inches="tight", pad_inches=.15)
        plt.close(fig)
    for index in SCENE_INDICES:
        fig, axes = plt.subplots(2, 4, figsize=(22, 11), dpi=130)
        for ax, group in zip(axes.flat[:7], GROUPS):
            row = results[group]["scenes"][index]
            _draw_map(ax, row, norm=norm, cmap=cmap, basemap=basemap,
                      extent=extents[index], compact=True)
            ax.set_title(_title(group, row, compact=True), fontsize=10)
        _legend_note(axes.flat[7])
        fig.suptitle(f"Scene {index:03d}: paired seven-group geometric coverage", fontsize=15)
        fig.subplots_adjust(bottom=.12, top=.92, hspace=.28, wspace=.16)
        colorbar = fig.add_axes([.32, .062, .36, .018])
        fig.colorbar(scalar, cax=colorbar, orientation="horizontal").set_label("Beam demand (Mbps)")
        file = output / f"compare_scene_{index:03d}.png"
        fig.savefig(file, dpi=130, bbox_inches="tight", pad_inches=.15)
        plt.close(fig)
    return records


def write_index(output, records):
    cards = []
    for item in records:
        label = (f"{item['group']} · scene {item['scene_index']:03d} · "
                 f"{item['selected_time_utc']} · cover {100*item['demand_coverage']:.2f}%")
        link = html.escape(item["image"], quote=True)
        cards.append(
            f'<article data-group="{item["group"]}" data-scene="{item["scene_index"]}">'
            f'<a href="{link}"><img loading="lazy" src="{link}" alt="{html.escape(label, quote=True)}"></a>'
            f'<p>{html.escape(label)}</p></article>')
    overview = " ".join(f'<a href="overview_{group}.png">{group} overview</a>' for group in GROUPS)
    comparisons = " ".join(f'<a href="compare_scene_{index:03d}.png">scene {index:03d}</a>'
                           for index in SCENE_INDICES)
    page = f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>第一阶段七组波束几何覆盖图库</title>
<style>body{{font:15px system-ui,sans-serif;margin:2rem;background:#f6f8fa;color:#17212b}}
header{{max-width:1200px;margin:auto}} select{{font:inherit;margin:.3rem;padding:.3rem}}
#gallery{{display:grid;grid-template-columns:repeat(auto-fill,minmax(310px,1fr));gap:1rem;margin-top:1rem}}
article{{background:white;border:1px solid #dce2e7;border-radius:.5rem;padding:.6rem}}
article img{{width:100%;height:230px;object-fit:contain;background:#fff}} article p{{margin:.4rem 0 0}}
.links a{{display:inline-block;margin:.25rem .5rem .25rem 0}}</style></head>
<body><header><h1>第一阶段波束几何覆盖图库</h1>
<p>七组共同选择场景 005、015、…、095：按100个考核时刻十等分取中点，不按成绩选图。</p>
<p>彩色轮廓是13°E静止卫星0.5°全宽波束与球形地表的交线。颜色为该束汇总业务需求，
深点为几何覆盖飞机，红×为未覆盖飞机。轮廓在地球边缘、欧洲服务区或最低10°仰角处截断，不假闭合。
这些图不表示实际容量、干扰或通信服务达标。同一场景七组采用完全相同的地图范围。</p>
<label>组别 <select id="group"><option value="all">全部</option>{''.join(f'<option>{g}</option>' for g in GROUPS)}</select></label>
<label>场景 <select id="scene"><option value="all">全部</option>{''.join(f'<option value="{i}">{i:03d}</option>' for i in SCENE_INDICES)}</select></label>
<div class="links"><strong>每组十图：</strong>{overview}</div>
<div class="links"><strong>同场景七组：</strong>{comparisons}</div></header>
<main id="gallery">{''.join(cards)}</main>
<script>const group=document.getElementById('group'),scene=document.getElementById('scene');
function filter(){{for(const card of document.querySelectorAll('#gallery article')){{
card.hidden=(group.value!=='all'&&card.dataset.group!==group.value)||
(scene.value!=='all'&&card.dataset.scene!==scene.value)}}}}
group.addEventListener('change',filter);scene.addEventListener('change',filter);</script>
</body></html>'''
    path = output / "index.html"
    path.write_text(page, encoding="utf-8")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--basemap", type=Path,
                        help="Optional local GeoJSON land polygons; no online dependency")
    parser.add_argument("--refresh-overviews", action="store_true",
                        help="Refresh only 17 overview images and layout provenance after a completed atlas")
    args = parser.parse_args()
    output = args.output.resolve()
    prior = None
    if args.refresh_overviews:
        prior = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        if prior["file_count"]["single_maps"] != 70:
            raise ValueError("refresh requires a completed 70-map atlas")
        for record in prior["single_maps"]:
            if sha256(output / record["image"]) != record["image_sha256"]:
                raise ValueError("single-map evidence changed before overview refresh")
    if not args.refresh_overviews and output.exists() and any(path.name != "assets" for path in output.iterdir()):
        raise ValueError(f"output directory must be new or contain only an assets/ folder: {output}")
    if args.basemap is not None and not args.basemap.is_file():
        raise FileNotFoundError(args.basemap)
    results, result_hashes = load_results(args.results_dir)
    basemap = load_basemap(args.basemap)
    output.mkdir(parents=True, exist_ok=True)
    _, maximum_load = build_footprints(results)
    extents = common_scene_extents(results)
    if args.refresh_overviews:
        if result_hashes != prior["results_sha256"]:
            raise ValueError("overview refresh inputs differ from the original atlas")
        if abs(maximum_load - prior["color_scale"]["maximum"]) > 1e-8:
            raise ValueError("overview refresh color scale changed")
        current_extents = {str(index): list(extents[index]) for index in SCENE_INDICES}
        if current_extents != prior["shared_extent_by_scene_lonlat"]:
            raise ValueError("overview refresh viewports changed")
        if args.basemap is not None and sha256(args.basemap) != prior["basemap"]["sha256"]:
            raise ValueError("overview refresh basemap changed")
        render(results, output, basemap, maximum_load, extents, single_maps=False)
        history_path = output / "overview_refresh_history.json"
        history = json.loads(history_path.read_text()) if history_path.exists() else []
        history.append({"at": datetime.now(timezone.utc).isoformat(),
                        "reason": "reserve lower margin and prevent colorbar text clipping",
                        "previous_overview_sha256": prior["overviews"],
                        "previous_renderer_sha256": prior.get("overview_renderer_sha256", prior["script_sha256"])})
        history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
        prior.setdefault("single_map_renderer_sha256", prior["script_sha256"])
        prior["overview_renderer_sha256"] = sha256(Path(__file__))
        prior["overview_refreshed_at"] = datetime.now(timezone.utc).isoformat()
        for record in prior["overviews"]:
            record["sha256"] = sha256(output / record["path"])
        (output / "manifest.json").write_text(json.dumps(prior, ensure_ascii=False, indent=2,
                                                        allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps({"status": "overviews_refreshed", "single_maps_unchanged": 70,
                          "overviews_updated": len(prior["overviews"])}))
        return
    records = render(results, output, basemap, maximum_load, extents)
    index = write_index(output, records)
    assets = [output / f"overview_{group}.png" for group in GROUPS]
    assets += [output / f"compare_scene_{index:03d}.png" for index in SCENE_INDICES]
    manifest = {"status": "completed_geometry_qa", "created_utc": datetime.now(timezone.utc).isoformat(),
                "script_sha256": sha256(Path(__file__)),
                "results_directory": str(args.results_dir.resolve()),
                "results_sha256": result_hashes,
                "selection_rule": "floor((i+0.5)*100/10), i=0..9",
                "selected_scene_indices": list(SCENE_INDICES),
                "shared_extent_by_scene_lonlat": {str(index): list(extents[index])
                                                  for index in SCENE_INDICES},
                "geometry": {"satellite_longitude_deg": SATELLITE_LONGITUDE_DEG,
                             "earth_radius_km": EARTH_RADIUS_KM,
                             "orbit_radius_km": ORBIT_RADIUS_KM,
                             "beam_full_width_deg": BEAM_FULL_WIDTH_DEG,
                             "cone_half_angle_deg": BEAM_HALF_ANGLE_DEG,
                             "minimum_elevation_deg": MIN_ELEVATION_DEG,
                             "latitude_bounds_deg": list(LAT_BOUNDS),
                             "longitude_bounds_deg": list(LON_BOUNDS),
                             "boundary_samples_per_beam": BOUNDARY_SAMPLES,
                             "off_axis_qa_tolerance_deg": ANGLE_QA_TOL_DEG,
                             "sphere_qa_tolerance_km": SPHERE_QA_TOL_KM,
                             "meaning": "geometric ray-cone footprint, not capacity or interference"},
                "basemap": ({"path": str(args.basemap.resolve()),
                             "sha256": sha256(args.basemap), "polygon_count": len(basemap)}
                            if args.basemap else None),
                "color_scale": {"unit": "Mbps assigned demand per beam", "minimum": 0.0,
                                "maximum": maximum_load, "cmap": "viridis",
                                "shared_across_all_70_maps": True},
                "single_maps": records,
                "overviews": [{"path": str(path.relative_to(output)).replace("\\", "/"),
                               "sha256": sha256(path)} for path in assets],
                "index": {"path": "index.html", "sha256": sha256(index)},
                "file_count": {"single_maps": len(records), "group_overviews": len(GROUPS),
                               "scene_comparisons": len(SCENE_INDICES)}}
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2,
                                        allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "manifest": str(manifest_path),
                      "index": str(index), "single_maps": len(records),
                      "group_overviews": len(GROUPS),
                      "scene_comparisons": len(SCENE_INDICES)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
