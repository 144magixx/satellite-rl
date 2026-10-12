"""Frozen scalar48 + preference5 inputs, computed before any task decoding.

Requests supplied by the aviation adapter are bps; only the log-total input
converts to Mbps. No task outcome or decoder-derived value is accepted here.
"""
import hashlib
import json
import numpy as np
from ..geometry import satellite_rays_numpy, ray_angle_numpy
from .config import GeometryConfig

FEATURE_VERSION = "scalar48_preference5_v2"
FEATURE_NAMES = (
    "global_aircraft_log1p", "global_N_over_200", "global_total_request_log1p_mbps",
    "global_peak_share", "global_tail_share", "global_CV", "global_solid_angle_over_200max",
    "global_mean_width_over_max", "global_min_width_over_max", "global_max_width_over_max",
    "global_nonzero_request_fraction", "global_mean_aircraft_request_share",
    "global_max_aircraft_request_share", "global_request_CV", "region_k_over_8",
    "region_aircraft_fraction", "region_request_share", "region_beam_peak_share",
    "region_beam_min_share", "region_beam_mean_share", "region_beam_std_share",
    "region_mean_width_over_max", "region_min_width_over_max", "region_max_width_over_max",
    "region_current_center_ray_span_over_maxwidth", "region_current_aircraft_ray_span_over_maxwidth",
    "region_min_current_coverage_slack_normalized", "region_mean_current_coverage_slack_normalized",
    "region_zero_request_fraction", "region_singleton_beam_fraction",
    "region_max_aircraft_request_share", "region_boundary_aircraft_fraction",
    "kind_transfer", "kind_swap", "kind_joint_reconstruct", "kind_merge", "kind_split",
    "kind_merge_split", "kind_center", "kind_width", "width_mode_03", "width_mode_05",
    "width_mode_10", "width_mode_mixed", "target_local_k_over_9", "local_delta_k_over_8",
    "variant_normalized", "width_policy_shrink")
PREFERENCE_FIELDS = ("N", "solid_angle_sum_sr", "maxload", "top10_mean", "CV")
FEATURE_SCHEMA = dict(version=FEATURE_VERSION, names=list(FEATURE_NAMES), input_dim=53,
    request_unit="bps", log_total_unit="Mbps", widths="full_angle_degrees",
    span="max_angle_from_normalized_mean_ray_degrees/1degree",
    slack="squared_chord/(1degree_full_width_half_angle_chord_limit)",
    preferences=list(PREFERENCE_FIELDS), source="parent_geometry_requests_and_predecode_task_only")


def scene_sha256(scene):
    h = hashlib.sha256()
    for array in (scene.positions, scene.demands):
        a = np.ascontiguousarray(array, dtype="<f8")
        h.update(str(a.shape).encode()); h.update(a.tobytes())
    h.update(json.dumps([scene.profile, scene.aircraft_ids], separators=(",", ":")).encode())
    return h.hexdigest()


def preference_index(value):
    if isinstance(value, str):
        value = PREFERENCE_FIELDS.index(value)
    if not isinstance(value, (int, np.integer)) or not 0 <= value < 5:
        raise ValueError("one of five frozen preference indices required")
    return int(value)


def _span(rays):
    if not len(rays): return 0.
    mean = rays.mean(axis=0)
    norm = np.linalg.norm(mean)
    if norm < 1e-12: raise ValueError("parent rays have undefined mean direction")
    return float(ray_angle_numpy(rays, mean / norm).max())


def task_features(scene, current, task, preference=0, geometry=GeometryConfig(), variant_count=8, budget=None):
    """Return 53 finite inputs; never consult Task.cheap_features or outcomes."""
    n, m = len(current.centers), len(scene.demands)
    region = np.asarray(task.region_ids, dtype=int)
    if not n or not len(region) or (region < 0).any() or (region >= n).any():
        raise ValueError("task region must refer to nonempty parent beams")
    assignment = current.assignment
    if len(assignment) != m or (assignment < 0).any() or (assignment >= n).any():
        raise ValueError("complete parent ownership required for predecode inputs")
    users = np.flatnonzero(np.isin(assignment, region))
    if not len(users): raise ValueError("empty parent task region")
    if budget is not None:
        budget.check()
        budget.geometry(len(users) + len(region))
    total = float(scene.demands.sum())
    loads = np.bincount(assignment, weights=scene.demands, minlength=n)
    counts = np.bincount(assignment, minlength=n)
    shares = loads / total if total else np.zeros(n)
    requests = scene.demands / total if total else np.zeros(m)
    rshares, rw = shares[region], current.widths[region]
    brays = satellite_rays_numpy(current.centers, scene.profile)
    chord = np.sum((scene.rays[users] - brays[assignment[users]]) ** 2, axis=1)
    limits = 4 * np.sin(np.deg2rad(current.widths[assignment[users]] / 2) / 2) ** 2
    maxlimit = 4 * np.sin(np.deg2rad(geometry.width_max_deg / 2) / 2) ** 2
    slack = (limits - chord) / maxlimit
    omega = np.sum(2 * np.pi * (1 - np.cos(np.deg2rad(current.widths / 2))))
    maxomega = 2 * np.pi * (1 - np.cos(np.deg2rad(geometry.width_max_deg / 2)))
    cv = float(loads.std() / loads.mean()) if total else 0.
    reqcv = float(scene.demands.std() / scene.demands.mean()) if total and m else 0.
    x = [np.log1p(m), n / 200, np.log1p(total / 1e6), float(shares.max()),
         float(np.sort(shares)[-max(1, int(np.ceil(.1*n))):].mean()), cv,
         float(omega / (200 * maxomega)), float(current.widths.mean() / geometry.width_max_deg),
         float(current.widths.min() / geometry.width_max_deg), float(current.widths.max() / geometry.width_max_deg),
         float(np.mean(scene.demands > 0)) if m else 0., float(requests.mean()) if m else 0.,
         float(requests.max()) if m else 0., reqcv, len(region)/8, len(users)/max(1,m),
         float(requests[users].sum()), float(rshares.max()), float(rshares.min()),
         float(rshares.mean()), float(rshares.std()), float(rw.mean()/geometry.width_max_deg),
         float(rw.min()/geometry.width_max_deg), float(rw.max()/geometry.width_max_deg),
         _span(brays[region])/geometry.width_max_deg, _span(scene.rays[users])/geometry.width_max_deg,
         float(slack.min()), float(slack.mean()), float(np.mean(scene.demands[users] == 0)),
         float(np.mean(counts[region] == 1)), float(requests[users].max()), float(np.mean(chord >= .9*limits))]
    kinds = ("transfer", "swap", "joint_reconstruct", "merge", "split", "merge_split", "center", "width")
    kind = task.kind
    if kind == "region_rebuild":
        kind = "merge" if task.target_k < len(region) else ("split" if task.target_k > len(region) else "joint_reconstruct")
    if kind not in kinds: raise ValueError("unknown frozen task kind: " + str(kind))
    x.extend(float(kind == k) for k in kinds)
    widths = np.asarray(task.widths, float)
    mode = next((i for i,w in enumerate((.3,.5,1.)) if np.allclose(widths,w,rtol=0,atol=1e-12)), 3)
    x.extend(float(mode == i) for i in range(4))
    x.extend((task.target_k/9, (task.target_k-len(region))/8,
              task.variant/max(1,variant_count-1), float(task.width_policy=="shrink")))
    pref = preference_index(preference)
    x.extend(float(pref == i) for i in range(5))
    result = np.asarray(x, dtype=np.float32)
    if result.shape != (53,) or not np.isfinite(result).all():
        raise ValueError("finite frozen 53-dimensional task inputs required")
    if budget is not None: budget.check()
    return result


def pool_features(scene, current, pool, preference=0, geometry=GeometryConfig(), variant_count=8, budget=None):
    return np.stack([task_features(scene,current,t,preference,geometry,variant_count,budget) for t in pool]) if pool else np.empty((0,53),np.float32)


class Normalizer:
    """Training inputs alone determine statistics; constant columns become zero."""
    def __init__(self, mean, std, constant):
        self.mean = np.asarray(mean, dtype=np.float64)
        self.std = np.asarray(std, dtype=np.float64)
        self.constant = np.asarray(constant, dtype=bool)
        if self.mean.shape != (53,) or self.std.shape != (53,) or self.constant.shape != (53,):
            raise ValueError("normalizer dimension mismatch")
        if not np.isfinite(self.mean).all() or not np.isfinite(self.std).all() or (self.std < 1e-8).any():
            raise ValueError("invalid training normalization")

    @classmethod
    def fit(cls, values, role="training"):
        if role != "training": raise ValueError("normalization may only use training inputs")
        x = np.asarray(values, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != 53 or not len(x) or not np.isfinite(x).all():
            raise ValueError("nonempty finite training task inputs required")
        std = x.std(axis=0)
        return cls(x.mean(axis=0), np.maximum(std,1e-8), std == 0)

    def transform(self, x):
        y = (np.asarray(x,dtype=np.float64)-self.mean)/self.std
        y[...,self.constant] = 0.
        if not np.isfinite(y).all(): raise ValueError("nonfinite normalized task inputs")
        return y.astype(np.float32)

    def to_record(self):
        return dict(mean=self.mean.tolist(),std=self.std.tolist(),constant=self.constant.tolist(),
                    fitted_role="training",feature_schema=FEATURE_SCHEMA)

    @classmethod
    def from_record(cls, value):
        if value.get("fitted_role") != "training" or value.get("feature_schema") != FEATURE_SCHEMA:
            raise ValueError("normalizer schema/source mismatch")
        return cls(value["mean"],value["std"],value["constant"])
