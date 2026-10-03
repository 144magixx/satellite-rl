"""Allocation and scene timing without counting repeated actions as new scenes."""
from collections import defaultdict

import numpy as np


def build_timing(initialization, setup, optimizer, callback_io, verification):
    values = np.asarray([initialization, setup, optimizer, callback_io, verification], dtype=float)
    if not np.isfinite(values).all() or np.any(values < 0) or callback_io > optimizer:
        raise ValueError("Invalid power-refinement timing")
    return {
        "initialization_seconds": float(initialization),
        "physics_setup_seconds": float(setup),
        "optimizer_wall_seconds": float(optimizer),
        "optimizer_callback_io_seconds": float(callback_io),
        "optimizer_excluding_callback_seconds": max(0.0, float(optimizer - callback_io)),
        "verification_seconds": float(verification),
        "total_pipeline_seconds": float(initialization + setup + optimizer + verification),
        "timing_scope": "sum_of_measured_stages_shared_initializer_charged_once_per_method",
    }


def aggregate_scene_timing(rows):
    """Report both one allocation and the sum of repeats in each actual scene."""
    if not rows or any("timing" not in row for row in rows):
        return {"timing_available": False}
    by_scene = defaultdict(list)
    seen = set()
    for row in rows:
        key = (row["source_scene_sha256"], row["repeat"])
        if key in seen:
            raise ValueError("Duplicate scene/repeat in method timing")
        seen.add(key)
        by_scene[key[0]].append(row)
    result = {"timing_available": True, "timed_scenes": len(by_scene),
              "timed_allocations": len(rows), "percentile_method": "numpy_linear"}
    for field in ("initialization_seconds", "physics_setup_seconds", "optimizer_wall_seconds",
                  "optimizer_callback_io_seconds", "optimizer_excluding_callback_seconds",
                  "verification_seconds", "total_pipeline_seconds"):
        allocations = np.array([row["timing"][field] for row in rows], dtype=float)
        scene_totals = np.array([sum(row["timing"][field] for row in group)
                                 for group in by_scene.values()])
        for prefix, data in (("single_allocation", allocations), ("scene_repeats_sum", scene_totals)):
            for suffix, value in (("mean", np.mean(data)), ("median", np.median(data)),
                                  ("p95", np.percentile(data, 95))):
                result[f"{prefix}_{field}_{suffix}"] = float(value)
    return result
