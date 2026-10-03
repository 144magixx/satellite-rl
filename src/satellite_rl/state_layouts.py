"""Shared, dependency-light definitions for the natural T-SAC observation layout."""

from math import prod


NATURAL_CONTEXT_LEVELS = ("none", "compact", "summary", "layout", "full")
BASE_FEATURES = ("beam", "power", "interference", "occupancy")
VARIABLE_BEAM_FEATURES = BASE_FEATURES + (
    "beta", "progress", "remaining_demand", "beam_count_normalized",
    "remaining_beams_normalized", "power_pressure",
)
VARIABLE_BEAM_STATE_DIM = 210
VARIABLE_BEAM_MAX = 200


def natural_shapes(level):
    """Return only fields visible at this level, in observation order."""
    if level not in NATURAL_CONTEXT_LEVELS:
        raise ValueError(f"Unknown natural context level: {level}")
    shapes = {"beam": [3], "power": [1], "interference": [100],
              "occupancy": [100]}
    if level != "none":
        shapes["natural_global_context"] = [3 if level == "compact" else 10]
    if level in ("layout", "full"):
        shapes["natural_beam_context"] = [10, 10, 5 if level == "layout" else 11]
    if level in ("summary", "layout", "full"):
        shapes["natural_resource_context"] = [8, 204 if level == "full" else 4]
    return shapes


def natural_dim(level):
    return sum(prod(shape) for shape in natural_shapes(level).values())


def natural_token_count(level):
    return 13 + (level != "none") + (10 if level in ("layout", "full") else 0) + (8 if level in ("summary", "layout", "full") else 0)
