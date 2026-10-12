"""Independent AI17 geometry and finite task-search contracts."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class GeometryConfig:
    width_min_deg: float = 0.3
    width_max_deg: float = 1.0
    width_menu_deg: tuple = (0.3, 0.5, 1.0)
    max_beams: int = 200
    tolerance: float = 1e-14
    profile: str = "europe_v1"

    def __post_init__(self):
        object.__setattr__(self, "width_menu_deg", tuple(self.width_menu_deg))
        if (not all(math.isfinite(x) for x in (self.width_min_deg, self.width_max_deg, self.tolerance))
                or self.width_min_deg <= 0 or self.width_max_deg < self.width_min_deg
                or self.width_max_deg >= 180 or self.tolerance < 0):
            raise ValueError("finite positive full-width bounds and nonnegative tolerance required")
        if type(self.max_beams) is not int or not 1 <= self.max_beams <= 200:
            raise ValueError("actual beam cap must be 1..200")
        if (not self.width_menu_deg or len(set(self.width_menu_deg)) != len(self.width_menu_deg)
                or any(not math.isfinite(w) or not self.width_min_deg <= w <= self.width_max_deg
                       for w in self.width_menu_deg)):
            raise ValueError("discrete widths must be unique and within declared bounds")
        if self.profile != "europe_v1":
            raise ValueError("AI17 currently freezes europe_v1 geometry")


@dataclass(frozen=True)
class SearchConfig:
    seconds: float = 10.0
    max_attempts: int = 8192
    max_geometry_calls: int = 100000
    pool_size: int = 24
    max_regions: int = 12
    region_sizes: tuple = (2, 4)
    max_region_users: int = 256
    variants: int = 8
    center_iterations: int = 4
    max_centers_per_task: int = 32
    max_assignment_augment_depth: int = 3
    archive_limit: int = 64
    task_candidate_limit: int = 64
    initial_seconds: float = 2.0
    initial_max_attempts: int = 2048
    nested_seconds: float = 2.0
    nested_max_attempts: int = 2048
    post_shrink_reserve_seconds: float = 1.0
    checkpoint_seconds: tuple = (10.0,)
    checkpoint_attempts: tuple = (8192,)
    max_restarts: int = 256
    objective_tolerance: float = 1e-9
    # Proposed scale-aware AI17 values; execution contract pins these explicitly.
    metric_abs_tolerances: tuple = (("N", 0.), ("solid_angle_sum_sr", 1e-12),
                                  ("maxload", 1e-6), ("top10_mean", 1e-6), ("CV", 1e-9))
    metric_relative_tolerance: float = 1e-12
    # Fixed before a scientific task; not a fitted weighted objective.
    pareto_fields: tuple = ("N", "solid_angle_sum_sr", "maxload", "top10_mean", "CV")

    def __post_init__(self):
        for field in ("region_sizes", "pareto_fields", "checkpoint_seconds", "checkpoint_attempts"):
            object.__setattr__(self, field, tuple(getattr(self, field)))
        object.__setattr__(self, "metric_abs_tolerances", tuple((str(k), float(v)) for k,v in self.metric_abs_tolerances))
        if any(not math.isfinite(x) or x < 0 for x in (self.seconds, self.initial_seconds, self.nested_seconds, self.post_shrink_reserve_seconds)):
            raise ValueError("finite nonnegative search seconds required")
        if (len(self.checkpoint_seconds) != len(self.checkpoint_attempts)
                or any(not math.isfinite(s) or s < 0 for s in self.checkpoint_seconds)
                or any(type(a) is not int or a < 0 for a in self.checkpoint_attempts)):
            raise ValueError("paired finite checkpoint budgets required")
        for name in ("max_attempts", "max_geometry_calls", "max_restarts", "initial_max_attempts", "nested_max_attempts"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(name + " must be a nonnegative integer")
        for name in ("pool_size", "max_regions", "max_region_users", "variants", "archive_limit", "task_candidate_limit", "max_centers_per_task", "max_assignment_augment_depth"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(name + " must be a positive integer")
        if type(self.center_iterations) is not int or not 0 <= self.center_iterations <= 8:
            raise ValueError("bounded center iterations required")
        if not self.region_sizes or any(type(x) is not int or not 1 <= x <= 8 for x in self.region_sizes):
            raise ValueError("bounded positive local region sizes required")
        if not math.isfinite(self.objective_tolerance) or self.objective_tolerance < 0:
            raise ValueError("finite nonnegative objective tolerance required")
        allowed = {"N", "solid_angle_sum_sr", "maxload", "top10_mean", "CV"}
        if not self.pareto_fields or set(self.pareto_fields) - allowed:
            raise ValueError("Pareto fields must be pure first-stage geometry/request metrics")
        if (set(dict(self.metric_abs_tolerances)) != allowed or any(not math.isfinite(v) or v < 0 for _,v in self.metric_abs_tolerances)
                or not math.isfinite(self.metric_relative_tolerance) or self.metric_relative_tolerance < 0):
            raise ValueError("complete finite scale-aware objective tolerances required")

