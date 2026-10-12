"""Explicit geometry and finite search budgets for pure stage one."""
from dataclasses import dataclass
import math

@dataclass(frozen=True)
class GeometryConfig:
    width_deg: float = 0.5
    max_beams: int = 200
    tolerance: float = 1e-14

    def __post_init__(self):
        if self.width_deg != 0.5 or not 1 <= self.max_beams <= 200:
            raise ValueError("G16 fixes full width at .5 degrees and at most 200 beams")

@dataclass(frozen=True)
class SearchConfig:
    seconds: float = 2.0
    max_attempts: int = 4096
    max_geometry_calls: int = 100000
    objective: str = "lex"
    max_joint_users: int = 4
    max_region_users: int = 128
    region_greedy_starts: int = 8
    max_joint_regions: int = 12
    joint_assignments_per_region: int = 64
    pool_size: int = 64
    centers_per_beam: int = 8
    allow_variable_n: bool = False
    target_n: int | None = None
    objective_tolerance: float = 1e-9
    checkpoint_seconds: tuple = ()
    checkpoint_attempts: tuple = ()

    def __post_init__(self):
        if not math.isfinite(self.seconds) or self.seconds < 0 or self.max_attempts < 0 or self.max_geometry_calls < 0:
            raise ValueError("budgets must be nonnegative")
        if self.objective not in ("lex", "legacyCV"):
            raise ValueError("unknown objective")
        if min(self.max_joint_users, self.max_region_users, self.region_greedy_starts,
               self.centers_per_beam, self.max_joint_regions,
               self.joint_assignments_per_region, self.pool_size) < 1:
            raise ValueError("candidate limits must be positive")
        if not math.isfinite(self.objective_tolerance) or self.objective_tolerance < 0:
            raise ValueError("finite nonnegative objective tolerance required")
        if self.target_n is not None and not 1 <= self.target_n <= 200:
            raise ValueError("target N must be within 1..200")
        if any(not math.isfinite(v) or v < 0 for v in self.checkpoint_seconds + self.checkpoint_attempts):
            raise ValueError("invalid checkpoint budget")
