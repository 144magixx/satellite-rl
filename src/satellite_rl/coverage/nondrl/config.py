"""Frozen common contract for the non-DRL fixed-count mechanism track."""
from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class NondrlConfig:
    online_seconds: float = 2.0
    max_candidates: int = 256
    max_physical_calls: int = 128
    max_geometry_calls: int = 8192
    max_beams: int = 200
    fixed_width_deg: float = 0.5
    seed: int = 51001
    max_rounds: int = 8
    max_menu: int = 32
    max_regions: int = 12
    max_rebuilds: int = 40
    local_aircraft_limit: int = 64
    centers_per_beam: int = 8
    mip_seconds: float = 0.1
    mip_relative_gap: float = 0.01
    square_segments: int = 16
    max_pool: int = 1000
    pool_expansions: int = 3
    ot_iterations: int = 100
    population: int = 16
    generations: int = 8
    evolution_proposals: int = 128
    carry_tolerance: float = 0.005
    cv_tolerance: float = 1e-10
    profile: str = "europe_v1"
    initial_trials: int = 256

    def __post_init__(self):
        for key, value in asdict(self).items():
            if key in ("profile",):
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"invalid {key}")
            if key != "seed" and value <= 0:
                raise ValueError(f"{key} must be positive")
        integers = ("max_candidates", "max_physical_calls", "max_geometry_calls", "max_beams",
                    "seed", "max_rounds", "max_menu", "max_regions", "max_rebuilds",
                    "local_aircraft_limit",
                    "centers_per_beam", "square_segments", "max_pool", "pool_expansions",
                    "ot_iterations", "population", "generations", "evolution_proposals", "initial_trials")
        if any(type(getattr(self, key)) is not int for key in integers):
            raise ValueError("count fields must be integers")
        if self.max_beams > 200 or self.fixed_width_deg != 0.5 or self.profile != "europe_v1":
            raise ValueError("first round uses fixed 0.5 degree Europe geometry and N <= 200")
        if self.carry_tolerance > 0.005 or self.mip_relative_gap > 1 or self.centers_per_beam > 8:
            raise ValueError("common protection or pool bound exceeded")

    @classmethod
    def from_dict(cls, value):
        return value if isinstance(value, cls) else cls(**(value or {}))

    def to_dict(self):
        return asdict(self)
