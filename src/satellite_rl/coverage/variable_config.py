"""Configuration for variable-count first-stage layouts (new physical version)."""

from __future__ import annotations

from dataclasses import dataclass
import math


ROUTES = tuple(f"R{i:02d}" for i in range(1, 13))
CORE_ROUTES = ("R01", "R02", "R03", "R04", "R05", "R06", "R07", "R11", "R12")
WIDTH_PHYSICS_VERSION = "idealized_circular_footprint_v1"


@dataclass(frozen=True)
class VariableConfig:
    """Explicit bounds; reference capacity is a screening assumption in bps.

    This is not the interference-aware downstream beam capacity.  Widths are
    full satellite-view angles.  R05 alone explores the three width choices.
    """

    max_beams: int = 200
    fixed_width_deg: float = 0.5
    width_choices_deg: tuple[float, ...] = (0.35, 0.5, 0.7)
    reference_capacity_bps: float = 600_000_000.0
    max_geometry_calls: int = 8192
    max_candidates: int = 32
    max_actions: int = 32
    max_initial_candidates: int = 4096
    max_mip_sites: int = 256
    mip_time_limit_s: float = 10.0
    mip_rel_gap: float = 0.05
    max_ot_options_per_user: int = 4
    ot_time_limit_s: float = 10.0
    robust_perturbation: float = 0.25
    switch_beam_cost: float = 0.02
    switch_pointing_cost: float = 0.005
    switch_assignment_cost: float = 0.001
    profile: str = "europe_v1"

    def __post_init__(self) -> None:
        for name in ("max_beams", "max_geometry_calls", "max_candidates",
                     "max_actions", "max_initial_candidates", "max_mip_sites",
                     "max_ot_options_per_user"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.max_beams > 200:
            raise ValueError("max_beams cannot exceed 200")
        for name in ("fixed_width_deg", "reference_capacity_bps",
                     "robust_perturbation", "switch_beam_cost",
                     "switch_pointing_cost", "switch_assignment_cost",
                     "mip_time_limit_s", "mip_rel_gap", "ot_time_limit_s"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.fixed_width_deg <= 0 or self.reference_capacity_bps <= 0:
            raise ValueError("width and reference capacity must be positive")
        if self.fixed_width_deg != 0.5:
            raise ValueError("fixed first-stage full beam width is 0.5 degree")
        if self.mip_time_limit_s <= 0 or self.ot_time_limit_s <= 0 or self.mip_rel_gap > 1:
            raise ValueError("solver limits or relative gap invalid")
        if self.robust_perturbation > 1:
            raise ValueError("robust_perturbation cannot exceed one")
        if self.width_choices_deg != (0.35, 0.5, 0.7):
            raise ValueError("R05 width choices are the explicit idealized 0.35/0.5/0.7 degree version")
        if self.profile != "europe_v1":
            raise ValueError("variable layout currently requires europe_v1 geometry")
