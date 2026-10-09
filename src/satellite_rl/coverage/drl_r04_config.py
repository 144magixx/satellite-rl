"""Frozen A-stage contract for R04-guided layout improvement."""

from dataclasses import dataclass

from .variable_config import VariableConfig


@dataclass(frozen=True)
class DrlR04Config:
    max_beams: int = 200
    width_deg: float = 0.5
    max_commits: int = 32
    max_trials: int = 256
    max_geometry_calls: int = 8192
    max_candidates: int = 32
    gamma: float = 1.0
    reward_scale: float = 100.0
    menu_version: str = "r04_split_merge_local_v1"
    objective_version: str = "r04_reference_pressure_v1"

    def __post_init__(self):
        if not 1 <= self.max_beams <= 200:
            raise ValueError("max_beams must be in 1..200")
        for name in ("max_commits", "max_trials", "max_geometry_calls", "max_candidates"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.width_deg != 0.5 or self.gamma != 1.0 or self.reward_scale != 100.0:
            raise ValueError("A-stage width, discount and reward scale are frozen")
        if self.menu_version != "r04_split_merge_local_v1" or self.objective_version != "r04_reference_pressure_v1":
            raise ValueError("unknown A-stage menu or objective")

    def variable_config(self) -> VariableConfig:
        return VariableConfig(max_beams=self.max_beams, fixed_width_deg=self.width_deg,
                              max_actions=self.max_commits, max_candidates=self.max_candidates,
                              max_geometry_calls=self.max_geometry_calls)
