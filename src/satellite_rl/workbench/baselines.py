"""Adapt the repository's fixed and greedy rules to a workbench action grid."""

from __future__ import annotations

import math

from satellite_rl.baselines.fixed import fixed_policy
from satellite_rl.baselines.greedy import greedy_policy


POLICIES = {"fixed": fixed_policy, "greedy": greedy_policy}


class BaselinePolicy:
    def __init__(self, name, environment):
        if name not in POLICIES:
            raise ValueError(f"Unsupported baseline: {name}")
        self.name = name
        self.environment = environment
        self.power_levels = tuple(float(x) for x in environment.config["environment"]["power_levels_w"])
        self.mapping_counts = {}
        self.last_action_info = {}

    def act(self, _state, deterministic=True):
        if not deterministic:
            raise ValueError("Fixed and greedy rules have one deterministic pass per scene")
        group, frequency, slots, power_db = POLICIES[self.name](self.environment.raw_env)
        frequency, slots = int(frequency), int(slots)
        if not 0 <= frequency < 100 or not 1 <= slots <= 10:
            raise ValueError(f"Baseline requested action outside common action grid: {(frequency, slots)}")
        wanted_w = 10 ** (float(power_db) / 10)
        if not math.isfinite(wanted_w) or wanted_w <= 0:
            raise ValueError("Baseline requested invalid power")
        # Closest legal watt level, breaking ties toward lower power.
        power_index = min(range(len(self.power_levels)),
                          key=lambda i: (abs(self.power_levels[i] - wanted_w), self.power_levels[i], i))
        mapped_w = self.power_levels[power_index]
        key = (round(wanted_w, 9), mapped_w)
        self.mapping_counts[key] = self.mapping_counts.get(key, 0) + 1
        self.last_action_info = {"source": self.name, "source_group": int(group),
                                 "source_frequency": frequency, "source_slots": slots,
                                 "source_power_w": wanted_w, "mapped_power_w": mapped_w,
                                 "power_mapping_changed": not math.isclose(wanted_w, mapped_w, rel_tol=1e-8)}
        return frequency, slots - 1, power_index

    def metadata(self):
        mappings = [{"source_power_w": source, "mapped_power_w": mapped, "count": count}
                    for (source, mapped), count in sorted(self.mapping_counts.items())]
        return {"variant": f"workbench_adapted_{self.name}_v1", "source_rule": self.name,
                "action_grid_power_levels_w": list(self.power_levels), "power_mappings": mappings,
                "exact_power_match": all(math.isclose(row["source_power_w"], row["mapped_power_w"], rel_tol=1e-8)
                                         for row in mappings),
                "evaluation_repeats": 1,
                "note": "原规则选物理动作；功率映射至本次配置的最近合法档位，频率和槽数沿用原规则。"}
