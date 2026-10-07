"""Recompute S2-P18 final-B gates from saved compact cases and P17 table."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean


ROOT = Path("D:/paper")
COMPACT = ROOT / "outputs/verification/s2p19_research_20261007/p18_existing_B_compact.json"
P17_SCENES = ROOT / "outputs/experiments/variable_beams_20261004/analysis/report_ef62040e88f1f02f936de8fb26699b2249feb1bb/per_scene_means.csv"
P17_CONFIGS = ROOT / "outputs/verification/s2p19_research_20261007/p17_sca_configs.json"
OUTPUT = ROOT / "outputs/verification/s2p19_research_20261007/p18_final_gate_reaudit.json"
LARGE_N = (125, 140, 150, 175, 180, 200)
SMALL_N = (50, 60, 75, 90, 100)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    compact = json.loads(COMPACT.read_text(encoding="utf-8"))
    prior_configs = json.loads(P17_CONFIGS.read_text(encoding="utf-8"))
    baseline = {}
    with P17_SCENES.open(encoding="utf-8-sig", newline="") as stream:
        for item in csv.DictReader(stream):
            if item["stage"] == "final" and item["role"] in ("V1", "greedy"):
                baseline[(item["role"], item["scene_sha256"])] = item
    config_anchor = prior_configs["V1"]["config_power_refinement"]
    assert config_anchor == prior_configs["B1"]["config_power_refinement"] | {
        "action_mode": "stochastic", "repeats": 3}, "P17 V1/B1 config mismatch beyond mode/repeats"
    out = {"scope": "saved S2-P18 final B; no new optimization",
           "input_sha256": {str(p.relative_to(ROOT)).replace('\\', '/'): sha(p)
                            for p in (COMPACT, P17_SCENES, P17_CONFIGS)},
           "roles": {}}
    for role, source in compact["runs"].items():
        assert source["status_stage"] == "completed"
        assert source["config_power_refinement"] == config_anchor
        grouped = defaultdict(list)
        for item in source["rows"]:
            scene_sha = item["source_scene_sha256"]
            for comparison in ("V1", "greedy"):
                old = baseline[(comparison, scene_sha)]
                assert int(old["beam_count"]) == item["beam_count"]
                assert old["mother_scene_id"] == item["mother_scene_id"]
                assert float(old["load_multiplier"]) == item["load_multiplier"]
            grouped[scene_sha].append(item)
        assert len(grouped) == 440 and all(len(items) == 3 for items in grouped.values())
        per_n = defaultdict(list)
        paired_mothers = defaultdict(lambda: defaultdict(list))
        for scene_sha, actions in grouped.items():
            action_mean = mean(row["si"] for row in actions)
            n = actions[0]["beam_count"]
            mother = actions[0]["mother_scene_id"]
            per_n[n].append(action_mean)
            for comparison in ("V1", "greedy"):
                old = baseline[(comparison, scene_sha)]
                paired_mothers[comparison][mother].append((n, action_mean - float(old["si"])))
        assert all(len(per_n[n]) == 40 for n in set(LARGE_N + SMALL_N))
        by_n = {str(n): mean(per_n[n]) for n in sorted(per_n)}
        greedy_by_n = {}
        v1_by_n = {}
        for n in sorted(per_n):
            greedy_by_n[str(n)] = mean(float(baseline[("greedy", scene_sha)]["si"])
                                          for scene_sha, actions in grouped.items() if actions[0]["beam_count"] == n)
            v1_by_n[str(n)] = mean(float(baseline[("V1", scene_sha)]["si"])
                                      for scene_sha, actions in grouped.items() if actions[0]["beam_count"] == n)
        large_gap = mean(greedy_by_n[str(n)] - by_n[str(n)] for n in LARGE_N)
        gap_200 = greedy_by_n["200"] - by_n["200"]
        small_declines = {str(n): (by_n[str(n)] - v1_by_n[str(n)]) for n in SMALL_N}
        out["roles"][role] = {
            "run_id": source["run_id"], "remote_config_sha256": source["config_sha256"],
            "remote_job_sha256": source["job_sha256"], "scene_sha_matches": 440,
            "action_repeats_per_scene": 3, "unique_mothers": len(paired_mothers["greedy"]),
            "final_B_si_by_n": by_n, "V1_final_B_si_by_n": v1_by_n,
            "greedy_final_B_si_by_n": greedy_by_n,
            "G1_large_n_equal_n_gap": large_gap,
            "G1_n200_gap": gap_200,
            "G1_pass_descriptive": large_gap <= .03 and gap_200 <= .04,
            "G2_small_n_minus_V1": small_declines,
            "G2_pass_descriptive": all(value >= -.01 for value in small_declines.values()),
            "overall_final_B_si": mean(by_n.values()),
        }
    OUTPUT.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for role, item in out["roles"].items():
        print(role, "overall", item["overall_final_B_si"], "G1", item["G1_large_n_equal_n_gap"],
              item["G1_n200_gap"], item["G1_pass_descriptive"], "G2", item["G2_pass_descriptive"])


if __name__ == "__main__":
    main()
