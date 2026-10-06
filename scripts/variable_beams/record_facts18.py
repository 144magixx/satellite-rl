"""Record the S2-P18 assessment facts and the G1-G3 comparison against frozen V1."""
import json
import sys

sys.path.insert(0, r"D:\paper\scripts\variable_beams")
import ops18  # noqa: E402

V1_DIRECT_SI = {"50": 0.915, "100": 0.934, "200": 0.898}
V1_SLOTS = {"50": 3.92, "100": 2.79, "200": 1.82}
GREEDY_SI_OVERALL = 0.8933191049569863
GREEDY_SLOTS_OVERALL = 569.2727272727273

facts = {
    "checked_utc": ops18.utc_now(),
    "reference": {
        "V1_direct_si_by_n": V1_DIRECT_SI,
        "V1_slots_per_beam_by_n": V1_SLOTS,
        "V1_direct_si_at_200_after_sca": 0.898,
        "greedy_direct_si_overall_440": GREEDY_SI_OVERALL,
        "greedy_slots_overall_440": GREEDY_SLOTS_OVERALL,
        "note": "V1 值取自报告13 冻结的最终B；S2-P18 现为原始分配，口径不同，比较须标注",
    },
    "V2": {"assessment_run": "evaluate_sac_20261006_181949_137a0534",
           "overall_si": 0.8795, "overall_sgm": 0.6717,
           "slots_total": 301, "power_w": 2478,
           "per_n": {"50": {"si": 0.8666, "slots_per_beam": 4.03, "vs_v1": -0.0484},
                     "100": {"si": 0.8972, "slots_per_beam": 2.85, "vs_v1": -0.0368},
                     "200": {"si": 0.8641, "slots_per_beam": 1.96, "vs_v1": -0.0339}}},
    "V3": {"assessment_run": "evaluate_sac_20261006_162655_a661dd3d",
           "sca_run": "refine_sac_20261006_173515_4b10636d",
           "overall_si": 0.8660, "overall_sgm": 0.6890,
           "slots_total": 281, "power_w": 2441,
           "per_n": {"50": {"si": 0.8973, "slots_per_beam": 4.05, "vs_v1": -0.0177},
                     "100": {"si": 0.8857, "slots_per_beam": 2.72, "vs_v1": -0.0483},
                     "200": {"si": 0.8205, "slots_per_beam": 1.68, "vs_v1": -0.0775}}},
    "gates_provisional": {
        "G1_large_beam_gap": {"verdict": "未达标",
                              "basis": "两臂在 N=200 的 SI 分别比 V1 低 3.4 与 7.8 个百分点；"
                                       "G1 要求与贪心差距相对 V1 缩小≥40%，实测差距未缩小而是扩大"},
        "G2_100_beam_no_regression": {"verdict": "未达标",
                                      "basis": "V2 N=100 为 0.8972、V3 为 0.8857，"
                                               "均低于 V1 的 0.934，超过允许的 0.5% 容差"},
        "G3_slots_up_30pct": {"verdict": "未达标",
                              "basis": "V2 N=200 每束 1.96 槽、V3 为 1.68 槽，"
                                       "相对 V1 的 1.82 为 +8% 与 −8%，均未达 +30%；"
                                       "42 个场景平均每束槽数也远低于贪心的 5 槽/束"},
        "G4": {"verdict": "按方案执行",
               "basis": "G1–G3 均未达标时如实报告本轮改进不足并转下一手段，不追加同手段预算"},
    },
    "pending": ["G0 的 440 景考核", "G0 的统一 SCA", "V2 的统一 SCA",
                "统一后的最终B口径比较与计时审计", "中文曲线与报告14填数"],
}
path = ops18.LOCAL / "run_manifest.json"
manifest = json.loads(path.read_text(encoding="utf-8-sig"))
manifest["assessment_facts"] = facts
manifest["updated_utc"] = ops18.utc_now()
ops18.write_json(path, manifest)
ops18.write_json(ops18.LOCAL / "analysis_inputs" / "assessment_facts.json", facts)
print("已记录考核事实与临时判定")
