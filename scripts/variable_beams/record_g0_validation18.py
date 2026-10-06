"""Record G0's final-endpoint per-N validation and the cross-arm slot comparison."""
import json
import sys

sys.path.insert(0, r"D:\paper\scripts\variable_beams")
import ops18  # noqa: E402

facts = {
    "checked_utc": ops18.utc_now(),
    "unit": "144 景验证集、40 万步终点、原始分配、未接 SCA",
    "G0": {"validation_step": 400000,
           "si": {"50": 0.828775, "80": 0.820443, "100": 0.819559,
                  "120": 0.805829, "160": 0.827461, "200": 0.82129},
           "slots_per_beam": {"50": 5.567, "80": 4.194, "100": 3.591,
                              "120": 3.187, "160": 2.634, "200": 2.359}},
    "V2_at_440_eval": {"si_at_200": 0.8641, "slots_per_beam_at_200": 1.96},
    "V3_at_440_eval": {"si_at_200": 0.8205, "slots_per_beam_at_200": 1.68},
    "V1_frozen": {"si_at_200": 0.898, "slots_per_beam_at_200": 1.82},
    "greedy_frozen": {"slots_per_beam_at_200": 4.00},
    "reading": ("G0 在 N=200 的每束槽数为 2.36，高于 V2 的 1.96、V3 的 1.68 与冻结 V1 的 1.82，"
                "更接近贪心的 4.00 槽/束，说明逐束编码＋注意力池化确实把频谱用量推高了；"
                "但其绝对需求满足率最低（验证集 0.82 量级），即资源被摊薄而未能换来更高的满足率。"
                "口径提示：G0 为 144 景验证集、V2/V3 为 440 景考核集，两者不可直接数值比较，"
                "此处只用于说明机制方向。"),
}
path = ops18.LOCAL / "run_manifest.json"
manifest = json.loads(path.read_text(encoding="utf-8-sig"))
manifest["g0_final_validation"] = facts
manifest["updated_utc"] = ops18.utc_now()
ops18.write_json(path, manifest)
ops18.write_json(ops18.LOCAL / "analysis" / "g0_final_validation.json", facts)
print("已记录 G0 终点逐N数据与跨臂槽数对照")
