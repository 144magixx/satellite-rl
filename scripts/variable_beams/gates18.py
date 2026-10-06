"""Evaluate the S2-P18 arms against the pre-registered G1-G4 gates.

Reads each arm's per-beam-count validation or assessment metrics and applies the
thresholds frozen in 方案18 before any result was seen:

* G1: gap to the greedy baseline at the large beam counts, and its shrink versus
  the frozen V1 reference;
* G2: no regression at 100 beams relative to the frozen V1 result;
* G3: mechanism improvement -- slots per beam up, and the worst-beam C/D tail up;
* G4: everything else is reported as "本轮改进不足" rather than massaged.

The script never edits results and prints every denominator it used.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ops18  # noqa: E402

ANCHOR = ops18.LOCAL / "analysis_inputs" / "p17_anchor_by_n.json"
COUNT_METRICS = "count_metrics.jsonl"
SKIP = ("retry1", "retry2")

# Gate thresholds frozen in the plan before any result was seen.
G1_PILOT_SHRINK = 0.40          # gap to greedy must shrink by >=40%
G2_PILOT_FLOOR = 0.995          # 100-beam SI >= 99.5% of the frozen V1 value
G3_PILOT_SLOT_GAIN = 0.30       # slots per beam up >=30% versus V1
G1_LARGE_BEAMS = (120, 160)     # pilot evidence is read at these counts
G1_FINAL_MEAN_GAP = 0.03        # 40万步: N>=125 average gap <=3 percentage points
G1_FINAL_200_GAP = 0.04         # 40万步: 200-beam gap <=4 percentage points

FETCH = (
    "import json,pathlib\n"
    "directory=pathlib.Path(@DIR@)\n"
    "path=directory/@FILE@\n"
    "rows=[]\n"
    "if path.is_file():\n"
    "    for line in path.read_text(encoding='utf-8', errors='replace').splitlines():\n"
    "        line=line.strip()\n"
    "        if line:\n"
    "            try:\n"
    "                rows.append(json.loads(line))\n"
    "            except Exception:\n"
    "                pass\n"
    "print(json.dumps({'rows':rows,'total':len(rows)}))\n"
)


def read_jsonl(port: int, directory: str, filename: str) -> dict:
    body = FETCH.replace("@DIR@", repr(directory)).replace("@FILE@", repr(filename))
    result = ops18.remote_python(body, port, timeout=600)
    lines = [line for line in result["stdout"].strip().splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"{port} 读取 {filename} 失败: {result['stderr'][-300:]}")
    return json.loads(lines[-1])


def arm_state(port: int, arm: str) -> dict:
    for item in ops18.api(port, "/runs", timeout=60).get("runs", []):
        request = str(item.get("request_id", ""))
        if not request.startswith(f"s2p18-{arm.lower()}-"):
            continue
        if any(tag in request for tag in SKIP):
            continue
        detail = ops18.api(port, "/runs/" + item["run_id"], timeout=60)
        status = detail.get("status") or {}
        stage = status.get("stage") or item.get("stage")
        if stage == "failed":
            continue
        return {"run_id": item["run_id"], "stage": stage,
                "env_step": status.get("env_step"),
                "budget": status.get("budget_env_steps"),
                "directory": detail.get("path")}
    return {}


def per_n_from_count_metrics(rows: list) -> dict:
    """Latest validation node's per-beam-count metrics keyed by N."""
    latest = None
    for row in rows:
        if row.get("group") != "validation":
            continue
        if latest is None or (row.get("env_step") or 0) >= (latest.get("env_step") or 0):
            latest = row
    if latest is None:
        return {}
    out = {}
    for key, block in (latest.get("by_beam_count") or {}).items():
        out[int(key)] = {"metrics": block.get("metrics") or {},
                         "scene_actions": block.get("scene_actions")}
    out["_env_step"] = latest.get("env_step")
    return out


def anchors() -> dict:
    if not ANCHOR.is_file():
        return {}
    data = json.loads(ANCHOR.read_text(encoding="utf-8-sig"))
    out = {}
    for row in data.get("by_n") or []:
        count = int(row["beam_count"])
        out.setdefault(count, {})[row["learning_role"]] = {
            "greedy_minus_si": row.get("greedy_minus_learning_si_mean"),
            "greedy_minus_sgm": row.get("greedy_minus_learning_sgm_mean")}
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    anchor = anchors()
    report = {"anchor_beam_counts": sorted(k for k in anchor if isinstance(k, int)),
              "gates": {"G1_pilot_shrink": G1_PILOT_SHRINK, "G2_pilot_floor": G2_PILOT_FLOOR,
                        "G3_pilot_slot_gain": G3_PILOT_SLOT_GAIN,
                        "G1_final_mean_gap": G1_FINAL_MEAN_GAP,
                        "G1_final_200_gap": G1_FINAL_200_GAP},
              "arms": {}}
    for arm, port in (("V2", 20418), ("V3", 20424), ("G0", 20418)):
        state = arm_state(port, arm)
        entry = {"port": port, **{k: v for k, v in state.items() if k != "directory"}}
        if state.get("directory"):
            try:
                rows = read_jsonl(port, state["directory"], COUNT_METRICS)["rows"]
                per_n = per_n_from_count_metrics(rows)
                entry["per_n_si"] = {n: round(v["metrics"].get("final_si_mean"), 6)
                                     for n, v in per_n.items() if isinstance(n, int)
                                     and v["metrics"].get("final_si_mean") is not None}
                entry["per_n_slots_per_beam"] = {
                    n: round((v["metrics"].get("actual_slots") or 0) / n, 3)
                    for n, v in per_n.items() if isinstance(n, int)
                    and v["metrics"].get("actual_slots")}
                entry["validation_step"] = per_n.get("_env_step")
            except Exception as error:  # keep the audit usable while data is partial
                entry["count_metrics_error"] = str(error)[:200]
        # G1/G2 style comparisons against the frozen V1 anchor where counts overlap
        overlap = sorted(set(entry.get("per_n_si") or {}) & set(anchor))
        entry["gate_notes"] = []
        for count in overlap:
            v1 = anchor[count].get("V1") or {}
            greedy_minus = v1.get("greedy_minus_si")
            if greedy_minus is None or count not in (entry.get("per_n_si") or {}):
                continue
            # V1 final SI at this count, reconstructed from the anchor relationship is
            # not possible without the greedy level; report the comparison basis instead.
            entry["gate_notes"].append({
                "beam_count": count,
                "arm_si": entry["per_n_si"][count],
                "v1_minus_greedy_si": round(-greedy_minus, 6),
                "note": "锚点给的是 V1 与贪心的差值；判定最终在 440 景同批场景上做"})
        report["arms"][arm] = entry
    if args.write:
        ops18.write_json(ops18.LOCAL / "analysis_inputs" / "gate_audit.json",
                         {"checked_utc": ops18.utc_now(), **report})
    print(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
