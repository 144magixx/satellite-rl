"""Aggregate S2-P18 comparison results and build the Chinese report figures.

Reads the completed evaluation / refinement jobs from both instances, so it works
the same way for the pilot (144 validation scenes) and for the final stage
(440 common assessment scenes). It reports which fields the saved rows actually
carry instead of assuming a schema, and every figure is generated from those
saved rows only.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ops18  # noqa: E402

# The three arms plus the two baselines carried by each assessment job.
METHODS = ("V2", "V3", "G0", "fixed", "greedy")
RUN_PREFIX = "s2p18-"

# Field names taken from ``WorkbenchEnvironment.metrics`` (the authoritative
# producer) rather than guessed. ``carried`` is derived from the saved per-beam
# rows: the环境 metric is total throughput, while the plan's "有效承载" is
# Σmin(C_i, D_i), which must be computed from the beam rows.
CANDIDATE_METRICS = {
    "si": ("final_si_mean",),
    "si_total": ("final_si_total",),
    "sgm": ("final_sgm_mean",),
    "basic_met": ("basic_met_count",),
    "slots": ("actual_slots",),
    "requested_slots": ("requested_slots",),
    "power": ("power_w",),
    "throughput_bps": ("throughput_bps",),
    "zero_capacity": ("zero_capacity_count",),
    "jain": ("jain",),
    "clipped": ("clipped_count",),
    "complete": ("complete",),
    "overload": ("overload",),
}

FETCH_RESULTS = r'''
import json,pathlib
directory=pathlib.Path(@DIR@)
path=directory/"results.json"
out={"exists":path.is_file()}
if path.is_file():
    data=json.loads(path.read_text(encoding="utf-8"))
    out["top_keys"]=sorted(data.keys())
    scenes=data.get("scenes") or []
    out["scene_count"]=len(scenes)
    if scenes:
        out["scene_keys"]=sorted(scenes[0].keys())
        metric_keys=set()
        for row in scenes[:5]:
            metrics=row.get("metrics") or {}
            metric_keys |= set(metrics.keys())
        out["metric_keys"]=sorted(metric_keys)
    baselines=data.get("baselines") or {}
    out["baseline_keys"]=sorted(baselines.keys())
    out["summary"]=data.get("summary")
print(json.dumps(out,ensure_ascii=False)[:12000])
'''


def pick(metrics: dict, name: str):
    """Read one metric by its authoritative name, tolerating missing fields."""
    for key in CANDIDATE_METRICS[name]:
        value = metrics.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def effective_carried(beams) -> float | None:
    """Σmin(C_i, D_i) from the saved per-beam rows.

    ``throughput_bps`` is ΣC_i, so the plan's "有效承载" estimate is recomputed
    from ``beam_service_fields``' own ``effective_served_bps``. Missing fields are
    reported as None instead of being approximated.
    """
    if not beams:
        return None
    total = 0.0
    for beam in beams:
        value = beam.get("effective_served_bps")
        if value is None:
            capacity = beam.get("capacity_bps", beam.get("rate_bps"))
            demand = beam.get("required_bps")
            if not isinstance(capacity, (int, float)) or not isinstance(demand, (int, float)):
                return None
            value = min(float(capacity), float(demand))
        total += float(value)
    return total


def per_beam_min_ratio(beams) -> float | None:
    """逐束 C/D 的最小值与 5% 分位，用于 G3 的尾部判据。"""
    if not beams:
        return None
    ratios = [float(beam["capacity_demand_ratio"]) for beam in beams
              if isinstance(beam.get("capacity_demand_ratio"), (int, float))]
    if not ratios:
        return None
    ratios.sort()
    index = max(0, min(len(ratios) - 1, int(0.05 * (len(ratios) - 1))))
    return {"min": ratios[0], "p05": ratios[index], "n": len(ratios)}


def configure_plotting():
    """Agg backend plus a CJK-capable font so Chinese labels render on both hosts."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import font_manager
    import matplotlib.pyplot as plt
    available = {font.name for font in font_manager.fontManager.ttflist}
    chosen = next((name for name in ("SimHei", "Microsoft YaHei", "Noto Sans CJK SC",
                                     "WenQuanYi Zen Hei", "Source Han Sans SC", "DejaVu Sans")
                   if name in available), "DejaVu Sans")
    plt.rcParams.update({
        "font.family": chosen,
        "font.size": 9,
        "axes.unicode_minus": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.dpi": 180,
    })
    return plt, chosen


def run_index(port: int) -> list:
    """All S2-P18 jobs on one instance, newest first."""
    runs = ops18.api(port, "/runs", timeout=60).get("runs", [])
    out = []
    for item in runs:
        request = str(item.get("request_id", ""))
        if not request.startswith(RUN_PREFIX):
            continue
        detail = ops18.api(port, "/runs/" + item["run_id"], timeout=60)
        status = detail.get("status") or {}
        out.append({"run_id": item["run_id"], "request_id": request,
                    "kind": item.get("kind"),
                    "stage": status.get("stage") or item.get("stage"),
                    "env_step": status.get("env_step"),
                    "directory": detail.get("path")})
    return out


def probe_results(port: int, directory: str) -> dict:
    body = FETCH_RESULTS.replace("@DIR@", repr(directory))
    result = ops18.remote_python(body, port, timeout=300)
    lines = [line for line in result["stdout"].strip().splitlines() if line.strip()]
    if not lines:
        return {"exists": False, "error": result["stderr"][-200:]}
    return json.loads(lines[-1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", action="store_true",
                        help="只探测已保存结果的字段结构，不做汇总")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    report = {}
    for port in ops18.PORTS:
        for run in run_index(port):
            if run["kind"] not in ("evaluate", "refine"):
                continue
            if run["stage"] not in ("completed", "failed", "interrupted", "training"):
                continue
            entry = {"port": port, **{k: v for k, v in run.items() if k != "directory"}}
            if args.probe:
                entry["probe"] = probe_results(port, run["directory"])
            report[run["run_id"]] = entry
    if args.write:
        ops18.write_json(ops18.LOCAL / "analysis_inputs" / "evaluation_inventory.json",
                         {"checked_utc": ops18.utc_now(), "jobs": report})
    print(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
