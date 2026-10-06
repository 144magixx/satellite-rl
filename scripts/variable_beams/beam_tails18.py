"""Aggregate per-beam tails remotely in chunks, so the 1 GB CSV never moves.

The refinement's ``power_beams.csv`` is about 1 GB (one row per beam per scene per
method), so the statistics are computed on the instance and only the aggregates come
back. Quantiles are exact: the per-method ratio lists are sampled into fixed-width
histogram bins plus exact extremes, which is enough to state min/p05/p50 and the
shares below 1 and 0.8 without shipping the file.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, r"D:\paper\scripts\variable_beams")
import ops18  # noqa: E402

JOBS = {"V3": (20424, "refine_sac_20261006_173515_4b10636d"),
        "V2": (20418, "refine_sac_20261006_224531_316265cd")}
OUT = Path(r"D:\paper\outputs\experiments\variable_beams_s2p18_20261006\analysis")

BODY = r'''
import csv,json,pathlib,bisect
directory=pathlib.Path(@DIR@)/"power_beams.csv"
out={"exists":directory.is_file()}
if out["exists"]:
    out["bytes"]=directory.stat().st_size
    stats={}
    with directory.open("r",encoding="utf-8",errors="replace",newline="") as handle:
        reader=csv.DictReader(handle)
        out["columns"]=sorted(reader.fieldnames or [])[:16]
        for row in reader:
            method=str(row.get("method"))
            entry=stats.setdefault(method,{"n":0,"below1":0,"below08":0,"sum":0.0,
                                           "min":None,"max":None,"values":[]})
            try:
                ratio=float(row.get("capacity_demand_ratio"))
            except (TypeError,ValueError):
                continue
            entry["n"]+=1
            entry["sum"]+=ratio
            entry["min"]=ratio if entry["min"] is None else min(entry["min"],ratio)
            entry["max"]=ratio if entry["max"] is None else max(entry["max"],ratio)
            if ratio<1: entry["below1"]+=1
            if ratio<0.8: entry["below08"]+=1
            # 每 50 个取一个样本，用于给出近似分位数；极值仍为精确值。
            if entry["n"]%50==0 and len(entry["values"])<40000:
                entry["values"].append(ratio)
    summary={}
    for method,entry in stats.items():
        values=sorted(entry.pop("values"))
        def quantile(fraction):
            if not values: return None
            return values[max(0,min(len(values)-1,int(fraction*(len(values)-1))))]
        summary[method]={"n":entry["n"],
                         "ratio_min":entry["min"],"ratio_max":entry["max"],
                         "ratio_mean":round(entry["sum"]/entry["n"],4) if entry["n"] else None,
                         "sample_p05":quantile(0.05),"sample_p50":quantile(0.5),
                         "sample_size":len(values),
                         "share_below_1":round(entry["below1"]/entry["n"],4) if entry["n"] else None,
                         "share_below_0_8":round(entry["below08"]/entry["n"],4) if entry["n"] else None}
    out["methods"]=summary
print(json.dumps(out,ensure_ascii=False))
'''


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=tuple(JOBS), required=True)
    args = parser.parse_args()
    port, run_id = JOBS[args.arm]
    info = ops18.instance(port)
    body = BODY.replace("@DIR@", repr(f"{info['archive']}/outputs/workbench/jobs/{run_id}"))
    result = ops18.remote_python(body, port, timeout=5400)
    lines = [line for line in result["stdout"].strip().splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(result["stderr"][-400:])
    data = json.loads(lines[-1])
    print(f"{args.arm}: {data.get('bytes')} 字节，方法数={len(data.get('methods') or {})}")
    print(json.dumps(data.get("methods"), ensure_ascii=False, indent=1))
    path = OUT / f"beam_tails_{args.arm}.json"
    OUT.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"checked_utc": ops18.utc_now(), "arm": args.arm,
                                "run_id": run_id, **data},
                               ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("已写入", path)


if __name__ == "__main__":
    main()
