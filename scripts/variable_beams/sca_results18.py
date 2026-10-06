"""Read each arm's SCA per-method results by streaming the CSV over SFTP.

The refinement's ``results.json`` is far too large to move, so its own
``power_results.csv`` export is read directly from the instance archive; the file is
small enough to stream and carries the per-scene/method metrics needed for the
direct-versus-SCA comparison.
"""
from __future__ import annotations

import csv
import io
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, r"D:\paper\scripts\variable_beams")
import ops18  # noqa: E402

JOBS = {"V3": (20424, "refine_sac_20261006_173515_4b10636d"),
        "V2": (20418, "refine_sac_20261006_224531_316265cd")}
OUT = Path(r"D:\paper\outputs\experiments\variable_beams_s2p18_20261006\analysis")
LABELS = {"none": "原分配（不调整）", "sca": "功率优化 SCA", "sca_si_lp": "SI优先 SCA"}


def stream_csv(port: int, run_id: str, filename: str, limit_mb: float = 400.0) -> str:
    """Stream one job file over SFTP with a size guard."""
    info = ops18.instance(port)
    directory = f"{info['archive']}/outputs/workbench/jobs/{run_id}/{filename}"
    probe = ops18.remote_python(
        "import json,pathlib\n"
        f"p=pathlib.Path({directory!r})\n"
        "print(json.dumps({'exists':p.is_file(),'bytes':p.stat().st_size if p.is_file() else 0}))\n",
        port, timeout=180)
    rows = [line for line in probe["stdout"].strip().splitlines() if line.strip()]
    meta = json.loads(rows[-1]) if rows else {"exists": False, "bytes": 0}
    if not meta.get("exists"):
        raise RuntimeError(f"{filename} 不存在于 {run_id}")
    if meta["bytes"] > limit_mb * 1024 * 1024:
        raise RuntimeError(f"{filename} 为 {meta['bytes']} 字节，超过阈值")
    print(f"  {filename}: {meta['bytes']} 字节，开始流式读取")
    import paramiko
    client = ops18.connect(port)
    try:
        with paramiko.SFTPClient.from_transport(client.get_transport()) as sftp:
            with sftp.open(directory) as handle:
                handle.prefetch()
                return handle.read().decode("utf-8", errors="replace")
    finally:
        client.close()


def mean(values):
    clean = [v for v in values if isinstance(v, (int, float))]
    return round(statistics.fmean(clean), 4) if clean else None


def main() -> None:
    report = {}
    for arm, (port, run_id) in JOBS.items():
        print(arm, run_id)
        text = stream_csv(port, run_id, "power_results.csv")
        rows = list(csv.DictReader(io.StringIO(text)))
        print(f"  行数={len(rows)}")
        by_method = {}
        for row in rows:
            by_method.setdefault(str(row.get("method")), []).append(row)
        entry = {}
        for method, items in sorted(by_method.items()):
            def col(key):
                out = []
                for row in items:
                    try:
                        out.append(float(row[key]))
                    except (KeyError, TypeError, ValueError):
                        continue
                return out
            entry[method] = {"label": LABELS.get(method, method), "rows": len(items),
                             "si": mean(col("final_si_mean")),
                             "sgm": mean(col("final_sgm_mean")),
                             "power_w": mean(col("power_w")),
                             "slots": mean(col("actual_slots"))}
        report[arm] = entry
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "sca_per_method.json").write_text(
        json.dumps({"checked_utc": ops18.utc_now(), "arms": report},
                   ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
