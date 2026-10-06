"""Summarize S2-P18 assessment results per beam count and build the Chinese figures.

Inputs are the saved ``results.json`` of each arm's assessment/refinement job (the
scenes rows carry ``beam_count``, ``metrics`` and the per-beam ``beams`` list). Every
aggregate states its denominator, and figures are produced locally with a CJK font
because the remote containers have none.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ops18  # noqa: E402
from analyse18 import (CANDIDATE_METRICS, configure_plotting, effective_carried,
                       per_beam_min_ratio, pick, run_index)  # noqa: E402

OUT = ops18.LOCAL / "analysis"
FETCH_RESULTS = "analysis_inputs/results_@RUN@.json"


def fetch_results(port: int, run_id: str, directory: str) -> dict:
    """Copy one job's results.json back to the local analysis inputs.

    The remote ``results.json`` is binary-read on the shared NFS by the remote
    helper writing a copy into the instance ``ops`` directory, then pulled with
    SFTP. Piping the bytes through stdout would run them through a text decode
    and can corrupt the payload, so that route is deliberately avoided.
    """
    target = OUT.parent / FETCH_RESULTS.replace("@RUN@", run_id)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.is_file():
        info = ops18.instance(port)
        staging = f"{info['ops']}/results_{run_id}.json"
        body = (
            "import shutil,pathlib,hashlib,json\n"
            f"source=pathlib.Path({(directory + '/results.json')!r})\n"
            f"target=pathlib.Path({staging!r})\n"
            "shutil.copy2(source,target)\n"
            "digest=hashlib.sha256(target.read_bytes()).hexdigest()\n"
            "print(json.dumps({'bytes':target.stat().st_size,'sha256':digest}))\n"
        )
        result = ops18.remote_python(body, port, timeout=900)
        if result["exit_code"]:
            raise RuntimeError(f"{run_id} 结果转存失败: {result['stderr'][-300:]}")
        digest = json.loads(result["stdout"].strip().splitlines()[-1])
        import paramiko
        client = ops18.connect(port)
        try:
            with paramiko.SFTPClient.from_transport(client.get_transport()) as sftp:
                sftp.get(staging, str(target))
        finally:
            client.close()
        local = ops18.sha256_file(target)
        if local != digest["sha256"]:
            raise RuntimeError(f"{run_id} 结果传输校验失败：{local} != {digest['sha256']}")
    return json.loads(target.read_text(encoding="utf-8"))


def summarize_scenes(rows: list, group_key) -> dict:
    buckets = {}
    for row in rows:
        buckets.setdefault(group_key(row), []).append(row)

    def mean(values):
        values = [value for value in values if isinstance(value, (int, float))]
        return statistics.fmean(values) if values else None

    out = {}
    for key, items in sorted(buckets.items(), key=lambda kv: (kv[0][0] or 0, str(kv[0][1]))):
        entry = {"scenes": len(items)}
        for name in CANDIDATE_METRICS:
            entry[name] = mean([pick(row.get("metrics") or {}, name) for row in items])
        carried = [effective_carried(row.get("beams")) for row in items]
        entry["effective_carried_bps"] = mean(carried)
        tails = [per_beam_min_ratio(row.get("beams")) for row in items]
        mins = [item["min"] for item in tails if item]
        p05 = [item["p05"] for item in tails if item]
        entry["beam_min_ratio"] = mean(mins)
        entry["beam_p05_ratio"] = mean(p05)
        out[key] = entry
    return out


def figure_terminal(table: dict, field: str, label: str, path: Path) -> dict:
    plt, font = configure_plotting()
    counts = sorted({key[0] for key in table})
    methods = sorted({key[1] for key in table})
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    for method in methods:
        xs, ys = [], []
        for count in counts:
            values = [table[key][field] for key in table
                      if key[0] == count and key[1] == method and table[key][field] is not None]
            if values:
                xs.append(count)
                ys.append(statistics.fmean(values))
        if xs:
            ax.plot(xs, ys, marker="o", label=method)
    ax.set_xlabel("目标波束数 N")
    ax.set_ylabel(label)
    ax.set_title(f"{label}随波束数变化（同批考核场景，每档两档负载平均）")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return {"figure": str(path), "font": font, "methods": methods, "counts": counts}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    inventory = []
    for port in ops18.PORTS:
        for run in run_index(port):
            if run["kind"] != "evaluate" or run["stage"] != "completed":
                continue
            inventory.append({"port": port, **{k: v for k, v in run.items() if k != "directory"},
                              "directory": run["directory"]})
    report = {"inventory": inventory}
    if args.probe:
        print(json.dumps(report, ensure_ascii=False, indent=1))
        return
    summaries = {}
    for entry in inventory:
        data = fetch_results(entry["port"], entry["run_id"], entry["directory"])
        scenes = data.get("scenes") or []
        key = entry["request_id"]
        model_rows = [row for row in scenes if (row.get("model") or row.get("policy") or "model") == "model"] or scenes
        table = summarize_scenes(model_rows, lambda row: (row.get("beam_count"),
                                                          (row.get("scene_metadata") or {}).get("load_multiplier")))
        summaries[key] = {"scenes": len(scenes), "per_group": table}
        for field, label in (("si", "需求满足率 SI"), ("sgm", "供需匹配程度 SGM"),
                             ("slots", "每束占用槽数")):
            path = OUT / f"中文_{field}_随束数变化.png"
            if field == "slots":
                for group_key in table:
                    entry_row = table[group_key]
                    if entry_row.get("slots") and group_key[0]:
                        entry_row["slots_per_beam"] = entry_row["slots"] / group_key[0]
                summaries[key]["slots_note"] = "按每束槽数绘制"
            summaries[key].setdefault("figures", []).append(
                figure_terminal(table, "slots_per_beam" if field == "slots" else field, label, path))
    ops18.write_json(OUT / "assessment_summary.json",
                     {"checked_utc": ops18.utc_now(), **report, "summaries": summaries})
    print(json.dumps({"jobs": len(inventory),
                      "outputs": [str(path) for path in sorted(OUT.glob("*.png"))]},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
