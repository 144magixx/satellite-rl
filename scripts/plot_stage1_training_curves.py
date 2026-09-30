"""Plot saved coverage PPO records without joining simultaneous episodes.

This is postprocessing only. It never imports a policy or starts a worker.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    args = parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    suite = args.suite.resolve()
    manifest = json.loads((suite / "run_manifest.json").read_text(encoding="utf-8"))
    output = suite / "curves"
    output.mkdir(exist_ok=True)
    ledger = {"source_commit": manifest["plan"]["source_commit"],
              "plot_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "smoothing": "none", "scale": "linear", "groups": {}}
    for label, job in manifest["training"].items():
        directory = suite / "raw" / job["run_id"]
        episodes = read_rows(directory / "episodes.jsonl")
        updates = read_rows(directory / "updates.jsonl")
        validation = [row for row in read_rows(directory / "metrics.jsonl")
                      if row.get("group") == "validation"]
        total = json.loads((directory / "config.json").read_text())["training"]["total_steps"]
        panels = []
        for key, title in (("episode_return", "Learning return: all raw episodes"),
                           ("coverage_demand", "Terminal demand coverage: all raw episodes"),
                           ("load_cv", "Terminal load CV: all raw episodes"),
                           ("empty_beams", "Empty beams: all raw episodes")):
            panels.append((title, [(row["env_step"], row["metrics"][key])
                                   for row in episodes], "scatter"))
        for key, title in (("coverage_demand", "Validation demand coverage: 20-scene mean"),
                           ("load_cv", "Validation load CV: valid-layout mean")):
            panels.append((title, [(row["step"], row["values"][key])
                                   for row in validation], "line"))
        panels.extend((key.replace("_", " ").title(),
                       [(row["env_step"], row[key]) for row in updates], "line")
                      for key in ("value_loss", "entropy"))
        figure, axes = plt.subplots(4, 2, figsize=(15, 13))
        for axis, (title, points, style) in zip(axes.flat, panels):
            if not points:
                raise ValueError(f"Missing original records for {label}/{title}")
            x, y = zip(*points)
            if style == "scatter":
                axis.scatter(x, y, s=6, alpha=0.18, linewidths=0, rasterized=True)
            else:
                axis.plot(x, y, marker=".", markersize=4, linewidth=0.9)
            axis.set_title(title, fontsize=10)
            axis.set_xlabel("Environment interactions")
            axis.set_xlim(0, total)
            axis.grid(alpha=0.2)
        figure.suptitle(f"{label} / seed 51001 / no smoothing / linear scale", fontsize=14)
        figure.tight_layout(rect=(0, 0, 1, 0.97))
        figure.savefig(output / f"{label}_training_diagnostics.png", dpi=160)
        plt.close(figure)
        ledger["groups"][label] = {
            "run_id": job["run_id"], "episode_rows": len(episodes),
            "completion_steps": sorted({row["env_step"] for row in episodes}),
            "rollout_update_rows": len(updates), "validation_rows": len(validation),
            "plotted_counts": {title: len(points) for title, points, _ in panels},
            "input_sha256": {name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
                             for name in ("episodes.jsonl", "updates.jsonl", "metrics.jsonl")}}
    (output / "training_diagnostics_ledger.json").write_text(
        json.dumps(ledger, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "completed", "groups": list(ledger["groups"]),
                      "raw_episode_scatter": True, "smoothing": "none"}))


if __name__ == "__main__":
    main()
