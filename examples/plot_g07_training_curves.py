"""Rebuild G07 learning figures from the frozen workbench JSONL logs.

Run from the repository root with ``python examples/plot_g07_training_curves.py``.
This script reads existing records only; it does not evaluate or train a model.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "train_sac_20260928_142100_38559f37"
RUN = ROOT / "outputs" / "workbench" / "jobs" / RUN_ID
OUT = ROOT / "outputs" / "experiments" / "europe_adsb_20260928" / "curves"
TRAIN_WINDOW = 100  # episodes, equivalent to 10,000 environment steps
LEARNING_WINDOW = 100  # recorded learning points, about 5,000 steps


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    # No centered window or future points: point x uses the previous window values.
    cumulative = np.cumsum(np.insert(values, 0, 0.0))
    count = np.minimum(np.arange(1, len(values) + 1), window)
    left = np.maximum(0, np.arange(1, len(values) + 1) - window)
    return (cumulative[1:] - cumulative[left]) / count


def series(rows: list[dict], name: str, scale: float = 1.0) -> np.ndarray:
    return np.asarray([row["values"][name] * scale for row in rows], dtype=float)


def save_figure(fig: plt.Figure, stem: str) -> list[Path]:
    paths = [OUT / f"{stem}.png", OUT / f"{stem}.svg"]
    for path in paths:
        options = {"metadata": {"Date": "2026-09-28"}} if path.suffix == ".svg" else {}
        fig.savefig(path, dpi=170, bbox_inches="tight", **options)
    plt.close(fig)
    return paths


def plot_service(training: list[dict], validation: list[dict]) -> list[Path]:
    panels = [
        ("final_si_mean", "Final demand satisfaction", 100, "%"),
        ("basic_met_count", "Beams meeting 10% demand", 1, "/ 100"),
        ("final_sgm_mean", "Supply-demand matching score", 1, ""),
        ("power_w", "Radiated power", 1, "W"),
        ("return", "Episode return", 1, ""),
        ("actual_slots", "Allocated slots", 1, ""),
    ]
    x_train = np.asarray([row["step"] for row in training])
    x_val = np.asarray([row["step"] for row in validation])
    fig, axes = plt.subplots(3, 2, figsize=(13.2, 11), sharex=True)
    for ax, (key, title, scale, unit) in zip(axes.flat, panels):
        raw = series(training, key, scale)
        val = series(validation, key, scale)
        ax.plot(x_train, raw, color="#9fc1d5", alpha=0.35, lw=0.6, label="Training episode")
        ax.plot(x_train, moving_average(raw, TRAIN_WINDOW), color="#12689a", lw=1.7,
                label=f"Training trailing mean ({TRAIN_WINDOW} episodes)")
        ax.plot(x_val, val, color="#cc6b28", lw=1.6, marker="o", markersize=2.8,
                label="Validation (20 scenes x 3)")
        ax.set_title(title)
        ax.set_ylabel(unit)
        ax.grid(alpha=0.22)
        if key == "final_si_mean":
            ax.set_ylim(0, 100)
        elif key == "basic_met_count":
            ax.set_ylim(80, 101)
        elif key == "power_w":
            ax.axhline(3000, color="#8a1717", ls="--", lw=1, label="Power budget")
            ax.set_ylim(0, 3200)
    for ax in axes[-1]:
        ax.set_xlabel("Environment steps")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.995))
    fig.suptitle("G07: service and resource curves", y=1.035, fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    return save_figure(fig, "service_resource_curves")


def plot_learning(learning: list[dict]) -> list[Path]:
    x = np.asarray([row["step"] for row in learning])
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 7.8), sharex=True)
    items = [
        ("actor_loss", "Actor loss", "#12689a"),
        ("critic_1_loss", "Critic 1 loss", "#cc6b28"),
        ("critic_2_loss", "Critic 2 loss", "#a14379"),
        ("alpha", "Temperature alpha", "#297e55"),
    ]
    for ax, (key, title, color) in zip(axes.flat, items):
        raw = series(learning, key)
        ax.plot(x, raw, color=color, alpha=0.16, lw=0.6, label="Logged value")
        ax.plot(x, moving_average(raw, LEARNING_WINDOW), color=color, lw=1.6,
                label=f"Trailing mean ({LEARNING_WINDOW} logged points)")
        ax.set_title(title)
        ax.set_yscale("symlog", linthresh=1 if key != "alpha" else 0.01)
        ax.grid(alpha=0.22)
    for ax in axes[-1]:
        ax.set_xlabel("Environment steps")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.01))
    fig.suptitle("G07: parameter-update diagnostics (symmetric-log y axes)", y=1.055, fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    return save_figure(fig, "learning_diagnostics")


def plot_test(test_rows: list[dict]) -> list[Path]:
    by_scene: dict[str, list[dict]] = defaultdict(list)
    for row in test_rows:
        by_scene[row["scene_sha256"]].append(row["metrics"])
    if len(by_scene) != 100 or any(len(rows) != 3 for rows in by_scene.values()):
        raise ValueError("Expected 100 distinct test scenes and three repeats each")
    panels = [
        ("final_si_mean", "Final satisfaction", 100, "%"),
        ("basic_met_count", "Beams meeting 10% demand", 1, "/ 100"),
        ("power_w", "Radiated power", 1, "W"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.8))
    for ax, (key, title, scale, unit) in zip(axes, panels):
        points = np.sort(np.asarray([
            np.mean([item[key] for item in records]) * scale
            for records in by_scene.values()
        ]))
        ax.step(points, np.arange(1, len(points) + 1) / len(points), where="post",
                color="#12689a", lw=1.8)
        ax.axvline(float(points.mean()), color="#cc6b28", ls="--", lw=1.3,
                   label=f"Mean {points.mean():.2f}")
        ax.set_title(title)
        ax.set_xlabel(unit)
        ax.set_ylabel("Fraction of test scenes")
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.22)
        ax.legend(loc="lower right", frameon=False)
    fig.suptitle("G07: final model on 100 held-out scenes (three actions averaged per scene)", y=1.04)
    fig.tight_layout()
    return save_figure(fig, "test_scene_distribution")


def main() -> None:
    inputs = {name: RUN / name for name in ("metrics.jsonl", "evaluations.jsonl", "results.json")}
    if not all(path.is_file() for path in inputs.values()):
        raise FileNotFoundError("G07 workbench records are incomplete")
    metrics = load_jsonl(inputs["metrics.jsonl"])
    evaluations = load_jsonl(inputs["evaluations.jsonl"])
    training = [row for row in metrics if row["group"] == "training"]
    validation = [row for row in metrics if row["group"] == "validation"]
    learning = [row for row in metrics if row["group"] == "learning"]
    test = [row for row in evaluations if row["group"] == "test"]
    if (len(training), len(validation), len(learning), len(test)) != (4000, 40, 7960, 300):
        raise ValueError("Unexpected G07 record counts; refusing to draw incomplete curves")
    if training[-1]["step"] != 400000 or validation[-1]["step"] != 400000:
        raise ValueError("Training or validation did not reach the planned budget")
    for group in (training, validation, learning):
        if not all(math.isfinite(value) for row in group for value in row["values"].values()):
            raise ValueError("Nonfinite logged value")
    result = json.loads(inputs["results.json"].read_text(encoding="utf-8"))
    if result["metadata"]["run_id"] != RUN_ID or result["metadata"]["actual_scene_count"] != 100:
        raise ValueError("Final result metadata does not match G07")
    if not math.isclose(validation[-1]["values"]["final_si_mean"],
                        np.mean([row["metrics"]["final_si_mean"] for row in evaluations
                                 if row["group"] == "validation" and row["env_step"] == 400000]),
                        rel_tol=0, abs_tol=1e-12):
        raise ValueError("Validation curve differs from per-scene records")
    if not math.isclose(result["summary"]["final_si_mean"],
                        np.mean([row["metrics"]["final_si_mean"] for row in test]),
                        rel_tol=0, abs_tol=1e-12):
        raise ValueError("Test distribution differs from summary")
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "svg.hashsalt": RUN_ID})
    OUT.mkdir(parents=True, exist_ok=True)
    figures = []
    figures += plot_service(training, validation)
    figures += plot_learning(learning)
    figures += plot_test(test)
    manifest = {
        "run_id": RUN_ID,
        "source_sha256": {name: sha256(path) for name, path in inputs.items()},
        "records": {"training_episodes": len(training), "validation_checkpoints": len(validation),
                    "learning_points": len(learning), "test_allocations": len(test),
                    "distinct_test_scenes": 100},
        "smoothing": {"training_trailing_episodes": TRAIN_WINDOW,
                      "learning_trailing_logged_points": LEARNING_WINDOW,
                      "validation": "none", "test": "three actions averaged within each scene"},
        "figures_sha256": {path.name: sha256(path) for path in figures},
    }
    (OUT / "plot_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
