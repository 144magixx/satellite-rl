"""Make publication figures from the completed, SHA-audited S2-P17 report.

Usage on the approved 20424 analysis checkout::

    python -B scripts/variable_beams/publication_figures.py \
      --audit-dir BASE/analysis/report_<audited-analysis-commit> \
      --output-dir BASE/analysis/publication_figures_<analysis-commit>

The audit directory must already contain a successful artifact_manifest.json.
The output directory must not exist. This script reads audited tables and the
two frozen training logs; it never runs a model, optimizer, or physical core.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import tempfile

import numpy as np


ROLES = ("V0", "V1", "H0", "fixed", "greedy")
STAGES = ("direct", "final")
LOADS = (0.75, 1.25)
N_VALUES = (50, 60, 75, 90, 100, 125, 140, 150, 175, 180, 200)
REPEATS = {"V0": 3, "V1": 3, "H0": 3, "fixed": 1, "greedy": 1}
LABELS = {
    "V0": "V0 · T-SAC, N=100 training",
    "V1": "V1 · T-SAC, mixed-N training",
    "H0": "H0 · frozen G11 T0",
    "fixed": "B0 · fixed rule + feasible projection",
    "greedy": "B1 · greedy rule + feasible projection",
}
COLORS = {"V0": "#386cb0", "V1": "#e07b28", "H0": "#7550a2",
          "fixed": "#3d9670", "greedy": "#ad586a"}
METRICS = {
    "si": ("Demand satisfaction SI", 1.0, "mean of all target beams"),
    "sgm": ("Supply-demand match SGM", 1.0, "mean of all target beams"),
    "basic_fraction": ("Basic-service beam fraction", 1.0, "fraction of all target beams"),
    "effective_carried_bps": ("Effective carried estimate (Gbps)", 1e-9,
                              "sum of min(C,D), not observed traffic"),
    "covered_demand_fraction": ("Covered demand fraction", 1.0,
                                "frozen scene coverage metadata; identical for all methods"),
    "power_w": ("Total allocated power (W)", 1.0, "3000 W shared limit"),
    "slots_total": ("Allocated slots summed across 8 reuse groups", 1.0,
                    "sum of beam slot allocations; not unique spectrum width"),
    "zero_slot_fraction": ("Zero-slot target-beam fraction", 1.0,
                           "fraction of all target beams with no allocated slot"),
}
TRAINING_SEED = 51001
SMOOTH_WINDOW = 30
TRAINING_PANELS_PER_PAGE = 12


class PublicationError(ValueError):
    """A required audited field, identity, or SHA is missing or inconsistent."""


def require(condition, message):
    if not condition:
        raise PublicationError(message)


def sha256_file(path):
    path = Path(path)
    require(path.is_file(), f"Missing frozen file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise PublicationError(f"Invalid JSON: {path}") from error
    require(isinstance(value, dict), f"Expected JSON object: {path}")
    return value


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def number(value, label):
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise PublicationError(f"Missing numeric {label}") from error
    require(math.isfinite(result), f"Nonfinite {label}")
    return result


def integer(value, label):
    result = number(value, label)
    require(result.is_integer(), f"Noninteger {label}")
    return int(result)


def close(actual, expected, label, *, atol=1e-9):
    require(math.isclose(actual, expected, rel_tol=1e-8, abs_tol=atol),
            f"Audited table mismatch: {label}: {actual} != {expected}")


def _verified_output(audit_dir, receipt, name, inputs):
    expected = (receipt.get("outputs_sha256") or {}).get(name)
    require(isinstance(expected, str) and len(expected) == 64,
            f"Audit receipt lacks {name} SHA")
    path = audit_dir / name
    actual = sha256_file(path)
    require(actual == expected, f"Audited output SHA changed: {name}")
    inputs[str(path.resolve())] = actual
    return path


def _verified_input(receipt, path, inputs):
    path = Path(path).resolve()
    expected = (receipt.get("inputs_sha256") or {}).get(str(path))
    require(isinstance(expected, str) and len(expected) == 64,
            f"Path absent from audit inputs: {path}")
    actual = sha256_file(path)
    require(actual == expected, f"Frozen audit input SHA changed: {path}")
    inputs[str(path)] = actual
    return path


def load_audit(audit_dir):
    audit_dir = Path(audit_dir).resolve()
    receipt_path = audit_dir / "artifact_manifest.json"
    receipt_sha = sha256_file(receipt_path)
    receipt = read_json(receipt_path)
    require(receipt.get("experiment_id") == "S2-P17" and
            receipt.get("status") == "results_audited_timing_incomplete" and
            receipt.get("training_interactions") == 800000 and
            receipt.get("independent_initial_allocations") == 4840 and
            receipt.get("workbench_refinement_method_records") == 14520 and
            receipt.get("main_terminal_records") == 9680 and
            receipt.get("physical_terminal_recalculations") == 19360 and
            receipt.get("highest_si_candidate_recalculations") == 4839 and
            receipt.get("highest_si_candidate_missing_count") == 1 and
            receipt.get("exact_total_time_missing_actions") == 2 and
            receipt.get("independent_user_pause_interruption_count") == 1,
            "Full S2-P17 arithmetic/source/physical audit has not passed")
    inputs = {str(receipt_path): receipt_sha}
    names = ("per_scene_means.csv", "by_n_load.csv", "timing_raw.csv",
             "timing_summary.csv", "interruptions.csv")
    tables = {name: read_csv(_verified_output(audit_dir, receipt, name, inputs))
              for name in names}
    require(len(tables["per_scene_means.csv"]) == 4400 and
            len(tables["by_n_load.csv"]) == 220 and
            len(tables["timing_raw.csv"]) == 4840 and
            len(tables["timing_summary.csv"]) == 55 and
            len(tables["interruptions.csv"]) == 1,
            "Audited output row counts differ")
    controller_paths = [Path(path) for path in receipt.get("inputs_sha256", {})
                        if Path(path).name == "controller_manifest.json" and
                        Path(path).parent.name == "ops"]
    require(len(controller_paths) == 1, "Cannot uniquely locate audited controller manifest")
    controller_path = _verified_input(receipt, controller_paths[0], inputs)
    controller = read_json(controller_path)
    require(controller.get("experiment_id") == "S2-P17" and
            controller.get("scientific_commit") == receipt.get("scientific_commit") and
            (controller.get("target") or {}).get("ssh_port") == 20424,
            "Controller and audit receipt disagree")
    report_scripts = [Path(path) for path in receipt.get("inputs_sha256", {})
                      if Path(path).name == "report_results.py"]
    require(len(report_scripts) == 1, "Cannot uniquely locate audited report script")
    audited_script = _verified_input(receipt, report_scripts[0], inputs)
    return receipt, tables, controller, inputs, audited_script


def terminal_points(tables):
    scenes = defaultdict(list)
    for row in tables["per_scene_means.csv"]:
        role, stage = row["role"], row["stage"]
        require(role in ROLES and stage in STAGES, "Unexpected terminal role/stage")
        n = integer(row["beam_count"], "terminal N")
        load = number(row["load_multiplier"], "terminal load")
        require(n in N_VALUES and load in LOADS and
                integer(row["action_repeats"], "action repeats") == REPEATS[role],
                "Terminal scene design differs")
        scenes[(role, stage, n, load)].append(row)
    expected = {(role, stage, n, load) for role in ROLES for stage in STAGES
                for n in N_VALUES for load in LOADS}
    require(set(scenes) == expected, "Terminal N/load matrix is incomplete")
    aggregates = {}
    for key, rows in scenes.items():
        require(len(rows) == 20 and
                len({row["mother_scene_id"] for row in rows}) == 20,
                f"Expected 20 distinct mother scenes for {key}")
        aggregates[key] = rows
    audited = {}
    for row in tables["by_n_load.csv"]:
        key = (row["role"], row["stage"], integer(row["beam_count"], "summary N"),
               number(row["load_multiplier"], "summary load"))
        require(key in aggregates and key not in audited and
                integer(row["mother_count"], "mother count") == 20,
                f"Invalid audited terminal summary key: {key}")
        audited[key] = row
    require(set(audited) == expected, "Audited terminal summary is incomplete")
    plotted, omissions = [], []
    for field, (title, scale, meaning) in METRICS.items():
        if not all(field in row and row[field] not in (None, "")
                   for rows in aggregates.values() for row in rows):
            if field in ("si", "sgm", "basic_fraction", "power_w",
                         "slots_total", "zero_slot_fraction"):
                raise PublicationError(f"Required terminal metric missing: {field}")
            omissions.append({"metric": field, "reason": "not recorded in audited scene rows"})
            continue
        for key in sorted(expected):
            source_rows = aggregates[key]
            measured = statistics.mean(number(row[field], field) for row in source_rows)
            audited_value = number(audited[key].get(field), f"audited {field}")
            close(measured, audited_value, f"{key}/{field}", atol=1e-5)
            plotted.append({"role": key[0], "combination": LABELS[key[0]],
                            "stage": key[1], "beam_count": key[2],
                            "load_multiplier": key[3], "mother_count": 20,
                            "metric": field, "metric_label": title,
                            "value": measured * scale, "unit_scale": scale,
                            "meaning": meaning})
    return plotted, omissions


def timing_points(tables):
    grouped = defaultdict(list)
    missing = defaultdict(int)
    for row in tables["timing_raw.csv"]:
        role = row["role"]
        require(role in ROLES, "Unexpected timing role")
        n = integer(row["beam_count"], "timing N")
        load = number(row["load_multiplier"], "timing load")
        repeat = integer(row["repeat_index"], "timing repeat")
        require(n in N_VALUES and load in LOADS and 0 <= repeat < REPEATS[role],
                "Timing design differs")
        grouped[(role, n, load, row["scene_sha256"])].append(row)
    expected_panels = {(role, n, load) for role in ROLES for n in N_VALUES for load in LOADS}
    per_panel = defaultdict(list)
    for (role, n, load, scene), rows in grouped.items():
        require(len(rows) == REPEATS[role] and
                {integer(row["repeat_index"], "repeat") for row in rows} ==
                set(range(REPEATS[role])), f"Missing/duplicate timing action: {scene}")
        if any(row["total_pipeline_seconds"] in (None, "") for row in rows):
            missing[(role, n, load)] += 1
            continue
        per_panel[(role, n, load)].append(
            statistics.mean(number(row["total_pipeline_seconds"], "route seconds")
                            for row in rows))
    require(set(per_panel) == expected_panels, "Complete-scene time panel missing")
    summary = {(row["role"], integer(row["beam_count"], "timing summary N")): row
               for row in tables["timing_summary.csv"]}
    require(len(summary) == len(ROLES) * len(N_VALUES), "Timing summary N matrix differs")
    points = []
    for role in ROLES:
        for n in N_VALUES:
            combined = []
            for load in LOADS:
                values = per_panel[(role, n, load)]
                require(len(values) + missing[(role, n, load)] == 20,
                        f"Timing mother count differs for {role}/N={n}/load={load}")
                combined.extend(values)
                points.append({"role": role, "combination": LABELS[role],
                               "beam_count": n, "load_multiplier": load,
                               "complete_mother_count": len(values),
                               "missing_mother_count": missing[(role, n, load)],
                               "mean_seconds_per_action": statistics.mean(values),
                               "median_seconds_per_action": statistics.median(values),
                               "p95_seconds_per_action": float(np.percentile(values, 95))})
            audited = summary[(role, n)]
            require(integer(audited["total_pipeline_seconds_route_scene_complete_count"],
                            "audited complete scenes") == len(combined) and
                    integer(audited["total_pipeline_seconds_route_scene_missing_count"],
                            "audited missing scenes") == 40 - len(combined),
                    f"Timing completeness differs for {role}/N={n}")
            close(statistics.mean(combined),
                  number(audited["total_pipeline_seconds_route_scene_mean_mean"],
                         "audited route-scene time"),
                  f"{role}/N={n} complete-scene mean", atol=1e-5)
    missing_actions = [row for row in tables["timing_raw.csv"]
                       if row["total_pipeline_seconds"] in (None, "")]
    require(len(missing_actions) == 2 and
            {row.get("timing_loss_kind") for row in missing_actions} ==
            {"failed_B_fallback_to_A", "user_pause_interrupted_A"},
            "Two distinct timing gaps were not preserved")
    return points


def _input_path(receipt, path, inputs):
    return _verified_input(receipt, path, inputs)


def training_points(receipt, controller, inputs):
    episode_rows, validation_rows, panel_counts = [], [], []
    for role in ("V0", "V1"):
        entry = controller["jobs"][f"{role}_train"]
        directory = Path(entry["job_path"]).parent
        config_path = _input_path(receipt, directory / "config.json", inputs)
        config = read_json(config_path)
        require((config.get("training") or {}).get("seed") == TRAINING_SEED and
                entry.get("stage") == "completed" and
                entry.get("env_step") == 400000,
                f"Frozen {role} training seed/budget differs")
        episodes_path = _input_path(receipt, directory / "episodes.jsonl", inputs)
        counts_path = _input_path(receipt, directory / "count_metrics.jsonl", inputs)
        last_step = -1
        with episodes_path.open(encoding="utf-8") as stream:
            for line in stream:
                require(line.endswith("\n"), f"Incomplete episode line: {role}")
                row = json.loads(line)
                step = integer(row.get("env_step"), "training step")
                n = integer(row.get("beam_count"), "training N")
                value = number(row.get("return"), "training return")
                require(last_step < step <= 400000 and 1 <= n <= 200,
                        f"Out-of-order training episode: {role}")
                last_step = step
                episode_rows.append({"role": role, "combination": LABELS[role],
                                     "seed": TRAINING_SEED, "beam_count": n,
                                     "env_step": step, "return": value})
        require(last_step == 400000, f"Missing {role} fixed-budget endpoint")
        nodes = set()
        with counts_path.open(encoding="utf-8") as stream:
            for line in stream:
                require(line.endswith("\n"), f"Incomplete validation line: {role}")
                row = json.loads(line)
                step = integer(row.get("env_step"), "validation step")
                by_n = row.get("by_beam_count") or {}
                require(row.get("group") == "validation" and
                        step in set(range(0, 400001, 20000)) and step not in nodes and
                        {int(n) for n in by_n} == {50, 80, 100, 120, 160, 200},
                        f"Validation node/N differs: {role}/{step}")
                nodes.add(step)
                for n_text, item in by_n.items():
                    require(item.get("scene_actions") == 24,
                            f"Validation scene actions differ: {role}/{n_text}")
                    for metric, source in (("si", "final_si_mean"),
                                           ("sgm", "final_sgm_mean")):
                        validation_rows.append({"role": role,
                                                "combination": LABELS[role],
                                                "seed": TRAINING_SEED, "env_step": step,
                                                "beam_count": int(n_text),
                                                "scene_actions": 24,
                                                "metric": metric,
                                                "value": number((item.get("metrics") or {}).get(source),
                                                                f"validation {source}")})
        require(nodes == set(range(0, 400001, 20000)),
                f"Validation checkpoints incomplete: {role}")
        for n in sorted({row["beam_count"] for row in episode_rows if row["role"] == role}):
            panel_counts.append({"role": role, "combination": LABELS[role],
                                 "seed": TRAINING_SEED, "beam_count": n,
                                 "raw_episode_count": sum(row["role"] == role and
                                                          row["beam_count"] == n
                                                          for row in episode_rows),
                                 "smoothing_window_same_N_episodes": SMOOTH_WINDOW})
    require(len(validation_rows) == 2 * 21 * 6 * 2,
            "Validation point count differs")
    return episode_rows, validation_rows, panel_counts


def write_csv(path, rows):
    require(rows, f"Empty output table: {path}")
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def configure_plotting():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.dpi": 180})
    return plt


def plot_terminal(stage, points, omitted):
    plt = configure_plotting()
    figures = []
    for field, (title, _, meaning) in METRICS.items():
        if any(item["metric"] == field for item in omitted):
            continue
        fig, axes = plt.subplots(2, 2, figsize=(13.6, 8.4), sharex=True)
        fig.subplots_adjust(left=0.08, right=0.98, top=0.85, bottom=0.20,
                            wspace=0.17, hspace=0.34)
        for row_index, load in enumerate(LOADS):
            for col_index, stage_name in enumerate(STAGES):
                ax = axes[row_index, col_index]
                for role in ROLES:
                    selected = sorted((p for p in points if p["metric"] == field and
                                       p["load_multiplier"] == load and
                                       p["stage"] == stage_name and p["role"] == role),
                                      key=lambda p: p["beam_count"])
                    require([p["beam_count"] for p in selected] == list(N_VALUES),
                            f"Incomplete figure points: {field}/{role}/{stage_name}/{load}")
                    ax.plot(N_VALUES, [p["value"] for p in selected],
                            color=COLORS[role], marker="o", markersize=3,
                            linewidth=1.4, label=LABELS[role])
                ax.set_title(f"Load ×{load:g} · {'Stage 2 direct' if stage_name == 'direct' else 'Final SCA B'}")
                ax.set_ylabel(title)
                ax.grid(alpha=0.25)
                if field == "zero_slot_fraction":
                    ax.set_ylim(0, 0.05)
                    if all(p["value"] == 0 for p in points if p["metric"] == field and
                           p["load_multiplier"] == load and p["stage"] == stage_name):
                        ax.text(0.02, 0.93, "All observed scene means = 0",
                                transform=ax.transAxes, va="top", fontsize=8)
                if row_index == 1:
                    ax.set_xlabel("Number of target beams N")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.015),
                   ncol=2, fontsize=8)
        fig.suptitle(f"{title} · 20 mother scenes per N and load\n{meaning}", y=0.98)
        name = f"terminal_{field}_by_n_load.png"
        fig.savefig(stage / name)
        plt.close(fig)
        figures.append(name)
    return figures


def plot_timing(stage, points):
    plt = configure_plotting()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), sharey=True)
    fig.subplots_adjust(left=0.09, right=0.98, top=0.77, bottom=0.29,
                        wspace=0.13)
    for ax, load in zip(axes, LOADS):
        for role in ROLES:
            selected = sorted((p for p in points if p["role"] == role and
                               p["load_multiplier"] == load), key=lambda p: p["beam_count"])
            require([p["beam_count"] for p in selected] == list(N_VALUES),
                    f"Incomplete timing curve: {role}/{load}")
            ax.plot(N_VALUES, [p["mean_seconds_per_action"] for p in selected],
                    color=COLORS[role], marker="o", markersize=3,
                    linewidth=1.4, label=LABELS[role])
        lost = [(p["role"], p["beam_count"], p["missing_mother_count"])
                for p in points if p["load_multiplier"] == load and
                p["missing_mother_count"]]
        note = "; ".join(f"{role} N={n}: {count} missing" for role, n, count in lost)
        ax.set_title(f"Load ×{load:g}" + (f"\n{note}" if note else ""), fontsize=9)
        ax.set_xlabel("Number of target beams N")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Seconds per action; complete repeated-action scenes only")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.02),
               ncol=2, fontsize=8)
    fig.suptitle("Measured initialization + A + B + checks · each scene averaged over all its actions",
                 y=0.98)
    name = "complete_scene_time_by_n_load.png"
    fig.savefig(stage / name)
    plt.close(fig)
    return [name]


def plot_training(stage, episodes):
    plt = configure_plotting()
    from matplotlib.lines import Line2D

    def draw_page(own, role, counts, name, page_title):
        cols = min(3, len(counts))
        rows = math.ceil(len(counts) / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(max(7.0, 4.6 * cols),
                                                 max(4.5, 3.15 * rows)),
                                 sharex=True)
        fig.subplots_adjust(left=0.08 if cols > 1 else 0.12, right=0.98,
                            top=0.86 if rows > 1 else 0.78,
                            bottom=0.13 if rows > 1 else 0.22,
                            wspace=0.28, hspace=0.52)
        flat = np.atleast_1d(axes).ravel()
        has_smooth = False
        for ax, n in zip(flat, counts):
            panel = [row for row in own if row["beam_count"] == n]
            x = np.asarray([row["env_step"] / 1000 for row in panel], dtype=float)
            y = np.asarray([row["return"] for row in panel], dtype=float)
            ax.plot(x, y, color=COLORS[role], alpha=0.55 if len(y) < 30 else 0.20,
                    linewidth=0.8, marker="o" if len(y) < 30 else None,
                    markersize=2)
            if len(y) >= SMOOTH_WINDOW:
                smooth = np.convolve(y, np.ones(SMOOTH_WINDOW) / SMOOTH_WINDOW,
                                     mode="valid")
                ax.plot(x[SMOOTH_WINDOW - 1:], smooth, color=COLORS[role],
                        linewidth=1.6)
                has_smooth = True
            ax.set_title(f"N={n} · {len(panel)} raw episodes" +
                         (" · raw only (<30)" if len(y) < SMOOTH_WINDOW else ""),
                         fontsize=9)
            ax.set_xlim(0, 400)
            ax.set_xlabel("Training interactions (thousands)", fontsize=8)
            ax.set_ylabel("Episode return", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(alpha=0.25)
        for ax in flat[len(counts):]:
            ax.axis("off")
        handles = [Line2D([0], [0], color=COLORS[role], alpha=0.55,
                          linewidth=1, marker="o", markersize=3,
                          label=f"{LABELS[role]} · raw episodes")]
        if has_smooth:
            handles.append(Line2D([0], [0], color=COLORS[role], linewidth=2,
                                  label=f"same-N {SMOOTH_WINDOW}-episode moving mean"))
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.01),
                   ncol=len(handles), fontsize=8)
        fig.suptitle(f"{LABELS[role]} · seed {TRAINING_SEED} · {page_title}\n"
                     "Exact-N raw episodes; smoothing only when that N has at least 30 episodes",
                     y=0.98, fontsize=11)
        fig.savefig(stage / name)
        plt.close(fig)
        return name

    figures = []
    v0 = [row for row in episodes if row["role"] == "V0"]
    require({row["beam_count"] for row in v0} == {100}, "V0 training N changed")
    figures.append(draw_page(v0, "V0", [100],
                             "training_return_by_actual_n_v0.png", "fixed N=100"))
    v1 = [row for row in episodes if row["role"] == "V1"]
    counts = sorted({row["beam_count"] for row in v1})
    page_total = math.ceil(len(counts) / TRAINING_PANELS_PER_PAGE)
    for page_index in range(page_total):
        selected = counts[page_index * TRAINING_PANELS_PER_PAGE:
                          (page_index + 1) * TRAINING_PANELS_PER_PAGE]
        name = f"training_return_by_actual_n_v1_page{page_index + 1:02d}_of{page_total:02d}.png"
        figures.append(draw_page(v1, "V1", selected, name,
                                 f"all actual N · page {page_index + 1}/{page_total} "
                                 f"(N={selected[0]}–{selected[-1]})"))
    representative = (50, 100, 200)
    require(set(representative).issubset(counts), "Prespecified boundary N missing")
    figures.append(draw_page(v1, "V1", representative,
                             "training_return_by_actual_n_v1_representative.png",
                             "prespecified boundary examples N=50/100/200; all N in pages"))
    return figures


def training_bucket_points(episodes):
    """Show exposure counts by the four frozen buckets, never pooled returns."""
    selected = [row for row in episodes if row["role"] == "V1"]
    def bucket(n):
        if 50 <= n <= 99 and n not in (60, 90):
            return "N=50–99 excluding 60/90"
        if n == 100:
            return "N=100"
        if 101 <= n <= 149 and n != 140:
            return "N=101–149 excluding 140"
        if 150 <= n <= 200 and n != 180:
            return "N=150–200 excluding 180"
        raise PublicationError(f"V1 training N outside frozen buckets: {n}")
    labels = ("N=50–99 excluding 60/90", "N=100",
              "N=101–149 excluding 140", "N=150–200 excluding 180")
    counts = Counter(bucket(row["beam_count"]) for row in selected)
    distinct = defaultdict(set)
    for row in selected:
        distinct[bucket(row["beam_count"])].add(row["beam_count"])
    require(all(counts[label] > 0 for label in labels), "V1 training bucket missing")
    return [{"bucket": label, "seed": TRAINING_SEED,
             "raw_episode_count": counts[label],
             "distinct_actual_N_count": len(distinct[label]),
             "quantity": "saved training episodes; not interactions or mean return"}
            for label in labels]


def plot_training_bucket_counts(stage, rows):
    plt = configure_plotting()
    fig, ax = plt.subplots(figsize=(10, 4.8))
    fig.subplots_adjust(left=0.12, right=0.98, top=0.78, bottom=0.22)
    labels = [row["bucket"] for row in rows]
    values = [row["raw_episode_count"] for row in rows]
    bars = ax.bar(range(len(rows)), values, color=["#5c90c3", COLORS["V1"],
                                                    "#8b74b5", "#54a58a"])
    ax.bar_label(bars, labels=[str(value) for value in values], padding=3)
    ax.set_xticks(range(len(rows)), labels, rotation=12, ha="right", fontsize=8)
    ax.set_ylabel("Saved training episode count")
    ax.set_ylim(0, max(values) * 1.13)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    fig.suptitle(f"{LABELS['V1']} · seed {TRAINING_SEED}\n"
                 "Four frozen sampling buckets; episode counts only, no cross-N return average",
                 y=0.98)
    name = "training_v1_four_bucket_episode_counts.png"
    fig.savefig(stage / name)
    plt.close(fig)
    return [name]


def plot_validation(stage, validation):
    plt = configure_plotting()
    figures = []
    for metric, name_text in (("si", "Demand satisfaction SI"),
                              ("sgm", "Supply-demand match SGM")):
        fig, axes = plt.subplots(2, 3, figsize=(14.5, 7.8), sharex=True,
                                 sharey=True)
        fig.subplots_adjust(left=0.07, right=0.99, top=0.87, bottom=0.17,
                            wspace=0.17, hspace=0.34)
        for ax, n in zip(axes.ravel(), (50, 80, 100, 120, 160, 200)):
            for role in ("V0", "V1"):
                selected = sorted((row for row in validation
                                   if row["role"] == role and row["metric"] == metric and
                                   row["beam_count"] == n), key=lambda row: row["env_step"])
                require([row["env_step"] for row in selected] ==
                        list(range(0, 400001, 20000)),
                        f"Missing validation series: {role}/{metric}/N={n}")
                ax.plot([row["env_step"] for row in selected],
                        [row["value"] for row in selected], color=COLORS[role],
                        marker=".", linewidth=1.2, label=LABELS[role])
            ax.set_title(f"N={n}; 24 allocations per checkpoint")
            ax.set_xlabel("Training interactions")
            ax.set_ylabel(name_text)
            ax.grid(alpha=0.25)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.02),
                   ncol=2, fontsize=8)
        fig.suptitle(f"Validation {name_text} · seed {TRAINING_SEED} · 21 fixed checkpoints",
                     y=0.98)
        name = f"validation_{metric}_by_n.png"
        fig.savefig(stage / name)
        plt.close(fig)
        figures.append(name)
    return figures


def generate(audit_dir, output_dir):
    audit_dir, output_dir = Path(audit_dir).resolve(), Path(output_dir).resolve()
    require(not output_dir.exists(), "Output directory exists; use a new publication directory")
    receipt, tables, controller, inputs, audited_script = load_audit(audit_dir)
    terminal, omissions = terminal_points(tables)
    timing = timing_points(tables)
    episodes, validation, panels = training_points(receipt, controller, inputs)
    buckets = training_bucket_points(episodes)
    script = Path(__file__).resolve()
    inputs[str(script)] = sha256_file(script)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="s2p17_publication_pending_",
                                     dir=output_dir.parent) as temporary:
        stage = Path(temporary)
        write_csv(stage / "terminal_figure_points.csv", terminal)
        write_csv(stage / "time_figure_points.csv", timing)
        write_csv(stage / "training_panel_counts.csv", panels)
        write_csv(stage / "training_v1_bucket_episode_counts.csv", buckets)
        write_csv(stage / "validation_figure_points.csv", validation)
        figures = (plot_terminal(stage, terminal, omissions) + plot_timing(stage, timing) +
                   plot_training(stage, episodes) +
                   plot_training_bucket_counts(stage, buckets) +
                   plot_validation(stage, validation))
        (stage / "README.md").write_text(
            "# S2-P17 publication figures\n\n"
            "All SI/SGM and resource points come from the SHA-audited report tables. "
            "Each terminal point represents 20 distinct mother scenes at one N and load; "
            "model actions were averaged within each derived scene in the audit. "
            "Both original and final terminal results retain the V0 failed-B fallback.\n\n"
            "Time points use only scenes with every repeated action timed. The V0 failed-B "
            "and V1 user-pause actions remain in performance results and are excluded with "
            "their whole scenes from main time curves. The separate load panels do not mix "
            "demand multipliers. V1 training used frozen scene files with four sampling "
            "buckets. Each mother contributes its own seven N values at both loads; the "
            "union of actual N spans many values and no scene is generated online. "
            "All actual-N raw training returns are retained in numbered pages with at most "
            f"{TRAINING_PANELS_PER_PAGE} panels each. A separate N=50/100/200 page contains "
            "prespecified boundary examples. The four-bucket chart counts saved episodes "
            "without averaging returns across N. "
            f"A {SMOOTH_WINDOW}-episode moving mean is drawn only within an N with enough "
            f"episodes; seed {TRAINING_SEED}. Short-N panels explicitly show raw only. "
            "Validation uses 24 allocations per N at each of 21 checkpoints.\n\n"
            "The slots_total plot sums all allocated beam slots across eight frequency-reuse "
            "groups. It is not a count of unique positions on the 100-slot frequency axis "
            "and must not be divided by 100 as a utilization rate. zero_slot_fraction is "
            "the fraction of target beams allocated no slots. Input and image SHA-256 hashes "
            "are in publication_manifest.json.\n",
            encoding="utf-8")
        outputs = {path.name: sha256_file(path) for path in sorted(stage.iterdir())
                   if path.is_file()}
        manifest = {"experiment_id": "S2-P17", "kind": "publication_figures",
                    "generated_utc": datetime.now(timezone.utc).isoformat(),
                    "source_audit_dir": str(audit_dir),
                    "source_audit_receipt_sha256": inputs[str(audit_dir / "artifact_manifest.json")],
                    "audited_report_script_sha256": inputs[str(audited_script)],
                    "publication_script_sha256": inputs[str(script)],
                    "scientific_commit": receipt["scientific_commit"],
                    "repair_commit": (receipt.get("boundary_repair") or {}).get("repair_commit"),
                    "training_seed": TRAINING_SEED,
                    "slots_total_semantics": "sum of allocated beam slots across 8 reuse groups; not unique 100-slot occupancy",
                    "training_smoothing_window_same_N_episodes": SMOOTH_WINDOW,
                    "training_panels_per_page": TRAINING_PANELS_PER_PAGE,
                    "training_actual_N_panel_count": len([row for row in panels if row["role"] == "V1"]),
                    "terminal_mother_scenes_per_N_load": 20,
                    "terminal_scene_point_count": len(terminal),
                    "timing_panel_point_count": len(timing),
                    "raw_episode_count": len(episodes),
                    "validation_point_count": len(validation),
                    "omitted_fields": omissions,
                    "figure_files": figures,
                    "inputs_sha256": dict(sorted(inputs.items())),
                    "outputs_sha256": outputs}
        (stage / "publication_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8")
        stage.rename(output_dir)
    return {"directory": str(output_dir),
            "receipt": str(output_dir / "publication_manifest.json"),
            "figure_count": len(figures), "omitted_fields": omissions}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(generate(args.audit_dir, args.output_dir), ensure_ascii=False))


if __name__ == "__main__":
    main()
