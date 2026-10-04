"""Read-only S1-03 65,536-step mechanism-gate diagnosis.

This analyzes existing Jan12 development validation and training records only.
It never invokes a policy, environment, workbench task, or SSH connection.
Run with the frozen remote DQN interpreter after copying this script to the
S1-03 ops/analysis_override directory. A failed run keeps an incomplete marker.
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
import numpy as np


COMMIT = "e1745eccc1f47aa0b2f4ede678ce7789e030c1ea"
RUN_ID = "train_ppo_20261005_051054_d2e8e76a"
STEPS = (16384, 32768, 49152, 65536)
LATEST_START = 49152
BIN = 4096
METRICS = ("coverage_users", "coverage_demand", "coverage_100", "subject_cv",
           "load_cv", "subject_in_band_beams", "quality", "j", "target_met")
TIMES = ("initialization_seconds", "policy_seconds", "reconstruction_seconds",
         "final_check_seconds", "total_seconds", "candidate_screen_seconds",
         "observation_seconds")
GEOMETRY_REASONS = {"no_legal_center_for_group", "covered_aircraft_lost",
                    "empty_or_nonpositive_beam", "edge_certificate_changed",
                    "demand_not_conserved", "action_geometry_budget_exhausted",
                    "episode_geometry_budget_exhausted", "no_legal_covered_assignment",
                    "split_group_empty", "uncovered_user_without_legal_beam",
                    "association outside actual satellite footprint"}


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def stream(path):
    with Path(path).open("rb") as source:
        for number, line in enumerate(source, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except ValueError as error:
                    raise ValueError(f"{path}:{number} invalid JSON") from error


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def numerical_summary(values, denominator):
    selected = [float(value) for value in values if finite(value)]
    return {"all_scene_mean": statistics.fmean(selected) if len(selected) == denominator and denominator else None,
            "defined_mean": statistics.fmean(selected) if selected else None,
            "median": statistics.median(selected) if selected else None,
            "p95": float(np.percentile(selected, 95)) if selected else None,
            "defined_count": len(selected), "missing_count": denominator - len(selected)}


def category(reason, accepted, operation):
    if operation == "stop":
        return "stop"
    if accepted:
        return "accepted"
    if reason == "no_change":
        return "no_change"
    if reason in GEOMETRY_REASONS or str(reason).startswith("no_legal_"):
        return "geometry_or_budget_infeasible"
    return "other_rejection"


def action_diagnosis(rows, maximum=65536):
    counts = {"all": Counter(), "recent": Counter()}
    groups = defaultdict(Counter)
    curve = defaultdict(Counter)
    repeats = Counter()
    seen_by_row = {}
    prior_by_row = {}
    strata = {"all": defaultdict(Counter), "recent": defaultdict(Counter)}
    strata_curve = defaultdict(lambda: defaultdict(Counter))
    seen_steps = set()
    count = 0
    for row in rows:
        step = row.get("env_step")
        if type(step) is not int or not 1 <= step <= maximum or step in seen_steps:
            raise ValueError("Action log has duplicate, missing-range or out-of-budget env_step")
        seen_steps.add(step)
        count += 1
        action = row.get("requested_action_fields") or {}
        operation = action.get("operation")
        if operation not in ("stop", "merge_split", "repartition2", "repartition3", "assign"):
            raise ValueError("Action log contains unknown operation")
        accepted = bool(row.get("accepted"))
        reason = str(row.get("reason"))
        label = category(reason, accepted, operation)
        owner = row.get("row")
        serial = row.get("episode_serial")
        scene = row.get("scene_index")
        if type(owner) is not int or type(serial) is not int or type(scene) is not int:
            raise ValueError("Cannot establish prior-accepted stratum without episode identity")
        episode_key = (serial, scene)
        old_episode, prior_accepted = prior_by_row.get(owner, (None, False))
        if old_episode != episode_key:
            prior_accepted = False
        stratum = "prior_accepted" if prior_accepted else "no_prior_accepted"
        window = "recent" if step > LATEST_START else "all"
        for scope in ("all",) if window == "all" else ("all", "recent"):
            counts[scope]["actions"] += 1
            counts[scope][f"category:{label}"] += 1
            counts[scope][f"operation:{operation}"] += 1
            if operation != "stop":
                counts[scope]["non_stop"] += 1
                counts[scope]["accepted_non_stop"] += accepted
            groups[(scope, operation, reason)]["attempted"] += 1
            groups[(scope, operation, reason)]["accepted"] += accepted
            strata[scope][stratum]["actions"] += 1
            strata[scope][stratum][f"reason:{reason}"] += 1
            strata[scope][stratum][f"category:{label}"] += 1
            if operation != "stop":
                strata[scope][stratum]["non_stop"] += 1
                strata[scope][stratum]["accepted"] += accepted
        bin_end = ((step - 1) // BIN + 1) * BIN
        curve[bin_end]["actions"] += 1
        curve[bin_end][f"category:{label}"] += 1
        if operation != "stop":
            curve[bin_end]["non_stop"] += 1
            curve[bin_end]["accepted"] += accepted
            strata_curve[bin_end][stratum]["non_stop"] += 1
            strata_curve[bin_end][stratum]["accepted"] += accepted
        prior_by_row[owner] = (episode_key, prior_accepted or (operation != "stop" and accepted))
        # The menu SHA plus complete identity is a reproducible *candidate*
        # repeat within this episode, not proof that the underlying state is equal.
        identity = row.get("action_identity")
        menu = row.get("candidate_identity_sha256")
        if (type(owner) is int and type(serial) is int and type(scene) is int and
                isinstance(identity, list) and isinstance(menu, str) and len(menu) == 64):
            old_key, known = seen_by_row.get(owner, (None, set()))
            if old_key != episode_key:
                known = set()
            candidate = (menu, tuple(identity))
            if candidate in known:
                repeats["same_menu_same_action_within_episode"] += 1
            known.add(candidate)
            seen_by_row[owner] = (episode_key, known)
            repeats["eligible_action_records"] += 1
        else:
            repeats["missing_identity_records"] += 1
    if count != maximum or seen_steps != set(range(1, maximum + 1)):
        raise ValueError(f"Expected exactly {maximum} continuous action interactions; got {count}")
    for scope in counts:
        total = counts[scope]["non_stop"]
        counts[scope]["acceptance_rate"] = counts[scope]["accepted_non_stop"] / total if total else None
    stratified = {}
    for scope, by_stratum in strata.items():
        stratified[scope] = {}
        for label, values in by_stratum.items():
            total = values["non_stop"]
            stratified[scope][label] = {**dict(values),
                "acceptance_rate": values["accepted"] / total if total else None}
    grouped = [{"window": scope, "operation": operation, "reason": reason,
                "attempted": values["attempted"], "accepted": values["accepted"],
                "rejected": values["attempted"] - values["accepted"],
                "category": category(reason, bool(values["accepted"]), operation)}
               for (scope, operation, reason), values in sorted(groups.items())]
    bins = [{"env_step_end": step, "actions": values["actions"],
             "non_stop": values["non_stop"], "accepted": values["accepted"],
             "acceptance_rate": values["accepted"] / values["non_stop"] if values["non_stop"] else None,
             "no_change": values["category:no_change"],
             "geometry_or_budget_infeasible": values["category:geometry_or_budget_infeasible"],
             "other_rejection": values["category:other_rejection"],
             "stop": values["category:stop"]}
            for step, values in sorted(curve.items())]
    stratified_bins = []
    for step, by_stratum in sorted(strata_curve.items()):
        for label in ("no_prior_accepted", "prior_accepted"):
            values = by_stratum[label]
            stratified_bins.append({"env_step_end": step, "stratum": label,
                "non_stop": values["non_stop"], "accepted": values["accepted"],
                "acceptance_rate": values["accepted"] / values["non_stop"]
                                   if values["non_stop"] else None})
    repeats["same_menu_same_action_repeat_rate"] = (
        repeats["same_menu_same_action_within_episode"] /
        repeats["eligible_action_records"] if repeats["eligible_action_records"] else None)
    return {"windows": {key: dict(value) for key, value in counts.items()},
            "operation_reasons": grouped, "bins": bins,
            "prior_acceptance_strata": stratified,
            "prior_acceptance_bins": stratified_bins,
            "repeats": dict(repeats)}


def episode_diagnosis(rows, maximum=65536):
    bins = defaultdict(lambda: {"episodes": 0, "zero_step": 0, "stop": 0,
                                "quality": [], "quality_gain": [], "subject_cv": [],
                                "load_cv": [], "coverage_users": [], "coverage_demand": [],
                                "return": []})
    total = 0
    raw = []
    for row in rows:
        step = row.get("terminal_env_step")
        if type(step) is not int or not 1 <= step <= maximum:
            raise ValueError("Episode terminal step outside stopped training budget")
        total += 1
        metric = row.get("metrics") or {}
        initial = row.get("initial_metrics") or {}
        bucket = bins[((step - 1) // BIN + 1) * BIN]
        bucket["episodes"] += 1
        bucket["zero_step"] += row.get("macro_steps") == 0
        bucket["stop"] += row.get("stop_reason") == "stop"
        for key in ("quality", "subject_cv", "load_cv", "coverage_users", "coverage_demand"):
            bucket[key].append(metric.get(key))
        bucket["quality_gain"].append(metric["quality"] - initial["quality"]
                                      if finite(metric.get("quality")) and finite(initial.get("quality"))
                                      else None)
        bucket["return"].append(row.get("episode_return"))
        raw.append({"terminal_env_step": step, "episode": row.get("episode"),
                    "scene_id": row.get("scene_id"),
                    "macro_steps": row.get("macro_steps"),
                    "stop_reason": row.get("stop_reason"),
                    "initial_quality": initial.get("quality"),
                    "terminal_quality": metric.get("quality"),
                    "quality_gain": (metric["quality"] - initial["quality"]
                                     if finite(metric.get("quality")) and
                                     finite(initial.get("quality")) else None),
                    "episode_return": row.get("episode_return"),
                    "coverage_users": metric.get("coverage_users"),
                    "coverage_demand": metric.get("coverage_demand"),
                    "subject_cv": metric.get("subject_cv"),
                    "load_cv": metric.get("load_cv")})
    output = []
    for step, bucket in sorted(bins.items()):
        record = {"env_step_end": step, "episodes": bucket["episodes"],
                  "zero_step": bucket["zero_step"], "stop": bucket["stop"]}
        for key in ("quality", "quality_gain", "subject_cv", "load_cv",
                    "coverage_users", "coverage_demand", "return"):
            result = numerical_summary(bucket[key], bucket["episodes"])
            record[key + "_mean"] = result["all_scene_mean"]
            record[key + "_defined_count"] = result["defined_count"]
        output.append(record)
    return {"completed_episodes": total, "bins": output, "raw": raw}


def job_identity(job_dir, dev_dirs, audit_path, output_dir):
    job_dir = Path(job_dir).resolve()
    archive = job_dir.parents[3]
    if (job_dir.name != RUN_ID or job_dir.parent.name != "jobs" or
            job_dir.parent.parent.name != "workbench" or
            archive.name != "20418" or archive.parent.name != "s1_03"):
        raise ValueError("Analysis job is outside authorized S1-03/20418 job root")
    if output_dir.resolve().parent != (archive / "ops" / "analysis_override").resolve():
        raise ValueError("Analysis output must be a new ops/analysis_override child")
    if output_dir.exists():
        raise FileExistsError("Analysis output already exists; preserve previous receipt")
    if not audit_path.resolve().is_relative_to((archive / "ops" / "audits").resolve()):
        raise ValueError("Mechanism audit must belong to the same S1-03 instance")
    job, config, status = (read(job_dir / name) for name in
                           ("job.json", "config.json", "status.json"))
    if (job.get("run_id") != RUN_ID or job.get("git_commit") != COMMIT or
            job.get("kind") != "train" or config.get("schema_version") != 4 or
            status.get("stage") != "stopped" or
            status.get("stop_reason") != "mechanism_gate_pending" or
            status.get("env_step") != 65536 or
            status.get("checkpoint_sha256") != sha(job_dir / "checkpoint.pt")):
        raise ValueError("Analysis requires the untouched 65,536-step S1-03 checkpoint")
    audit = read(audit_path)
    checks = audit.get("gate_checks", {})
    if (audit.get("mode") != "mechanism" or audit.get("run_id") != RUN_ID or
            audit.get("env_step") != 65536 or
            audit.get("checkpoint_sha256") != status["checkpoint_sha256"] or
            audit.get("passed") is not False or
            checks.get("recent_action_validity") is not False or
            not all(value is True for key, value in checks.items()
                    if key != "recent_action_validity")):
        raise ValueError("Analysis input is not the original failed acceptance gate")
    dev = {}
    dev_inputs = {}
    frozen = config["datasets"]["validation"]["sha256"]
    starts = config["reconstruction_starts"]["validation"]["sha256"]
    for route in ("R0", "R1", "R2", "R3"):
        directory = Path(dev_dirs[route]).resolve()
        if directory.parent != job_dir.parent:
            raise ValueError("Development comparator outside same workbench")
        j, c, s = (read(directory / name) for name in
                   ("job.json", "config.json", "status.json"))
        if (j.get("git_commit") != COMMIT or j.get("kind") != "evaluate" or
                c.get("coverage", {}).get("reconstruction", {}).get("route") != route or
                c.get("datasets", {}).get("validation", {}).get("sha256") != frozen or
                c.get("reconstruction_starts", {}).get("validation", {}).get("sha256") != starts or
                s.get("stage") != "completed"):
            raise ValueError(f"Development {route} is not the same completed Jan12 comparison")
        result = read(directory / "results.json")
        raw = Path(result.get("rows_path", "")).resolve()
        if (raw != directory / "results.partial.jsonl" or
                result.get("rows_sha256") != sha(raw) or
                result.get("scene_count") != 20):
            raise ValueError(f"Development {route} raw scenes differ from frozen result")
        rows = list(stream(raw))
        if len(rows) != 20 or len({row["scene_sha256"] for row in rows}) != 20:
            raise ValueError("Development scene denominator or uniqueness differs")
        dev[route] = {row["scene_sha256"]: row for row in rows}
        dev_inputs[route] = {"result_path": str(directory / "results.json"),
                             "result_sha256": sha(directory / "results.json"),
                             "raw_path": str(raw), "raw_sha256": result["rows_sha256"]}
    return {"job": job, "config": config, "status": status, "audit": audit,
            "dev": dev, "dev_inputs": dev_inputs,
            "job_dir": job_dir, "archive": archive,
            "audit_path": audit_path}


def validation_diagnosis(job_dir, config, dev):
    paths = []
    scenes = []
    summaries = []
    reference = set(dev["R0"])
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True,
        ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
    if set(dev["R3"]) != reference:
        raise ValueError("R0/R3 scene sets differ")
    for step in STEPS:
        path = job_dir / "validation_nodes" / f"{step}.json"
        node = read(path)
        if (node.get("scheduled_step") != step or node.get("actual_env_step") != step or
                node.get("scene_count") != 20 or len(node.get("rows", [])) != 20 or
                node.get("config_hash") != config_hash):
            raise ValueError("Frozen validation node step/count differs")
        identifiers = [row["scene_sha256"] for row in node["rows"]]
        if len(set(identifiers)) != 20 or set(identifiers) != reference:
            raise ValueError("Validation and development do not share the same 20 Jan12 scenes")
        paths.append({"path": str(path), "sha256": sha(path)})
        for row in node["rows"]:
            scene_id = row["scene_sha256"]
            metric = row.get("metrics") or {}
            init = row.get("initial_metrics") or {}
            record = {"step": step, "scene_id": row["scene_id"],
                      "scene_sha256": scene_id, "dataset_role": "Jan12_development_validation",
                      "initialization_valid": row.get("initialization_valid"),
                      "stop_reason": row.get("stop_reason"),
                      "macro_steps": row.get("macro_steps"),
                      "best_quality_diagnostic": row.get("best_quality_diagnostic"),
                      "best_minus_raw_quality_diagnostic": (
                          row["best_quality_diagnostic"] - metric["quality"]
                          if finite(row.get("best_quality_diagnostic")) and
                          finite(metric.get("quality")) else None)}
            for key in METRICS:
                record[key] = metric.get(key)
                record["initial_" + key] = init.get(key)
                record["R0_" + key] = dev["R0"][scene_id].get("metrics", {}).get(key)
                record["R3_" + key] = dev["R3"][scene_id].get("metrics", {}).get(key)
            for key in TIMES:
                record[key] = row.get(key)
                record["R0_" + key] = dev["R0"][scene_id].get(key)
                record["R3_" + key] = dev["R3"][scene_id].get(key)
            record["delta_quality_vs_R0"] = (metric["quality"] - dev["R0"][scene_id]["metrics"]["quality"]
                if finite(metric.get("quality")) and finite(dev["R0"][scene_id]["metrics"].get("quality")) else None)
            record["delta_quality_vs_R3"] = (metric["quality"] - dev["R3"][scene_id]["metrics"]["quality"]
                if finite(metric.get("quality")) and finite(dev["R3"][scene_id]["metrics"].get("quality")) else None)
            for key in ("coverage_users", "coverage_demand", "subject_cv", "load_cv",
                        "subject_in_band_beams"):
                for route in ("R0", "R3"):
                    comparison = dev[route][scene_id]["metrics"].get(key)
                    record[f"delta_{key}_vs_{route}"] = (
                        metric[key] - comparison if finite(metric.get(key)) and
                        finite(comparison) else None)
            scenes.append(record)
        current = [row for row in scenes if row["step"] == step]
        summary = {"step": step, "scenes": 20,
                   "strict_coverage_scenes": sum(row["coverage_100"] is True for row in current),
                   "target_scenes": sum(row["target_met"] is True for row in current)}
        paired = ("delta_quality_vs_R0", "delta_quality_vs_R3") + tuple(
            f"delta_{key}_vs_{route}" for key in
            ("coverage_users", "coverage_demand", "subject_cv", "load_cv",
             "subject_in_band_beams") for route in ("R0", "R3"))
        for key in METRICS + TIMES + paired + (
                "best_quality_diagnostic", "best_minus_raw_quality_diagnostic"):
            summary[key] = numerical_summary([row.get(key) for row in current], 20)
        summaries.append(summary)
    return scenes, summaries, paths


def csv_write(path, rows):
    rows = list(rows)
    if not rows:
        return
    with Path(path).open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_curves(output, actions, episodes, validation, font_path=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    candidates = ("Noto Sans CJK SC", "Noto Sans CJK", "Source Han Sans SC",
                  "WenQuanYi Micro Hei", "SimHei", "Microsoft YaHei")
    known_paths = ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                   "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc")
    font = None
    if font_path is not None:
        font_path = Path(font_path)
        if not font_path.is_file() or font_path.suffix.lower() not in (".ttf", ".ttc", ".otf"):
            raise RuntimeError("指定的中文字体文件不存在")
        with font_path.open("rb") as source:
            if not source.read(4):
                raise RuntimeError("指定的中文字体为空或不可读")
        font_manager.fontManager.addfont(str(font_path))
        font = font_manager.FontProperties(fname=str(font_path))
    if font is None:
        font = next((font for font in font_manager.fontManager.ttflist
                     if font.name in candidates), None)
    if font is None:
        for path in known_paths:
            if Path(path).is_file():
                font_manager.fontManager.addfont(path)
                font = font_manager.FontProperties(fname=path)
                break
    if font is None:
        raise RuntimeError("未找到中文字体；拒绝生成无法阅读的中文曲线")
    font_name = font.get_name() if hasattr(font, "get_name") else font.name
    if not any(token in font_name.lower() for token in
               ("cjk", "source han", "wenquanyi", "simhei", "yahei")):
        raise RuntimeError("指定字体未识别为可用中文字体；拒绝生成曲线")
    plt.rcParams["font.family"] = font_name
    plt.rcParams["axes.unicode_minus"] = False
    def chart(filename, title, x, series, ylabel):
        fig, axis = plt.subplots(figsize=(9, 4.5))
        for label, values in series:
            axis.plot(x, values, marker="o", linewidth=1.4, label=label)
        axis.set(title=title, xlabel="训练宏动作交互步数", ylabel=ylabel)
        axis.grid(alpha=.25)
        axis.legend()
        fig.tight_layout()
        fig.savefig(output / filename, dpi=160)
        plt.close(fig)
    bins = actions["bins"]
    x = [row["env_step_end"] for row in bins]
    chart("01_动作接受率与拒绝曲线.png", "训练动作接受率及拒绝类型（原始4096步分箱）", x,
          [("非STOP接受率", [row["acceptance_rate"] for row in bins]),
           ("no_change占比", [row["no_change"] / row["actions"] for row in bins]),
           ("几何/额度不可行占比", [row["geometry_or_budget_infeasible"] / row["actions"]
                              for row in bins])], "比例")
    ebins = episodes["bins"]
    fig, axis = plt.subplots(figsize=(9, 4.5))
    raw = episodes["raw"]
    axis.scatter([row["terminal_env_step"] for row in raw],
                 [row["terminal_quality"] for row in raw], s=5, alpha=.17,
                 label="P1 共同初始化＋区域重构PPO：逐回合原终态Q")
    axis.scatter([row["terminal_env_step"] for row in raw],
                 [row["episode_return"] for row in raw], s=5, alpha=.13,
                 label="P1：逐回合训练回报")
    axis.plot([row["env_step_end"] for row in ebins],
              [row["quality_mean"] for row in ebins], marker="o", linewidth=1.5,
              label="P1：4096步箱原终态Q均值")
    axis.plot([row["env_step_end"] for row in ebins],
              [row["return_mean"] for row in ebins], marker="o", linewidth=1.5,
              label="P1：4096步箱回报均值")
    axis.set(title="训练逐回合原始波动与箱均值", xlabel="训练宏动作交互步数", ylabel="Q/回报")
    axis.grid(alpha=.25)
    axis.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output / "02_训练原终态Q与回报.png", dpi=160)
    plt.close(fig)
    steps = [row["step"] for row in validation]
    chart("03_验证原终态Q同景对照.png", "Jan12验证20景原终态Q与共同对照", steps,
          [("P1 共同初始化＋区域重构PPO", [row["quality"]["all_scene_mean"] for row in validation]),
           ("R0 共同初始化", [row.get("R0_quality", {}).get("all_scene_mean") for row in validation]),
           ("R3 共同初始化＋同重构器有界搜索", [row.get("R3_quality", {}).get("all_scene_mean")
                                             for row in validation])], "Q")
    chart("04_验证主体与全体CV.png", "Jan12验证20景原终态主体/全体CV", steps,
          [("P1 共同初始化＋区域重构PPO：主体CV",
            [row["subject_cv"]["all_scene_mean"] for row in validation]),
           ("P1 共同初始化＋区域重构PPO：全体CV",
            [row["load_cv"]["all_scene_mean"] for row in validation])], "CV")
    strata = actions["prior_acceptance_bins"]
    chart("05_此前接受动作分层.png", "训练动作接受率：按本回合此前是否接受过动作分层",
          sorted({row["env_step_end"] for row in strata}),
          [("本回合此前无已接受动作" if label == "no_prior_accepted" else
            "本回合此前已有已接受动作", [next((row["acceptance_rate"] for row in strata
                          if row["env_step_end"] == step and row["stratum"] == label), None)
                    for step in sorted({row["env_step_end"] for row in strata})])
           for label in ("no_prior_accepted", "prior_accepted")], "非STOP接受率")
    return {"font_name": font_name,
            "font_file": str(font_path) if font_path is not None else
                         getattr(font, "fname", None)}


def run(job_dir, dev_dirs, audit_path, output_dir, font_path=None):
    context = job_identity(job_dir, dev_dirs, audit_path, output_dir)
    if font_path is not None and not Path(font_path).resolve().is_relative_to(
            (context["archive"] / "ops" / "analysis_override").resolve()):
        raise ValueError("中文字体只允许来自本实验ops/analysis_override")
    job_dir = context["job_dir"]
    sources = {name: {"path": str(job_dir / name), "sha256_before": sha(job_dir / name)}
               for name in ("actions.jsonl", "episodes.jsonl", "updates.jsonl",
                            "evaluations.jsonl", "checkpoint.pt")}
    actions = action_diagnosis(stream(job_dir / "actions.jsonl"))
    episodes = episode_diagnosis(stream(job_dir / "episodes.jsonl"))
    validation_scenes, validation, nodes = validation_diagnosis(
        job_dir, context["config"], context["dev"])
    sources.update({"audit": {"path": str(audit_path), "sha256_before": sha(audit_path)}})
    for route, record in context["dev_inputs"].items():
        sources[f"dev_{route}_result"] = {"path": record["result_path"],
            "sha256_before": record["result_sha256"]}
        sources[f"dev_{route}_raw"] = {"path": record["raw_path"],
            "sha256_before": record["raw_sha256"]}
    for name, record in sources.items():
        if sha(record["path"]) != record["sha256_before"]:
            raise ValueError(f"Input changed during read-only diagnosis: {name}")
    gate = context["audit"]["gate_evidence"]
    recent = actions["windows"]["recent"]
    if (recent["non_stop"] != gate["recent_non_stop_proposals"] or
            recent["accepted_non_stop"] != gate["recent_accepted_actions"] or
            context["audit"]["gate_checks"].get("recent_action_validity") is not False):
        raise ValueError("Independent action recount differs from frozen failed gate")
    for step in STEPS:
        matching = [row for row in stream(job_dir / "evaluations.jsonl")
                    if row.get("scheduled_step") == step]
        node = next(item for item in nodes if item["path"].endswith(f"/{step}.json"))
        if len(matching) != 1 or matching[0].get("node_sha256") != node["sha256"]:
            raise ValueError("Validation node SHA differs from training ledger")
    for item in nodes:
        if sha(item["path"]) != item["sha256"]:
            raise ValueError("Validation node changed during diagnosis")
    for summary in validation:
        reference = [row for row in validation_scenes if row["step"] == summary["step"]]
        for route in ("R0", "R3"):
            for key in ("quality", "coverage_users", "subject_cv", "load_cv"):
                summary[f"{route}_{key}"] = numerical_summary(
                    [row.get(f"{route}_{key}") for row in reference], 20)
    output_dir.mkdir(parents=True)
    marker = output_dir / "analysis_status.json"
    marker.write_text(json.dumps({"status": "incomplete", "started_utc": now()},
                                 ensure_ascii=False), encoding="utf-8")
    csv_write(output_dir / "action_reason_counts.csv", actions["operation_reasons"])
    csv_write(output_dir / "action_bins.csv", actions["bins"])
    csv_write(output_dir / "prior_acceptance_bins.csv", actions["prior_acceptance_bins"])
    csv_write(output_dir / "training_episodes_light.csv", episodes["raw"])
    csv_write(output_dir / "episode_bins.csv", episodes["bins"])
    csv_write(output_dir / "validation_scenes.csv", validation_scenes)
    csv_write(output_dir / "validation_summary.csv", [
        {"step": row["step"], "scenes": 20,
         **{key + "_" + field: value[source]
            for key, value in row.items()
            if isinstance(value, dict) and "all_scene_mean" in value
            for field, source in (("mean", "all_scene_mean"), ("median", "median"),
                                  ("p95", "p95"), ("defined_count", "defined_count"),
                                  ("missing_count", "missing_count"))}}
        for row in validation])
    font_receipt = plot_curves(output_dir, actions, episodes, validation, font_path)
    if font_path is not None:
        font_receipt["sha256"] = sha(font_path)
    result = {"status": "analysis_complete", "completed_utc": now(),
              "analysis_scope": "S1-03 Jan12 development validation, single seed 51001, first 65536/262144 interactions",
              "science_commit": COMMIT, "run_id": RUN_ID,
              "frozen_gate_threshold": 0.50,
              "gate_passed": False,
              "gate_interpretation": "50% is an engineering mechanism threshold; failing it does not prove PPO has no overall value",
              "actions": actions,
              "episodes": {"completed_episodes": episodes["completed_episodes"],
                           "bins": episodes["bins"],
                           "raw_csv": "training_episodes_light.csv"},
              "validation": validation, "input_files": sources,
              "validation_nodes": nodes,
              "figure_font": font_receipt,
              "development_results": {route: {"scenes": len(rows),
                                                **context["dev_inputs"][route]}
                                      for route, rows in context["dev"].items()},
              "limitations": ["Jan12 is development validation, not independent Jan31 assessment",
                              "One training seed; no added training or evaluation",
                              "Repeated menu/action identity is not proof of an identical geometry state",
                              "Frozen e1745 common reconstruction has an accepted-action geometry cache defect; R1/R2/R3/P1 share it",
                              "Association after prior acceptance is observational; do not assign every rejection to the cache defect",
                              "A cache repair has not been trained/evaluated here; improvement is unproven",
                              "Missing per-scene timing is left undefined, never filled with zero"]}
    (output_dir / "summary.json").write_text(json.dumps(result, ensure_ascii=False,
        indent=2, allow_nan=False) + "\n", encoding="utf-8")
    latest = validation[-1]
    strata_all = actions["prior_acceptance_strata"]["all"]
    prior0 = strata_all.get("no_prior_accepted", {})
    prior1 = strata_all.get("prior_accepted", {})
    recent_reasons = sorted((row for row in actions["operation_reasons"]
                             if row["window"] == "recent" and row["operation"] != "stop"),
                            key=lambda row: row["attempted"], reverse=True)[:8]
    def percent(value):
        return f"{value:.2%}" if finite(value) else "缺项"
    report = ["# S1-03 65536步机制停止与几何缓存缺陷诊断（只读）", "",
        "本报告使用既有训练日志和Jan12开发验证，不是Jan31独立考核，也未增加训练或评价。",
        "", "## 闸门结论", "",
        f"最近16384步非STOP动作 {recent['non_stop']} 次，接受 {recent['accepted_non_stop']} 次，"
        f"接受率 {recent['acceptance_rate']:.2%}；预冻结工程门槛为50%。",
        "已定位冻结重构器在已接受动作后未同步几何距离缓存。该缺陷影响后续重构判定；"
        "训练与开发R1/R2/R3共用这一版本，现有对照不能排序无缺陷算法性能。",
        "接受率分层按同回合此前是否出现已接受动作统计；这是相关性证据，"
        "不能据此认定每条拒绝均由缓存缺陷造成。修复后效果仍待重新验证。",
        f"全期无先前接受动作的非STOP提议 {prior0.get('non_stop', 0)} 次、接受率 "
        f"{percent(prior0.get('acceptance_rate'))}；已有先前接受动作的提议 "
        f"{prior1.get('non_stop', 0)} 次、接受率 {percent(prior1.get('acceptance_rate'))}。",
        "50%为工程机制闸门，不是PPO整体有效性的证明或否定。",
        f"四个验证节点各20景；65536步原终态Q均值为 {latest['quality']['all_scene_mean']}。",
        "", "## 最近16384步主要动作结果", "",
        "| 操作 | 原因 | 次数 | 类别 |", "| --- | --- | ---: | --- |",
        *[f"| {row['operation']} | {row['reason']} | {row['attempted']} | {row['category']} |"
          for row in recent_reasons],
        "", "## 65536步验证分段时间（20景）", "",
        "| 阶段 | 均值秒 | 中位数秒 | P95秒 | 有效/缺失 |",
        "| --- | ---: | ---: | ---: | ---: |",
        *[f"| {key} | {latest[key]['all_scene_mean']} | {latest[key]['median']} | "
          f"{latest[key]['p95']} | {latest[key]['defined_count']}/{latest[key]['missing_count']} |"
          for key in TIMES],
        "", "## 口径与缺项", "",
        "曲线采用原始4096交互步分箱；训练回合按完成终态记录，验证每节点同一20景。",
        "训练Q与回报图同时展示逐回合原始散点与4096步箱均值，不用平滑曲线遮盖波动。",
        "主体CV与全体CV、覆盖用户与需求、均值带内束数及Q分别保存。",
        "时间分别给均值、中位数、P95及有效/缺失场景数，缺项为null，不补零。",
        "candidate_screen_seconds与observation_seconds已包含在policy_seconds，不能再次相加计算总耗时。",
        "同景R0/R3差值仅在双方有限值时计算。best_quality_diagnostic只作轨迹最好布局诊断，"
        "主结果始终是原终态；best−raw差值不替换原终态，也不推断未保存的best状态。",
        "重复候选仅指同回合相同菜单SHA与完整动作身份重复，不代表几何状态相同。",
        "", "## 原始数据与图", "",
        "action_reason_counts.csv、action_bins.csv、prior_acceptance_bins.csv、"
        "training_episodes_light.csv、episode_bins.csv、"
        "validation_scenes.csv、validation_summary.csv；",
        "01至05号PNG均由这些原始日志/节点生成。", ""]
    (output_dir / "report.md").write_text("\n".join(report), encoding="utf-8")
    marker.write_text(json.dumps({"status": "analysis_complete", "completed_utc": now(),
                                  "summary_sha256": sha(output_dir / "summary.json")},
                                 ensure_ascii=False), encoding="utf-8")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-dir", type=Path, required=True)
    for route in ("R0", "R1", "R2", "R3"):
        parser.add_argument("--dev-" + route.lower(), type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--font-path", type=Path)
    args = parser.parse_args(argv)
    dev = {route: getattr(args, "dev_" + route.lower()) for route in
           ("R0", "R1", "R2", "R3")}
    result = run(args.job_dir, dev, args.audit, args.output_dir, args.font_path)
    print(json.dumps({"status": result["status"], "run_id": RUN_ID,
                      "output_dir": str(args.output_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
