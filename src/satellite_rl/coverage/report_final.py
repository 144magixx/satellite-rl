"""Read-only S1-03-R1 final tables and Chinese curves from archived workbench data.

Run with the frozen archive root and a NEW output directory. This module never
loads a model or reruns an evaluation. Official execution uses remote DQN.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


ROUTES = ("R0", "R1", "R2", "R3", "P1")
STAGES = ("initialization_seconds", "policy_seconds", "reconstruction_seconds",
          "final_check_seconds", "total_seconds")
METRICS = ("quality", "coverage_users", "coverage_demand", "subject_cv",
           "load_cv", "subject_in_band_beams", "target_met")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_lines(path: Path):
    with path.open(encoding="utf-8") as source:
        for line in source:
            if line.strip():
                yield json.loads(line)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def distribution(values):
    values = sorted(float(x) for x in values if finite(x))
    if not values:
        return {"n": 0, "mean": None, "median": None, "p95": None}
    position = .95 * (len(values) - 1)
    low = math.floor(position)
    p95 = values[low] + (values[min(low + 1, len(values) - 1)] - values[low]) * (position - low)
    return {"n": len(values), "mean": statistics.fmean(values),
            "median": statistics.median(values), "p95": p95}


def flatten(row, route):
    metrics = row["metrics"]
    initial = row["initial_metrics"]
    result = {"route": route, "scene_id": row["scene_id"],
              "scene_sha256": row["scene_sha256"], "scene_index": row["scene_index"],
              "training_seed": row["training_seed"], "model_step": row["model_step"],
              "initialization_valid": row["initialization_valid"],
              "stop_reason": row["stop_reason"], "macro_steps": row["macro_steps"]}
    for metric in METRICS:
        result[metric] = metrics.get(metric)
        result["initial_" + metric] = initial.get(metric)
    result["subject_beams"] = metrics["subject_beams"]
    result["edge_beam_count"] = metrics["edge_beam_count"]
    result["empty_beams"] = metrics["empty_beams"]
    result["coverage_100"] = metrics["coverage_100"]
    result["coverage_999"] = (metrics["coverage_users"] >= .999 and
                              metrics["coverage_demand"] >= .999)
    result["best_quality_diagnostic"] = row.get("best_quality_diagnostic")
    loads = row["beam_load_demand_bps"]
    require(len(loads) == 100 and all(finite(x) and x >= 0 for x in loads),
            "invalid final 100-beam loads")
    all_mean = statistics.fmean(loads)
    result["all_100_in_band_beams"] = (sum(abs(x / all_mean - 1) <= .2 + 1e-12
                                            for x in loads) if all_mean > 0 else 0)
    result["peak_to_all_mean"] = max(loads) / all_mean if all_mean > 0 else None
    result["in_band_fraction"] = (metrics["subject_in_band_beams"] / metrics["subject_beams"]
                                  if metrics["subject_beams"] else None)
    for stage in STAGES:
        result[stage] = row.get(stage)
    return result


def load_formal(root: Path, control: dict, science: str, plan: str,
                formal_jobs_root: Path | None = None, audit_path: Path | None = None):
    audit_path = audit_path or root / "ops/remote_20424/audits/formal_audit.json"
    formal_jobs_root = formal_jobs_root or root / "jobs"
    require(sha256(audit_path) == control["audits"]["formal"]["audit_sha256"],
            "formal audit SHA mismatch")
    audit = read_json(audit_path)
    require(audit.get("passed") is True and not audit.get("issues") and
            audit.get("science_commit") == science and audit.get("plan_sha256") == plan,
            "formal audit did not pass for this source and plan")
    jobs = control["jobs"]
    by_route = {}
    provenance = {}
    common = None
    for route in ROUTES:
        run_id = jobs["formal_" + route]
        job_dir = formal_jobs_root / run_id
        metadata = read_json(job_dir / "results.json")
        job = read_json(job_dir / "job.json")
        require(metadata["run_id"] == run_id and metadata["route"] == route and
                metadata["source_commit"] == science and metadata["scene_count"] == 100,
                f"{route} result provenance/count mismatch")
        raw_path = job_dir / "results.partial.jsonl"
        digest = sha256(raw_path)
        require(digest == metadata["rows_sha256"], f"{route} raw SHA mismatch")
        raw = list(read_lines(raw_path))
        require(len(raw) == 100, f"{route} raw count mismatch")
        keyed = {}
        for row in raw:
            require(row["route"] == route and row["group"] == "test" and
                    row["training_seed"] == 51001,
                    f"{route} invalid formal row")
            key = (row["scene_id"], row["scene_sha256"])
            require(key not in keyed, f"{route} duplicate scene")
            keyed[key] = flatten(row, route)
        require(common is None or set(keyed) == common, "five routes do not share scenes")
        common = set(keyed)
        by_route[route] = keyed
        provenance[route] = {"run_id": run_id, "raw_sha256": digest,
                             "config_hash": metadata["config_hash"],
                             "execution_target": metadata["execution_target"]}
        if route == "P1":
            require(job.get("model_source_run_id") == jobs["train_P1"] and
                    len(job.get("model_sha256", "")) == 64,
                    "formal P1 model is not linked to completed training")
            provenance[route]["model_source_run_id"] = job["model_source_run_id"]
            provenance[route]["model_sha256"] = job["model_sha256"]
    return by_route, provenance


def analyze_formal(by_route):
    keys = sorted(by_route["P1"])
    rows = [by_route[route][key] for route in ROUTES for key in keys]
    summary = {"routes": {}, "paired": {}, "timing_scope":
               "训练后共同考核的已记录分段；缺失起点生成时间不补零，不代表完整端到端耗时"}
    for route in ROUTES:
        rr = list(by_route[route].values())
        summary["routes"][route] = {
            "scenes": len(rr),
            "initialization_valid": sum(x["initialization_valid"] is True for x in rr),
            "coverage_100": sum(x["coverage_100"] is True for x in rr),
            "coverage_999": sum(x["coverage_999"] is True for x in rr),
            "target_met": sum(x["target_met"] is True for x in rr),
            "subject_in_band_beams": distribution(x["subject_in_band_beams"] for x in rr),
            "all_100_in_band_beams": distribution(x["all_100_in_band_beams"] for x in rr),
            "subject_beams": distribution(x["subject_beams"] for x in rr),
            "edge_beam_count": distribution(x["edge_beam_count"] for x in rr),
            "peak_to_all_mean": distribution(x["peak_to_all_mean"] for x in rr),
            "in_band_fraction": distribution(x["in_band_fraction"] for x in rr),
            "cv": distribution(x["subject_cv"] for x in rr),
            "all_beam_cv": distribution(x["load_cv"] for x in rr),
            "quality": distribution(x["quality"] for x in rr),
            "timing": {stage: distribution(x[stage] for x in rr) for stage in STAGES},
        }
        summary["routes"][route]["coverage_failures"] = len(rr) - summary["routes"][route]["coverage_100"]
        summary["routes"][route]["coverage_999_failures"] = len(rr) - summary["routes"][route]["coverage_999"]
        summary["routes"][route]["target_failures_all_scenes"] = len(rr) - summary["routes"][route]["target_met"]
        summary["routes"][route]["coverage_subsets"] = {}
        for subset, selected in (("strict_covered", [x for x in rr if x["coverage_100"]]),
                                 ("not_strict_covered", [x for x in rr if not x["coverage_100"]]),
                                 ("at_least_99_9_both", [x for x in rr if x["coverage_999"]]),
                                 ("below_99_9_either", [x for x in rr if not x["coverage_999"]])):
            summary["routes"][route]["coverage_subsets"][subset] = {
                "denominator": len(selected),
                "initialization_valid": sum(x["initialization_valid"] is True for x in selected),
                "subject_cv": distribution(x["subject_cv"] for x in selected),
                "all_beam_cv": distribution(x["load_cv"] for x in selected),
                "subject_in_band_beams": distribution(x["subject_in_band_beams"] for x in selected),
                "target_met": sum(x["target_met"] is True for x in selected),
            }
    paired_rows = []
    for key in keys:
        p, r = by_route["P1"][key], by_route["R3"][key]
        require(p["initial_quality"] == r["initial_quality"], "P1/R3 initial quality mismatch")
        line = {"scene_id": key[0], "scene_sha256": key[1],
                "P1_quality": p["quality"], "R3_quality": r["quality"],
                "initial_quality": p["initial_quality"],
                "P1_minus_R3_quality": p["quality"] - r["quality"],
                "P1_minus_initial_quality": p["quality"] - p["initial_quality"],
                "P1_best_quality_diagnostic": p["best_quality_diagnostic"],
                "P1_best_minus_original_quality": (p["best_quality_diagnostic"] - p["quality"]
                                                    if finite(p["best_quality_diagnostic"]) else None),
                "P1_minus_R3_subject_cv": p["subject_cv"] - r["subject_cv"],
                "P1_minus_R3_in_band_beams": p["subject_in_band_beams"] - r["subject_in_band_beams"],
                "P1_target_met": p["target_met"], "R3_target_met": r["target_met"]}
        paired_rows.append(line)
    for field in ("P1_minus_R3_quality", "P1_minus_initial_quality", "P1_best_minus_original_quality",
                  "P1_minus_R3_subject_cv", "P1_minus_R3_in_band_beams"):
        vals = [row[field] for row in paired_rows if finite(row[field])]
        summary["paired"][field] = {**distribution(vals), "positive": sum(v > 0 for v in vals),
                                     "zero": sum(v == 0 for v in vals), "negative": sum(v < 0 for v in vals)}
    return rows, paired_rows, summary


def load_training(root, control, science, train_jobs_root=None):
    job_dir = (train_jobs_root or root / "jobs") / control["jobs"]["train_P1"]
    job = read_json(job_dir / "job.json")
    status = read_json(job_dir / "status.json")
    require(job["git_commit"] == science and status["stage"] == "completed" and
            status["env_step"] == status["budget_env_steps"] == 262144 and
            status["update_step"] > 0, "training source or final budget mismatch")
    episodes_path, updates_path, actions_path = (job_dir / name for name in
                                                 ("episodes.jsonl", "updates.jsonl", "actions.jsonl"))
    episode_curve = []
    for row in read_lines(episodes_path):
        episode_curve.append({"env_step": row["terminal_env_step"],
                              "episode": row["episode_serial"], "return": row["episode_return"],
                              "quality": row["metrics"]["quality"],
                              "coverage_users": row["metrics"]["coverage_users"],
                              "subject_cv": row["metrics"]["subject_cv"],
                              "in_band_beams": row["metrics"]["subject_in_band_beams"]})
    updates = list(read_lines(updates_path))
    require(episode_curve and updates and updates[-1]["env_step"] == 262144,
            "training episode/update records incomplete")
    actions = {"count": 0, "accepted_non_stop": 0, "non_stop": 0, "stop": 0}
    for row in read_lines(actions_path):
        actions["count"] += 1
        operation = row.get("requested_action_fields", {}).get("operation")
        if operation == "stop":
            actions["stop"] += 1
        else:
            actions["non_stop"] += 1
            actions["accepted_non_stop"] += int(row.get("accepted") is True)
    require(actions["count"] == 262144, "training action count does not match budget")
    nodes = []
    for event in read_lines(job_dir / "evaluations.jsonl"):
        step = event["scheduled_step"]
        path = job_dir / "validation_nodes" / f"{step}.json"
        require(sha256(path) == event["node_sha256"], f"validation node {step} SHA mismatch")
        node = read_json(path)
        require(node["scheduled_step"] == step and len(node["rows"]) == 20,
                f"validation node {step} count mismatch")
        nodes.append({"step": step, "actual_env_step": event["actual_env_step"],
                      "scene_count": 20,
                      "quality_mean": statistics.fmean(r["metrics"]["quality"] for r in node["rows"]),
                      "coverage_100": sum(r["metrics"]["coverage_100"] for r in node["rows"]),
                      "target_met": sum(r["metrics"]["target_met"] for r in node["rows"]),
                      "subject_cv_mean": statistics.fmean(r["metrics"]["subject_cv"] for r in node["rows"]),
                      "node_sha256": event["node_sha256"]})
    require([n["step"] for n in nodes] == list(range(16384, 262145, 16384)),
            "validation is not the complete 16-node schedule")
    return episode_curve, updates, actions, nodes, status["update_step"], status["checkpoint_sha256"], {name: sha256(path) for name, path in
                                                     (("episodes", episodes_path), ("updates", updates_path),
                                                      ("actions", actions_path))}


def write_csv(path, rows):
    require(rows, f"empty CSV {path.name}")
    with path.open("w", encoding="utf-8-sig", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_curves(output, episodes, updates, nodes, formal, formal_rows, paired_rows, font_path=None):
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    if font_path:
        require(font_path.is_file(), f"Chinese font missing: {font_path}")
        font_manager.fontManager.addfont(str(font_path))
        family = font_manager.FontProperties(fname=str(font_path)).get_name()
        glyphs = font_manager.ft2font.FT2Font(str(font_path)).get_charmap()
        require(all(ord(c) in glyphs for c in "训练验证覆盖主体全体束景耗时均值中位策略重构终态"),
                "font lacks required Chinese glyphs")
    else:
        available = {font.name for font in font_manager.fontManager.ttflist}
        family = next((name for name in ("Noto Sans CJK SC", "Noto Sans CJK JP", "Microsoft YaHei", "SimHei")
                       if name in available), None)
        require(family is not None, "Chinese font unavailable; pass --font /path/to/cjk-font.ttf")
    plt.rcParams["font.sans-serif"] = [family]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    window = 100
    xs = [r["env_step"] for r in episodes]
    raw = [r["return"] for r in episodes]
    axes[0, 0].plot(xs, raw, alpha=.23, lw=.65, label="逐回合原始回报")
    smooth = [statistics.fmean(raw[max(0, i-window+1):i+1]) for i in range(len(raw))]
    axes[0, 0].plot(xs, smooth, lw=1.3, label="回合滑动均值，窗口100")
    axes[0, 0].set_title("训练回报（seed 51001）")
    axes[0, 0].legend()
    axes[0, 1].plot([r["env_step"] for r in updates], [r["entropy"] for r in updates], label="熵")
    axes[0, 1].set_title("PPO更新熵")
    axes[1, 0].plot([r["step"] for r in nodes], [r["quality_mean"] for r in nodes], marker="o", label="验证均值Q")
    axes[1, 0].set_title("固定20景验证：16个节点")
    names = list(ROUTES)
    labels = ["R0 共同起点", "R1 规则重构", "R2 随机重构", "R3 同重构器搜索", "P1 PPO策略"]
    axes[1, 1].bar(labels, [formal["routes"][r]["quality"]["mean"] for r in names])
    axes[1, 1].set_title("Jan31共同100景：最终Q均值")
    axes[1, 1].tick_params(axis="x", labelrotation=25)
    for ax in axes.flat:
        ax.grid(alpha=.2)
        ax.set_xlabel("训练交互步" if ax is not axes[1, 1] else "方法")
    fig.tight_layout()
    fig.savefig(output / "中文训练验证与考核曲线.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    ax = axes[0, 0]
    ax.bar(labels, [formal["routes"][r]["coverage_100"] for r in names], label="严格双100%")
    ax.bar(labels, [formal["routes"][r]["coverage_failures"] for r in names],
           bottom=[formal["routes"][r]["coverage_100"] for r in names], label="未严格覆盖")
    ax.set_ylim(0, 105)
    ax.set_title("共同100景：覆盖分母含失败场景")
    ax.legend()
    ax = axes[0, 1]
    ax.boxplot([[r["subject_cv"] for r in formal_rows if r["route"] == route]
                for route in names], tick_labels=labels)
    ax.set_title("主体CV逐景分布（含未严格覆盖）")
    ax = axes[1, 0]
    ax.boxplot([[r["load_cv"] for r in formal_rows if r["route"] == route]
                for route in names], tick_labels=labels)
    ax.set_title("全体100束CV逐景分布（含失败）")
    ax = axes[1, 1]
    ax.bar(labels, [formal["routes"][r]["subject_in_band_beams"]["mean"] for r in names],
           label="主体均值±20%带内束数")
    ax.plot(labels, [formal["routes"][r]["all_100_in_band_beams"]["mean"] for r in names],
            color="black", marker="o", label="全体均值±20%带内束数")
    ax.set_title("带内束数；主体按认证边缘例外排除")
    ax.legend()
    for ax in axes.flat:
        ax.tick_params(axis="x", labelrotation=25)
        ax.grid(axis="y", alpha=.2)
    fig.tight_layout()
    fig.savefig(output / "中文共同考核覆盖均衡分布.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(13, 8))
    indices = [i + 1 for i in range(len(paired_rows))]
    for ax, field, title in ((axes[0], "P1_minus_R3_quality", "P1 PPO策略－R3 同重构器搜索：同景原终态Q差"),
                              (axes[1], "P1_best_minus_original_quality", "P1 best诊断－P1策略原终态Q：同景差")):
        ax.axhline(0, color="black", lw=.8)
        ax.plot(indices, [r[field] for r in paired_rows], marker=".", lw=.8)
        ax.set_title(title)
        ax.set_xlabel("共同场景序号（按scene_id排序）")
        ax.set_ylabel("Q差")
        ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(output / "中文P1同景配对与best诊断.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, stage, title in ((axes[0], "policy_seconds", "策略段耗时"),
                             (axes[1], "total_seconds", "已记录分段合计")):
        for statistic, marker in (("mean", "o"), ("median", "s"), ("p95", "^")):
            ax.plot(labels, [formal["routes"][r]["timing"][stage][statistic] for r in names],
                    marker=marker, label=statistic)
        ax.set_title(title + "（均值/中位/P95）")
        ax.set_ylabel("秒/场景")
        ax.tick_params(axis="x", labelrotation=25)
        ax.grid(alpha=.2)
        ax.legend()
    fig.tight_layout()
    fig.savefig(output / "中文共同考核分段耗时.png", dpi=160)
    plt.close(fig)


def build(root: Path, output: Path, train_jobs_root: Path | None = None,
          formal_jobs_root: Path | None = None, audit_path: Path | None = None,
          font_path: Path | None = None):
    require(not output.exists(), "output already exists; choose a new directory")
    control = read_json(root / "controller_state.json")
    science, plan = control["science_commit"], control["plan_sha256"]
    require(control["phase"] == "awaiting_final_report" and
            control["audits"]["formal"]["passed"] is True,
            "controller has not reached the audited final report phase")
    by_route, provenance = load_formal(root, control, science, plan, formal_jobs_root, audit_path)
    formal_rows, paired_rows, formal = analyze_formal(by_route)
    episodes, updates, actions, nodes, optimizer_updates, final_checkpoint_sha, training_sha = load_training(
        root, control, science, train_jobs_root)
    output.mkdir(parents=True)
    write_csv(output / "formal_scenes.csv", formal_rows)
    write_csv(output / "paired_scenes.csv", paired_rows)
    (output / "formal_scenes.json").write_text(
        json.dumps(formal_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "paired_scenes.json").write_text(
        json.dumps(paired_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv(output / "training_episodes.csv", episodes)
    write_csv(output / "training_updates.csv", updates)
    write_csv(output / "validation_nodes.csv", nodes)
    summary = {"scope": "S1-03-R1 final read-only analysis", "science_commit": science,
               "plan_sha256": plan, "formal_audit_sha256": control["audits"]["formal"]["audit_sha256"],
               "training_run_id": control["jobs"]["train_P1"], "training_source_sha256": training_sha,
               "formal_provenance": provenance, "training_seed": 51001,
               "model_selection": {"rule": "fixed_budget_endpoint", "model_step": 262144,
                                   "training_checkpoint_sha256": final_checkpoint_sha,
                                   "formal_P1_model_sha256": provenance["P1"]["model_sha256"]},
               "training_env_steps": 262144, "training_episodes": len(episodes),
               "logged_update_batches": len(updates), "optimizer_updates": optimizer_updates,
               "training_actions": actions,
               "validation_nodes": len(nodes), "formal": formal,
               "interpretation": ["Q为本实验覆盖与负载重构指标；非通信容量、SI或SGM。",
                                  "五方法逐景为同一Jan31场景；单训练种子，不是多种子稳定性结论。",
                                  "all_100_in_band_beams使用全体100束均值的±20%描述统计；主体带内数来自冻结指标的主体均值与认证边缘例外。",
                                  "candidate_screen与observation的耗时已包含在policy_seconds中，不重复相加。",
                                  "起点生成时间未记录；现有initialization_seconds只含工作台已记录的初始化段，不把分段合计称作完整端到端。",
                                  "正式结论须等待独立原始结果复算及报告审计。"]}
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    plot_curves(output, episodes, updates, nodes, formal, formal_rows, paired_rows, font_path)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="archived S1-03-R1 experiment root")
    parser.add_argument("--output", type=Path, required=True, help="new directory for derived files")
    parser.add_argument("--train-jobs-root", type=Path, help="optional original 20418 jobs directory")
    parser.add_argument("--formal-jobs-root", type=Path, help="optional original 20424 jobs directory")
    parser.add_argument("--audit-path", type=Path, help="optional original formal_audit.json")
    parser.add_argument("--font", type=Path, help="Chinese TTF/OTF font for verified readable figures")
    args = parser.parse_args()
    result = build(args.root, args.output, args.train_jobs_root, args.formal_jobs_root,
                   args.audit_path, args.font)
    print(json.dumps({"output": str(args.output), "formal_audit_sha256": result["formal_audit_sha256"],
                      "training_env_steps": result["training_env_steps"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
