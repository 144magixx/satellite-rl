"""Create Chinese S1-02 reports and plots from independently audited results."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path

import numpy as np

from audit_stage1_balance import parse_results, read_bundle, sha, write_json
from select_stage1_menus import summarize

NAMES = {
    "A0": "A0 旧C1聚类加局部搜索", "A1": "A1 C1加覆盖修补",
    "A2": "A2 再加负载感知归属", "A3": "A3 再加四向格点调位",
    "X0": "X0 四向固定步长", "X1": "X1 八方向固定步长",
    "X2": "X2 四向多步长", "X3": "X3 业务引导位置", "X4": "X4 相邻双束配合",
}


def display(label):
    if label in NAMES:
        return NAMES[label]
    for menu in ("X0", "X1", "X2", "X3", "X4"):
        if menu in label:
            return NAMES[menu] + ("＋PPO" if "ppo" in label.lower() else "＋同菜单搜索")
    return label


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def number(value, digits=4):
    return "未记录/不适用" if value is None or not np.isfinite(value) else f"{value:.{digits}f}"


def _matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    selected_font = None
    for path in ("/home/u2024110224/second_stage_satellite_rl/fonts/NotoSansSC-VF.ttf",
                 "C:/Windows/Fonts/msyh.ttc", "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                 "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"):
        if Path(path).exists():
            font_manager.fontManager.addfont(path)
            matplotlib.rcParams["font.family"] = font_manager.FontProperties(fname=path).get_name()
            selected_font = {"path": path, "sha256": sha(path), "family": matplotlib.rcParams["font.family"]}
            break
    if selected_font is None:
        raise RuntimeError("No verified CJK font available for readable Chinese report figures")
    matplotlib.rcParams["axes.unicode_minus"] = False
    return plt, selected_font


def create_report(results, audit_path, output, phase, training_runs=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    audit_path = Path(audit_path)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if not audit.get("passed") or set(audit["sources"]) != set(results):
        raise ValueError("An audit of exactly these methods must pass before the report gate")
    for label, path in results.items():
        if audit["sources"][label]["sha256"] != sha(path):
            raise ValueError("A result changed after its independent audit")
    if phase in ("A", "B") and audit["expected_scenes"] != 100:
        raise ValueError("Formal A/B reports require the pre-registered 100 scenes")
    rows, configs, summaries = {}, {}, {}
    for label, path in results.items():
        rows[label], configs[label] = read_bundle(path)
        rows[label] = sorted(rows[label], key=lambda row: row["scene_id"])
        summaries[label] = summarize(rows[label])
    labels = list(results)
    count = len(rows[labels[0]])
    plt, selected_font = _matplotlib()
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    assets = []
    def save(fig, name):
        filename = figures / (name + ".png")
        fig.savefig(filename, dpi=160, bbox_inches="tight")
        plt.close(fig)
        assets.append({"path": "figures/" + filename.name, "sha256": sha(filename),
                       "bytes": filename.stat().st_size})
    def vals(label, key):
        return np.array([(r.get("metrics") or {}).get(key, np.nan) for r in rows[label]], dtype=float)
    x = np.arange(count)
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for label in labels:
        axes[0].plot(x, vals(label, "coverage_demand") * 100, ".-", ms=3, lw=.7, label=display(label))
        axes[1].plot(x, vals(label, "coverage_users") * 100, ".-", ms=3, lw=.7, label=display(label))
    for ax, title in zip(axes, ("需求覆盖率 / %", "用户覆盖率 / %")):
        ax.set_ylabel(title)
        ax.axhline(99.9, color="black", linestyle="--", alpha=.5)
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8, ncol=2)
    axes[-1].set_xlabel("共同场景序号（按场景ID固定排序，无平滑）")
    fig.suptitle(f"S1-02 {phase} 全部场景覆盖")
    save(fig, "coverage_all_scenes")
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for label in labels:
        axes[0].plot(x, vals(label, "load_cv"), ".", label=display(label))
        axes[1].plot(x, vals(label, "subject_cv"), ".", label=display(label))
    axes[0].set_ylabel("全体100束CV")
    axes[1].set_ylabel("主体波束CV")
    axes[1].axhline(.2, color="black", linestyle="--")
    axes[0].legend(fontsize=8, ncol=2)
    for ax in axes:
        ax.grid(alpha=.2)
    axes[-1].set_xlabel("共同场景序号（保留失败布局；无平滑）")
    save(fig, "all_and_subject_cv")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for label in labels:
        axes[0].scatter(vals(label, "coverage_users")*100, vals(label, "subject_cv"), s=15, alpha=.65, label=display(label))
        timing = np.array([r.get("total_pipeline_seconds", r["balance_diagnostics"].get("total_pipeline_seconds")) for r in rows[label]])
        axes[1].plot(x, timing, ".", label=display(label))
    axes[0].set(xlabel="用户覆盖率 / %", ylabel="主体波束CV")
    axes[1].set(xlabel="共同场景序号", ylabel="完整路线时间 / 秒")
    axes[0].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=.2)
    save(fig, "coverage_balance_and_runtime")
    fig, ax = plt.subplots(figsize=(11, 5))
    for label in labels:
        ax.plot(x, vals(label, "subject_in_band_beams"), ".", label=display(label))
    ax.axhline(90, color="black", linestyle="--")
    ax.set(xlabel="共同场景序号", ylabel="主体均值±20%范围内束数（分母始终100）")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=.2)
    save(fig, "balanced_beam_count")
    reference = "A0" if "A0" in rows else next((label for label in labels if "search" in label.lower() and "X0" in label), labels[0])
    paired = {}
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for label in labels:
        if label == reference:
            continue
        dc = (vals(label, "coverage_demand")-vals(reference, "coverage_demand"))*100
        dcv = vals(label, "subject_cv")-vals(reference, "subject_cv")
        paired[label] = {"reference": reference,
                         "paired_valid_layout_scenes": int(np.sum(np.isfinite(dc) & np.isfinite(dcv))),
                         "coverage_demand_delta_percentage_points_mean": float(np.nanmean(dc)) if np.isfinite(dc).any() else None,
                         "subject_cv_delta_mean": float(np.nanmean(dcv)) if np.isfinite(dcv).any() else None,
                         "subject_cv_improved": int(np.sum(dcv < -1e-9)),
                         "subject_cv_worse": int(np.sum(dcv > 1e-9)),
                         "subject_cv_equal": int(np.sum(np.abs(dcv) <= 1e-9))}
        axes[0].plot(x, dc, ".", label=display(label))
        axes[1].plot(x, dcv, ".", label=display(label))
    for ax in axes:
        ax.axhline(0, color="black", lw=.7)
        ax.grid(alpha=.2)
    axes[0].set_ylabel("需求覆盖差 / 百分点")
    axes[1].set_ylabel("主体CV差（负值为降低）")
    axes[0].legend(fontsize=8, ncol=2)
    axes[-1].set_xlabel("共同场景序号；参照：" + display(reference))
    save(fig, "paired_changes")
    fig, ax = plt.subplots(figsize=(12, 5))
    representative = count // 2
    for label in labels:
        if rows[label][representative].get("beam_loads_bps") is not None:
            load = np.asarray(rows[label][representative]["beam_loads_bps"], dtype=float) / 1e6
            ax.plot(np.arange(1, 101), np.sort(load), label=display(label))
    ax.set(xlabel="按负载排序的全部100束", ylabel="业务需求 / Mbps",
           title="固定中间序号场景（不按成绩挑选）")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=.2)
    save(fig, "representative_loads")
    process_missing = []
    fig, ax = plt.subplots(figsize=(11, 5))
    drew = False
    for label in labels:
        stage = rows[label][representative].get("balance_diagnostics", {}).get("stages", {}).get("adjustment", {})
        history = stage.get("counts", {}).get("trace", stage.get("counts", {}).get("history", stage.get("history", [])))
        if not history:
            history = rows[label][representative].get("action_trace", [])
        points = [(h.get("beam_processes", h.get("decision", h.get("round"))), h.get("subject_cv", h.get("metrics", {}).get("subject_cv"))) for h in history]
        points = [(a, b) for a, b in points if a is not None and b is not None]
        if points:
            ax.plot([p[0] for p in points], [p[1] for p in points], label=display(label))
            drew = True
        else:
            process_missing.append(label)
    if drew:
        ax.set(xlabel="已记录的束处理次数", ylabel="主体CV", title="固定场景记录过程：搜索保留值或PPO实际轨迹，无补点")
        ax.legend(fontsize=8)
        save(fig, "recorded_adjustment_process")
    else:
        plt.close(fig)
    training_evidence = {}
    training_runs = training_runs or {}
    if phase == "B" and len(training_runs) != 3:
        raise ValueError("B delivery requires all three actual training runs for learning curves")
    for menu, directory in training_runs.items():
        directory = Path(directory)
        episode_path, update_path = directory / "episodes.jsonl", directory / "updates.jsonl"
        if not episode_path.is_file() or not update_path.is_file():
            raise ValueError(f"Missing actual training curves for {menu}")
        episodes, updates = jsonl(episode_path), jsonl(update_path)
        fig, axes = plt.subplots(2, 1, figsize=(11, 7))
        episode_points = [(r.get("env_step", r.get("terminal_env_step")), r.get("episode_return", r.get("return"))) for r in episodes]
        episode_points = [(a, b) for a, b in episode_points if a is not None and b is not None]
        if not episode_points:
            raise ValueError(f"{menu}: no actual complete-episode reward records for learning curves")
        if episode_points:
            axes[0].scatter([a for a, b in episode_points], [b for a, b in episode_points], s=5, alpha=.5)
        keys = ("policy_loss", "value_loss", "entropy")
        for key in keys:
            points = [(r.get("env_step", r.get("step")), r.get(key, r.get("metrics", {}).get(key))) for r in updates]
            points = [(a, b) for a, b in points if a is not None and b is not None]
            if points:
                axes[1].plot([a for a, b in points], [b for a, b in points], label=key)
        if not axes[1].lines:
            raise ValueError(f"{menu}: no recorded learning diagnostics")
        axes[0].set_ylabel("完整回合累计奖励（原始散点）")
        axes[1].set(xlabel="累计环境交互", ylabel="已记录学习指标")
        if axes[1].lines:
            axes[1].legend()
        fig.suptitle(display(menu) + " PPO，seed51001，无平滑")
        save(fig, "training_" + menu)
        evaluation_path = directory / "evaluations.jsonl"
        training_evidence[menu] = {"directory": str(directory), "episodes_sha256": sha(episode_path),
                                   "updates_sha256": sha(update_path), "complete_episode_points": len(episode_points)}
        if evaluation_path.is_file():
            evaluations = [r for r in jsonl(evaluation_path) if r.get("group") == "validation"]
            steps = sorted({r.get("env_step", r.get("step")) for r in evaluations if r.get("env_step", r.get("step")) is not None})
            if phase == "B":
                if len(steps) != 20 or len(evaluations) != 400:
                    raise ValueError(f"{menu}: expected 20 validation nodes and 400 complete scene rows")
                for step in steps:
                    node = [r for r in evaluations if r.get("env_step", r.get("step")) == step]
                    if len(node) != 20 or len({r.get("scene_id") for r in node}) != 20:
                        raise ValueError(f"{menu}: validation node is incomplete or repeats a scene")
            fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
            for ax, metric in zip(axes, ("coverage_demand", "subject_cv")):
                means = []
                for step in steps:
                    values = [r["metrics"][metric] for r in evaluations if r.get("env_step", r.get("step")) == step and r.get("metrics", {}).get(metric) is not None]
                    means.append(float(np.mean(values)) if values else np.nan)
                ax.plot(steps, means, "o-")
                ax.set_ylabel(metric)
                ax.grid(alpha=.2)
            axes[-1].set_xlabel("累计环境交互；固定20验证场景均值，无平滑")
            save(fig, "validation_" + menu)
            training_evidence[menu]["evaluation_sha256"] = sha(evaluation_path)
            training_evidence[menu]["validation_nodes"] = steps
        elif phase == "B":
            raise ValueError(f"{menu}: missing per-scene validation evidence")
    lines = [f"# S1-02 {phase}阶段覆盖与主体负载均衡报告", "",
             f"生成日期（北京时间）：{datetime.now(timezone(timedelta(hours=8))).date()}。共同{count}场景，可核布局的独立几何/需求/指标及计时审计通过；未形成布局的场景只核失败来源记录，不进行不存在的物理复算。", "",
             "本报告只描述保存结果。业务需求为ADS-B真实位置驱动的合成量，覆盖不是链路容量；第一阶段SI/SGM不适用。", "",
             "## 共同条件与全部场景结果", "",
             "100束、全宽0.5°、共同用户输入与A2来源；严格全覆盖优先，双99.9%单列。主体CV及合理负载范围统计包含未认证轻载束、过载束和空束，不删除失败场景。", "",
             "| 编号与实验组合 | 全覆盖且合法 | 双99.9%可行 | 覆盖及均衡达标 | 主体CV均值 | 全体CV均值 | 完整时间均值/秒 | 中位/秒 | P95/秒 |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for label in labels:
        s = summaries[label]
        lines.append(f"| {display(label)} | {s['full_coverage_legal_scenes']}/{count} | {s['near_coverage_legal_scenes']}/{count} | {s['balanced_target_scenes']}/{count} | {number(s['subject_cv_mean'])} | {number(s['all_cv_mean'])} | {number(s['full_pipeline_seconds_mean'])} | {number(s['full_pipeline_seconds_median'])} | {number(s['full_pipeline_seconds_p95'])} |")
    lines.extend(["", "覆盖/达标计数始终以全部场景为分母；CV只在实际形成布局时有定义，其均值分母逐组如下，不能将缺失填0：", ""])
    for label in labels:
        s = summaries[label]
        lines.append(f"- {display(label)}：有效布局{s['valid_layout_scenes']}/{count}，初始化失败{s['initialization_failed_scenes']}；覆盖未达但有布局的场景仍计入CV。")
    lines.extend(["", "## 同场景变化", "", "| 双方实验组合 | 需求覆盖差/百分点 | 主体CV差 | CV改善/持平/退步场景 | 双方有效布局数 |", "| --- | ---: | ---: | ---: | ---: |"])
    for label, p in paired.items():
        lines.append(f"| {display(label)} − {display(reference)} | {number(p['coverage_demand_delta_percentage_points_mean'], 6)} | {number(p['subject_cv_delta_mean'], 6)} | {p['subject_cv_improved']}/{p['subject_cv_equal']}/{p['subject_cv_worse']} | {p['paired_valid_layout_scenes']}/{count} |")
    lines.extend(["", "## 停止原因与计算口径", "",
                  "每方法独立获得算法额度；公共初始化/修补/归属仅实际计算一次，完整路线时间按其分段实测值加本方法新增部分合计，不代表各路线重新运行了全部公共步骤。候选生成、归属与最后核验均需计入；每景串行延迟与批量训练吞吐分开。", ""])
    for label in labels:
        stops = Counter(str(r.get("stop_reason", r.get("balance_diagnostics", {}).get("stages", {}).get("adjustment", {}).get("counts", {}).get("stop_reason", r.get("status")))) for r in rows[label])
        lines.append(f"- {display(label)}：" + "；".join(f"{key} {value}景" for key, value in sorted(stops.items())) + "。")
    lines.extend(["", "## 原始曲线", "", "全部为保存数据，无平滑、线性坐标。场景序号不是训练步数；同日场景可能相关。固定中间序号示例不按成绩筛选。", ""])
    for asset in assets:
        lines.append(f"![{Path(asset['path']).stem}]({asset['path']})\n")
    if process_missing:
        lines.append("未记录逐轮调整过程或该方法不调整：" + "、".join(display(label) for label in process_missing) + "。未补造曲线。")
    lines.extend(["", "## 结论范围与证据", "",
                  "覆盖达标与主体均衡必须一起看，不能凭单项CV最低宣布整体最好。完整时间更短也不代表覆盖更好；表内变化仅限本批共同场景、冻结方法和预算。B阶段只有训练seed51001；非学习搜索、同一场景重复处理都不是额外训练种子。没有全局最优或跨季节稳定性的证明。", "",
                  "机器可核对的汇总、配对差值、原始文件SHA、独立审计来源与图像SHA见同目录report_manifest.json。正式交付还需完成原始资产归档、中文结论复核、目录及实验状态同步。"])
    report_path = output / "report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest = {"experiment_id": "S1-02", "phase": phase, "created_at": datetime.now(timezone.utc).isoformat(),
                "audit_path": str(audit_path), "audit_sha256": sha(audit_path), "summaries": summaries,
                "paired": paired, "sources": audit["sources"], "training": training_evidence,
                "figures": assets, "report_sha256": sha(report_path), "scene_count": count,
                "process_curve_missing": process_missing, "plot_font": selected_font}
    write_json(output / "report_manifest.json", manifest)
    write_json(output / "gate.json", {"experiment_id": "S1-02", "status": "passed", "phase": phase,
                "report_sha256": sha(report_path), "audit_sha256": sha(audit_path),
                "report_manifest_sha256": sha(output / "report_manifest.json"),
                "meaning": "Data audit and reproducible report generation complete; final delivery still requires archive and human-readable review"})
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", action="append", required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("A", "B", "development"), required=True)
    parser.add_argument("--training-run", action="append", default=[])
    args = parser.parse_args()
    manifest = create_report(parse_results(args.result), args.audit, args.output, args.phase,
                             parse_results(args.training_run))
    print(json.dumps({"phase": args.phase, "figures": len(manifest["figures"]), "scene_count": manifest["scene_count"]}))


if __name__ == "__main__":
    main()
