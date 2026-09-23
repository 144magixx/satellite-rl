"""Build Chinese, raw-data-backed reports for the 18 paired SAC experiments."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import statistics

import numpy as np

from satellite_rl.training.experiment_runner import sha256_file
from satellite_rl.training.run_monitor import atomic_json, utc_now


MODELS = ("transformer", "cnn_matched", "mlp_matched")
FORMULAS = ("legacy", "corrected")
MODEL_NAMES = {"transformer": "Transformer SAC", "cnn_matched": "参数量接近的 CNN SAC",
               "mlp_matched": "参数量接近的 MLP SAC"}
FORMULA_NAMES = {"legacy": "原学习公式", "corrected": "修正学习公式"}
FINAL_FIELDS = (
    "si_sum", "si_mean", "sgm_mean", "throughput_bps", "total_power_w",
    "served_beams", "total_beams", "power_exceeded", "completed",
    "occupancy_fraction", "zero_slot_rate", "allocated_slots",
    "spectral_efficiency", "interference_beams", "in_range_interference_beams",
)
MAIN_FIELDS = FINAL_FIELDS


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _numeric(value):
    if isinstance(value, bool):
        return float(value)
    return float(value)


def _sample_rows(result):
    return [row for row in result["rows"]
            if row.get("action_mode", "sample") == "sample"]


def _load_run(directory, job, plan):
    run_dir = directory / job["run_id"]
    status = _read_json(run_dir / "status.json")
    if status.get("stage") != "completed" or status.get("env_step") != plan["steps"]:
        raise RuntimeError(f"{job['run_id']} 尚未完成 400000 步及最终考核")
    manifest = _read_json(run_dir / "manifest.json")
    config = manifest["config"]
    for key, value in (("model", job["model"]), ("formula", job["formula"]),
                       ("train_seed", job["seed"]), ("budget_env_steps", plan["steps"])):
        if config[key] != value:
            raise RuntimeError(f"{job['run_id']} 清单字段 {key} 与队列不一致")
    if config["runtime"]["git_commit"] != plan["source"]["git_commit"]:
        raise RuntimeError(f"{job['run_id']} 源码版本与队列不一致")
    result = _read_json(run_dir / "evaluation" / f"final_test_{plan['steps']}.json")
    rows = _sample_rows(result)
    expected_keys = {(str(seed), str(action_seed))
                     for seed in plan["test_scenes"]
                     for action_seed in plan["action_seeds"]}
    actual_keys = {(str(row["scene_id"]), str(row["action_seed"])) for row in rows}
    if actual_keys != expected_keys or len(rows) != len(expected_keys):
        raise RuntimeError(f"{job['run_id']} 最终考核缺失或重复场景/动作种子")
    for row in rows:
        missing = set(FINAL_FIELDS) - row.keys()
        if missing:
            raise RuntimeError(f"{job['run_id']} 最终考核缺失字段：{sorted(missing)}")
    for path_key, hash_key in (("arrays_path", "arrays_sha256"),
                               ("trace_path", "trace_sha256")):
        path = result.get(path_key)
        if not path or not Path(path).exists() or sha256_file(path) != result[hash_key]:
            raise RuntimeError(f"{job['run_id']} 原始考核资产校验失败：{path_key}")
    # Every run must use the shared, saved test scenes, not a reconstructed seed alone.
    if result["scene_references"] != plan["test_scenes"]:
        raise RuntimeError(f"{job['run_id']} 考核场景引用与队列不一致")
    episodes = _read_csv(run_dir / "episodes.csv")
    evaluations = _read_csv(run_dir / "evaluations.csv")
    if not episodes:
        raise RuntimeError(f"{job['run_id']} 缺少训练逐回合指标")
    if int(float(episodes[-1]["env_step"])) > plan["steps"]:
        raise RuntimeError(f"{job['run_id']} 逐回合记录超出训练预算")
    by_key = {(str(row["scene_id"]), str(row["action_seed"])): row for row in rows}
    argmax_rows = [row for row in result["rows"] if row.get("action_mode") == "argmax"]
    if len(argmax_rows) != len(plan["test_scenes"]):
        raise RuntimeError(f"{job['run_id']} 缺少逐头最大概率动作补充考核")
    best_path = run_dir / "evaluation" / f"best_test_{plan['steps']}.json"
    best_test = _read_json(best_path) if best_path.exists() else None
    return {"job": job, "run_dir": run_dir, "status": status,
            "manifest": manifest, "result": result, "rows": rows,
            "argmax_rows": argmax_rows, "best_test": best_test,
            "by_key": by_key, "episodes": episodes,
            "evaluations": evaluations}


def _scene_means(rows, field):
    grouped = defaultdict(list)
    for row in rows:
        grouped[str(row["scene_id"])].append(_numeric(row[field]))
    return {scene: statistics.mean(values) for scene, values in grouped.items()}


def _field_mean(rows, field):
    return statistics.mean(_numeric(row[field]) for row in rows)


def _seed_stats(values):
    return {"mean": statistics.mean(values),
            "std_across_train_seeds": statistics.stdev(values) if len(values) > 1 else 0.0,
            "by_train_seed": values}


def summarize(runs):
    by_model = defaultdict(dict)
    for run in runs:
        job = run["job"]
        by_model[(job["model"], job["formula"])][job["seed"]] = run
    if len(runs) != 18 or any(len(group) != 3 for group in by_model.values()):
        raise RuntimeError("正式报告需要完整 18 组训练结果")
    variants = {}
    comparisons = {}
    for model in MODELS:
        for formula in FORMULAS:
            seed_runs = by_model[(model, formula)]
            fields = {field: _seed_stats([_field_mean(seed_runs[seed]["rows"], field)
                                          for seed in sorted(seed_runs)])
                      for field in MAIN_FIELDS}
            supplemental_fields = ("si_mean", "sgm_mean", "power_exceeded", "completed",
                                   "total_power_w", "throughput_bps")
            supplemental = {"argmax_" + field: _seed_stats([
                _field_mean(seed_runs[seed]["argmax_rows"], field)
                for seed in sorted(seed_runs)]) for field in supplemental_fields}
            if all(seed_runs[seed]["best_test"] for seed in seed_runs):
                for field in supplemental_fields:
                    supplemental["best_validation_selected_test_" + field] = _seed_stats([
                        _field_mean(_sample_rows(seed_runs[seed]["best_test"]), field)
                        for seed in sorted(seed_runs)])
            variants[f"{model}_{formula}"] = {
                "model": model, "formula": formula,
                "train_seeds": sorted(seed_runs), "metrics": fields,
                "supplemental": supplemental,
                "run_ids": [seed_runs[seed]["job"]["run_id"] for seed in sorted(seed_runs)],
            }
        paired = {}
        for field in MAIN_FIELDS:
            seed_differences = []
            scene_wins = []
            for seed in sorted(by_model[(model, "legacy")]):
                old = by_model[(model, "legacy")][seed]
                new = by_model[(model, "corrected")][seed]
                if old["by_key"].keys() != new["by_key"].keys():
                    raise RuntimeError(f"{model} 种子 {seed} 的配对场景不同")
                differences = [_numeric(new["by_key"][key][field])
                               - _numeric(old["by_key"][key][field])
                               for key in sorted(old["by_key"])]
                seed_differences.append(statistics.mean(differences))
                old_scenes = _scene_means(old["rows"], field)
                new_scenes = _scene_means(new["rows"], field)
                scene_wins.append(sum(new_scenes[scene] > old_scenes[scene]
                                      for scene in old_scenes))
            paired[field] = {**_seed_stats(seed_differences),
                             "improved_scenes_by_train_seed": scene_wins,
                             "scenes_per_train_seed": 100}
        comparisons[model] = paired
    return {"variants": variants, "paired_corrected_minus_legacy": comparisons}


def _fmt(value, digits=2):
    return f"{value:.{digits}f}"


def _percent(value):
    return 100 * value


def _variant_table(summary):
    lines = ["| 算法与公式 | 训练种子 | 终态 SI / % | SGM / % | 总吞吐量 / Gbps | 超功率 / % | 完整分配 / % | 总功率 / W | 实际槽数 | 频谱效率 / bit·s⁻¹·Hz⁻¹ | 干扰波束数 | 近距离干扰波束数 | 频谱占用 / % | 零槽 / % |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for model in MODELS:
        for formula in FORMULAS:
            variant = summary["variants"][f"{model}_{formula}"]
            m = variant["metrics"]
            seed_text = ", ".join(str(seed) for seed in variant["train_seeds"])
            def metric(field, scale=1):
                stat = m[field]
                return f"{_fmt(stat['mean'] * scale)} ± {_fmt(stat['std_across_train_seeds'] * scale)}"
            lines.append("| " + " | ".join((
                f"{MODEL_NAMES[model]} · {FORMULA_NAMES[formula]}", seed_text,
                metric("si_mean", 100), metric("sgm_mean", 100),
                metric("throughput_bps", 1e-9), metric("power_exceeded", 100),
                metric("completed", 100), metric("total_power_w"),
                metric("allocated_slots"), metric("spectral_efficiency"),
                metric("interference_beams"), metric("in_range_interference_beams"),
                metric("occupancy_fraction", 100), metric("zero_slot_rate", 100),
            )) + " |")
    return "\n".join(lines)


def _paired_table(summary):
    lines = ["| 算法 | SI 差 / 百分点 | SGM 差 / 百分点 | 吞吐量差 / Gbps | 超功率率差 / 百分点 | 完整分配率差 / 百分点 | SI 改善场景数（各训练种子 / 100） |",
             "| --- | ---: | ---: | ---: | ---: | ---: | --- |"]
    for model in MODELS:
        m = summary["paired_corrected_minus_legacy"][model]
        def diff(field, scale=1):
            item = m[field]
            return f"{_fmt(item['mean'] * scale)} ± {_fmt(item['std_across_train_seeds'] * scale)}"
        lines.append("| " + " | ".join((
            MODEL_NAMES[model], diff("si_mean", 100), diff("sgm_mean", 100),
            diff("throughput_bps", 1e-9), diff("power_exceeded", 100),
            diff("completed", 100),
            " / ".join(str(count) for count in m["si_mean"]["improved_scenes_by_train_seed"]),
        )) + " |")
    return "\n".join(lines)


def _supplemental_table(summary):
    lines = ["| 算法与公式 | 模型及动作选择 | SI / % | SGM / % | 超功率 / % | 完整分配 / % | 总功率 / W | 吞吐量 / Gbps |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for model in MODELS:
        for formula in FORMULAS:
            extra = summary["variants"][f"{model}_{formula}"]["supplemental"]
            def cell(key, scale=100):
                stat = extra.get(key)
                if stat is None:
                    return "未保存"
                return f"{_fmt(scale * stat['mean'])} ± {_fmt(scale * stat['std_across_train_seeds'])}"
            for prefix, label in (("argmax_", "预算终点；逐头最大概率"),
                                  ("best_validation_selected_test_", "按验证选择；采样动作")):
                lines.append("| " + " | ".join((
                    f"{MODEL_NAMES[model]} · {FORMULA_NAMES[formula]}", label,
                    cell(prefix + "si_mean"), cell(prefix + "sgm_mean"),
                    cell(prefix + "power_exceeded"), cell(prefix + "completed"),
                    cell(prefix + "total_power_w", 1), cell(prefix + "throughput_bps", 1e-9),
                )) + " |")
    return "\n".join(lines)


def _plot_curves(directory, runs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    figure_dir = directory / "figures"
    figure_dir.mkdir(exist_ok=True)
    paths = []
    for model in MODELS:
        fig, grid = plt.subplots(2, 2, figsize=(13, 8))
        axes = grid.ravel()
        for run in runs:
            job = run["job"]
            if job["model"] != model:
                continue
            label = f"{FORMULA_NAMES[job['formula']]}，种子 {job['seed']}"
            color = "tab:blue" if job["formula"] == "legacy" else "tab:orange"
            x = np.array([float(row["env_step"]) for row in run["episodes"]])
            y = np.array([float(row["si_mean"]) * 100 for row in run["episodes"]])
            axes[0].plot(x, y, color=color, alpha=0.12, linewidth=0.7)
            window = min(50, len(y))
            if window:
                smoothed = np.convolve(y, np.ones(window) / window, mode="valid")
                axes[0].plot(x[window - 1:], smoothed, color=color,
                             label=label, linewidth=1.3, alpha=0.8)
            for ax, field, scale in ((axes[1], "si_mean", 100),
                                     (axes[2], "return", 1),
                                     (axes[3], "throughput_bps", 1e-9)):
                grouped = defaultdict(list)
                for row in run["evaluations"]:
                    if (row["checkpoint"] == "validation" and
                            row.get("action_mode", "sample") == "sample" and
                            row.get(field) not in (None, "")):
                        grouped[int(row["env_step"])].append(float(row[field]) * scale)
                if grouped:
                    steps = sorted(grouped)
                    ax.plot(steps, [statistics.mean(grouped[step]) for step in steps],
                            color=color, label=label, linewidth=1.3, alpha=0.8)
        labels = (("训练回合最终需求满足率", "需求满足率 SI（%）"),
                  ("固定验证场景需求满足率", "需求满足率 SI（%）"),
                  ("固定验证场景累计奖励", "平均回合累计奖励"),
                  ("固定验证场景总吞吐量", "平均总吞吐量（Gbps）"))
        for ax, (title, ylabel) in zip(axes, labels):
            ax.set_title(title)
            ax.set_xlabel("环境交互步数")
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.25)
            if ax.lines:
                ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        path = figure_dir / f"{model}_si_curves.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(path)
        # 使用实际保存的回报和学习记录，补充训练回报、温度、熵和损失曲线。
        fig, axes = plt.subplots(3, 3, figsize=(16, 12))
        for run in runs:
            job = run['job']
            if job['model'] != model:
                continue
            label = f"{FORMULA_NAMES[job['formula']]}，种子 {job['seed']}"
            color = 'tab:blue' if job['formula'] == 'legacy' else 'tab:orange'
            x = np.array([float(row['env_step']) for row in run['episodes']])
            y = np.array([float(row['return']) for row in run['episodes']])
            axes[0, 0].plot(x, y, color=color, alpha=.1, linewidth=.6)
            window = min(50, len(y))
            if window:
                axes[0, 0].plot(x[window-1:], np.convolve(y, np.ones(window)/window, 'valid'),
                                color=color, label=label, alpha=.8)
            updates_path = run['run_dir'] / 'updates.csv'
            updates = _read_csv(updates_path) if updates_path.exists() else []
            for ax, key in ((axes[0, 1], 'alpha'), (axes[0, 2], 'entropy'),
                            (axes[1, 0], 'entropy_freq'), (axes[1, 1], 'entropy_slots'),
                            (axes[1, 2], 'entropy_power'), (axes[2, 0], 'actor_loss'),
                            (axes[2, 1], 'critic_1_loss'), (axes[2, 2], 'critic_2_loss')):
                valid = [row for row in updates if row.get(key) not in ('', None)]
                if valid:
                    ax.plot([float(row['env_step']) for row in valid],
                            [float(row[key]) for row in valid], color=color, label=label, alpha=.75)
        titles = ('回合累计奖励', '温度系数 α', '动作分布总熵', '频率起点分布熵',
                  '槽数分布熵', '功率分布熵', '策略网络损失', '价值网络一损失', '价值网络二损失')
        for ax, title in zip(axes.flat, titles):
            ax.set_title(title)
            ax.set_xlabel('环境交互步数')
            ax.grid(alpha=.25)
            if ax.lines:
                ax.legend(fontsize=7)
        fig.tight_layout()
        path = figure_dir / f'{model}_learning_curves.png'
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(path)
    return paths


def _behavior_from_traces(runs):
    import gzip
    result = {}
    for model in MODELS:
        for formula in FORMULAS:
            requested_slots = defaultdict(int)
            actual_slots = defaultdict(int)
            power_w = defaultdict(int)
            starts = defaultdict(int)
            steps = 0
            clipped = 0
            for run in runs:
                job = run["job"]
                if job["model"] != model or job["formula"] != formula:
                    continue
                with gzip.open(run["result"]["trace_path"], "rt", encoding="utf-8") as stream:
                    for line in stream:
                        item = json.loads(line)
                        if item.get("action_mode", "sample") != "sample":
                            continue
                        requested = item["requested"]
                        actual = item["actual"]
                        steps += 1
                        requested_slots[int(requested[2])] += 1
                        actual_slots[int(actual[2])] += 1
                        power_w[round(10 ** (float(requested[3]) / 10), 3)] += 1
                        starts[int(requested[1])] += 1
                        clipped += requested[2] != actual[2]
            result[f"{model}_{formula}"] = {
                "steps": steps, "clipped_rate": clipped / steps if steps else None,
                "requested_slots": dict(sorted(requested_slots.items())),
                "actual_slots": dict(sorted(actual_slots.items())),
                "requested_power_w": dict(sorted(power_w.items())),
                "requested_start": dict(sorted(starts.items())),
            }
    return result


def _behavior_table(behavior):
    lines = ["| 算法与公式 | 请求功率最多的档位 / W | 请求槽数最多的档位 | 被环境裁剪的动作 / % |",
             "| --- | ---: | ---: | ---: |"]
    for model in MODELS:
        for formula in FORMULAS:
            item = behavior[f"{model}_{formula}"]
            power = max(item["requested_power_w"], key=item["requested_power_w"].get)
            slots = max(item["requested_slots"], key=item["requested_slots"].get)
            lines.append(f"| {MODEL_NAMES[model]} · {FORMULA_NAMES[formula]} | "
                         f"{power} | {slots} | {_fmt(_percent(item['clipped_rate']))} |")
    return "\n".join(lines)


def generate_report(directory):
    directory = Path(directory).resolve()
    plan = _read_json(directory / "suite_plan.json")
    if len(plan["jobs"]) != 18 or any(job["state"] != "completed" for job in plan["jobs"]):
        raise RuntimeError("18 组作业尚未全部完成，不能生成正式报告")
    for group in ("validation_scenes", "test_scenes"):
        for reference in plan[group].values():
            if (not Path(reference["path"]).exists() or
                    sha256_file(reference["path"]) != reference["sha256"]):
                raise RuntimeError(f"固定场景文件校验失败：{reference['path']}")
    runs = [_load_run(directory, job, plan) for job in plan["jobs"]]
    summary = summarize(runs)
    behavior = _behavior_from_traces(runs)
    curves = _plot_curves(directory, runs)
    payload = {"generated_at": utc_now(), "experiment_id": plan["experiment_id"],
               "source_commit": plan["source"]["git_commit"],
               "source_root": plan["source"]["source_root"],
               "report_script_path": str(Path(__file__).resolve()),
               "report_script_sha256": sha256_file(Path(__file__)),
               "steps_per_run": plan["steps"], "training_runs": 18,
               "validation_scenes": len(plan["validation_scenes"]),
               "test_scenes": len(plan["test_scenes"]),
               "action_seeds": plan["action_seeds"],
               "results": summary, "behavior": behavior,
               "curves": [str(path) for path in curves]}
    atomic_json(directory / "summary.json", payload)
    lines = [
        "# SAC 学习公式与网络对照实验报告",
        "",
        f"生成时间：{payload['generated_at']}。训练源码提交：`{payload['source_commit']}`。",
        "",
        "本实验在修正后的卫星环境中比较 Transformer SAC、参数量接近的 CNN SAC、参数量接近的 MLP SAC，"
        "每种网络分别使用原学习公式和修正学习公式，各训练 3 个独立种子。每组固定 400000 次环境交互。"
        "以下均为训练后的独立最终考核，不以考核结果挑选模型。",
        "",
        "**术语：**SI 是全部目标波束的终态需求满足率；未服务波束计零，后续干扰造成的速率变化也已计入。"
        "SGM 是供需匹配程度，资源给得过多也会扣分。两项指标分别报告。",
        "",
        "## 共同条件与判断口径",
        "",
        f"训练种子：{', '.join(str(seed) for seed in sorted({job['seed'] for job in plan['jobs']}))}。"
        f"验证场景 {len(plan['validation_scenes'])} 个，独立考核场景 {len(plan['test_scenes'])} 个；"
        f"每个场景的采样动作种子为 {plan['action_seeds']}。三次动作采样属于同一场景的重复观察，"
        "不能算作三倍独立场景。图和表来自原始逐回合与逐场景记录。",
        "",
        "每个单元格为 3 个训练种子的平均值 ± 种子间样本标准差。超功率、完整分配、总功率与成绩并列，"
        "不能只凭单项 SI 判定整体方案更好。训练回报用于观察学习过程，主比较采用相同环境中的终态服务与资源指标。",
        "单个 Actor/Critic 参数量：Transformer 417272，CNN 447176（多约7.2%），MLP 429176（多约2.9%）；属于近似匹配。",
        "",
        "## 最终考核成绩与资源约束",
        "",
        _variant_table(summary),
        "",
        "本表频谱效率为总吞吐量除以各组实际占用槽数之和与 25 MHz 的乘积，"
        "表示已分配带宽上的平均频谱效率，并非按整个系统频带计算的频谱效率。"
        "干扰波束数是分配该束时检出与历史波束存在同极化、同频占用的次数（每束最多计一次）；"
        "近距离干扰还要求在环境既有的波束地面覆盖距离内。它们不是干扰强度或终态受扰波束总数。",
        "",
        "## 同网络下的配对差值",
        "",
        "差值定义为“修正公式 − 原公式”；每个训练种子先在相同场景和动作种子上配对，"
        "再统计三个训练种子。表中“改善场景数”按每场景三个动作种子的均值判断，"
        "依次列出三个训练种子，不能把这些计数看成独立训练次数。",
        "",
        _paired_table(summary),
        "",
        "## 补充考核：动作选法与模型保存点",
        "",
        "主表使用预先固定的采样动作规则和预算终点模型。下表另外报告每个动作头取最大概率的确定选择，"
        "以及仅依据日常验证场景选出的保存模型；这些结果不替代主表。",
        "",
        _supplemental_table(summary),
        "",
        "## 策略选择与环境裁剪",
        "",
        _behavior_table(behavior),
        "",
        "功率、请求槽数、实际槽数和频率起点的完整分布在 `summary.json` 的 `behavior` 中。"
        "环境裁剪使用请求动作与实际动作逐步比较；裁剪后的动作不算作模型主动选对。",
        "",
        "## 训练与日常验证曲线",
        "",
        "每种网络的首张图包含训练终态 SI（左上）、验证 SI（右上）、验证累计奖励（左下）及验证总吞吐量（右下）。"
        "训练 SI 的淡线是逐回合原始值，深线为 50 回合滑动平均；验证数据为各固定场景和动作采样的原始均值。"
        "第二张图显示训练回报、温度、各动作分布熵与损失。平滑只用于观察趋势，终态成绩使用未经平滑的独立考核数据。",
        "",
    ]
    for path in curves:
        lines.extend((f"![{path.stem}](figures/{path.name})", ""))
    lines.extend((
        "## 解释范围与可追溯文件", "",
        "这份报告给出实际观察到的差值及种子间波动；不预设修正公式一定提高 SI。"
        "三种网络即便使用相近规模，也可能有不同学习速度和推理开销；如需断言某一网络在所有场景更好，"
        "还应结合图中收敛过程、每种压力场景及运行耗时检查。", "",
        "每组运行目录保留 `manifest.json`、`status.json`、训练 `episodes.csv`/`updates.csv`、"
        "逐场景 `evaluations.csv`、最终考核 JSON/NPZ/逐步压缩日志、场景数组及模型断点。"
        "本报告的数值摘要为 `summary.json`；运行状态为 `suite_status.json`。", "",
    ))
    path = directory / "中文实验对照报告.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def generate_progress_report(directory):
    directory = Path(directory)
    status = _read_json(directory / "suite_status.json")
    lines = ["# SAC 对照实验阶段进度", "",
             f"更新时间：{status['updated_at']}。已完成 {status['counts']['completed']}/18，"
             f"运行中 {status['counts']['running']}，待运行 {status['counts']['pending']}，"
             f"失败 {status['counts']['failed']}。", "",
             "这是一份运行状态记录。18 组独立训练及最终考核完成前，效果待验证。", "",
             "| 作业 | 阶段 | 环境交互步 | 尝试次数 | 最近状态时间 |", "| --- | --- | ---: | ---: | --- |"]
    for job in status["jobs"]:
        lines.append(f"| {job['run_id']} | {job['stage'] or job['state']} | "
                     f"{job['env_step']}/{status['budget_env_steps_per_job']} | "
                     f"{job['attempts']} | {job['heartbeat_at'] or '—'} |")
    path = directory / "阶段进度.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main():
    parser = argparse.ArgumentParser(description="SAC 实验结果中文报告")
    parser.add_argument("suite_dir", type=Path)
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()
    print(generate_progress_report(args.suite_dir) if args.progress
          else generate_report(args.suite_dir))


if __name__ == "__main__":
    main()
