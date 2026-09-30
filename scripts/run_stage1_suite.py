"""Persistent seven-group controller for the first-stage workbench.

Run on the remote host against its localhost API.  All training and evaluation
is submitted through the workbench; this process only coordinates durable jobs
and archives their raw records.  ``--self-check`` never contacts the server.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen


TRAIN = {"R2": "rule", "C2": "cluster", "P0": "empty"}
EVALUATE = {"R0": ("rule", "none"), "R1": ("rule", "local_search"),
            "C0": ("cluster", "none"), "C1": ("cluster", "local_search"),
            "R2": ("rule", "ppo"), "C2": ("cluster", "ppo"),
            "P0": ("empty", "ppo")}
ROLES = ("train", "validation", "test")


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_bytes(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(value).hexdigest()
    if path.exists():
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"已归档原始证据内容变化，保留现场：{path}")
        return digest
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    try:
        with temporary.open("wb") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return digest


@contextmanager
def controller_lock(output):
    if os.name != "posix":
        raise RuntimeError("正式控制器必须在远端 Linux 主机运行")
    import fcntl
    path = output / "controller.lock"
    with path.open("a+", encoding="utf-8") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("已有同一套实验控制器在运行") from error
        stream.seek(0)
        stream.truncate()
        stream.write(json.dumps({"pid": os.getpid(), "started_at": now()}))
        stream.flush()
        yield


def api(base, path, body=None):
    url = base.rstrip("/") + "/api/" + path.lstrip("/")
    data = None if body is None else json.dumps(body, ensure_ascii=False).encode("utf-8")
    request = Request(url, data=data,
                      headers={"Content-Type": "application/json"} if data is not None else {})
    for attempt in range(3):
        try:
            with urlopen(request, timeout=60) as response:
                return json.load(response)
        except HTTPError as error:
            detail = error.read().decode("utf-8", "replace")[:1000]
            if error.code < 500 or attempt == 2:
                raise RuntimeError(f"工作台 HTTP {error.code}: {detail}") from error
        except (URLError, TimeoutError, OSError):
            if attempt == 2:
                raise
        time.sleep(2 ** attempt)
    raise AssertionError("unreachable")


def download(base, run_id, filename):
    query = urlencode({"file": filename})
    url = (base.rstrip("/") + "/api/runs/" + quote(run_id, safe="") +
           "/download?" + query)
    with urlopen(Request(url), timeout=120) as response:
        return response.read()


def json_lines(data, label):
    rows = []
    for line_number, line in enumerate(data.decode("utf-8-sig").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError as error:
            raise ValueError(f"{label}:{line_number} 不是完整JSON记录") from error
        if not isinstance(row, dict):
            raise ValueError(f"{label}:{line_number} 记录须为对象")
        rows.append(row)
    return rows


def config(plan, initializer, method, name):
    result = deepcopy(plan["config"])
    result["name"] = name
    result["coverage"].update(initialization=initializer, method=method,
                              sweeps=1 if initializer == "empty" else 2)
    return result


def plan_key(plan):
    canonical = json.dumps(plan, ensure_ascii=False, sort_keys=True,
                           separators=(",", ":")).encode("utf-8")
    return "S1_01_" + hashlib.sha256(canonical).hexdigest()[:16]


def check_plan(base, plan, *, repo):
    if not isinstance(plan, dict) or not isinstance(plan.get("config"), dict):
        raise ValueError("实验方案缺少冻结配置")
    training = plan["config"].get("training", {})
    network = plan["config"].get("network", {})
    if (plan.get("experiment_id") != "S1-01" or
            plan.get("selection") != "fixed_budget_final" or
            plan.get("max_workers") != 2 or
            training.get("seed") != 51001 or training.get("total_steps") != 409600 or
            training.get("num_envs") != 512 or training.get("rollout_steps") != 16 or
            training.get("reward_scale") != 0.01 or training.get("gamma") != 1.0 or
            network.get("kind") != "coverage_candidate_mlp"):
        raise ValueError("方案偏离已冻结的S1-01三PPO共同训练条件")
    commit = plan.get("source_commit")
    if not isinstance(commit, str) or len(commit) != 40 or any(
            character not in "0123456789abcdef" for character in commit):
        raise ValueError("source_commit须为完整小写40位提交号")
    actual_commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"],
                                            text=True).strip()
    if actual_commit != commit:
        raise ValueError(f"当前工作台源码提交{actual_commit}与方案{commit}不一致")
    dirty = subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain",
                                     "--", "src", "data", "pyproject.toml"], text=True)
    if dirty.strip():
        raise ValueError("训练源码仍有未提交差异，不能冻结正式任务")
    datasets = plan.get("datasets")
    if not isinstance(datasets, dict) or set(datasets) != set(TRAIN.values()):
        raise ValueError("必须提供规则、聚类、空构建三套数据集编号")
    catalog = api(base, "coverage/datasets")
    records = {record["id"]: record for record in catalog["datasets"]}
    selected = {}
    for initializer in TRAIN.values():
        ids = datasets.get(initializer)
        if not isinstance(ids, dict) or set(ids) != set(ROLES):
            raise ValueError(f"{initializer}须分别绑定train/validation/test")
        selected[initializer] = {}
        for role, identifier in ids.items():
            record = records.get(identifier)
            if (record is None or record.get("research_stage") != "coverage" or
                    record.get("initialization") != initializer or
                    record.get("partition_role") != role or
                    not record.get("sha256") or not record.get("dates") or
                    not isinstance(record.get("scenes"), list) or not record["scenes"]):
                raise ValueError(f"{initializer}/{role}数据编号、来源或用途不完整")
            if role == "train" and record.get("failed_initializations"):
                raise ValueError(f"{initializer}训练集有初始化失败场景，训练入口会拒绝")
            selected[initializer][role] = record
        dates = {role: set(record["dates"]) for role, record in selected[initializer].items()}
        if any(dates[left] & dates[right] for left in ROLES for right in ROLES if left < right):
            raise ValueError(f"{initializer}训练/验证/考核航空日期重叠")
    for role in ROLES:
        fingerprints = [[(scene["scene_id"], scene["source_hash"])
                         for scene in selected[initializer][role]["scenes"]]
                        for initializer in TRAIN.values()]
        if not (fingerprints[0] == fingerprints[1] == fingerprints[2]):
            raise ValueError(f"{role}三组原始场景编号/SHA或顺序不一致")
    for label, (initializer, method) in EVALUATE.items():
        response = api(base, "coverage/validate", config(plan, initializer, method, label))
        if (response["config"]["coverage"]["initialization"] != initializer or
                response["config"]["coverage"]["method"] != method):
            raise ValueError("工作台验证未保留基础布局或方法配置")
    return selected


def check_job(base, job, *, commit, kind, initializer, method, request_id):
    if (job.get("git_commit") != commit or job.get("kind") != kind or
            job.get("research_stage") != "coverage" or
            job.get("experiment_role") != "formal" or
            job.get("request_id") != request_id):
        raise ValueError("工作台返回的冻结任务版本、类型、用途或幂等编号不一致")
    detail = api(base, "runs/" + quote(job["run_id"], safe=""))
    frozen = detail["config"]
    if (detail["job"]["git_commit"] != commit or
            frozen["coverage"]["initialization"] != initializer or
            frozen["coverage"]["method"] != method or
            frozen["research_stage"] != "coverage" or
            frozen["schema_version"] != 2 or
            frozen["training"]["seed"] != 51001):
        raise ValueError("工作台冻结配置与预定科学组不一致")
    return detail


def await_jobs(base, jobs, output, phase):
    while True:
        states = {label: api(base, "runs/" + quote(job["run_id"], safe=""))["status"]
                  for label, job in jobs.items()}
        atomic(output / "suite_status.json", {"phase": phase, "updated_at": now(),
                                               "states": states})
        failures = {label: status for label, status in states.items()
                    if status.get("stage") in ("failed", "interrupted", "stopped", "cancelled")}
        if failures:
            raise RuntimeError("工作台任务需要修复或恢复：" +
                               json.dumps(failures, ensure_ascii=False))
        if all(status.get("stage") == "completed" for status in states.values()):
            return
        time.sleep(5)


def check_metric_series(payload):
    if not isinstance(payload, dict) or not isinstance(payload.get("series"), list):
        raise ValueError("工作台指标API没有series数组")
    tags = {item.get("tag"): item for item in payload["series"]}
    for required in ("terminal/episode_return", "validation/coverage_demand",
                     "ppo_update/value_loss", "ppo_update/entropy"):
        item = tags.get(required)
        if not item or not item.get("points") or item.get("raw_count", 0) < 1:
            raise ValueError(f"训练任务缺少工作台可见曲线：{required}")
        if any("step" not in point or "value" not in point for point in item["points"]):
            raise ValueError(f"工作台曲线点格式错误：{required}")


def archive_run(base, job, output, *, train):
    run_id = job["run_id"]
    directory = output / "raw" / run_id
    names = (["episodes.jsonl", "updates.jsonl", "metrics.jsonl", "events.jsonl",
              "config.json", "job.json", "status.json", "artifacts.json"] if train else
             ["evaluations.jsonl", "events.jsonl", "config.json", "job.json",
              "status.json", "results.json", "artifacts.json"])
    digests = {name: atomic_bytes(directory / name, download(base, run_id, name))
               for name in names}
    if train:
        episodes = json_lines((directory / "episodes.jsonl").read_bytes(), "episodes.jsonl")
        updates = json_lines((directory / "updates.jsonl").read_bytes(), "updates.jsonl")
        metrics = json_lines((directory / "metrics.jsonl").read_bytes(), "metrics.jsonl")
        if not episodes or not updates or not any(row.get("group") == "validation" for row in metrics):
            raise ValueError(f"{run_id}缺少原始回合/更新/验证数据")
        if not all(isinstance(row.get("metrics", {}).get("episode_return"), (int, float))
                   for row in episodes):
            raise ValueError(f"{run_id}原始回合缺少学习回报")
        ui_metrics = api(base, "runs/" + quote(run_id, safe="") + "/metrics?limit=20000")
        check_metric_series(ui_metrics)
        atomic(directory / "metrics_api.json", ui_metrics)
    atomic(directory / "checksums.json", digests)
    return directory


def check_results(results, selected, commit):
    if set(results) != set(EVALUATE):
        raise ValueError("共同考核不是预定七组")
    expected = [(scene["scene_id"], scene["source_hash"])
                for scene in selected["rule"]["test"]["scenes"]]
    for label, result in results.items():
        rows = result.get("scenes")
        summary = result.get("summary")
        if (not isinstance(rows, list) or len(rows) != len(expected) or
                not isinstance(summary, dict) or summary.get("scene_count") != len(expected) or
                result.get("metadata", {}).get("git_commit") != commit):
            raise ValueError(f"{label}考核结果不完整或来源版本不一致")
        observed = [(row.get("scene_id"), row.get("scene_sha256")) for row in rows]
        if observed != expected:
            raise ValueError(f"{label}逐场景来源、顺序或SHA与共同考核输入不同")
        if summary.get("failed_scene_count", 0) + summary.get("valid_scene_count", 0) != len(rows):
            raise ValueError(f"{label}失败场景没有纳入汇总分母")


def _numeric_points(rows, key, *, nested=None):
    points = []
    for row in rows:
        values = row.get(nested, {}) if nested else row
        value = values.get(key) if isinstance(values, dict) else None
        if isinstance(value, (int, float)) and isinstance(row.get("env_step", row.get("step")), (int, float)):
            points.append((row.get("env_step", row.get("step")), value))
    return points


def plot_curves(training, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for label, job in training.items():
        directory = output / "raw" / job["run_id"]
        episodes = json_lines((directory / "episodes.jsonl").read_bytes(), "episodes.jsonl")
        updates = json_lines((directory / "updates.jsonl").read_bytes(), "updates.jsonl")
        metrics = json_lines((directory / "metrics.jsonl").read_bytes(), "metrics.jsonl")
        validation = [row for row in metrics if row.get("group") == "validation"]
        groups = [
            ("Learning return (raw episodes)", _numeric_points(episodes, "episode_return", nested="metrics")),
            ("Terminal demand coverage (raw episodes)", _numeric_points(episodes, "coverage_demand", nested="metrics")),
            ("Validation demand coverage (scene mean)", [(row["step"], row["values"]["coverage_demand"])
                                          for row in validation if "coverage_demand" in row.get("values", {})]),
            ("Value loss", _numeric_points(updates, "value_loss")),
            ("Policy entropy", _numeric_points(updates, "entropy")),
        ]
        if any(not points for _, points in groups):
            raise ValueError(f"{label}原始曲线缺少回报、覆盖、验证或学习诊断")
        fig, axes = plt.subplots(len(groups), 1, figsize=(11, 12), squeeze=False)
        for axis, (title, points) in zip(axes[:, 0], groups):
            axis.plot([point[0] for point in points], [point[1] for point in points],
                      linewidth=0.6, alpha=0.75)
            axis.set_title(f"{label} / {title} / seed 51001")
            axis.set_xlabel("Environment interactions")
            axis.grid(alpha=0.2)
        fig.tight_layout()
        destination = output / "curves" / (label + "_raw_curves.png")
        destination.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(destination, dpi=160)
        plt.close(fig)


def plot_results(results, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = list(EVALUATE)
    figure, axes = plt.subplots(1, 3, figsize=(14, 4))
    for axis, key, title in zip(axes, ("coverage_demand", "load_cv", "feasible"),
                                ("Demand coverage (all scenes)", "Load CV (valid layouts)",
                                 "Feasible fraction")):
        values = [results[label]["summary"].get(key, float("nan")) for label in labels]
        axis.bar(labels, values)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(output / "summary.png", dpi=160)
    plt.close(figure)

    distributions = []
    for label in labels:
        values = [float(row["metrics"]["coverage_demand"])
                  if row.get("metrics") is not None and
                  row.get("input", {}).get("metadata", {}).get("initialization_valid") is not False
                  else 0.0 for row in results[label]["scenes"]]
        distributions.append(values)
    figure, axis = plt.subplots(figsize=(11, 4))
    axis.boxplot(distributions, labels=labels, showfliers=True)
    axis.set_title("Paired test demand coverage; failed initialization = 0")
    axis.set_ylabel("Demand coverage")
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(output / "scene_distribution.png", dpi=160)
    plt.close(figure)


def write_report(results, plan, output):
    lines = ["# 第一阶段七组覆盖规划自动汇总", "",
             f"冻结源码：`{plan['source_commit']}`。训练种子51001；三种PPO每组{plan['config']['training']['total_steps']}交互步。",
             "最终固定预算权重用于共同考核；此处仅按保存结果汇总，学习效果结论须独立逐场景审查。",
             "覆盖与可行比例包含所有场景及初始化失败；负载只在可计算布局上平均。P0移动项不适用。",
             "", "|组|需求覆盖|用户覆盖|负载CV|可行比例|初始化失败|",
             "|---|---:|---:|---:|---:|---:|"]
    for label in EVALUATE:
        summary = results[label]["summary"]
        lines.append(f"|{label}|{summary['coverage_demand']:.6f}|{summary['coverage_users']:.6f}|"
                     f"{summary.get('load_cv', float('nan')):.6f}|{summary['feasible']:.6f}|"
                     f"{summary['failed_scene_count']}|")
    lines.extend(["", "![七组汇总](summary.png)", "",
                  "![逐场景考核分布](scene_distribution.png)", "",
                  "## 训练原始曲线", ""])
    for label in TRAIN:
        lines.append(f"![{label}训练/验证原始曲线](curves/{label}_raw_curves.png)")
    lines.extend(["", "P0与微调组的动作数、回合长度和移动项适用范围不同，"
                  "差异不能纯归因于初始化。三个模型共用候选评分网络结构和学习奖励缩放0.01，"
                  "但P0合法动作数更多，策略熵量级不同。单训练种子、冬季来源和合成业务需求限制保留。"])
    (output / "自动汇总报告.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def self_check():
    example = {"config": {"training": {"seed": 51001}}, "source_commit": "a" * 40}
    if plan_key(example) != plan_key(dict(reversed(list(example.items())))):
        raise AssertionError("request ID depends on JSON key insertion order")
    rows = json_lines(b'{"env_step":8,"metrics":{"episode_return":0.4}}\n', "example")
    if _numeric_points(rows, "episode_return", nested="metrics") != [(8, 0.4)]:
        raise AssertionError("nested episode return was lost")
    metrics = {"series": [{"tag": tag, "raw_count": 1,
                           "points": [{"step": 8, "value": 0.4}]}
                          for tag in ("terminal/episode_return", "validation/coverage_demand",
                                      "ppo_update/value_loss", "ppo_update/entropy")]}
    check_metric_series(metrics)
    try:
        check_metric_series({"series": []})
    except ValueError:
        pass
    else:
        raise AssertionError("empty workbench curves were accepted")
    selected = {initializer: {"test": {"scenes": [{"scene_id": "scene",
                                                   "source_hash": "sha"}]}}
                for initializer in TRAIN.values()}
    result = {"summary": {"scene_count": 1, "failed_scene_count": 0,
                          "valid_scene_count": 1},
              "scenes": [{"scene_id": "scene", "scene_sha256": "sha"}],
              "metadata": {"git_commit": "a" * 40}}
    results = {label: deepcopy(result) for label in EVALUATE}
    check_results(results, selected, "a" * 40)
    results["P0"]["scenes"][0]["scene_sha256"] = "wrong"
    try:
        check_results(results, selected, "a" * 40)
    except ValueError:
        pass
    else:
        raise AssertionError("mismatched P0 source scene was accepted")
    print(json.dumps({"status": "passed", "checks": ["idempotent_key", "raw_episode_reader",
                                                        "metrics_api_schema", "missing_curves_rejected",
                                                        "seven_group_pairing"]}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--base", help="远端工作台 localhost 地址，正式运行必须显式指定")
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--check-only", action="store_true", help="仅核对源码与数据，不提交任务")
    args = parser.parse_args()
    if args.self_check:
        self_check()
        return
    if not args.plan or not args.output or not args.base:
        parser.error("正式/只读预检均须提供--plan、--output、--base")
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parents[1]
    with controller_lock(output):
        selected = check_plan(args.base, plan, repo=repo)
        if args.check_only:
            atomic(output / "preflight.json", {"checked_at": now(), "plan_key": plan_key(plan),
                                                "source_commit": plan["source_commit"],
                                                "datasets": {method: {role: record["id"]
                                                    for role, record in roles.items()}
                                                    for method, roles in selected.items()}})
            return
        manifest = output / "run_manifest.json"
        state = json.loads(manifest.read_text(encoding="utf-8")) if manifest.exists() else {
            "experiment_id": plan.get("experiment_id", "S1-01"), "created_at": now(),
            "plan": plan, "plan_key": plan_key(plan), "training": {}, "evaluation": {}}
        if state["plan"] != plan or state.get("plan_key") != plan_key(plan):
            raise ValueError("既有实验计划与本次输入不同，不能覆盖")
        atomic(manifest, state)
        try:
            for label, initializer in TRAIN.items():
                request_id = plan_key(plan) + "_" + label + "_train"
                body = {"kind": "train", "experiment_role": "formal",
                        "config": config(plan, initializer, "ppo", f"第一阶段/{label}"),
                        "dataset_ids": {role: plan["datasets"][initializer][role]
                                        for role in ("train", "validation")},
                        "request_id": request_id}
                job = api(args.base, "coverage/runs", body)
                check_job(args.base, job, commit=plan["source_commit"], kind="train",
                          initializer=initializer, method="ppo", request_id=request_id)
                state["training"][label] = job
                atomic(manifest, state)
            await_jobs(args.base, state["training"], output, "training")
            for label, (initializer, method) in EVALUATE.items():
                request_id = plan_key(plan) + "_" + label + "_test"
                body = {"kind": "evaluate", "experiment_role": "formal",
                        "config": config(plan, initializer, method, f"第一阶段/{label}共同考核"),
                        "dataset_ids": {"test": plan["datasets"][initializer]["test"]},
                        "request_id": request_id}
                if method == "ppo":
                    body["model_id"] = state["training"][label]["run_id"]
                job = api(args.base, "coverage/runs", body)
                check_job(args.base, job, commit=plan["source_commit"], kind="evaluate",
                          initializer=initializer, method=method, request_id=request_id)
                state["evaluation"][label] = job
                atomic(manifest, state)
            await_jobs(args.base, state["evaluation"], output, "evaluation")
            results = {label: api(args.base, "runs/" + quote(job["run_id"], safe="") + "/results")
                       for label, job in state["evaluation"].items()}
            check_results(results, selected, plan["source_commit"])
            for label, job in state["training"].items():
                state.setdefault("raw_archive", {})[label] = str(
                    archive_run(args.base, job, output, train=True))
            for label, job in state["evaluation"].items():
                directory = archive_run(args.base, job, output, train=False)
                state.setdefault("raw_archive", {})[label] = str(directory)
                atomic(output / "results" / (label + ".json"), results[label])
            atomic(output / "summary.json", {label: result["summary"]
                                              for label, result in results.items()})
            plot_curves(state["training"], output)
            plot_results(results, output)
            write_report(results, plan, output)
            state.update(status="completed_pending_independent_audit", finished_at=now())
            atomic(manifest, state)
            atomic(output / "suite_status.json", {
                "phase": "completed_pending_independent_audit", "updated_at": now()})
        except BaseException as error:
            state.update(status="needs_attention", error=str(error), updated_at=now())
            atomic(manifest, state)
            atomic(output / "suite_status.json", {"phase": "needs_attention",
                                                  "error": str(error), "updated_at": now()})
            raise


if __name__ == "__main__":
    main()
