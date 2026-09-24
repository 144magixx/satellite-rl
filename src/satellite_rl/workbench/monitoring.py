"""Read canonical workbench and historical TensorBoard/CSV diagnostics."""

from __future__ import annotations

from collections import deque
import csv
from datetime import datetime, timezone
import math
from pathlib import Path
import time

from .storage import clean_json, read_json

GROUP_LABELS = {
    "learning": "学习过程", "training": "训练业务效果", "validation": "验证效果",
    "evaluation": "测试效果", "reward": "奖励分项", "resources": "运行资源",
    "actions": "动作与资源分配", "stability": "数值稳定性",
}


def read_jsonl(path, limit=10000):
    import json
    rows = deque(maxlen=limit) if limit else []
    try:
        with Path(path).open(encoding="utf-8-sig", errors="replace") as stream:
            for line in stream:
                try:
                    row = json.loads(line)
                    if isinstance(row, dict):
                        rows.append(row)
                except ValueError:
                    # A worker can currently be appending the final line.
                    continue
    except OSError:
        pass
    return list(rows)


def log_tail(directory, limit=200):
    directory = Path(directory)
    lines = []
    for name in ("worker.log", "stderr.log", "stdout.log", "train.log"):
        path = directory / name
        if path.is_file():
            with path.open(encoding="utf-8-sig", errors="replace") as stream:
                lines.extend(deque(stream, maxlen=limit))
    return {"lines": [line.rstrip() for line in lines[-limit:]],
            "events": read_jsonl(directory / "events.jsonl", limit)}


def _number(value):
    if isinstance(value, bool):
        return float(value)
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (ValueError, TypeError):
        return None


def _time_value(value):
    if value is None or value == "":
        return None
    number = _number(value)
    if number is not None:
        return number
    try:
        return datetime.fromisoformat(str(value)).timestamp()
    except (ValueError, TypeError):
        return None


def reduce_points(points, limit):
    """Keep time coverage and each bin's min/max so spikes remain visible."""
    if len(points) <= limit:
        return points
    if limit < 4:
        return [points[0], points[-1]][:limit]
    result = [points[0]]
    bins = max(1, (limit - 2) // 2)
    interior = points[1:-1]
    for index in range(bins):
        block = interior[index * len(interior) // bins:(index + 1) * len(interior) // bins]
        if not block:
            continue
        picks = {min(range(len(block)), key=lambda i: block[i]["value"]),
                 max(range(len(block)), key=lambda i: block[i]["value"])}
        result.extend(block[i] for i in sorted(picks))
    result.append(points[-1])
    return result


class MetricsReader:
    def __init__(self):
        self.cache = {}

    def read(self, directory, limit=2000):
        directory = Path(directory)
        limit = min(20000, max(50, int(limit)))
        key = (str(directory), limit)
        cached = self.cache.get(key)
        if cached and time.monotonic() - cached[0] < 4:
            return cached[1]
        series, sources, invalid, time_bases = {}, set(), [], {}

        def add(tag, step, value, wall_time=None):
            numeric = _number(value)
            if numeric is None:
                if str(value).lower() in ("nan", "inf", "-inf", "infinity"):
                    invalid.append({"tag": tag, "step": step, "value": str(value)})
                return
            step = _number(step)
            if step is None:
                return
            series.setdefault(tag, {})[float(step)] = {
                "step": float(step), "value": numeric, "wall_time": _time_value(wall_time),
            }

        json_path = directory / "metrics.jsonl"
        if json_path.exists():
            sources.add("metrics.jsonl")
            for row in read_jsonl(json_path, limit=0):
                group = row.get("group", "training")
                for name, value in row.get("values", {}).items():
                    add(name if name.startswith(group + "/") else group + "/" + name,
                        row.get("step", row.get("env_step", 0)), value,
                        row.get("wall_time", row.get("wall_seconds")))
        skip = {"run_id", "env_step", "update_step", "episode", "scene_id", "action_seed", "wall_seconds"}
        for filename, group in (("updates", "learning"), ("episodes", "training"), ("evaluations", "evaluation")):
            path = directory / (filename + ".csv")
            if path.exists():
                sources.add(path.name)
                with path.open(encoding="utf-8-sig", newline="", errors="replace") as stream:
                    rows = list(csv.DictReader(stream))
                # Evaluation rows often have many scenes at the same checkpoint.
                # Aggregate their numeric metrics rather than silently keep the last scene.
                if group == "evaluation":
                    buckets = {}
                    for row in rows:
                        context = str(row.get("checkpoint", "evaluation")) + "/" + str(row.get("action_mode") or "sample")
                        step = row.get("env_step", 0)
                        for name, value in row.items():
                            number = _number(value)
                            if name not in skip and number is not None:
                                buckets.setdefault((context, name, step), []).append(number)
                    for (context, name, step), values in buckets.items():
                        add("evaluation/" + context + "/" + name, step, sum(values) / len(values))
                else:
                    cumulative_episode_seconds = 0.0
                    for row in rows:
                        record_time = _number(row.get("wall_seconds"))
                        if group == "training" and record_time is not None:
                            cumulative_episode_seconds += record_time
                            record_time = cumulative_episode_seconds
                        for name, value in row.items():
                            if name not in skip:
                                add(group + "/" + name, row.get("env_step", row.get("episode", 0)), value,
                                    record_time)
                                if group == "training":
                                    time_bases[group + "/" + name] = "累计回合交互耗时，不含验证与保存"
        # Also accept per-category records if a worker crashed before its metrics file flush.
        if not json_path.exists():
            for filename, group in (("updates", "learning"), ("episodes", "training")):
                path = directory / (filename + ".jsonl")
                if path.exists():
                    sources.add(path.name)
                    for row in read_jsonl(path, limit=0):
                        for name, value in row.items():
                            if name not in skip and not isinstance(value, (dict, list)):
                                add(group + "/" + name, row.get("env_step", row.get("step", 0)), value,
                                    row.get("wall_seconds"))
        tb_path = directory / "tensorboard"
        tb_error = None
        if tb_path.is_dir():
            try:
                from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
                accumulator = EventAccumulator(str(tb_path), size_guidance={"scalars": 10000})
                accumulator.Reload()
                tags = accumulator.Tags().get("scalars", [])
                if tags:
                    sources.add("TensorBoard（每项最多读取10000个事件）")
                for tag in tags:
                    if tag in series:
                        for event in accumulator.Scalars(tag):
                            point = series[tag].get(float(event.step))
                            if point is not None:
                                point["wall_time"] = event.wall_time
                        time_bases[tag] = "TensorBoard实际记录时间"
                        continue
                    for event in accumulator.Scalars(tag):
                        add(tag, event.step, event.value, event.wall_time)
            except (ImportError, OSError, ValueError, KeyError) as error:
                tb_error = str(error)
        result = []
        for tag, by_step in sorted(series.items()):
            points = sorted(by_step.values(), key=lambda point: point["step"])
            result.append({"tag": tag, "group": tag.split("/")[0],
                           "points": reduce_points(points, limit), "raw_count": len(points),
                           "time_basis": time_bases.get(tag, "记录时间；未记录时为空"),
                           "downsampled": len(points) > limit})
        status = read_json(directory / "status.json", {})
        warnings = []
        if invalid:
            warnings.append({"level": "error", "message": "记录中出现非有限数值，请检查学习损失、Q值和最后正常断点", "details": invalid[-30:]})
        if tb_error:
            warnings.append({"level": "warning", "message": "TensorBoard读取失败，其他原始记录仍可查看", "details": tb_error})
        failure = read_json(directory / "failure.json")
        if failure:
            warnings.append({"level": "error", "message": "该任务保留了异常现场", "details": clean_json(failure)})
        if status.get("stage") in ("training", "running", "warmup", "validating", "evaluating"):
            try:
                age = (datetime.now(timezone.utc) - datetime.fromisoformat(status["heartbeat_at"])).total_seconds()
                if age > 180:
                    warnings.append({"level": "warning", "message": f"状态已 {int(age)} 秒未更新；请结合实际进程判断，不能仅凭running认定仍在训练"})
            except (KeyError, ValueError, TypeError):
                pass
        output = {"series": result, "available_tags": [s["tag"] for s in result],
                  "diagnostics": warnings, "sources": sorted(sources), "groups": GROUP_LABELS,
                  "note": "图来自原始记录；稠密曲线按时间分段保留极值。历史回合CSV的时间为累计交互耗时，不含验证与保存；同名TensorBoard标签优先提供实际时间。缺失时间不补零；不同奖励回报不直接排名。"}
        self.cache[key] = (time.monotonic(), output)
        return output
