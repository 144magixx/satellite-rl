"""Read canonical workbench and historical TensorBoard/CSV diagnostics."""

from __future__ import annotations

from collections import OrderedDict, deque
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import threading

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
    """Incremental canonical JSONL and TensorBoard reader with bounded LRU state."""

    MAX_RUNS = 4
    MAX_POINTS = 1_000_000

    def __init__(self):
        self.cache = OrderedDict()
        self.lock = threading.RLock()

    def _state(self, directory):
        key = str(directory)
        state = self.cache.get(key)
        if state is None:
            state = {"json_identity": None, "json_size": 0, "json_mtime": None,
                     "json_offset": 0, "json_pending_size": None,
                     "json_series": {}, "invalid": deque(maxlen=30),
                     "tb_signature": None, "tb_accumulator": None, "tb_events": {}}
            self.cache[key] = state
        self.cache.move_to_end(key)
        return state

    @staticmethod
    def _identity(stat):
        # Windows ctime is file creation time; it catches an atomic replacement
        # even where st_ino is not provided by the filesystem.
        return (stat.st_dev, stat.st_ino, stat.st_ctime_ns if os.name == "nt" else None)

    @staticmethod
    def _add_json_metric(state, row):
        group = row.get("group", "training")
        values = row.get("values", {})
        if not isinstance(values, dict):
            return
        for name, value in values.items():
            tag = name if name.startswith(group + "/") else group + "/" + name
            numeric = _number(value)
            if numeric is None:
                if str(value).lower() in ("nan", "inf", "-inf", "infinity"):
                    state["invalid"].append({"tag": tag, "step": row.get("step", row.get("env_step", 0)),
                                             "value": str(value)})
                continue
            step = _number(row.get("step", row.get("env_step", 0)))
            if step is not None:
                state["json_series"].setdefault(tag, {})[float(step)] = (
                    numeric, _time_value(row.get("wall_time", row.get("wall_seconds"))))

    def _read_json_increment(self, path, state):
        try:
            stat = path.stat()
        except OSError:
            state.update(json_identity=None, json_size=0, json_mtime=None, json_offset=0,
                         json_pending_size=None, json_series={}, invalid=deque(maxlen=30))
            return False
        identity = self._identity(stat)
        if (state["json_identity"] == identity and state["json_pending_size"] == stat.st_size and
                state["json_mtime"] == stat.st_mtime_ns):
            return True
        if (state["json_identity"] != identity or stat.st_size < state["json_size"] or
                state["json_pending_size"] is not None or
                (stat.st_size == state["json_size"] and state["json_mtime"] != stat.st_mtime_ns)):
            state.update(json_identity=identity, json_size=0, json_mtime=None, json_offset=0,
                         json_pending_size=None, json_series={}, invalid=deque(maxlen=30))
        if stat.st_size > state["json_offset"]:
            with path.open("rb") as stream:
                stream.seek(state["json_offset"])
                chunk = stream.read(stat.st_size - state["json_offset"])
            offset = state["json_offset"]
            for line in chunk.splitlines(keepends=True):
                complete = line.endswith(b"\n")
                try:
                    row = json.loads(line.decode("utf-8-sig", errors="replace"))
                except ValueError:
                    if not complete:
                        state["json_pending_size"] = stat.st_size
                        break  # Re-read this partial record after its next append.
                else:
                    if isinstance(row, dict):
                        self._add_json_metric(state, row)
                    if not complete:
                        state["json_pending_size"] = stat.st_size
                        break  # Valid unterminated line may still gain its newline.
                offset += len(line)
            state["json_offset"] = offset
        state["json_identity"] = identity
        state["json_size"] = stat.st_size
        state["json_mtime"] = stat.st_mtime_ns
        return True

    def _tensorboard_events(self, path, state):
        signature = []
        for file in sorted(path.glob("events.out.tfevents.*")):
            stat = file.stat()
            signature.append((file.name, self._identity(stat), stat.st_size, stat.st_mtime_ns))
        signature = tuple(signature)
        if signature == state["tb_signature"] and state["tb_accumulator"] is not None:
            return state["tb_events"]
        current = {name: (identity, size, mtime) for name, identity, size, mtime in signature}
        replaced = any(name not in current or current[name][0] != identity or
                       current[name][1] < size or
                       (current[name][1] == size and current[name][2] != mtime)
                       for name, identity, size, mtime in (state["tb_signature"] or ()))
        if state["tb_accumulator"] is None or replaced:
            from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
            state["tb_accumulator"] = EventAccumulator(str(path), size_guidance={"scalars": 10000})
        accumulator = state["tb_accumulator"]
        accumulator.Reload()
        state["tb_events"] = {tag: [(float(event.step), event.value, event.wall_time)
                                     for event in accumulator.Scalars(tag)]
                              for tag in accumulator.Tags().get("scalars", [])}
        state["tb_signature"] = signature
        return state["tb_events"]

    def _bound_cache(self):
        while (len(self.cache) > self.MAX_RUNS or
               sum(sum(len(points) for points in state["json_series"].values()) +
                   sum(len(points) for points in state["tb_events"].values())
                   for state in self.cache.values()) > self.MAX_POINTS):
            self.cache.popitem(last=False)

    def read(self, directory, limit=2000):
        with self.lock:
            return self._read_locked(directory, limit)

    def _read_locked(self, directory, limit):
        directory = Path(directory)
        limit = min(20000, max(50, int(limit)))
        state = self._state(directory)
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
        if self._read_json_increment(json_path, state):
            sources.add("metrics.jsonl")
            series = {tag: {step: {"step": step, "value": value, "wall_time": wall_time}
                            for step, (value, wall_time) in points.items()}
                      for tag, points in state["json_series"].items()}
            invalid = list(state["invalid"])
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
                events = self._tensorboard_events(tb_path, state)
                if events:
                    sources.add("TensorBoard（每项最多读取10000个事件）")
                for tag, rows in events.items():
                    if tag in series:
                        for step, value, wall_time in rows:
                            point = series[tag].get(step)
                            if point is not None:
                                point["wall_time"] = wall_time
                        time_bases[tag] = "TensorBoard实际记录时间"
                        continue
                    for step, value, wall_time in rows:
                        add(tag, step, value, wall_time)
            except (ImportError, OSError, ValueError, KeyError) as error:
                tb_error = str(error)
        else:
            state.update(tb_signature=None, tb_accumulator=None, tb_events={})
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
        self._bound_cache()
        return output
