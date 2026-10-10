"""Offline S1-D12A schema-6 evidence audit and figures.

Accept a downloaded jobs directory or ZIP without executing frozen science
code. Missing evidence stays missing. A partial download produces QA only.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import hashlib
import io
import json
import math
from pathlib import Path
from statistics import mean, median
import zipfile

try:
    from scripts.stage1_drl_r04.strict_audit import audit_scene_row, scenes_from_npz
except ModuleNotFoundError:  # Direct script invocation from the repository root.
    from strict_audit import audit_scene_row, scenes_from_npz


METHODS = ("I0", "H1", "H2", "BC", "P0", "P1")
LEVELS = (.75, 1.0, 1.25)
REQUIRED_BEAM = ("demand_bps", "capacity_bps", "carried_bps",
                 "unclipped_capacity_demand_ratio", "si", "sgm")


class Evidence:
    def __init__(self, root):
        self.root = Path(root)
        self.zip = zipfile.ZipFile(self.root) if self.root.is_file() else None
        self.nested = {}
        if self.zip:
            self.names = [name for name in self.zip.namelist() if not name.endswith("/")]
        elif self.root.is_dir():
            self.names = [p.relative_to(self.root).as_posix() for p in self.root.rglob("*") if p.is_file()]
            # A verified archive directory has per-task ZIP volumes and no
            # extracted job files. Index members virtually, without copying.
            if not any(name.endswith(("/job.json", "/status.json", "/results.jsonl",
                                      "/actions.jsonl")) for name in self.names):
                receipts = sorted(self.root.rglob("receipt.json"))
                if receipts:
                    for receipt_path in receipts:
                        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
                        for volume in receipt.get("volumes", []):
                            path = Path(volume.get("canonical_path") or
                                        receipt_path.parent / volume["file"])
                            if not path.is_file() or path.stat().st_size != volume["bytes"]:
                                raise ValueError("canonical archive ZIP absent or wrong size")
                            checksum = hashlib.sha256()
                            with path.open("rb") as stream:
                                for block in iter(lambda: stream.read(8 << 20), b""):
                                    checksum.update(block)
                            if checksum.hexdigest() != volume["sha256"]:
                                raise ValueError("canonical archive ZIP SHA differs")
                            with zipfile.ZipFile(path) as bundle:
                                names = [name for name in bundle.namelist() if not name.endswith("/")]
                            if set(names) != set(volume["members"]) or len(names) != len(volume["members"]):
                                raise ValueError("canonical archive member inventory differs")
                            for member in names:
                                if member in self.nested:
                                    raise ValueError("archive member appears in two ZIP volumes")
                                self.nested[member] = path
                else:
                    for name in list(self.names):
                        if not name.endswith(".zip"):
                            continue
                        path = self.root / name
                        with zipfile.ZipFile(path) as bundle:
                            for member in bundle.namelist():
                                if member.endswith("/"):
                                    continue
                                if member in self.nested:
                                    raise ValueError("archive member appears in two ZIP volumes")
                                self.nested[member] = path
                self.names.extend(self.nested)
        else:
            raise FileNotFoundError(root)
        self._digest_cache = {}

    def close(self):
        if self.zip:
            self.zip.close()

    @contextmanager
    def _open(self, name):
        if self.zip:
            with self.zip.open(name) as stream:
                yield stream
        elif name in self.nested:
            with zipfile.ZipFile(self.nested[name]) as bundle:
                with bundle.open(name) as stream:
                    yield stream
        else:
            with (self.root / name).open("rb") as stream:
                yield stream

    def _read(self, name):
        with self._open(name) as stream:
            return stream.read()

    def _sha_name(self, name):
        checksum = hashlib.sha256()
        with self._open(name) as stream:
            for block in iter(lambda: stream.read(8 << 20), b""):
                checksum.update(block)
        return checksum.hexdigest()

    def _job_name(self, port, run_id, filename):
        suffixes = (f"/{port}/{run_id}/{filename}", f"/{run_id}/{filename}")
        matches = [name for name in self.names if any(("/" + name).endswith(s) for s in suffixes)]
        return (matches[0], None) if len(matches) == 1 else (
            None, f"expected one {filename} for {port}/{run_id}; found {len(matches)}")

    def job_file(self, port, run_id, filename):
        name, error = self._job_name(port, run_id, filename)
        return (None, error) if error else (self._read(name), None)

    def job_json(self, port, run_id, filename):
        data, error = self.job_file(port, run_id, filename)
        if error:
            return None, error
        try:
            return json.loads(data), None
        except (ValueError, UnicodeDecodeError) as exc:
            return None, f"invalid {filename}: {exc}"

    def job_jsonl(self, port, run_id, filename):
        data, error = self.job_file(port, run_id, filename)
        if error:
            return [], error
        try:
            return [json.loads(line) for line in data.splitlines() if line.strip()], None
        except (ValueError, UnicodeDecodeError) as exc:
            return [], f"invalid {filename}: {exc}"

    def job_iter_jsonl(self, port, run_id, filename):
        name, error = self._job_name(port, run_id, filename)
        if error:
            raise FileNotFoundError(f"expected one {filename} for {port}/{run_id}")
        with self._open(name) as binary:
            with io.TextIOWrapper(binary, encoding="utf-8") as stream:
                for line in stream:
                    if line.strip():
                        yield json.loads(line)

    def sha(self, port, run_id, filename):
        name, error = self._job_name(port, run_id, filename)
        return (None, error) if error else (self._sha_name(name), None)

    def has_digest(self, digest, basename):
        if not isinstance(digest, str) or len(digest) != 64:
            return False
        key = (digest, basename)
        if key not in self._digest_cache:
            self._digest_cache[key] = any(
                self._sha_name(name) == digest
                for name in self.names if name.endswith("/" + basename) or name == basename)
        return self._digest_cache[key]


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _near(a, b, *, atol=1e-5, rtol=1e-8):
    return _number(a) and _number(b) and abs(a - b) <= atol + rtol * max(abs(a), abs(b))


def _layout_sha(layout):
    return hashlib.sha256(json.dumps(layout, sort_keys=True,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def _audit_row(row):
    errors = []
    initial = row.get("initialization", {}).get("metrics", {})
    if not initial.get("coverage_100") or not initial.get("demand_conserved"):
        errors.append("initial_geometry_or_demand_failed")
    for field, limit in (("commits", 32), ("trials", 256), ("geometry_calls", 8192)):
        value = row.get(field)
        if not isinstance(value, int) or not 0 <= value <= limit:
            errors.append(f"{field}_budget_invalid")
    layouts = row.get("terminal_physics", {}).get("layouts_by_sha256", {})
    alias = row.get("terminal_physics", {}).get("layout_aliases", {}).get("best")
    best = layouts.get(alias) if isinstance(layouts, dict) else None
    if alias != row.get("best_sha256") or not isinstance(best, dict):
        errors.append("best_physical_layout_missing_or_sha_mismatch")
        best = {}
    beams = best.get("beams")
    layout = row.get("best_layout") or {}
    if not layout or _layout_sha(layout) != row.get("best_sha256"):
        errors.append("best_layout_content_sha_mismatch")
    initial_layout = row.get("initial_layout")
    if not isinstance(initial_layout, dict) or _layout_sha(initial_layout) != row.get("initial_sha256"):
        errors.append("initial_layout_content_sha_mismatch")
    if not isinstance(beams, list) or len(beams) != len(layout.get("centers", [])) or len(beams) > 200:
        errors.append("best_beam_count_invalid")
        beams = []
    owners = layout.get("assignment", [])
    if not owners or any(not isinstance(x, int) or not 0 <= x < len(beams) for x in owners):
        errors.append("best_aircraft_ownership_invalid")
    if any(not _near(w, .5, atol=1e-12) for w in layout.get("widths", [])):
        errors.append("best_width_invalid")
    total_d = 0.0
    carried = 0.0
    for beam in beams:
        if not all(key in beam for key in REQUIRED_BEAM):
            errors.append("per_beam_field_missing")
            continue
        d, c, take = (beam[key] for key in ("demand_bps", "capacity_bps", "carried_bps"))
        if not all(_number(value) and value >= 0 for value in (d, c, take)):
            errors.append("per_beam_nonfinite_or_negative")
            continue
        total_d += d
        carried += take
        if not _near(take, min(c, d)):
            errors.append("per_beam_carried_not_min_capacity_demand")
        ratio, si, sgm = (beam[key] for key in ("unclipped_capacity_demand_ratio", "si", "sgm"))
        if d > 0:
            if not _near(ratio, c / d, atol=1e-8) or not _near(si, min(c / d, 1), atol=1e-8):
                errors.append("per_beam_ratio_or_si_invalid")
            if not _number(sgm):
                errors.append("per_beam_sgm_missing")
        elif ratio is not None or si is not None or sgm is not None:
            errors.append("zero_demand_beam_not_null")
    expected_d = initial.get("total_demand_bps")
    if beams and not _near(total_d, expected_d, atol=1e-2):
        errors.append("beam_demand_not_conserved")
    physical = best.get("metrics", {})
    aircraft_service = best.get("aircraft_service", {})
    if beams and not _number(aircraft_service.get("demand_weighted_carried_fraction")):
        errors.append("aircraft_tdma_carried_fraction_missing")
    if beams and (not _near(carried, physical.get("carried_bps"), atol=1e-2) or
                  not _near(physical.get("total_demand_bps"), expected_d, atol=1e-2)):
        errors.append("physical_summary_differs_from_beams")
    j0, jb = row.get("initial_j"), row.get("best_j")
    if not _number(j0) or not _number(jb) or jb > j0 + 1e-10:
        errors.append("best_j_invalid")
    ledger = row.get("ledger")
    ledger_rewards = ([x.get("reward") for x in ledger if isinstance(x, dict)]
                      if isinstance(ledger, list) else [])
    if (not isinstance(ledger, list) or len(ledger_rewards) != len(ledger) or
            not all(_number(x) for x in ledger_rewards) or
            not _near(sum(ledger_rewards) if all(_number(x) for x in ledger_rewards) else None,
                      row.get("reward_sum"), atol=1e-7)):
        errors.append("action_reward_ledger_missing_or_mismatch")
    if _number(j0) and _number(jb) and not _near(row.get("reward_sum"), 100 * (j0 - jb), atol=1e-7):
        errors.append("reward_not_telescoping")
    if not all(_number(row.get(field)) and row[field] >= 0 for field in
               ("algorithm_seconds", "total_seconds")):
        errors.append("algorithm_or_total_timing_missing")
    return sorted(set(errors)), best


def _stats(values):
    data = sorted(float(x) for x in values if _number(x))
    if not data:
        return {"count": 0, "mean": None, "median": None, "p95": None}
    return {"count": len(data), "mean": mean(data), "median": median(data),
            "p95": data[math.ceil(.95 * len(data)) - 1]}


def audit_training_terminal(scene_lookup, terminal):
    """Replay recorded terminal physics without inventing episode wall time."""
    if (terminal.get("purpose") == "budget_tail_snapshot" and
            terminal.get("action_trace", {}).get("count") == 0):
        return [], False
    scene = scene_lookup.get((terminal.get("scene_id"), terminal.get("scene_sha256")))
    if scene is None:
        return ["training_frozen_scene_unavailable"], False
    errors, _ = audit_scene_row(scene, terminal, require_total_timing=False)
    if not _near(terminal.get("reward_sum"), terminal.get("episode_return"), atol=1e-7):
        errors.append("training_terminal_reward_sum_differs")
    if not (_number(terminal.get("initial_j")) and _number(terminal.get("best_j"))) or not _near(
            terminal.get("episode_return"),
            100 * (terminal["initial_j"] - terminal["best_j"]), atol=1e-7):
        errors.append("training_terminal_reward_not_telescoping")
    return errors, True


def _plot(output, scene_rows, training, validation):
    if not scene_rows and not any(training.values()) and not any(validation.values()):
        return [], None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except (ImportError, MemoryError):
        return [], "matplotlib_unavailable"
    plots = []
    if scene_rows:
        fig, ax = plt.subplots(figsize=(9, 5))
        for method in METHODS:
            values = [x["aircraft_tdma_carried_fraction"] for x in scene_rows
                      if x["method"] == method and _number(x["aircraft_tdma_carried_fraction"])]
            if values:
                ax.plot(sorted(values), label=f"{method} (n={len(values)})")
        ax.set(xlabel="Sorted scene result", ylabel="Aircraft TDMA demand-weighted carried fraction",
               title="Independent test distributions; three levels per mother")
        ax.legend()
        fig.tight_layout()
        name = "test_carried_distribution.png"
        fig.savefig(output / name, dpi=140)
        plt.close(fig)
        plots.append(name)
    if training:
        fig, ax = plt.subplots(figsize=(9, 5))
        for method, rows in training.items():
            points = sorted((step, reward) for step, reward in rows
                            if _number(step) and _number(reward))
            if points:
                ax.scatter(*zip(*points), s=3, alpha=.25, label=f"{method} raw episodes")
                window = 100
                if len(points) >= window:
                    running = sum(value for _, value in points[:window])
                    smoothed = [(points[window - 1][0], running / window)]
                    for i in range(window, len(points)):
                        running += points[i][1] - points[i - window][1]
                        smoothed.append((points[i][0], running / window))
                    ax.plot(*zip(*smoothed), linewidth=1.3,
                            label=f"{method} trailing mean {window} episodes")
        ax.set(xlabel="RL decisions", ylabel="Episode reward (100 × best J improvement)",
               title="Training episodes; budget tails excluded")
        ax.legend()
        fig.tight_layout()
        name = "train_episode_reward_raw.png"
        fig.savefig(output / name, dpi=140)
        plt.close(fig)
        plots.append(name)
    if validation:
        fig, ax = plt.subplots(figsize=(9, 5))
        for method, rows in validation.items():
            points = [(x.get("env_step"), x.get("mean_best_j")) for x in rows
                      if _number(x.get("env_step")) and _number(x.get("mean_best_j"))]
            if points:
                ax.plot(*zip(*points), marker="o", label=method)
        ax.set(xlabel="RL decisions", ylabel="Mean best R04 J", title="Fixed validation nodes")
        ax.legend()
        fig.tight_layout()
        name = "validation_best_j.png"
        fig.savefig(output / name, dpi=140)
        plt.close(fig)
        plots.append(name)
    return plots, None


def analyze(state, source, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if state.get("experiment_id") != "S1-D12A":
        raise ValueError("controller state is not S1-D12A")
    evidence = Evidence(source)
    issues, rows, training, validation, task_audit = [], [], {}, {}, {}
    strict_rows = 0
    try:
        tasks = state.get("tasks", {})
        for key, task in tasks.items():
            port, run_id = task.get("port"), task.get("run_id")
            if port is None or not run_id:
                issues.append(f"{key}: task identity missing")
                continue
            job, e1 = evidence.job_json(port, run_id, "job.json")
            status, e2 = evidence.job_json(port, run_id, "status.json")
            config, e3 = evidence.job_json(port, run_id, "config.json")
            if e1 or e2 or e3:
                issues.append(f"{key}: {e1 or e2 or e3}")
                continue
            method = task.get("variant")
            valid = (job.get("run_id") == run_id and job.get("variant") == method and
                     job.get("git_commit") == state.get("scientific_commit") and
                     job.get("experiment_id") == "S1-D12A")
            if not valid:
                issues.append(f"{key}: frozen job identity differs")
            if config.get("schema_version") != 6:
                issues.append(f"{key}: schema-6 configuration missing")
            task_audit[key] = {"port": port, "run_id": run_id, "method": method,
                               "stage": status.get("stage"), "identity_valid": valid,
                               "scientific_stop": status.get("scientific_stop") is True,
                               "config_schema": config.get("schema_version")}
            if key.startswith("test_"):
                result, error = evidence.job_jsonl(port, run_id, "results.jsonl")
                if error:
                    issues.append(f"{key}: {error}")
                    continue
                digest, _ = evidence.sha(port, run_id, "results.jsonl")
                summary, serr = evidence.job_json(port, run_id, "summary.json")
                if serr or summary.get("result_sha256") != digest or summary.get("scene_count") != len(result):
                    issues.append(f"{key}: result SHA/count mismatch")
                if summary and summary.get("method") != method:
                    issues.append(f"{key}: result method identity differs")
                shard = job.get("shard_id")
                expected_indices = list(range(shard, 100, 4)) if isinstance(shard, int) and 0 <= shard < 4 else []
                if len(result) != 75 or job.get("scene_indices") != expected_indices:
                    issues.append(f"{key}: shard must be 25 mothers x 3 levels")
                dataset_record = config.get("datasets", {}).get("test", {})
                dataset_sha, dataset_error = evidence.sha(port, run_id, "scene_inputs/test/inputs.npz")
                if dataset_error or dataset_sha != dataset_record.get("sha256"):
                    issues.append(f"{key}: frozen test dataset SHA unavailable or differs")
                    frozen_scenes = None
                else:
                    try:
                        data, _ = evidence.job_file(port, run_id, "scene_inputs/test/inputs.npz")
                        frozen_scenes = scenes_from_npz(data)
                        if len(frozen_scenes) != 100:
                            raise ValueError("independent test NPZ needs 100 mothers")
                    except (ValueError, KeyError, IndexError) as exc:
                        frozen_scenes = None
                        issues.append(f"{key}: frozen test NPZ decode failed: {exc}")
                model = (summary or {}).get("model", {})
                if method in ("BC", "P0", "P1"):
                    if (model.get("policy_format") != "drl_r04_set_ppo_v1" or
                            model.get("train_mode") != method or
                            not evidence.has_digest(model.get("policy_sha256"), "policy.pt")):
                        issues.append(f"{key}: learned model weight SHA unavailable or differs")
                for ordinal, row in enumerate(result):
                    errors, best = _audit_row(row)
                    index = row.get("dataset_index")
                    level = row.get("demand_multiplier")
                    if ordinal // 3 >= len(expected_indices) or index != expected_indices[ordinal // 3] or level != LEVELS[ordinal % 3]:
                        errors.append("shard_scene_or_level_order_invalid")
                    if row.get("checkpoint_step") is None:
                        errors.append("model_step_missing")
                    elif row.get("checkpoint_step") != model.get("env_step"):
                        errors.append("evaluation_model_step_differs")
                    strict_facts = {}
                    if frozen_scenes is not None and isinstance(index, int) and 0 <= index < len(frozen_scenes):
                        try:
                            strict_errors, strict_facts = audit_scene_row(frozen_scenes[index], row)
                            errors.extend(strict_errors)
                            strict_rows += 1
                        except (ValueError, TypeError, KeyError, IndexError, ArithmeticError) as exc:
                            errors.append(f"strict_frozen_scene_replay_failed:{type(exc).__name__}")
                    else:
                        errors.append("strict_frozen_scene_replay_unavailable")
                    physical = best.get("metrics", {})
                    aircraft_service = best.get("aircraft_service", {})
                    beam_demands = [b["demand_bps"] for b in best.get("beams", [])
                                    if _number(b.get("demand_bps"))]
                    average_load = mean(beam_demands) if beam_demands else None
                    best_cv = (math.sqrt(mean((d - average_load) ** 2 for d in beam_demands)) /
                               average_load if average_load and average_load > 0 else None)
                    item = {"method": method, "task_key": key, "dataset_index": index,
                            "scene_id": row.get("scene_id"), "scene_sha256": row.get("scene_sha256"),
                            "demand_multiplier": level, "initial_sha256": row.get("initial_sha256"),
                            "checkpoint_step": row.get("checkpoint_step"),
                            "initial_j": row.get("initial_j"), "best_j": row.get("best_j"),
                            "beam_count": len((row.get("best_layout") or {}).get("centers", [])),
                            "beam_carried_fraction": physical.get("demand_weighted_carried_fraction"),
                            "aircraft_tdma_carried_fraction":
                                aircraft_service.get("demand_weighted_carried_fraction"),
                            "load_cv": best_cv,
                            "beam_mean_si": physical.get("beam_mean_si"),
                            "beam_mean_sgm": physical.get("beam_mean_sgm"),
                            "algorithm_seconds": row.get("algorithm_seconds"),
                            "total_seconds": row.get("total_seconds"),
                            "physical_seconds": (row["total_seconds"] - row["algorithm_seconds"]
                                                 if _number(row.get("total_seconds")) and
                                                 _number(row.get("algorithm_seconds")) else None),
                            "initialization_seconds": row.get("initialization_seconds"),
                            "menu_seconds": row.get("menu_seconds"),
                            "trial_seconds": row.get("trial_seconds"),
                            "commits": row.get("commits"), "trials": row.get("trials"),
                            "geometry_calls": row.get("geometry_calls"),
                            "best_N_lt_50": strict_facts.get("best", {}).get("N_lt_50"),
                            "best_N_eq_200": strict_facts.get("best", {}).get("N_eq_200"),
                            "positive_band_0p8_1p2": strict_facts.get("best", {}).get("positive_band_0p8_1p2"),
                            "positive_demand_beams": strict_facts.get("best", {}).get("positive_demand_beams"),
                            "strict_facts": strict_facts,
                            "errors": sorted(set(errors))}
                    rows.append(item)
            elif key in ("P0", "P1"):
                checkpoint_sha = status.get("checkpoint_sha256")
                actual_checkpoint, checkpoint_error = evidence.sha(port, run_id, "checkpoint.pt")
                if checkpoint_error or not checkpoint_sha or checkpoint_sha != actual_checkpoint:
                    issues.append(f"{key}: terminal checkpoint SHA unavailable or differs")
                for role in ("train", "validation"):
                    expected = config.get("datasets", {}).get(role, {}).get("sha256")
                    actual, dataset_error = evidence.sha(port, run_id, f"scene_inputs/{role}/inputs.npz")
                    if dataset_error or actual != expected:
                        issues.append(f"{key}: frozen {role} dataset SHA unavailable or differs")
                training_scene_lookup = {}
                try:
                    train_bytes, train_error = evidence.job_file(port, run_id, "scene_inputs/train/inputs.npz")
                    if train_error or hashlib.sha256(train_bytes).hexdigest() != config["datasets"]["train"]["sha256"]:
                        raise ValueError("training NPZ is missing or has another SHA")
                    training_scenes = scenes_from_npz(train_bytes)
                    training_scene_lookup = {(scene.scene_id, scene.source_hash): scene
                                             for scene in training_scenes}
                    if len(training_scene_lookup) != len(training_scenes):
                        raise ValueError("training scene identity duplicated")
                except (KeyError, TypeError, ValueError) as exc:
                    issues.append(f"{key}: training scene replay unavailable: {exc}")
                tails, error = evidence.job_jsonl(port, run_id, "budget_tails.jsonl")
                if error:
                    issues.append(f"{key}: {error}")
                if len(tails) > 32:
                    issues.append(f"{key}: budget tails exceed 32 scene streams")
                nodes, error = evidence.job_jsonl(port, run_id, "evaluations.jsonl")
                if error:
                    issues.append(f"{key}: {error}")
                training[key], validation[key] = [], nodes
                action_groups = {}
                action_total = 0
                try:
                    for event in evidence.job_iter_jsonl(port, run_id, "actions.jsonl"):
                        identity = (event.get("stream_id"), event.get("stream_episode_serial"))
                        group = action_groups.setdefault(identity, {"count": 0, "reward": 0.0,
                                                                    "first": None, "last": None})
                        step = event.get("env_step")
                        reward = event.get("reward")
                        if not isinstance(step, int) or not _number(reward):
                            issues.append(f"{key}: invalid action step/reward")
                            continue
                        group["count"] += 1
                        group["reward"] += reward
                        group["first"] = step if group["first"] is None else min(group["first"], step)
                        group["last"] = step if group["last"] is None else max(group["last"], step)
                        action_total += 1
                        info = event.get("info", {})
                        before, after = event.get("best_j_before"), event.get("best_j_after")
                        if _number(before) and _number(after) and not _near(
                                reward, 100 * (before - after), atol=1e-7):
                            issues.append(f"{key}: action reward differs from best J delta")
                        for field, limit in (("commits", 32), ("trials", 256),
                                             ("geometry_calls", 8192)):
                            value = info.get(field)
                            if not isinstance(value, int) or not 0 <= value <= limit:
                                issues.append(f"{key}: action {field} budget invalid")
                except (FileNotFoundError, ValueError, UnicodeDecodeError) as exc:
                    issues.append(f"{key}: action ledger unavailable or invalid: {exc}")
                def check_terminal(terminal):
                    trace = terminal.get("action_trace", {})
                    if trace.get("count") == 0 and terminal.get("purpose") == "budget_tail_snapshot":
                        return
                    identity = (terminal.get("stream_id"), terminal.get("stream_episode_serial"))
                    actual = action_groups.pop(identity, {})
                    if (actual.get("count") != trace.get("count") or
                            actual.get("first") != trace.get("first_env_step") or
                            actual.get("last") != trace.get("last_env_step") or
                            not _near(actual.get("reward"), terminal.get("episode_return"), atol=1e-7)):
                        issues.append(f"{key}: terminal action trace or reward differs {identity}")
                episode_count = 0
                strict_training_terminals = 0
                strict_training_error_count = 0
                try:
                    for terminal in evidence.job_iter_jsonl(port, run_id, "episodes.jsonl"):
                        episode_count += 1
                        check_terminal(terminal)
                        replay_errors, replayed = audit_training_terminal(training_scene_lookup, terminal)
                        strict_training_terminals += int(replayed)
                        strict_training_error_count += len(replay_errors)
                        issues.extend(f"{key}: {message}" for message in replay_errors)
                        if terminal.get("completed_episode") is not True:
                            issues.append(f"{key}: completed episode ledger contains partial tail")
                        else:
                            step, reward = terminal.get("terminal_env_step"), terminal.get("episode_return")
                            if _number(step) and _number(reward):
                                training[key].append((step, reward))
                except (FileNotFoundError, ValueError, UnicodeDecodeError) as exc:
                    issues.append(f"{key}: episode ledger unavailable or invalid: {exc}")
                for terminal in tails:
                    check_terminal(terminal)
                    replay_errors, replayed = audit_training_terminal(training_scene_lookup, terminal)
                    strict_training_terminals += int(replayed)
                    strict_training_error_count += len(replay_errors)
                    issues.extend(f"{key}: {message}" for message in replay_errors)
                if action_total != status.get("env_step"):
                    issues.append(f"{key}: action count differs from recorded decision budget")
                task_audit[key].update(env_step=status.get("env_step"),
                                       action_rows=action_total, completed_episodes=episode_count,
                                       budget_tails=len(tails),
                                       validation_nodes=len(nodes),
                                       strict_training_terminals=strict_training_terminals,
                                       strict_training_error_count=strict_training_error_count,
                                       training_total_timing="not_recorded_not_filled")
                if any(x.get("purpose") != "budget_tail_snapshot" or x.get("completed_episode") is not False for x in tails):
                    issues.append(f"{key}: budget tail mixed with completed episodes")
                for node in nodes:
                    step = node.get("env_step")
                    name = f"validation/step_{step:06d}.jsonl" if isinstance(step, int) else "invalid"
                    content, verr = evidence.job_jsonl(port, run_id, name)
                    sha, sherr = evidence.sha(port, run_id, name)
                    if (verr or sherr or len(content) != 60 or node.get("rows") != 60 or
                            node.get("sha256") != sha or
                            {(r.get("dataset_index"), r.get("demand_multiplier")) for r in content} !=
                            {(i, level) for i in range(20) for level in LEVELS}):
                        issues.append(f"{key}: invalid 20-mother validation node {step}")
            elif key == "teacher":
                actual, data_error = evidence.sha(port, run_id, "scene_inputs/train/inputs.npz")
                if data_error or actual != config.get("datasets", {}).get("train", {}).get("sha256"):
                    issues.append("teacher: frozen train dataset SHA unavailable or differs")
                manifest, error = evidence.job_json(port, run_id, "teacher_manifest.json")
                if error or not manifest or manifest.get("completed") is not True:
                    issues.append("teacher: completed manifest missing")
                else:
                    if manifest.get("dataset_sha256") != config.get("datasets", {}).get("train", {}).get("sha256"):
                        issues.append("teacher: frozen train dataset SHA differs")
                    if manifest.get("labels", 0) > 20000 or manifest.get("trials", 0) + manifest.get("unverified_trials", 0) > 442368:
                        issues.append("teacher: demonstration or trial budget exceeded")
                    for item in manifest.get("scenes", []):
                        name = "demonstrations/" + Path(item.get("path", "")).name
                        digest, digest_error = evidence.sha(port, run_id, name)
                        if digest_error or digest != item.get("sha256"):
                            issues.append("teacher: demonstration shard SHA unavailable or differs")
                            break
            elif key == "BC":
                actual, data_error = evidence.sha(port, run_id, "scene_inputs/train/inputs.npz")
                if data_error or actual != config.get("datasets", {}).get("train", {}).get("sha256"):
                    issues.append("BC: frozen train dataset SHA unavailable or differs")
                if status.get("stage") == "completed":
                    artifact, error = evidence.job_json(port, run_id, "artifacts.json")
                    if error or not artifact:
                        issues.append("BC: artifact manifest unavailable")
        by_method = {method: [r for r in rows if r["method"] == method] for method in METHODS}
        for method, subset in by_method.items():
            grid = {(r["dataset_index"], r["demand_multiplier"]) for r in subset}
            expected_grid = {(i, level) for i in range(100) for level in LEVELS}
            if len(subset) != 300 or grid != expected_grid:
                issues.append(f"{method}: independent test grid incomplete")
        for index in range(100):
            for level in LEVELS:
                match = [r for r in rows if r["dataset_index"] == index and r["demand_multiplier"] == level]
                if len(match) == 6 and (len({r["scene_sha256"] for r in match}) != 1 or
                                        len({r["initial_sha256"] for r in match}) != 1):
                    issues.append(f"test mother {index} level {level}: scene/initial SHA differs")
        for row in rows:
            issues.extend(f"{row['task_key']} mother {row['dataset_index']} level {row['demand_multiplier']}: {error}"
                          for error in row["errors"])
        for method in ("P0", "P1"):
            task = task_audit.get(method)
            if task is None or task["stage"] not in ("completed", "stopped") or not _number(task.get("env_step")):
                issues.append(f"{method}: learning terminal budget unavailable")
            elif task["stage"] == "completed" and task["env_step"] != 262144:
                issues.append(f"{method}: completed learning budget is not 262144")
            elif task["stage"] == "stopped" and not task["scientific_stop"]:
                issues.append(f"{method}: stopped without certified scientific stop")
            if task and not task["validation_nodes"]:
                issues.append(f"{method}: no 20-mother validation nodes")
            if task and isinstance(task.get("env_step"), int) and task["env_step"] % 16384 == 0:
                actual_nodes = {node.get("env_step") for node in validation.get(method, [])}
                expected_nodes = set(range(16384, task["env_step"] + 1, 16384))
                if actual_nodes != expected_nodes:
                    issues.append(f"{method}: fixed validation node grid differs from actual endpoint")
            if task and isinstance(task.get("env_step"), int):
                steps = {row.get("checkpoint_step") for row in by_method[method]}
                if steps and steps != {task["env_step"]}:
                    issues.append(f"{method}: independent test model step differs from learning terminal")
        for key in ("teacher", "BC"):
            if key not in task_audit or task_audit[key]["stage"] != "completed":
                issues.append(f"{key}: completed provenance task missing")
        for method in METHODS:
            for shard in range(4):
                key = f"test_{method}_{(20418,20424,20473,20474)[shard]}"
                if key not in task_audit or task_audit[key]["stage"] != "completed":
                    issues.append(f"{key}: completed shard missing")
        summary = {}
        for method in METHODS:
            for level in LEVELS:
                subset = [r for r in by_method[method] if r["demand_multiplier"] == level]
                strict_complete = bool(subset) and all(
                    r["best_N_lt_50"] is not None and
                    isinstance(r["positive_band_0p8_1p2"], int) and
                    isinstance(r["positive_demand_beams"], int) for r in subset)
                summary[f"{method}@{level}"] = {"scene_count": len(subset),
                    "mother_count": len({r["dataset_index"] for r in subset}),
                    "best_j": _stats(r["best_j"] for r in subset),
                    "beam_carried_fraction": _stats(r["beam_carried_fraction"] for r in subset),
                    "aircraft_tdma_carried_fraction":
                        _stats(r["aircraft_tdma_carried_fraction"] for r in subset),
                    "load_cv": _stats(r["load_cv"] for r in subset),
                    "beam_mean_si": _stats(r["beam_mean_si"] for r in subset),
                    "beam_mean_sgm": _stats(r["beam_mean_sgm"] for r in subset),
                    "total_seconds": _stats(r["total_seconds"] for r in subset),
                    "algorithm_seconds": _stats(r["algorithm_seconds"] for r in subset),
                    "physical_seconds": _stats(r["physical_seconds"] for r in subset),
                    "beam_count": _stats(r["beam_count"] for r in subset),
                    "N_lt_50_count": (sum(r["best_N_lt_50"] is True for r in subset)
                                      if strict_complete else None),
                    "N_eq_200_count": (sum(r["best_N_eq_200"] is True for r in subset)
                                      if strict_complete else None),
                    "positive_band_0p8_1p2_beams":
                        (sum(r["positive_band_0p8_1p2"] for r in subset)
                         if strict_complete else None),
                    "positive_demand_beams":
                        (sum(r["positive_demand_beams"] for r in subset)
                         if strict_complete else None)}
        mothers = []
        for method in METHODS:
            for index in range(100):
                group = [r for r in by_method[method] if r["dataset_index"] == index]
                if len(group) == 3 and {r["demand_multiplier"] for r in group} == set(LEVELS):
                    mothers.append({"method": method, "dataset_index": index,
                                    "mean_best_j": mean(r["best_j"] for r in group if _number(r["best_j"]))
                                    if all(_number(r["best_j"]) for r in group) else None,
                                    "mean_aircraft_tdma_carried_fraction":
                                        mean(r["aircraft_tdma_carried_fraction"] for r in group
                                             if _number(r["aircraft_tdma_carried_fraction"]))
                                        if all(_number(r["aircraft_tdma_carried_fraction"]) for r in group)
                                        else None})
        plots, plot_error = _plot(output, rows, training, validation)
        if plot_error:
            issues.append(plot_error)
        if not plots or not {"test_carried_distribution.png", "train_episode_reward_raw.png",
                             "validation_best_j.png"}.issubset(plots):
            issues.append("required real-data curves incomplete")
        if strict_rows != 1800:
            issues.append("strict_frozen_scene_replay_incomplete")
        curve_metadata = {
            "train_episode_reward_raw.png": {"source": "P0/P1 episodes.jsonl",
                "excluded": "budget_tails.jsonl", "x": "RL decisions", "y": "episode reward",
                "raw_points": True, "trailing_mean_window_episodes": 100, "training_seed": 51001},
            "validation_best_j.png": {"source": "P0/P1 evaluations.jsonl with verified 60-row nodes",
                "x": "RL decisions", "y": "mean best R04 J", "training_seed": 51001},
            "test_carried_distribution.png": {"source": "independent results.jsonl best aircraft_service",
                "x": "sorted scene result", "y": "aircraft TDMA carried/all scene demand",
                "correlation_unit": "mother scene; three demand levels are related"}}
        (output / "curve_metadata.json").write_text(json.dumps(curve_metadata,
            ensure_ascii=False, indent=2), encoding="utf-8")
        for name, data in (("per_scene.json", rows), ("per_mother.json", mothers),
                           ("summary.json", summary), ("task_audit.json", task_audit)):
            (output / name).write_text(json.dumps(data, ensure_ascii=False, indent=2,
                                                 allow_nan=False), encoding="utf-8")
        if rows:
            with (output / "per_scene.csv").open("w", newline="", encoding="utf-8-sig") as stream:
                writer = csv.DictWriter(stream, fieldnames=[key for key in rows[0]
                                                            if key not in ("errors", "strict_facts")])
                writer.writeheader()
                writer.writerows({key: value for key, value in row.items()
                                  if key not in ("errors", "strict_facts")} for row in rows)
        audit = {"experiment_id": "S1-D12A", "scientific_commit": state.get("scientific_commit"),
                 "source": str(source), "formal_ready": not issues,
                 "scope": "formal evidence complete" if not issues else "partial QA only",
                 "task_count": len(task_audit), "test_rows": len(rows),
                 "strict_replayed_rows": strict_rows,
                 "method_test_counts": {key: len(value) for key, value in by_method.items()},
                 "limitations": ["GEO coverage replay reuses the project's pure ray implementation.",
                                 "Logged beam capacity and off-axis aircraft rate are checked algebraically; RF interference is not recomputed.",
                                 "Single training seed 51001; three demand levels share each mother scene."],
                 "plots": plots, "issues": sorted(set(issues))}
        (output / "audit.json").write_text(json.dumps(audit, ensure_ascii=False,
                                                     indent=2, allow_nan=False), encoding="utf-8")
        return audit
    finally:
        evidence.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", required=True, type=Path)
    parser.add_argument("--jobs", required=True, type=Path,
                        help="Downloaded jobs directory or ZIP stream")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    state = json.loads(args.state.read_text(encoding="utf-8"))
    result = analyze(state, args.jobs, args.output)
    print(json.dumps({"formal_ready": result["formal_ready"],
                      "test_rows": result["test_rows"], "issue_count": len(result["issues"]),
                      "audit": str(args.output / "audit.json")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
