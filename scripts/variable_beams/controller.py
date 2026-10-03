"""Persistent, idempotent S2-P17 workbench progression on 20424 only.

Requires a user-confirmed training scope and a clean frozen scientific commit.
It imports frozen inputs, submits V0/V1, waits for both terminal policies, then
switches this isolated workbench to one worker before sequential evaluation and
SCA. It never changes experiment budget or resumes a failed task automatically.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import urllib.request

import numpy as np

from remote_server import ensure_server, proc_birth


BASE = Path("/home/u2024110224/second_stage_satellite_rl/variable_beams_20261004/20424")
REPO = BASE / "repo"
ARCHIVE = BASE / "archive"
OPS = BASE / "ops"
PORT = 8781
MANIFEST = OPS / "controller_manifest.json"
SCOPE = OPS / "approved_scope.json"
LOCK = OPS / "controller.lock"
VALIDATION = BASE / "data/prepared/validation_eligible12_v2/record.json"
TEST = BASE / "data/prepared/test/record.json"
VALIDATION_SHA = "fa049ce2a881d1c5da81d2d6e2338821f7dfb6052684a999d438805d9b5e6525"
TEST_SHA = "202d3972c6e668af0f814585604ed3201e464490fc4cbcfea89e79fe3426e404"
T0 = Path("/home/u2024110224/second_stage_satellite_rl/g11/20418/archive/models/runs/workbench/train_sac_20261001_162126_33e89e95/final")
T0_SHA = "ae095e8246d22b2716ba154315cee18584917d215580821cb53f9aeba78eb824"
T0_METADATA_SHA = "1d4055e40ba6314ad8753ae64c1321a30fff7d7c3b25429b86ffca92c5388dde"
REQUESTS = {
    "V0_train": "s2p17-v0-train-51001", "V1_train": "s2p17-v1-train-51001",
    "V0_eval": "s2p17-v0-jan29-eval", "V1_eval": "s2p17-v1-jan29-eval",
    "H0_eval": "s2p17-h0-jan29-eval", "V0_sca": "s2p17-v0-jan29-sca",
    "V1_sca": "s2p17-v1-jan29-sca", "H0_sca": "s2p17-h0-jan29-sca",
    "fixed_sca": "s2p17-fixed-jan29-sca", "greedy_sca": "s2p17-greedy-jan29-sca",
}
EVALUATIONS = ("V0_eval", "V1_eval", "H0_eval")
REFINEMENTS = ("V0_sca", "V1_sca", "H0_sca", "fixed_sca", "greedy_sca")
SOURCES = {"V0_sca": ("V0_eval", "model"), "V1_sca": ("V1_eval", "model"),
           "H0_sca": ("H0_eval", "model"), "fixed_sca": ("V0_eval", "fixed"),
           "greedy_sca": ("V0_eval", "greedy")}
HELD_OUT = {60, 90, 140, 180}
ACTIVE = {"starting", "initializing", "running", "validating", "evaluating", "queued", "stopping"}
ERROR = {"failed", "stopped", "cancelled", "interrupted", "invalid"}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".pending")
    with pending.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    pending.replace(path)


def api(method: str, path: str, payload=None, timeout=120):
    data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(f"http://127.0.0.1:{PORT}/api{path}",
                                     data=data, method=method,
                                     headers={"Content-Type": "application/json", "Cache-Control": "no-cache"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def scope_contract(expected_sha: str) -> dict:
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha) or sha(SCOPE) != expected_sha:
        raise RuntimeError("Approved training scope file/SHA missing or changed")
    scope = json.loads(SCOPE.read_text(encoding="utf-8"))
    if (scope.get("experiment_id") != "S2-P17" or scope.get("status") != "user_confirmed"
            or scope.get("decision") not in ("eligible520", "replace576")
            or scope.get("mother_count") not in (520, 576)
            or scope.get("derived_scene_count") != scope["mother_count"] * 14
            or not scope.get("user_message_reference")
            or not re.fullmatch(r"[0-9a-f]{64}", scope.get("train_record_sha256", ""))
            or scope.get("h0_compatibility_status") != "passed"
            or not scope.get("h0_compatibility_receipt")):
        raise RuntimeError("User-confirmed mother count and H0 compatibility proof required")
    if (scope["decision"] == "eligible520") != (scope["mother_count"] == 520):
        raise RuntimeError("Approved mother count contradicts decision")
    source = Path(scope.get("train_record_path", ""))
    if not source.is_file() or not source.is_relative_to(BASE / "data/prepared"):
        raise RuntimeError("Approved train record must be under isolated 20424 prepared inputs")
    return scope


def source_record(path: Path, expected_sha: str, role: str, count: int) -> tuple[dict, list[str]]:
    if sha(path) != expected_sha:
        raise RuntimeError(f"{role} source record SHA differs")
    record = json.loads(path.read_text(encoding="utf-8"))
    source = record.get("source") or {}
    dates = {"train": {"2024-01-08", "2024-01-09"},
             "validation": {"2024-01-12"}, "test": {"2024-01-29"}}[role]
    if (record.get("count") != count or record.get("profile") != "europe_v1"
            or source.get("experiment_id") != "S2-P17"
            or source.get("partition_role") != role
            or set(source.get("source_dates", [])) != dates
            or len(record.get("files", [])) != count):
        raise RuntimeError(f"{role} prepared record count/profile/provenance differs")
    mothers = set()
    groups = {}
    ordered = []
    details = {}
    for index, item in enumerate(record["files"]):
        path_item = Path(item["path"])
        if not path_item.is_file() or path_item.parent != path.parent or sha(path_item) != item["sha256"]:
            raise RuntimeError(f"{role} prepared NPZ SHA/path mismatch at {index}")
        n = item.get("beam_count")
        load = item.get("load_multiplier")
        mother = item.get("mother_scene_id")
        if (type(n) is not int or not 50 <= n <= 200 or load not in (0.75, 1.25)
                or not mother or item.get("partition_role") != role
                or item.get("selected_time_utc", "")[:10] not in dates):
            raise RuntimeError(f"{role} N/load/mother metadata invalid at {index}")
        if role != "test" and n in HELD_OUT:
            raise RuntimeError("Final held-out N appeared in train or validation")
        mothers.add(mother)
        groups.setdefault((mother, n), set()).add(load)
        ordered.append(item["sha256"])
        detail = item.get("observation_detail") or {}
        if detail.get("path"):
            details[detail["path"]] = detail["sha256"]
    for name, digest in details.items():
        if sha(Path(name)) != digest:
            raise RuntimeError(f"{role} observation detail SHA differs")
    expected_mothers = count // (14 if role == "train" else 12 if role == "validation" else 22)
    expected_n = 7 if role == "train" else 6 if role == "validation" else 11
    if (len(mothers) != expected_mothers or len(groups) != expected_mothers * expected_n
            or any(loads != {0.75, 1.25} for loads in groups.values())):
        raise RuntimeError(f"{role} mother/N/load grid incomplete")
    return record, ordered


def validate_system(commit: str) -> dict:
    head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain"],
                           capture_output=True, text=True, check=True).stdout.strip()
    system = api("GET", "/system")
    if (head != commit or dirty or system.get("cuda") is not True
            or Path(system.get("source_root", "")).resolve() != REPO
            or Path(system.get("archive_root", "")).resolve() != ARCHIVE
            or Path(system.get("package_file", "")).resolve() != REPO / "src/satellite_rl/__init__.py"):
        raise RuntimeError("Live workbench source/CUDA/archive differs from clean scientific commit")
    return system


def initial_manifest(commit: str, scope_sha: str, scope: dict) -> dict:
    jobs = {}
    for key, request_id in REQUESTS.items():
        kind = "train" if key.endswith("_train") else "evaluate" if key.endswith("_eval") else "refine"
        jobs[key] = {"kind": kind, "request_id": request_id, "run_id": None, "stage": "not_submitted",
                     "source_run_id": None, "source_group": None, "model_id": None,
                     "scene_id": None, "job_path": None, "status_path": None,
                     "results_path": None, "config_sha256": None, "job_sha256": None,
                     "status_sha256": None, "results_sha256": None,
                     "model_path": None, "model_sha256": None,
                     "model_metadata_sha256": None, "scene_record_sha256": None,
                     "source_manifest_sha256": None, "last_checked_utc": None}
    return {"schema_version": 1, "experiment_id": "S2-P17",
            "scientific_commit": commit, "scope_sha256": scope_sha,
            "scope": {"decision": scope["decision"], "mother_count": scope["mother_count"],
                      "derived_scene_count": scope["derived_scene_count"],
                      "user_message_reference": scope["user_message_reference"]},
            "target": {"ssh_port": 20424, "workbench_port": PORT,
                       "repo": str(REPO), "archive": str(ARCHIVE)},
            "source": {"t0_policy_sha256": T0_SHA, "t0_model_sha256": T0_METADATA_SHA,
                       "h0_compatibility_receipt": scope["h0_compatibility_receipt"]},
            "datasets": {}, "jobs": jobs,
            "totals": {"train_env_steps": 800000, "validation_allocations": 6048,
                       "initial_allocations": 4840, "sca_refinements": 4840,
                       "reported_original_and_final_rows": 9680},
            "phase": "prepared_no_jobs", "checked_utc": now(), "last_error": None}


def load_manifest(commit: str, scope_sha: str, scope: dict) -> dict:
    if MANIFEST.is_file():
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        if (manifest.get("experiment_id") != "S2-P17" or manifest.get("scientific_commit") != commit
                or manifest.get("scope_sha256") != scope_sha
                or manifest.get("scope", {}).get("mother_count") != scope["mother_count"]
                or set(manifest.get("jobs", {})) != set(REQUESTS)):
            raise RuntimeError("Persistent controller manifest conflicts with frozen scope/source")
        return manifest
    return initial_manifest(commit, scope_sha, scope)


def _match_imported(source_id: str, role: str) -> list[dict]:
    matches = []
    for record in api("GET", "/scenes").get("scenes", []):
        provenance = record.get("source") or {}
        old = provenance.get("reimport") or {}
        if (old.get("record_id") == source_id and provenance.get("experiment_id") == "S2-P17"
                and provenance.get("partition_role") == role):
            matches.append(record)
    return matches


def import_scene_set(manifest: dict, role: str, path: Path, expected_sha: str, count: int) -> None:
    if role in manifest["datasets"]:
        item = manifest["datasets"][role]
        record_path = Path(item["record_path"])
        if (not record_path.is_file() or sha(record_path) != item["record_sha256"]
                or sha(path) != item["source_record_sha256"]):
            raise RuntimeError(f"{role} imported or prepared record changed")
        return
    source, source_hashes = source_record(path, expected_sha, role, count)
    found = _match_imported(source["id"], role)
    if len(found) > 1:
        raise RuntimeError(f"Duplicate {role} scene imports")
    if not found:
        try:
            api("POST", "/scenes/import",
                {"path": str(path.parent), "name": f"S2-P17 {role} frozen inputs", "profile": "europe_v1"},
                timeout=3600)
        except Exception:
            found = _match_imported(source["id"], role)
            if len(found) != 1:
                raise
        else:
            found = _match_imported(source["id"], role)
    if len(found) != 1:
        raise RuntimeError(f"{role} scene import did not produce one identified record")
    imported = found[0]
    if imported.get("count") != count or len(imported.get("files", [])) != count:
        raise RuntimeError(f"{role} imported count differs")
    reimport = (imported.get("source") or {}).get("reimport") or {}
    if [item["sha256"] for item in reimport.get("files", [])] != source_hashes:
        raise RuntimeError(f"{role} imported original file SHA/order differs")
    for index, (old, new) in enumerate(zip(source["files"], imported["files"], strict=True)):
        for key in ("beam_count", "mother_scene_id", "load_multiplier", "partition_role",
                    "selected_time_utc", "source_sha256"):
            if old.get(key) != new.get(key):
                raise RuntimeError(f"{role} imported metadata differs at {index}: {key}")
        old_file, new_file = Path(old["path"]), Path(new["path"])
        if not new_file.is_file() or sha(new_file) != new["sha256"]:
            raise RuntimeError(f"{role} imported NPZ SHA differs at {index}")
        with np.load(old_file, allow_pickle=False) as left, np.load(new_file, allow_pickle=False) as right:
            if set(left.files) != set(right.files) or any(
                    left[key].dtype != right[key].dtype or not np.array_equal(left[key], right[key])
                    for key in left.files):
                raise RuntimeError(f"{role} imported array differs at {index}")
    imported_path = Path(imported["path"]) / "record.json"
    if not imported_path.is_file():
        raise RuntimeError("Workbench imported record file missing")
    manifest["datasets"][role] = {"scene_id": imported["id"], "count": count,
                                  "record_path": str(imported_path),
                                  "record_sha256": sha(imported_path),
                                  "source_record_path": str(path),
                                  "source_record_sha256": expected_sha,
                                  "ordered_source_npz_sha256": hashlib.sha256(
                                      "".join(source_hashes).encode()).hexdigest(),
                                  "ordered_imported_npz_sha256": hashlib.sha256(
                                      "".join(item["sha256"] for item in imported["files"]).encode()).hexdigest()}
    atomic_json(MANIFEST, manifest)


def import_h0(manifest: dict) -> str:
    if sha(T0 / "policy.pt") != T0_SHA or sha(T0 / "model.json") != T0_METADATA_SHA:
        raise RuntimeError("Frozen T0/H0 endpoint SHA differs")
    models = api("GET", "/models").get("models", [])
    def matches():
        return [model for model in models
                if (model.get("provenance") or {}).get("run_id") == "train_sac_20261001_162126_33e89e95"
                and (model.get("provenance") or {}).get("env_step") == 150000
                and (model.get("provenance") or {}).get("git_commit") == "85f667cd5731e284c19b2590b5dcf5308a58f234"
                and Path(model.get("path", "")).is_dir()
                and sha(Path(model["path"]) / "policy.pt") == T0_SHA]
    found = matches()
    if len(found) > 1:
        raise RuntimeError("Duplicate imported H0 models")
    if not found:
        try:
            api("POST", "/models/import", {"path": str(T0), "name": "S2-P17 H0 · G11 T0 150000"}, timeout=300)
        except Exception:
            models = api("GET", "/models").get("models", [])
            found = matches()
            if len(found) != 1:
                raise
        else:
            models = api("GET", "/models").get("models", [])
            found = matches()
    if len(found) != 1:
        raise RuntimeError("Exact H0 model was not registered")
    manifest["source"]["h0_model_id"] = found[0]["id"]
    manifest["source"]["h0_model_path"] = found[0]["path"]
    atomic_json(MANIFEST, manifest)
    return found[0]["id"]


def training_config(mode: str) -> dict:
    catalog = api("GET", "/catalog")
    preset = next((item for item in catalog.get("network_presets", [])
                   if item.get("id") == "tsac_14_variable_context"), None)
    if preset is None:
        raise RuntimeError("Frozen workbench catalog lacks the variable-beam preset")
    config = deepcopy(catalog["defaults"])
    config["name"] = "S2-P17 V0 固定100束" if mode == "fixed_100" else "S2-P17 V1 混合束数"
    config["algorithm"] = "sac"
    config["network"].update(preset["network"])
    config["state"].update(features=preset["state_features"], demand_visibility="known",
                           context_level="none")
    config["reward"]["preset"] = "legacy_repaired"
    config["environment"].update(profile="europe_v1", beam_count_max=200,
                                 power_budget_w=3000, action_mode="legacy_clip",
                                 action_mask="joint_feasible",
                                 power_levels_w=[5, 10, 15, 20, 25, 30, 35, 40, 45, 50])
    config["training"].update(seed=51001, scene_order="variable_beams", scene_order_seed=51001,
                              beam_count_mode=mode, beam_count_min=50, beam_count_max=200,
                              held_out_beam_counts=[60, 90, 140, 180], total_steps=400000,
                              batch_size=256, replay_capacity=20000, warmup_steps=2000,
                              checkpoint_interval=10000, update_every=1, gradient_steps=1,
                              gamma=0.99, tau=0.005, actor_lr=1e-5, critic_lr=1e-4,
                              gradient_clip=10, alpha_mode="fixed", fixed_alpha=0.01,
                              device="cuda")
    config["evaluation"].update(validation_count=144, test_count=0,
                                validation_seed=53001, interval_steps=20000,
                                extra_steps=[0], repeats=1, action_mode="stochastic")
    validated = api("POST", "/validate", config)["config"]
    if validated != config:
        raise RuntimeError("Workbench normalized the formal config unexpectedly")
    return validated


def find_request(request_id: str) -> dict | None:
    found = [item for item in api("GET", "/runs").get("runs", [])
             if item.get("request_id") == request_id and not item.get("read_only")]
    if len(found) > 1:
        raise RuntimeError("Duplicate workbench request ID")
    return found[0] if found else None


def submit_job(path: str, body: dict, request_id: str, *, timeout=180) -> dict:
    current = find_request(request_id)
    if current:
        return api("GET", "/runs/" + current["run_id"])["job"]
    try:
        created = api("POST", path, body, timeout=timeout)
    except Exception:
        current = find_request(request_id)
        if current:
            return api("GET", "/runs/" + current["run_id"])["job"]
        raise
    if created.get("request_id") != request_id:
        raise RuntimeError("Workbench returned a different request ID")
    return created


def refresh_job(manifest: dict, key: str) -> dict:
    entry = manifest["jobs"][key]
    if not entry["run_id"]:
        return entry
    detail = api("GET", "/runs/" + entry["run_id"])
    job, status = detail["job"], detail["status"]
    if (job.get("run_id") != entry["run_id"] or job.get("request_id") != entry["request_id"]
            or job.get("git_commit") != manifest["scientific_commit"]
            or job.get("experiment_id") != "S2-P17"):
        raise RuntimeError(f"{key} workbench job identity/scientific source differs")
    root = ARCHIVE / "outputs/workbench/jobs" / entry["run_id"]
    source_manifest = Path(job["source_root"]) / "source_manifest.json"
    source_data = json.loads(source_manifest.read_text(encoding="utf-8"))
    if source_data.get("git_commit") != manifest["scientific_commit"]:
        raise RuntimeError(f"{key} frozen job source commit differs")
    config_path, job_path, status_path = (root / name for name in ("config.json", "job.json", "status.json"))
    scene_role = "test" if entry["kind"] != "train" else "train"
    scene_record = root / "scene_inputs" / scene_role / "record.json"
    if not scene_record.is_file():
        raise RuntimeError(f"{key} frozen {scene_role} scene record missing")
    imported = manifest["datasets"][scene_role]
    frozen = json.loads(scene_record.read_text(encoding="utf-8"))
    original = json.loads(Path(imported["record_path"]).read_text(encoding="utf-8"))
    if ([item["sha256"] for item in frozen["files"]]
            != [item["sha256"] for item in original["files"]]):
        raise RuntimeError(f"{key} task-frozen scene SHA/order differs from imported dataset")
    entry.update(stage=status.get("stage"), job_path=str(job_path), status_path=str(status_path),
                 results_path=str(root / "results.json"), config_sha256=sha(config_path),
                 job_sha256=sha(job_path), status_sha256=sha(status_path),
                 scene_record_sha256=sha(scene_record),
                 source_manifest_sha256=sha(source_manifest),
                 results_sha256=sha(root / "results.json") if (root / "results.json").is_file() else None,
                 last_checked_utc=now())
    entry["env_step"] = status.get("env_step")
    entry["budget_env_steps"] = status.get("budget_env_steps")
    entry["evaluation_units"] = job.get("evaluation_units")
    if entry["kind"] == "train":
        model_dir = Path(job["model_dir"]) / "final"
    else:
        model_dir = root / "model_input"
    policy = model_dir / "policy.pt"
    metadata = model_dir / "model.json"
    entry["model_path"] = str(policy) if policy.is_file() else None
    entry["model_sha256"] = sha(policy) if policy.is_file() else None
    entry["model_metadata_sha256"] = sha(metadata) if metadata.is_file() else None
    pid = status.get("pid")
    birth = proc_birth(pid) if isinstance(pid, int) else None
    entry["worker_pid"] = pid
    entry["worker_start_ticks"] = birth
    launch = root / "launch.json"
    if launch.is_file():
        launch_data = json.loads(launch.read_text(encoding="utf-8"))
        entry["worker_launch_pid"] = launch_data.get("pid")
        entry["worker_launch_start_time"] = launch_data.get("process_start_time")
        if status.get("stage") in {"running", "validating", "evaluating"}:
            if launch_data.get("pid") != pid or birth is None:
                raise RuntimeError(f"{key} live worker PID/birth unavailable")
            boot = next(int(line.split()[1]) for line in Path("/proc/stat").read_text().splitlines()
                        if line.startswith("btime "))
            actual_birth = boot + birth / os.sysconf("SC_CLK_TCK")
            if (launch_data.get("process_start_time") is None
                    or abs(actual_birth - launch_data["process_start_time"]) >= 0.02):
                raise RuntimeError(f"{key} worker creation time differs from launch receipt")
    elif status.get("stage") in {"running", "validating", "evaluating"}:
        raise RuntimeError(f"{key} active worker lacks launch receipt")
    if status.get("stage") == "completed" and (entry["results_sha256"] is None
            or entry["model_sha256"] is None):
        raise RuntimeError(f"{key} completed without immutable results/model asset")
    return entry


def final_model(manifest: dict, training_key: str) -> str:
    entry = refresh_job(manifest, training_key)
    if entry["stage"] != "completed":
        raise RuntimeError("Training endpoint not complete")
    run_id = entry["run_id"]
    model_path = Path(entry["model_path"]).parent
    metadata = json.loads((model_path / "model.json").read_text(encoding="utf-8"))
    provenance = metadata.get("provenance") or {}
    if (provenance.get("run_id") != run_id or provenance.get("env_step") != 400000
            or provenance.get("git_commit") != manifest["scientific_commit"]
            or (metadata.get("config") or {}).get("training", {}).get("seed") != 51001):
        raise RuntimeError(f"{training_key} final policy provenance differs from fixed endpoint")
    models = api("GET", "/models").get("models", [])
    found = [model for model in models if Path(model.get("path", "")) == model_path
             and (model.get("provenance") or {}).get("env_step") == 400000]
    if len(found) != 1:
        raise RuntimeError(f"{training_key} exact final model not in workbench catalog")
    if sha(model_path / "policy.pt") != entry["model_sha256"]:
        raise RuntimeError("Final policy SHA changed during model selection")
    return found[0]["id"]


def register_job(manifest: dict, key: str, job: dict, *, model_id=None,
                 scene_id=None, source_run_id=None, source_group=None) -> None:
    entry = manifest["jobs"][key]
    if entry["run_id"] and entry["run_id"] != job["run_id"]:
        raise RuntimeError(f"{key} already bound to a different workbench run")
    if job.get("request_id") != entry["request_id"]:
        raise RuntimeError("Workbench returned another request identity")
    entry.update(run_id=job["run_id"], model_id=model_id, scene_id=scene_id,
                 source_run_id=source_run_id, source_group=source_group)
    refresh_job(manifest, key)
    atomic_json(MANIFEST, manifest)


def submit_training(manifest: dict) -> None:
    for key, mode in (("V0_train", "fixed_100"), ("V1_train", "mixed_four_buckets")):
        if manifest["jobs"][key]["run_id"]:
            continue
        config = training_config(mode)
        body = {"config": config,
                "scene_id": manifest["datasets"]["train"]["scene_id"],
                "validation_scene_id": manifest["datasets"]["validation"]["scene_id"],
                "request_id": REQUESTS[key]}
        job = submit_job("/runs", body, REQUESTS[key], timeout=1800)
        actual = api("GET", "/runs/" + job["run_id"])
        if (actual["config"] != config or job.get("kind") != "train"
                or job.get("scene_sets", {}).get("train", {}).get("id") != body["scene_id"]
                or job.get("scene_sets", {}).get("validation", {}).get("id") != body["validation_scene_id"]):
            raise RuntimeError(f"{key} existing request has different training config or inputs")
        register_job(manifest, key, job, scene_id=body["scene_id"])
    manifest["phase"] = "training"
    manifest["checked_utc"] = now()
    atomic_json(MANIFEST, manifest)


def submit_evaluation(manifest: dict, key: str) -> None:
    model_key = {"V0_eval": "V0_train", "V1_eval": "V1_train"}.get(key)
    model_id = final_model(manifest, model_key) if model_key else manifest["source"]["h0_model_id"]
    scene_id = manifest["datasets"]["test"]["scene_id"]
    body = {"model_id": model_id, "scene_id": scene_id,
            "experiment_id": "S2-P17", "action_mode": "stochastic",
            "action_seed": 53001, "repeats": 3,
            "baselines": ["fixed", "greedy"] if key == "V0_eval" else [],
            "request_id": REQUESTS[key]}
    job = submit_job("/evaluations", body, REQUESTS[key], timeout=1800)
    if (job.get("experiment_id") != "S2-P17" or job.get("repeats") != 3
            or job.get("model_id") != model_id or job.get("scene_id") != scene_id
            or job.get("action_mode") != "stochastic" or job.get("action_seed") != 53001
            or job.get("baselines") != sorted(body["baselines"])):
        raise RuntimeError("S2-P17 evaluation source identity or repeats differ")
    expected_units = 2200 if key == "V0_eval" else 1320
    if job.get("evaluation_units") not in (None, expected_units):
        raise RuntimeError("S2-P17 evaluation allocation budget differs")
    register_job(manifest, key, job, model_id=model_id, scene_id=scene_id)


def submit_refinement(manifest: dict, key: str) -> None:
    source_key, group = SOURCES[key]
    source_run_id = manifest["jobs"][source_key]["run_id"]
    if not source_run_id or manifest["jobs"][source_key]["stage"] != "completed":
        raise RuntimeError("SCA source evaluation incomplete")
    defaults = api("GET", "/power-refinement/defaults")["defaults"]
    controls = deepcopy(defaults)
    controls.update(methods=["none", "sca", "sca_si_lp"], power_budget_mode="total",
                    total_power_w=3000.0,
                    repeats=1 if group in ("fixed", "greedy") else 3,
                    action_mode="deterministic" if group in ("fixed", "greedy") else "stochastic",
                    action_seed=53001)
    checked = api("POST", "/power-refinement/validate", controls)["config"]
    if checked != controls:
        raise RuntimeError("SCA controls changed during workbench validation")
    body = {"source_run_id": source_run_id, "source_group": group,
            "config": controls, "request_id": REQUESTS[key]}
    job = submit_job("/power-refinement/runs", body, REQUESTS[key], timeout=1800)
    if (job.get("experiment_id") != "S2-P17" or job.get("initial_source", {}).get("group") != group
            or job.get("initial_source", {}).get("run_id") != source_run_id
            or job.get("power_refinement") != controls):
        raise RuntimeError("SCA job did not freeze the approved saved source")
    if job.get("evaluation_units") != (1320 if group in ("fixed", "greedy") else 3960):
        raise RuntimeError("SCA initial/A/B method budget differs")
    register_job(manifest, key, job, model_id=job.get("model_id"),
                 scene_id=manifest["datasets"]["test"]["scene_id"],
                 source_run_id=source_run_id, source_group=group)


def check_server_mode(commit: str, required: str) -> None:
    launch_path = OPS / "workbench_8781_launch.json"
    launch = json.loads(launch_path.read_text(encoding="utf-8"))
    limit = 2 if required == "training" else 1
    if (launch.get("commit") != commit or launch.get("max_workers") != limit
            or proc_birth(launch.get("pid")) != launch.get("start_ticks")):
        raise RuntimeError("Current 20424 workbench PID/birth or worker limit differs")


def step(commit: str, scope_sha: str) -> dict:
    scope = scope_contract(scope_sha)
    validate_system(commit)
    manifest = load_manifest(commit, scope_sha, scope)
    if not manifest["datasets"]:
        check_server_mode(commit, "training")
        sources = (("train", Path(scope["train_record_path"]), scope["train_record_sha256"],
                    scope["derived_scene_count"]),
                   ("validation", VALIDATION, VALIDATION_SHA, 144),
                   ("test", TEST, TEST_SHA, 440))
        for role, path, expected, count in sources:
            import_scene_set(manifest, role, path, expected, count)
        import_h0(manifest)
        manifest["phase"] = "inputs_imported_no_training"
        manifest["checked_utc"] = now()
        atomic_json(MANIFEST, manifest)
    elif set(manifest["datasets"]) != {"train", "validation", "test"}:
        # A prior import may have ended between roles; resume only missing roles.
        sources = (("train", Path(scope["train_record_path"]), scope["train_record_sha256"],
                    scope["derived_scene_count"]),
                   ("validation", VALIDATION, VALIDATION_SHA, 144),
                   ("test", TEST, TEST_SHA, 440))
        for role, path, expected, count in sources:
            import_scene_set(manifest, role, path, expected, count)
        import_h0(manifest)
    if not manifest["source"].get("h0_model_id"):
        import_h0(manifest)
    if any(not manifest["jobs"][key]["run_id"] for key in ("V0_train", "V1_train")):
        check_server_mode(commit, "training")
        submit_training(manifest)
        return manifest
    for key in ("V0_train", "V1_train"):
        entry = refresh_job(manifest, key)
        if entry["stage"] in ERROR:
            manifest["phase"] = "attention_required"
            manifest["last_error"] = f"{key} stage={entry['stage']}; no automatic resume or extra steps"
            manifest["checked_utc"] = now()
            atomic_json(MANIFEST, manifest)
            return manifest
    if any(manifest["jobs"][key]["stage"] != "completed" for key in ("V0_train", "V1_train")):
        manifest["phase"] = "training"
        manifest["checked_utc"] = now()
        atomic_json(MANIFEST, manifest)
        return manifest
    for key in ("V0_train", "V1_train"):
        final_model(manifest, key)
    ensure_server(commit, "evaluation")
    check_server_mode(commit, "evaluation")
    for key in EVALUATIONS:
        entry = manifest["jobs"][key]
        if not entry["run_id"]:
            submit_evaluation(manifest, key)
            manifest["phase"] = "evaluation"
            manifest["checked_utc"] = now()
            atomic_json(MANIFEST, manifest)
            return manifest
        refresh_job(manifest, key)
        if entry["stage"] in ERROR:
            manifest["phase"] = "attention_required"
            manifest["last_error"] = f"{key} stage={entry['stage']}; no automatic repeat"
            manifest["checked_utc"] = now()
            atomic_json(MANIFEST, manifest)
            return manifest
        if entry["stage"] != "completed":
            manifest["phase"] = "evaluation"
            manifest["checked_utc"] = now()
            atomic_json(MANIFEST, manifest)
            return manifest
    for key in REFINEMENTS:
        entry = manifest["jobs"][key]
        if not entry["run_id"]:
            submit_refinement(manifest, key)
            manifest["phase"] = "sca"
            manifest["checked_utc"] = now()
            atomic_json(MANIFEST, manifest)
            return manifest
        refresh_job(manifest, key)
        if entry["stage"] in ERROR:
            manifest["phase"] = "attention_required"
            manifest["last_error"] = f"{key} stage={entry['stage']}; no automatic repeat"
            manifest["checked_utc"] = now()
            atomic_json(MANIFEST, manifest)
            return manifest
        if entry["stage"] != "completed":
            manifest["phase"] = "sca"
            manifest["checked_utc"] = now()
            atomic_json(MANIFEST, manifest)
            return manifest
    manifest["phase"] = "awaiting_report_and_archive"
    manifest["checked_utc"] = now()
    atomic_json(MANIFEST, manifest)
    return manifest


def plan(commit: str, scope_sha: str) -> dict:
    scope = scope_contract(scope_sha)
    system = validate_system(commit)
    sources = (("train", Path(scope["train_record_path"]), scope["train_record_sha256"],
                scope["derived_scene_count"]),
               ("validation", VALIDATION, VALIDATION_SHA, 144),
               ("test", TEST, TEST_SHA, 440))
    checks = {}
    for role, path, expected, count in sources:
        record, ordered = source_record(path, expected, role, count)
        checks[role] = {"record_path": str(path), "record_sha256": expected,
                        "count": count, "mother_count": len(set(item["mother_scene_id"] for item in record["files"])),
                        "ordered_npz_sha256": hashlib.sha256("".join(ordered).encode()).hexdigest()}
    if sha(T0 / "policy.pt") != T0_SHA or sha(T0 / "model.json") != T0_METADATA_SHA:
        raise RuntimeError("T0/H0 frozen endpoint differs")
    return {"status": "ready_to_start_controller_no_jobs_created", "checked_utc": now(),
            "experiment_id": "S2-P17", "scientific_commit": commit,
            "scope_sha256": scope_sha, "decision": scope["decision"],
            "workbench_source_root": system.get("source_root"),
            "datasets": checks, "t0_policy_sha256": T0_SHA,
            "planned_request_ids": REQUESTS,
            "planned_eval_allocations": {"V0_eval": 2200, "V1_eval": 1320, "H0_eval": 1320},
            "planned_sca_initial_sources": 4840,
            "formal_task_count_before_execution": len([item for item in api("GET", "/runs").get("runs", [])
                                                       if item.get("experiment_id") == "S2-P17"])}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--commit", required=True)
    parser.add_argument("--scope-sha256", required=True)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--plan", action="store_true")
    action.add_argument("--run-once", action="store_true")
    action.add_argument("--daemon", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.commit.lower()):
        raise ValueError("Full scientific commit required")
    commit = args.commit.lower()
    if args.plan:
        print(json.dumps(plan(commit, args.scope_sha256), ensure_ascii=False))
        return
    import fcntl
    OPS.mkdir(parents=True, exist_ok=True)
    with LOCK.open("a+b") as locked:
        try:
            fcntl.flock(locked, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another S2-P17 controller already owns the lock") from error
        while True:
            try:
                manifest = step(commit, args.scope_sha256)
            except BaseException as error:
                failure = {"status": "controller_error_no_auto_resume", "checked_utc": now(),
                           "error": repr(error), "commit": commit,
                           "scope_sha256": args.scope_sha256}
                atomic_json(OPS / "controller_failure.json", failure)
                raise
            print(json.dumps({"checked_utc": now(), "phase": manifest["phase"],
                              "run_ids": {key: item["run_id"] for key, item in manifest["jobs"].items()}},
                             ensure_ascii=False), flush=True)
            if (not args.daemon or manifest["phase"] in
                    ("attention_required", "awaiting_report_and_archive")):
                break
            time.sleep(120)


if __name__ == "__main__":
    main()
