"""Durable S1-02 workbench coordinator for the authorized A-then-B sequence.

Run only on instance 20418 after the science commit, data imports, and gates in
plan.json have been reviewed. Every formal calculation is a workbench job.
The controller never repairs or resubmits failed jobs automatically.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen


REMOTE = Path("/home/u2024110224/first_stage_satellite_rl/s1_02/20418")
S101 = Path("/home/u2024110224/first_stage_satellite_rl")
TERMINAL = {"completed", "failed", "stopped", "cancelled"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path, default=None):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def api(base: str, endpoint: str, body=None, *, timeout=120):
    url = base.rstrip("/") + "/api/" + endpoint.lstrip("/")
    payload = None if body is None else json.dumps(body, ensure_ascii=False).encode("utf-8")
    request = Request(url, data=payload,
                      headers={"Content-Type": "application/json"} if payload is not None else {})
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except HTTPError as error:
        message = error.read().decode("utf-8", "replace")[:1500]
        raise RuntimeError(f"Workbench HTTP {error.code}: {message}") from error


def canonical_sha(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


class Controller:
    def __init__(self, plan_file: Path, state_file: Path):
        self.plan_file = plan_file.resolve()
        self.plan = read_json(self.plan_file)
        if not isinstance(self.plan, dict):
            raise ValueError("Frozen controller plan is missing")
        self.plan_sha = digest(self.plan_file)
        self.state_file = state_file.resolve()
        self.state = read_json(self.state_file, {"experiment_id": "S1-02",
                                                  "created_utc": utc_now(),
                                                  "plan_sha256": self.plan_sha,
                                                  "phase": "preflight",
                                                  "datasets": {}, "dataset_records": {},
                                                  "jobs": {}, "events": []})
        if self.state["plan_sha256"] != self.plan_sha:
            raise RuntimeError("Controller plan changed after first use")
        replacements = self.plan.get("recovery_context", {}).get("previous_failed_jobs", {})
        if "replaces_run_ids" in self.state and self.state["replaces_run_ids"] != replacements:
            raise RuntimeError("Controller replacement-task contract changed")
        self.state["replaces_run_ids"] = replacements
        self.base = f"http://127.0.0.1:{self.plan['api_port']}"
        self.request_prefix = "S1_02_" + self.plan_sha[:12]
        self.validate_plan()

    def log(self, event: str, **details) -> None:
        self.state["events"].append({"at_utc": utc_now(), "event": event, **details})
        self.state["updated_utc"] = utc_now()
        save_json(self.state_file, self.state)

    def phase(self, value: str) -> None:
        if self.state["phase"] != value:
            self.state["phase"] = value
            self.log("phase", value=value)

    def validate_plan(self) -> None:
        plan = self.plan
        expected = {"train": ["2024-01-08", "2024-01-09"],
                    "validation": ["2024-01-12"],
                    "A_test": ["2024-01-28"], "B_test": ["2024-01-30"]}
        if (plan.get("experiment_id") != "S1-02" or
                plan.get("execution_target") != "u2024110224@10.160.17.102:20418" or
                plan.get("source_commit") is None or
                len(plan["source_commit"]) != 40 or
                plan.get("data_dates") != expected or
                not 1024 < int(plan.get("api_port", 0)) < 65536 or
                not isinstance(plan.get("base_config"), dict) or
                set(plan.get("input_records", {})) != set(expected)):
            raise ValueError("S1-02 plan identity, dates, source, or datasets differ")
        config = plan["base_config"]
        balance = config.get("coverage", {}).get("balance", {})
        train = config.get("training", {})
        if (config.get("schema_version") != 3 or config.get("research_stage") != "coverage" or
                config.get("coverage", {}).get("initialization") != "cluster" or
                balance.get("max_rounds") != 10 or balance.get("max_beam_decisions") != 1000 or
                train.get("seed") != 51001 or train.get("total_steps") != 409600 or
                train.get("num_envs") != 128 or train.get("rollout_steps") != 64):
            raise ValueError("S1-02 frozen base configuration differs from approved budget")
        for role, record in plan["input_records"].items():
            path = Path(record["path"])
            if not path.is_file() and role in ("train", "B_test"):
                continue
            if not path.is_file() or digest(path) != record["record_sha256"]:
                raise ValueError(f"{role} staged record is missing or changed")
        repo = Path(plan["remote_repo"])
        actual = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"],
                                         text=True).strip()
        if actual != plan["source_commit"]:
            raise ValueError("Remote workbench source commit differs")
        dirty = subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain",
                                         "--", "src", "data", "pyproject.toml"], text=True)
        if dirty.strip():
            raise ValueError("Remote source is dirty")
        if not str(repo.resolve()).startswith(str(REMOTE)):
            raise ValueError("S1-02 repo escaped its isolated 20418 root")
        recovery = plan.get("recovery_context")
        if recovery:
            expected_labels = {"A_dev_A0", "A_dev_A1", "A_dev_A2", "A_dev_A3"}
            if (recovery.get("version") != "zero_scene_startup_recovery_v1" or
                    set(recovery.get("allowed_completed_prepare", {})) != {"validation"} or
                    set(recovery.get("previous_failed_jobs", {})) != expected_labels or
                    recovery.get("previous_plan_sha256") !=
                    "5895e5d2712eb8db84942bbdfd1719af2d75ccfd32e390ca72282f15d46ef8f4"):
                raise ValueError("S1-02 recovery scope differs from four zero-scene A jobs")

    def verify_completed_prepare(self) -> str:
        recovery = self.plan.get("recovery_context", {})
        allowed = recovery.get("allowed_completed_prepare", {}).get("validation")
        if not allowed:
            raise RuntimeError("Recovery plan lacks the sole allowed validation prepare")
        old_plan = REMOTE / "ops/controller_plan.json"
        old_state = REMOTE / "ops/controller_state.json"
        if (digest(old_plan) != recovery["previous_plan_sha256"] or
                digest(old_state) != recovery["previous_state_sha256"]):
            raise RuntimeError("Original frozen plan/controller failure state changed")
        prior_state = read_json(old_state)
        if (prior_state.get("phase") != "needs_attention" or
                prior_state["jobs"]["prepare_validation"]["run_id"] != allowed["run_id"]):
            raise RuntimeError("Original successful prepare identity changed")
        for label, expected_run in recovery["previous_failed_jobs"].items():
            if prior_state["jobs"][label]["run_id"] != expected_run:
                raise RuntimeError("A development failed run identity changed")
            old_dir = REMOTE / "outputs/workbench/jobs" / expected_run
            failed = read_json(old_dir / "status.json", {})
            if (failed.get("stage") != "failed" or
                    failed.get("evaluated_scenes", 0) not in (0, None) or
                    (old_dir / "results.partial.jsonl").exists() or
                    (old_dir / "results.json").exists()):
                raise RuntimeError("Old A method generated results or changed status")
        old_dir = REMOTE / "outputs/workbench/jobs" / allowed["run_id"]
        job_file, config_file = old_dir / "job.json", old_dir / "config.json"
        status_file, artifacts_file = old_dir / "status.json", old_dir / "artifacts.json"
        for file, key in ((job_file, "job_sha256"), (config_file, "config_sha256"),
                          (status_file, "status_sha256"),
                          (artifacts_file, "artifacts_sha256")):
            if digest(file) != allowed[key]:
                raise RuntimeError("Whitelisted prepare provenance file changed: " + file.name)
        job, config, status, artifacts = (read_json(item) for item in
                                           (job_file, config_file, status_file, artifacts_file))
        old_source = REMOTE / "outputs/workbench/source_snapshots" / allowed["source_commit"]
        if (job.get("run_id") != allowed["run_id"] or
                job.get("request_id") != allowed["request_id"] or
                job.get("git_commit") != allowed["source_commit"] or
                Path(job.get("source_root", "")).resolve() != old_source or
                status.get("stage") != "completed" or status.get("evaluated_scenes") != 20 or
                config.get("schema_version") != 3 or
                config.get("coverage", {}).get("balance", {}).get("route") != "prepare" or
                config.get("coverage") != self.config("prepare")["coverage"] or
                set(config.get("datasets", {})) != {"validation"}):
            raise RuntimeError("Whitelisted completed prepare status/config differs")
        dataset = config["datasets"]["validation"]
        if (dataset.get("id") != allowed["dataset_id"] or
                dataset.get("sha256") != allowed["dataset_sha256"] or
                dataset.get("sha256") != self.state["dataset_records"]["validation"]["sha256"] or
                self.state["datasets"]["validation"] != allowed["dataset_id"]):
            raise RuntimeError("Whitelisted prepare uses different validation data")
        starts = Path(artifacts.get("balance_starts", ""))
        if (starts.resolve() != old_dir / "starts.jsonl" or
                digest(starts) != allowed["starts_sha256"] or
                artifacts.get("balance_starts_sha256") != allowed["starts_sha256"]):
            raise RuntimeError("Whitelisted A2 starts path/SHA differs")
        lines = [json.loads(line) for line in starts.read_text(encoding="utf-8").splitlines() if line]
        scenes = self.state["dataset_records"]["validation"]["scenes"]
        if (len(lines) != 20 or len(scenes) != 20 or
                any(start.get("scene_id") != scene.get("scene_id") or
                    start.get("scene_sha256") != scene.get("source_hash")
                    for start, scene in zip(lines, scenes))):
            raise RuntimeError("Whitelisted A2 starts do not cover the same 20 scenes")
        manifest = old_source / "source_manifest.json"
        if (digest(manifest) != allowed["source_manifest_sha256"] or
                read_json(manifest).get("git_commit") != allowed["source_commit"]):
            raise RuntimeError("Old completed prepare source manifest differs")
        previous = self.state["jobs"].get("prepare_validation")
        if previous and previous["run_id"] != allowed["run_id"]:
            raise RuntimeError("Recovery prepare dependency changed")
        self.state["jobs"]["prepare_validation"] = {
            "run_id": allowed["run_id"], "request_id": allowed["request_id"],
            "kind": "evaluate", "route": "prepare", "menu": "X0",
            "roles": ["validation"], "purpose": "verification",
            "reuse_only_completed_prepare_from_commit": allowed["source_commit"],
            "starts_sha256": allowed["starts_sha256"]}
        self.log("completed_prepare_reused_by_whitelist", run_id=allowed["run_id"],
                 prior_commit=allowed["source_commit"],
                 starts_sha256=allowed["starts_sha256"])
        return allowed["run_id"]

    def check_runtime(self) -> None:
        python = S101 / "envs/DQN/bin/python"
        result = subprocess.run([str(python), "-c",
                                 "import json,sys,torch,satellite_rl;print(json.dumps({'python':sys.executable,'cuda':torch.version.cuda,'available':torch.cuda.is_available(),'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,'package':satellite_rl.__file__}))"],
                                capture_output=True, text=True, check=True, timeout=45)
        data = json.loads(result.stdout)
        if (not data["available"] or "4090 D" not in data["gpu"] or
                not data["package"].startswith(str(Path(self.plan["remote_repo"]) / "src"))):
            raise RuntimeError("DQN/CUDA or actual package import path differs")
        self.log("runtime_checked", **data)

    def config(self, route: str, menu: str = "X0") -> dict:
        config = deepcopy(self.plan["base_config"])
        config["name"] = f"S1-02 {route} {menu}"
        config["coverage"]["balance"].update(route=route, action_menu=menu)
        config["coverage"]["method"] = "none" if route == "prepare" else (
            "ppo" if route == "ppo" else "local_search")
        return config

    def ensure_dataset(self, role: str) -> dict:
        record = self.plan["input_records"][role]
        source = Path(record["path"])
        if not source.is_file() or digest(source) != record["record_sha256"]:
            raise RuntimeError(f"Staged {role} record is missing or changed")
        source_dir = source.parent
        expected_dates = self.plan["data_dates"][role]
        actual_role = "test" if role.endswith("_test") else role
        catalog = api(self.base, "coverage/datasets")["datasets"]
        matching = [item for item in catalog if item.get("source_path") == str(source_dir)
                    and item.get("partition_role") == actual_role
                    and item.get("dates") == expected_dates
                    and item.get("initialization") == "cluster"]
        if len(matching) > 1:
            raise RuntimeError(f"Duplicate imported {role} records need audit")
        if matching:
            selected = matching[0]
        else:
            self.phase("import_" + role)
            selected = api(self.base, "coverage/datasets/import", {
                "path": str(source_dir), "partition_role": actual_role,
                "coverage": self.config("prepare")["coverage"],
                "name": f"S1-02 {role} {','.join(expected_dates)}"}, timeout=7200)
        if (selected.get("count") != (576 if role == "train" else 20 if role == "validation" else 100)
                or selected.get("dates") != expected_dates or not selected.get("sha256")):
            raise RuntimeError(f"Imported {role} scene count, source dates, or SHA differs")
        scene_rows = selected.get("scenes")
        if (not isinstance(scene_rows, list) or len(scene_rows) != selected["count"] or
                len({row.get("scene_id") for row in scene_rows}) != selected["count"] or
                any(not row.get("source_hash") for row in scene_rows)):
            raise RuntimeError(f"Imported {role} scene IDs/source hashes are incomplete")
        prior = self.state["datasets"].get(role)
        if prior and prior != selected["id"]:
            raise RuntimeError(f"Dataset ID changed for {role}")
        slim_record = {"sha256": selected["sha256"], "count": selected["count"],
                       "scenes": [{"scene_id": row["scene_id"],
                                   "source_hash": row["source_hash"]}
                                  for row in scene_rows]}
        prior_record = self.state["dataset_records"].get(role)
        if prior_record and prior_record != slim_record:
            raise RuntimeError(f"Imported {role} scene identity changed")
        self.state["datasets"][role] = selected["id"]
        self.state["dataset_records"][role] = slim_record
        self.log("dataset_verified", role=role, dataset_id=selected["id"],
                 sha256=selected["sha256"], count=selected["count"],
                 failed_initializations=selected.get("failed_initializations", 0))
        return selected

    def ensure_job(self, label: str, *, route: str, menu: str = "X0",
                   roles: tuple[str, ...], kind="evaluate", purpose="formal",
                   model_id=None, starts: dict | None = None) -> dict:
        request_id = self.request_prefix + "_" + label
        dataset_ids = {("test" if role.endswith("_test") else role): self.state["datasets"][role]
                       for role in roles}
        if len(dataset_ids) != len(roles):
            raise ValueError("Dataset roles collide")
        config = self.config(route, menu)
        body = {"kind": kind, "config": config, "dataset_ids": dataset_ids,
                "experiment_role": purpose, "request_id": request_id}
        if kind == "evaluate":
            body["dataset_role"] = next(iter(dataset_ids))
        if starts:
            body["start_run_ids"] = starts
        if model_id:
            body["model_id"] = model_id
        old = self.state["jobs"].get(label)
        if old:
            detail = api(self.base, "runs/" + quote(old["run_id"], safe=""))
            if (detail["job"].get("request_id") != request_id or
                    detail["job"].get("git_commit") != self.plan["source_commit"]):
                raise RuntimeError(f"Saved {label} job identity differs")
            return detail
        response = api(self.base, "coverage/runs", body, timeout=7200)
        if (response.get("request_id") != request_id or
                response.get("git_commit") != self.plan["source_commit"] or
                response.get("kind") != kind):
            raise RuntimeError(f"Workbench response for {label} differs")
        detail = api(self.base, "runs/" + quote(response["run_id"], safe=""))
        if (detail["job"].get("request_id") != request_id or
                detail["config"]["coverage"]["balance"]["route"] != route):
            raise RuntimeError(f"Frozen {label} job differs")
        self.state["jobs"][label] = {"run_id": response["run_id"],
                                     "request_id": request_id, "kind": kind,
                                     "route": route, "menu": menu,
                                     "roles": roles, "purpose": purpose,
                                     "replaces_run_id": self.state["replaces_run_ids"].get(label),
                                     "submitted_utc": utc_now()}
        self.log("job_submitted", label=label, run_id=response["run_id"],
                 request_id=request_id,
                 replaces_run_id=self.state["replaces_run_ids"].get(label))
        return detail

    def await_jobs(self, labels: list[str]) -> None:
        while True:
            stages = {}
            for label in labels:
                job = self.state["jobs"][label]
                detail = api(self.base, "runs/" + quote(job["run_id"], safe=""))
                stage = detail["status"].get("stage")
                stages[label] = stage
                if stage in TERMINAL - {"completed"}:
                    self.log("job_attention", label=label, run_id=job["run_id"], stage=stage)
                    raise RuntimeError(f"{label} requires diagnosis: {stage}")
            if all(stage == "completed" for stage in stages.values()):
                self.log("jobs_completed", labels=labels,
                         run_ids=[self.state["jobs"][label]["run_id"] for label in labels])
                return
            self.state["last_observed_stages"] = stages
            self.state["last_observed_utc"] = utc_now()
            save_json(self.state_file, self.state)
            time.sleep(30)

    def wait_gate(self, key: str) -> dict:
        gate = Path(self.plan["gates"][key])
        while True:
            payload = read_json(gate)
            if payload is not None:
                if payload.get("experiment_id") != "S1-02" or payload.get("status") != "passed":
                    raise RuntimeError(f"Gate {key} is present but not a passed S1-02 audit")
                self.log("gate_passed", key=key, path=str(gate), sha256=digest(gate))
                return payload
            self.phase("awaiting_" + key)
            time.sleep(30)

    def postprocess(self, phase: str) -> None:
        script = Path(__file__).with_name("postprocess.py")
        if not script.is_file():
            raise RuntimeError("Frozen S1-02 postprocessor is missing")
        log_dir = REMOTE / "ops/postprocess_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        python = str(S101 / "envs/DQN/bin/python")
        command = [python, "-B", str(script), phase,
                   "--plan", str(self.plan_file), "--state", str(self.state_file)]
        with (log_dir / (phase + ".stdout.log")).open("ab") as stdout, \
                (log_dir / (phase + ".stderr.log")).open("ab") as stderr:
            result = subprocess.run(command, stdout=stdout, stderr=stderr,
                                    cwd=self.plan["remote_repo"], timeout=7200)
        if result.returncode:
            raise RuntimeError(f"S1-02 {phase} postprocess failed; inspect {log_dir}")
        self.log("postprocess_passed", phase=phase, log_dir=str(log_dir))

    def run(self) -> None:
        self.check_runtime()
        prep = {}
        self.ensure_dataset("validation")
        if self.plan.get("recovery_context"):
            prep["validation"] = self.verify_completed_prepare()
        else:
            self.ensure_job("prepare_validation", route="prepare", roles=("validation",),
                            purpose="verification")
            prep["validation"] = self.state["jobs"]["prepare_validation"]["run_id"]
        self.phase("A_validation_prepare")
        self.await_jobs(["prepare_validation"])
        for route in ("A0", "A1", "A2", "A3"):
            self.ensure_job("A_dev_" + route, route=route, roles=("validation",),
                            purpose="verification", starts={"validation": prep["validation"]})
        self.phase("A_development")
        self.await_jobs(["A_dev_" + route for route in ("A0", "A1", "A2", "A3")])
        self.postprocess("A_development")
        self.ensure_dataset("A_test")
        self.ensure_job("prepare_A_test", route="prepare", roles=("A_test",))
        prep["A_test"] = self.state["jobs"]["prepare_A_test"]["run_id"]
        self.phase("A_test_prepare")
        self.await_jobs(["prepare_A_test"])
        for route in ("A0", "A1", "A2", "A3"):
            self.ensure_job("A_test_" + route, route=route, roles=("A_test",),
                            starts={"test": prep["A_test"]})
        self.phase("A_formal")
        self.await_jobs(["A_test_" + route for route in ("A0", "A1", "A2", "A3")])
        self.postprocess("A_report")
        self.wait_gate("A_report")
        for menu in ("X0", "X1", "X2", "X3", "X4"):
            self.ensure_job("B_screen_" + menu, route=menu, menu=menu,
                            roles=("validation",), purpose="verification",
                            starts={"validation": prep["validation"]})
        self.phase("B_screen")
        self.await_jobs(["B_screen_" + menu for menu in ("X0", "X1", "X2", "X3", "X4")])
        self.postprocess("B_menu_selection")
        selected = self.wait_gate("B_menu_selection")
        menus = selected.get("menus")
        if (not isinstance(menus, list) or len(menus) != 3 or menus[0] != "X0" or
                len(set(menus)) != 3 or not set(menus) <= {"X0", "X1", "X2", "X3", "X4"} or
                selected.get("selection_source") != "Jan12_validation_only"):
            raise RuntimeError("B menu selection changed or used independent test results")
        self.ensure_dataset("train")
        self.ensure_dataset("B_test")
        for role in ("train", "B_test"):
            self.ensure_job("prepare_" + role, route="prepare", roles=(role,),
                            purpose="verification" if role == "train" else "formal")
            prep[role] = self.state["jobs"]["prepare_" + role]["run_id"]
        self.phase("B_train_and_test_prepare")
        self.await_jobs(["prepare_train"])
        for menu in menus:
            self.ensure_job("B_train_" + menu, route="ppo", menu=menu,
                            kind="train", roles=("train", "validation"),
                            starts={"train": prep["train"], "validation": prep["validation"]})
        self.phase("B_training")
        self.await_jobs(["B_train_" + menu for menu in menus])
        self.await_jobs(["prepare_B_test"])
        for menu in menus:
            self.ensure_job("B_test_PPO_" + menu, route="ppo", menu=menu,
                            roles=("B_test",), starts={"test": prep["B_test"]},
                            model_id=self.state["jobs"]["B_train_" + menu]["run_id"])
            self.ensure_job("B_test_search_" + menu, route=menu, menu=menu,
                            roles=("B_test",), starts={"test": prep["B_test"]})
        self.phase("B_formal")
        self.await_jobs(["B_test_" + kind + "_" + menu
                         for menu in menus for kind in ("PPO", "search")])
        self.postprocess("B_report")
        self.wait_gate("B_report")
        self.phase("awaiting_final_delivery")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=REMOTE / "ops/controller_plan.json")
    parser.add_argument("--state", type=Path, default=REMOTE / "ops/controller_state.json")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    controller = Controller(args.plan, args.state)
    if args.check:
        controller.check_runtime()
        print(json.dumps({"status": "preflight_passed", "plan_sha256": controller.plan_sha}))
        return
    lock_file = args.state.with_suffix(".lock")
    with lock_file.open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another S1-02 controller owns the lock") from error
        try:
            controller.run()
        except BaseException as error:
            controller.state["phase"] = "needs_attention"
            controller.log("controller_stopped", error=type(error).__name__ + ": " + str(error))
            raise


if __name__ == "__main__":
    main()
