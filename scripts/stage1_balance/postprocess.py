"""Run S1-02 independent audits, reports, and validation-only menu selection.

The durable controller invokes this after each workbench phase. A failed audit
or report blocks the next phase and leaves its logs intact.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess

from controller import REMOTE, S101, digest, read_json, save_json


ANALYSIS = REMOTE / "outputs/experiments/first_stage_s1_02_20261004/analysis"
WORKBENCH_JOBS = REMOTE / "outputs/workbench/jobs"


def result(state: dict, label: str) -> Path:
    job = state.get("jobs", {}).get(label)
    if not job:
        raise RuntimeError("Missing workbench task: " + label)
    directory = WORKBENCH_JOBS / job["run_id"]
    status = read_json(directory / "status.json", {})
    frozen = read_json(directory / "job.json", {})
    if (status.get("stage") != "completed" or
            frozen.get("request_id") != job["request_id"] or
            frozen.get("git_commit") != state.get("source_commit")):
        raise RuntimeError("Completed frozen workbench task differs: " + label)
    path = directory / "results.json"
    if not path.is_file():
        raise RuntimeError("Missing workbench result: " + label)
    return path


def command(args: list[str], output: Path, name: str, env: dict) -> None:
    output.mkdir(parents=True, exist_ok=True)
    out = output / (name + ".stdout.log")
    err = output / (name + ".stderr.log")
    with out.open("ab") as stdout, err.open("ab") as stderr:
        process = subprocess.run(args, stdout=stdout, stderr=stderr,
                                 cwd=env["S1_02_REPO"], env=env, timeout=7200)
    if process.returncode:
        raise RuntimeError(f"{name} failed with exit {process.returncode}; inspect {err}")


def run_phase(phase: str, state: dict, plan: dict) -> dict:
    repo = Path(plan["remote_repo"])
    commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"],
                                     text=True).strip()
    if commit != plan["source_commit"]:
        raise RuntimeError("Postprocess repo differs from frozen scientific commit")
    state = dict(state, source_commit=commit)
    python = str(S101 / "envs/DQN/bin/python")
    env = dict(os.environ, PYTHONPATH=str(repo / "src"), S1_02_REPO=str(repo),
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    script = repo / "scripts"
    destination = ANALYSIS / phase
    if phase == "A_development":
        labels = {route: "A_dev_" + route for route in ("A0", "A1", "A2", "A3")}
        expected = 20
        mode = "audit_only"
    elif phase == "A_report":
        labels = {route: "A_test_" + route for route in ("A0", "A1", "A2", "A3")}
        expected = 100
        mode = "report"
    elif phase == "B_menu_selection":
        labels = {menu: "B_screen_" + menu for menu in ("X0", "X1", "X2", "X3", "X4")}
        expected = 20
        mode = "select"
    elif phase == "B_report":
        gate = read_json(Path(plan["gates"]["B_menu_selection"]), {})
        menus = gate.get("menus") or gate.get("selected_menus")
        if not isinstance(menus, list) or len(menus) != 3:
            raise RuntimeError("Validated B menu selection is missing")
        labels = {"PPO_" + menu: "B_test_PPO_" + menu for menu in menus}
        labels.update({"search_" + menu: "B_test_search_" + menu for menu in menus})
        expected = 100
        mode = "report"
    else:
        raise ValueError("Unknown S1-02 postprocessing phase")
    paths = {label: result(state, job_label) for label, job_label in labels.items()}
    sources = {label: {"path": str(path), "sha256": digest(path)}
               for label, path in paths.items()}
    prior = read_json(destination / "input_receipt.json")
    if prior and prior.get("sources") != sources:
        raise RuntimeError("Postprocessing inputs changed after first run")
    gate_key = "B_menu_selection" if mode == "select" else phase
    if mode != "audit_only" and prior:
        previous_gate = read_json(Path(plan["gates"][gate_key]))
        if (previous_gate and previous_gate.get("experiment_id") == "S1-02"
                and previous_gate.get("status") == "passed"):
            return previous_gate
    destination.mkdir(parents=True, exist_ok=True)
    if not prior:
        save_json(destination / "input_receipt.json", {
            "experiment_id": "S1-02", "phase": phase,
            "source_commit": commit, "captured_utc": datetime.now(timezone.utc).isoformat(),
            "expected_scenes": expected, "sources": sources})
    audit_args = [python, "-B", str(script / "audit_stage1_balance.py")]
    for label, path in paths.items():
        audit_args += ["--result", label + "=" + str(path)]
    audit_args += ["--expected-scenes", str(expected), "--output", str(destination)]
    command(audit_args, destination, "audit", env)
    audit = read_json(destination / "audit.json", {})
    if audit.get("passed") is not True or len(audit.get("methods", {})) != len(paths):
        raise RuntimeError("Independent S1-02 audit did not pass")
    if mode == "select":
        output = destination / "selection.json"
        command([python, "-B", str(script / "select_stage1_menus.py"),
                 *[item for label, path in paths.items()
                   for item in ("--result", label + "=" + str(path))],
                 "--output", str(output)], destination, "select", env)
        selection = read_json(output, {})
        menus = selection.get("selected_menus")
        if (selection.get("passed") is not True or not isinstance(menus, list) or
                len(menus) != 3 or menus[0] != "X0" or len(set(menus)) != 3 or
                not set(menus) <= {"X0", "X1", "X2", "X3", "X4"}):
            raise RuntimeError("Validation-only selection did not pass")
        gate = {"experiment_id": "S1-02", "status": "passed", "phase": phase,
                "selection_source": "Jan12_validation_only", "menus": menus,
                "selected_menus": menus, "selection_sha256": digest(output),
                "audit_sha256": digest(destination / "audit.json"),
                "created_utc": datetime.now(timezone.utc).isoformat()}
        save_json(Path(plan["gates"]["B_menu_selection"]), gate)
        return gate
    if mode == "audit_only":
        return {"phase": phase, "passed": True, "audit_sha256": digest(destination / "audit.json")}
    report_args = [python, "-B", str(script / "report_stage1_balance.py"),
                   "--phase", "A" if phase == "A_report" else "B",
                   "--output", str(destination), "--audit", str(destination / "audit.json")]
    for label, path in paths.items():
        report_args += ["--result", label + "=" + str(path)]
    if phase == "B_report":
        for menu in menus:
            run_id = state["jobs"]["B_train_" + menu]["run_id"]
            report_args += ["--training-run", menu + "=" + str(WORKBENCH_JOBS / run_id)]
    command(report_args, destination, "report", env)
    gate = read_json(destination / "gate.json", {})
    if gate.get("status") != "passed" or gate.get("experiment_id") != "S1-02":
        raise RuntimeError("Formal report did not produce a passed gate")
    target = Path(plan["gates"][phase])
    if target != destination / "gate.json":
        old = read_json(target)
        if old and old != gate:
            raise RuntimeError("Existing formal gate differs")
        save_json(target, gate)
    return gate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("A_development", "A_report", "B_menu_selection", "B_report"))
    parser.add_argument("--plan", type=Path, default=REMOTE / "ops/controller_plan.json")
    parser.add_argument("--state", type=Path, default=REMOTE / "ops/controller_state.json")
    args = parser.parse_args()
    plan = read_json(args.plan)
    state = read_json(args.state)
    if not isinstance(plan, dict) or not isinstance(state, dict):
        raise RuntimeError("Controller plan/state are required")
    print(json.dumps(run_phase(args.phase, state, plan), ensure_ascii=False))


if __name__ == "__main__":
    main()
