"""S1-D12A paired PPO pilot audit and checkpoint-bound release."""
from __future__ import annotations

import json
import math
from pathlib import Path

from satellite_rl.training.run_monitor import atomic_json
from satellite_rl.workbench.storage import digest, now, read_json, validate_id
from satellite_rl.workbench.coverage_drl_r04 import GROUP, MULTIPLIERS, PARENT, PORTS, deployment
from satellite_rl.workbench.runtime import pid_alive


MISCLASSIFIED_PILOTS = {
    "train_ppo_20261010_034306_03e7f00f": ("P0", "20418"),
    "train_ppo_20261010_034937_d980cee4": ("P1", "20424"),
}
FROZEN_PILOT_COMMIT = "dfe7c394267e94bd2084ae72add3aa2c855b578f"


def _status(job, status, config, run_id):
    if (job.get("experiment_id") != GROUP or job.get("kind") != "train" or
            job.get("train_mode") not in ("P0", "P1") or
            job.get("experiment_role") != "formal" or config.get("schema_version") != 6 or
            status.get("stage") != "pilot_pending" or
            status.get("stop_reason") != "pilot_gate" or status.get("env_step") != 65536 or
            job.get("run_id") != run_id):
        raise ValueError("S1-D12A gate requires exact formal pilot task/state")


def _valid_terminal(row):
    initial, best = row.get("initial_j"), row.get("best_j")
    if (not all(isinstance(v, (float, int)) and math.isfinite(v)
                for v in (initial, best)) or best > initial + 1e-8 or
            row.get("failure_reason") or row.get("geometry_calls", 8193) > 8192 or
            row.get("trials", 257) > 256 or row.get("commits", 33) > 32 or
            row.get("matrix_calls", -1) < 0 or
            row.get("matrix_calls", 8193) > row.get("geometry_calls", -1) or
            row.get("ray_pair_checks", -1) < 0 or
            not isinstance(row.get("reward_sum"), (int, float)) or
            not math.isclose(row["reward_sum"], 100 * (initial - best),
                             rel_tol=1e-8, abs_tol=1e-7)):
        return False
    layout = row.get("best_layout") or {}
    assigned = layout.get("assignment")
    centers = layout.get("centers")
    widths = layout.get("widths")
    aircraft_ids = layout.get("aircraft_ids")
    if (not isinstance(centers, list) or not 1 <= len(centers) <= 200 or
            not isinstance(widths, list) or len(widths) != len(centers) or
            any(w != .5 for w in widths) or not isinstance(assigned, list) or
            not isinstance(aircraft_ids, list) or len(assigned) != len(aircraft_ids) or
            len(set(aircraft_ids)) != len(aircraft_ids) or
            any(type(v) is not int or not 0 <= v < len(centers) for v in assigned)):
        return False
    physics = row.get("terminal_physics") or {}
    aliases = physics.get("layout_aliases") or {}
    measured = (physics.get("layouts_by_sha256") or {}).get(aliases.get("best"), {})
    metrics = measured.get("metrics") or {}
    demand = metrics.get("total_demand_bps")
    beam_demand = [beam.get("demand_bps") for beam in measured.get("beams", [])]
    if (not isinstance(demand, (int, float)) or not math.isfinite(demand) or
            not all(isinstance(v, (int, float)) and math.isfinite(v) and v >= 0
                    for v in beam_demand) or
            not math.isclose(sum(beam_demand), demand, rel_tol=1e-10, abs_tol=1e-5) or
            metrics.get("uncovered_demand_bps") != 0 or
            not isinstance(metrics.get("total_power_w"), (int, float)) or
            not 0 <= metrics["total_power_w"] <= 3000 + 1e-7 or
            physics.get("status") != "evaluated_all_actual_beams_and_users"):
        return False
    return True


def _real_refresh(row):
    previous = row["initial_j"]
    for step in row.get("ledger", []):
        current = step.get("best_j")
        action = step.get("action") or {}
        if (action.get("operation") != "stop" and step.get("accepted") is True and
                isinstance(current, (int, float)) and current < previous - 1e-10):
            return True
        if isinstance(current, (int, float)):
            previous = current
    return False


def _audit(application, run_id, *, status_override=None):
    directory = application.store.job_dir(run_id)
    job = read_json(directory / "job.json", {})
    status = status_override if status_override is not None else read_json(directory / "status.json", {})
    config = read_json(directory / "config.json", {})
    _status(job, status, config, run_id)
    checkpoint = directory / "checkpoint.pt"
    checkpoint_sha = digest(checkpoint)
    if status.get("checkpoint_sha256") != checkpoint_sha:
        raise ValueError("pilot checkpoint SHA differs from status")
    expected_steps = (16384, 32768, 49152, 65536)
    ledger = [json.loads(line) for line in
              (directory / "evaluations.jsonl").read_text(encoding="utf-8").splitlines()]
    if len(ledger) != 4 or tuple(item.get("env_step") for item in ledger) != expected_steps:
        raise ValueError("pilot lacks four committed validation nodes")
    nodes = []
    for step, summary in zip(expected_steps, ledger):
        path = directory / "validation" / f"step_{step:06d}.jsonl"
        lines = path.read_text(encoding="utf-8").splitlines()
        if (len(lines) != 60 or summary.get("rows") != 60 or
                summary.get("path") != str(path) or summary.get("sha256") != digest(path)):
            raise ValueError("validation file/ledger SHA or row count differs")
        rows = [json.loads(line) for line in lines]
        for ordinal, row in enumerate(rows):
            if (row.get("dataset_index") != ordinal // 3 or
                    row.get("demand_multiplier") != MULTIPLIERS[ordinal % 3] or
                    not _valid_terminal(row)):
                raise ValueError(f"validation scene correctness failed at {step}/{ordinal}")
        real_refresh = sum(_real_refresh(row) for row in rows)
        if real_refresh != summary.get("best_j_improved_count"):
            raise ValueError("real non-STOP best refresh differs from summary")
        first_stop = sum(bool(row.get("ledger") and
                              row["ledger"][0]["action"]["operation"] == "stop")
                         for row in rows)
        if first_stop != summary.get("first_step_stop_count"):
            raise ValueError("first-step STOP count differs from raw validation")
        improved = [mother for mother in range(20)
                    if sum(rows[mother * 3 + level]["initial_j"] -
                           rows[mother * 3 + level]["best_j"]
                           for level in range(3)) / 3 > 1e-8 and
                    any(_real_refresh(rows[mother * 3 + level]) for level in range(3))]
        all_stop_zero = first_stop == 60 and real_refresh == 0
        if (summary.get("three_level_improved_mother_count") != len(improved) or
                summary.get("nonstop_best_refresh_rows") != real_refresh or
                summary.get("feasible_row_count") != 60 or
                summary.get("all_60_first_step_stop_zero_progress") is not all_stop_zero):
            raise ValueError("validation mother/refresh/feasibility summary differs")
        nodes.append({"env_step": step, "sha256": digest(path),
                      "first_step_stop_count": first_stop,
                      "real_best_refresh_count": real_refresh,
                      "improved_mother_indices": improved,
                      "improved_mother_count": len(improved),
                      "all_60_first_step_stop_zero_progress": all_stop_zero,
                      "correctness_passed": True})
    if nodes[-1]["improved_mother_count"] >= 6:
        progress_pass = True
    else:
        progress_pass = False
    zero_progress_stop = (all(node["first_step_stop_count"] == 60 and
                              node["real_best_refresh_count"] == 0
                              for node in nodes[-2:]))
    audits = application.store.archive_root / "ops" / "audits"
    audits.mkdir(parents=True, exist_ok=True)
    audit_path = audits / f"s1d12a_pilot_{run_id}.json"
    content = {"run_id": run_id, "env_step": 65536,
               "checkpoint_sha256": checkpoint_sha,
               "source_commit": job["git_commit"],
               "validation_dataset_sha256": config["datasets"]["validation"]["sha256"],
               "reachability_contract_sha256": job.get("reachability_contract", {}).get("sha256"),
               "progress_pass": progress_pass,
               "zero_progress_stop": zero_progress_stop,
               "correctness_passed": True, "nodes": nodes}
    existing = read_json(audit_path)
    if existing:
        if existing != content:
            raise ValueError("existing pilot audit differs")
    else:
        atomic_json(audit_path, content)
    return directory, job, status, audit_path, content


def reconcile_misclassified_pilot(application, run_id, body):
    """Repair only the two scheduler-misclassified, normally exited D12-A pilots."""
    validate_id(run_id)
    expected = MISCLASSIFIED_PILOTS.get(run_id)
    if expected is None or deployment(application.store) != expected[1]:
        raise ValueError("pilot reconciliation is limited to the two affected lanes")
    directory = application.store.job_dir(run_id)
    job = read_json(directory / "job.json", {})
    status_path = directory / "status.json"
    status = read_json(status_path, {})
    launch = read_json(directory / "launch.json", {})
    failure_path = directory / "failure.json"
    checkpoint = directory / "checkpoint.pt"
    receipt_path = directory / "pilot_pending_reconciliation.json"
    prior = read_json(receipt_path, {})
    if prior:
        if (status.get("stage") != "pilot_pending" or
                status.get("checkpoint_sha256") != prior.get("checkpoint_sha256") or
                digest(checkpoint) != prior.get("checkpoint_sha256")):
            raise ValueError("existing pilot reconciliation differs from current status")
        return {"status": status, "receipt": prior}
    if (not isinstance(body, dict) or
            set(body) != {"expected_status_sha256", "expected_checkpoint_sha256",
                          "expected_failure_sha256"} or
            body["expected_status_sha256"] != digest(status_path) or
            body["expected_checkpoint_sha256"] != digest(checkpoint) or
            body["expected_failure_sha256"] != digest(failure_path)):
        raise ValueError("pilot reconciliation requires exact status/checkpoint/failure SHA")
    failure = read_json(failure_path, {})
    config = read_json(directory / "config.json", {})
    if (job.get("run_id") != run_id or job.get("experiment_id") != GROUP or
            job.get("kind") != "train" or job.get("train_mode") != expected[0] or
            job.get("experiment_role") != "formal" or
            job.get("git_commit") != FROZEN_PILOT_COMMIT or
            config.get("schema_version") != 6 or
            status.get("stage") != "interrupted" or
            status.get("stop_reason") != "pilot_gate" or
            status.get("env_step") != 65536 or
            status.get("checkpoint_sha256") != digest(checkpoint) or
            status.get("exit_code") != 0 or
            status.get("scientific_stop") is True or
            launch.get("pid") != status.get("pid") or
            not isinstance(launch.get("process_start_time"), (int, float)) or
            pid_alive(launch["pid"], launch["process_start_time"]) or
            failure.get("run_id") != run_id or failure.get("exit_code") != 0 or
            failure.get("pid") != status.get("pid") or
            failure.get("git_commit") != FROZEN_PILOT_COMMIT or
            failure.get("error") !=
            "进程在没有Python异常记录的情况下结束；原生退出应结合Windows事件和驱动日志排查"):
        raise ValueError("pilot reconciliation evidence is not an exact normal gate exit")
    repaired = {**status, "stage": "pilot_pending"}
    repaired.pop("exit_code", None)
    repaired.pop("exit_hex", None)
    repaired.pop("error", None)
    _, _, _, audit_path, audit = _audit(application, run_id, status_override=repaired)
    receipt = {"run_id": run_id, "source_commit": FROZEN_PILOT_COMMIT,
               "previous_status_sha256": body["expected_status_sha256"],
               "failure_sha256": body["expected_failure_sha256"],
               "checkpoint_sha256": body["expected_checkpoint_sha256"],
               "launch_pid": launch["pid"],
               "launch_process_start_time": launch["process_start_time"],
               "pilot_audit_path": str(audit_path), "pilot_audit_sha256": digest(audit_path),
               "pilot_progress_pass": audit["progress_pass"],
               "pilot_zero_progress_stop": audit["zero_progress_stop"],
               "reconciled_at": now()}
    atomic_json(receipt_path, receipt)
    atomic_json(status_path, {**repaired, "heartbeat_at": now()})
    return {"status": read_json(status_path), "receipt": receipt, "audit": audit}


def _peer(job, body, source_commit, validation_sha):
    port = str(body.get("paired_port"))
    run_id = body.get("paired_run_id")
    if port not in PORTS:
        raise ValueError("paired source port invalid")
    validate_id(run_id)
    directory = PARENT / port / "archive" / "outputs" / "workbench" / "jobs" / run_id
    peer_job = read_json(directory / "job.json", {})
    peer_status = read_json(directory / "status.json", {})
    if (peer_job.get("run_id") != run_id or peer_job.get("experiment_id") != GROUP or
            peer_job.get("train_mode") not in ("P0", "P1") or
            peer_job.get("train_mode") == job["train_mode"] or
            peer_job.get("git_commit") != source_commit or
            peer_job.get("dataset_sha256", {}).get("validation") != validation_sha):
        raise ValueError("paired PPO pilot identity/scene/source differs")
    audit_path = PARENT / port / "archive" / "ops" / "audits" / f"s1d12a_pilot_{run_id}.json"
    audit = read_json(audit_path, {})
    if audit:
        if (audit.get("run_id") != run_id or
                audit.get("source_commit") != source_commit or
                audit.get("validation_dataset_sha256") != validation_sha):
            raise ValueError("paired pilot audit differs")
    elif not (peer_status.get("stage") == "stopped" and
              peer_status.get("scientific_stop") is True):
        raise ValueError("paired PPO has not reached a reviewable pilot/stop state")
    return {"run_id": run_id, "port": port, "audit": audit,
            "audit_sha256": digest(audit_path) if audit else None,
            "stage": peer_status.get("stage"),
            "scientific_stop": peer_status.get("scientific_stop") is True}


def approve_pilot_gate(application, run_id, body=None):
    """Audit, release or scientifically stop one pilot from paired evidence."""
    validate_id(run_id)
    body = body or {}
    mode = body.get("mode", "audit")
    if mode not in ("audit", "release", "stop"):
        raise ValueError("unknown S1-D12A pilot gate operation")
    directory, job, status, audit_path, audit = _audit(application, run_id)
    if mode == "audit":
        if audit["zero_progress_stop"]:
            status.update(stage="stopped", stop_reason="consecutive_validation_all_stop_zero_progress",
                          scientific_stop=True, ended_at=now(), heartbeat_at=now())
            atomic_json(directory / "status.json", status)
        return {"proof": None, "audit": audit, "audit_path": str(audit_path),
                "audit_sha256": digest(audit_path)}
    peer = _peer(job, body, job["git_commit"], audit["validation_dataset_sha256"])
    allowed = audit["progress_pass"] or (peer["audit"].get("progress_pass") and
                                           not audit["zero_progress_stop"])
    if mode == "stop":
        if allowed:
            raise ValueError("scientific stop cannot discard a reviewable pilot/reference")
        status.update(stage="stopped", stop_reason="paired_pilot_no_progress",
                      scientific_stop=True, ended_at=now(), heartbeat_at=now())
        atomic_json(directory / "status.json", status)
        return {"proof": None, "audit": audit, "paired": peer}
    if not allowed:
        raise ValueError("pilot failed six-mother gate and paired reference exception")
    release_path = audit_path.with_name(audit_path.stem + "_release.json")
    release = {"passed": True, "run_id": run_id, "env_step": 65536,
               "checkpoint_sha256": audit["checkpoint_sha256"],
               "source_commit": audit["source_commit"],
               "validation_dataset_sha256": audit["validation_dataset_sha256"],
               "individual_audit_path": str(audit_path),
               "individual_audit_sha256": digest(audit_path),
               "paired_run_id": peer["run_id"], "paired_port": peer["port"],
               "paired_audit_sha256": peer["audit_sha256"],
               "reason": "six_mothers_progress" if audit["progress_pass"] else
                         "necessary_paired_reference_without_zero_progress_stop"}
    prior = read_json(release_path)
    if prior and prior != release:
        raise ValueError("existing pilot release audit differs")
    if not prior:
        atomic_json(release_path, release)
    proof = {"passed": True, "run_id": run_id, "env_step": 65536,
             "checkpoint_sha256": audit["checkpoint_sha256"],
             "audit_path": str(release_path), "audit_sha256": digest(release_path),
             "approved_at": now()}
    old_proof = read_json(directory / "pilot_gate.json")
    if old_proof:
        for key in ("passed", "run_id", "env_step", "checkpoint_sha256",
                    "audit_path", "audit_sha256"):
            if old_proof.get(key) != proof[key]:
                raise ValueError("existing pilot proof differs")
        proof = old_proof
    else:
        atomic_json(directory / "pilot_gate.json", proof)
    return {"proof": proof, "audit": audit, "paired": peer}
