"""Bind the S3-P03 submission receipts to one local handoff manifest."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re


ROOT = Path("D:/paper/outputs/experiments/stage3_si_priority_20261003")
PREFLIGHT = ROOT / "preflight"
POLICY_SHA = "ae095e8246d22b2716ba154315cee18584917d215580821cb53f9aeba78eb824"


def read(name: str) -> dict:
    return json.loads((PREFLIGHT / name).read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--commit", required=True)
    args = parser.parse_args()
    commit = args.commit.lower()
    if not re.fullmatch("[0-9a-f]{40}", commit):
        raise ValueError("Full frozen source commit required")
    deployment = read(f"remote_deployment_{commit[:12]}.json")
    data = read("remote_jan27_data_receipt.json")
    if deployment["commit"] != commit or data["status"] != "jan27_inputs_verified_on_both_instances_no_evaluation":
        raise RuntimeError("Deployment or Jan27 source evidence differs")
    groups = {}
    for port, mode, wb, local in ((20418, "total", 8776, 8776),
                                  (20424, "initial", 8777, 8777)):
        launch = read(f"workbench_launch_{port}.json")
        model = read(f"t0_import_{port}.json")
        scene = read(f"scene_import_{port}.json")
        submission = read(f"jan27_submission_{port}.json")
        if (launch["commit"] != commit or model["commit"] != commit or
                scene["commit"] != commit or submission["commit"] != commit or
                model["policy_sha256"] != POLICY_SHA or
                scene["original_record_sha256"] != data["remote_checks"][str(port)]["staged_test/record.json"]["sha256"] or
                submission["config"]["power_budget_mode"] != mode or
                submission["config"]["methods"] != ["sca", "sca_si_lp", "sca_si_lb"] or
                submission["evaluation_units_expected"] != 900):
            raise RuntimeError(f"Frozen S3-P03 handoff contract differs on {port}")
        groups[mode] = {
            "ssh_port": port, "workbench_port": wb, "local_port": local,
            "root": f"/home/u2024110224/third_stage_satellite_rl/s3_si_priority_20261003/{port}",
            "run_id": submission["job"]["run_id"], "request_id": submission["request_id"],
            "model_id": model["model_id"], "scene_id": scene["scene_id"],
            "config": submission["config"], "evaluation_units_budget": 900,
            "status": "submitted", "env_step": 0,
            "gpu_uuid": launch["gpu_uuid"], "server_pid": launch["pid"],
            "server_start_ticks": launch["start_ticks"],
            "thread_environment": launch["thread_environment"],
            "cpu": launch["cpu"], "versions": launch["versions"],
            "solver_versions": launch["solver_versions"],
            "submission_receipt": str(PREFLIGHT / f"jan27_submission_{port}.json"),
        }
    result = {
        "experiment_id": "S3-P03", "status": "submitted_first_case_pending",
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "science_commit": commit, "new_training_interactions": 0,
        "evaluation_units_total": 1800,
        "methods": ["sca", "sca_si_lp", "sca_si_lb"],
        "scenario": {"date_utc": "2024-01-27", "scene_count": 100,
                     "record_sha256": data["remote_checks"]["20418"]["staged_test/record.json"]["sha256"],
                     "raw_archive_sha256": data["source_archive_sha256"],
                     "data_receipt": str(PREFLIGHT / "remote_jan27_data_receipt.json")},
        "initializer": {"name": "G11 T0", "run_id": "train_sac_20261001_162126_33e89e95",
                        "training_seed": 51001, "terminal_env_step": 150000,
                        "source_commit": "85f667cd5731e284c19b2590b5dcf5308a58f234",
                        "policy_sha256": POLICY_SHA},
        "action_seed_base": 53001, "repeats": 3,
        "budgets": {"A": {"maxeval": 2000, "maxiter": 300, "max_time_seconds": 30.0},
                    "B_C_additional": {"maxeval": 2000, "maxiter": 300,
                                        "max_time_seconds": 30.0}},
        "groups": groups,
        "deployment_receipt": str(PREFLIGHT / f"remote_deployment_{commit[:12]}.json"),
        "planned_report": "D:/paper/docs/第三阶段/实验报告/03_SCA需求满足率优先三组对照.md",
        "limitations": ["single T0 training seed", "one independent Jan27 date",
                        "A shared once per scene/action/budget", "B/C may lower SGM to improve SI"],
    }
    target = ROOT / "run_manifest.json"
    if target.exists():
        old = json.loads(target.read_text(encoding="utf-8"))
        if old.get("science_commit") != commit or old.get("groups", {}).keys() != result["groups"].keys():
            raise RuntimeError("Existing run manifest belongs to another submission")
        raise RuntimeError("Existing run manifest requires an explicit status update, not replacement")
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"manifest": str(target), "run_ids": {k: v["run_id"] for k, v in groups.items()}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
