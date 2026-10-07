"""Build the S2-P19 typed API plan from frozen contracts and service defaults.

Bootstrap contains data preparation, both train-scene imports, teacher,
non-learning gate, three authorized SAC pilots and their source-backed audit
gates. It uses task-result references so no run ID or output record path is
filled by hand. Later phases are appended with a parent SHA chain.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import subprocess

from . import controller, ops


ROOT = Path("C:/Users/Admin/.codex/worktrees/s2-p19-sac/paper")
CONTRACTS = ops.LOCAL / "contracts"
TRAIN_RECORD = "/home/u2024110224/second_stage_satellite_rl/variable_beams_20261004/20424/data/prepared/train"


def ref(task: str, *keys: str) -> dict:
    return {"$task": {"id": task, "path": list(keys)}}


def task(identifier: str, phase: str, port: int, kind: str, body: dict,
         disk_class: str, dependencies: list[str], **extra) -> dict:
    return {"id": identifier, "phase": phase, "port": port, "kind": kind,
            "body": body, "disk_class": disk_class, "requires": dependencies,
            "request_id": body.get("request_id") or "s2p19-" + identifier.replace("_", "-"), **extra}


def _source_commit() -> str:
    result = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                            text=True, capture_output=True, check=True)
    commit = result.stdout.strip()
    dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain"],
                           text=True, capture_output=True, check=True).stdout.strip()
    if dirty:
        raise RuntimeError("plan builder requires clean frozen S2-P19 worktree")
    return commit


def bootstrap() -> dict:
    commit = _source_commit()
    launch_path = ops.LOCAL / "preflight" / f"workbench_{commit[:12]}.json"
    if not launch_path.is_file():
        raise RuntimeError("deploy+serve exact frozen commit before building controller plan")
    launch = json.loads(launch_path.read_text(encoding="utf-8"))
    if launch.get("scientific_commit") != commit or set(launch.get("instances", {})) != {"20418", "20424"}:
        raise RuntimeError("two-instance workbench launch does not match frozen commit")
    catalog = ops.api(20418, "/api/s2p19/catalog")
    if catalog.get("experiment_id") != "S2-P19":
        raise RuntimeError("deployed typed catalog differs")
    defaults = catalog["defaults"]
    c0 = deepcopy(defaults["C0"])
    c1 = deepcopy(defaults["C1"])
    i0 = deepcopy(defaults["I0"])
    if (c0["training"]["total_steps"] != 400000 or
            c1["training"]["total_steps"] != 400000 or
            i0["total_steps"] != 400000 or
            c0["training"]["pilot_stop_steps"] != 100000 or
            c1["training"]["pilot_stop_steps"] != 100000 or
            i0["pilot_stop_steps"] != 100000):
        raise RuntimeError("deployed defaults differ from approved pilot/total decisions")
    c1["training"]["demonstrations"].update(
        manifest_path=ref("teacher_20424", "result", "summary", "manifest_path"),
        manifest_sha256=ref("teacher_20424", "result", "summary", "manifest_sha256"),
        source_set_sha256=ref("teacher_20424", "result", "summary", "source_set_sha256"))

    dev_body = json.loads((CONTRACTS / "api_prepare_dev_20418.json").read_text(encoding="utf-8"))["body"]
    mother_body = json.loads((CONTRACTS / "api_prepare_mothers_20424.json").read_text(encoding="utf-8"))["body"]
    v2_pin = json.loads((ops.LOCAL / "preflight/p18_v2_model.json").read_text(encoding="utf-8"))
    if (v2_pin.get("provenance", {}).get("run_id") != "train_sac_20261006_084454_365eff71" or
            v2_pin.get("provenance", {}).get("env_step") != 400000 or
            v2_pin.get("files", {}).get("policy.pt", {}).get("sha256")
            != "78ed38b164141af3968c4498dc4a462b8baf060764f133076ea7c02c7d868cea"):
        raise ValueError("historical V2 final model pin differs")
    tasks = [
        task("prepare_dev_20418", "preparation", 20418, "prepare", dev_body,
             "dev_validation", []),
        task("prepare_mothers_20424", "preparation", 20424, "prepare", mother_body,
             "mother_sources", []),
        task("import_V2_model_20418", "preparation", 20418, "import_model",
             {"path": v2_pin["source_dir"], "name": "S2-P18 V2 frozen 400k historical anchor"},
             "dev_validation", [], source_path=v2_pin["source_dir"],
             expected_sha256=v2_pin["files"]["policy.pt"]["sha256"]),
    ]
    for port in ops.SSH_PORTS:
        tasks.append(task(f"import_train_{port}", "preparation", port, "import_scene",
            {"path": TRAIN_RECORD, "name": f"S2-P19 frozen 8064 train {port}", "profile": "europe_v1"},
            "dev_validation", [], source_record_id="variable_beams_train_20261004", expected_count=8064))
        for partition in ("development", "validation"):
            tasks.append(task(f"import_{partition}_{port}", "preparation", port, "import_scene",
                {"path": ref("prepare_dev_20418", "result", "summary", "outputs", partition, "path"),
                 "name": f"S2-P19 {partition} {port}", "profile": "europe_v1"},
                "dev_validation", ["prepare_dev_20418"],
                source_record_id=f"s2p19_{partition}_20261007", expected_count=72))
    tasks.append(task("prepare_final_20424", "preparation", 20424, "prepare",
        {"preparation": {"preparation_type": "final",
                         "source_record": {
                             "path": ref("prepare_mothers_20424", "result", "summary", "path"),
                             "sha256": ref("prepare_mothers_20424", "result", "summary", "sha256")},
                         "seed": 51001, "environment_profile": "europe_v1"}},
        "final", ["prepare_mothers_20424"]))
    for port in ops.SSH_PORTS:
        tasks.append(task(f"import_final_{port}", "preparation", port, "import_scene",
            {"path": ref("prepare_final_20424", "result", "summary", "outputs", "final", "path"),
             "name": f"S2-P19 final 880 {port}", "profile": "europe_v1"},
            "final", ["prepare_final_20424"],
            source_record_id="s2p19_final_20261007", expected_count=880))
    tasks.append(task("teacher_20424", "preparation", 20424, "prepare",
        {"preparation": {"preparation_type": "demonstrations"}, "teacher_config": c0,
         "scene_id": ref("import_train_20424", "scene_id")},
        "demonstrations", ["import_train_20424"]))
    tasks.append(task("i0_development_gate_20418", "engineering_gate", 20418, "improvement",
        {"mode": "gate", "partition": "development", "i0_config": i0,
         "scene_id": ref("import_development_20418", "scene_id")},
        "gate_B_216", ["import_development_20418"]))
    tasks.append({"id": "i0_scientific_gate", "phase": "engineering_gate", "port": 20418,
                  "kind": "milestone", "requires": ["i0_development_gate_20418"],
                  "receipt_file": str(ops.LOCAL / "controller/gates/i0_nonlearning_receipt.json")})
    tasks.append(task("C0_pilot_20418", "pilot", 20418, "constructive_train",
        {"config": c0, "scene_id": ref("import_train_20418", "scene_id"),
         "validation_scene_id": ref("import_validation_20418", "scene_id")},
        "C0_pilot", ["import_train_20418", "import_validation_20418"]))
    tasks.append(task("C1_pilot_20424", "pilot", 20424, "constructive_train",
        {"config": c1, "scene_id": ref("import_train_20424", "scene_id"),
         "validation_scene_id": ref("import_validation_20424", "scene_id")},
        "C1_pilot", ["teacher_20424", "import_validation_20424"]))
    tasks.append(task("I0_pilot_20418", "pilot", 20418, "improvement",
        {"mode": "train", "i0_config": i0,
         "scene_id": ref("import_train_20418", "scene_id"),
         "validation_scene_id": ref("import_validation_20418", "scene_id")},
        "I0_pilot", ["i0_scientific_gate", "import_validation_20418"]))
    for arm, port in (("C0", 20418), ("C1", 20424), ("I0", 20418)):
        pilot = f"{arm}_pilot_{port}"
        tasks.append({"id": f"approve_{arm}_pilot", "phase": "pilot_gate",
                      "port": port, "kind": "approve_pilot", "pilot_task_id": pilot,
                      "requires": [pilot],
                      "gate_receipt_file": str(ops.LOCAL / f"controller/gates/{arm}_pilot_receipt.json")})
    plan = {"schema": 1, "experiment_id": "S2-P19", "scientific_commit": commit,
            "revision": 1, "terminal": False, "tasks": tasks}
    controller.load_plan  # keep shared schema import explicit
    if controller.PLAN.exists():
        prior = json.loads(controller.PLAN.read_text(encoding="utf-8"))
        if prior != plan:
            raise FileExistsError("existing controller plan differs; use append-only phase advance")
    else:
        controller.write_json(controller.PLAN, plan)
    controller.load_plan()
    return {"plan": str(controller.PLAN), "scientific_commit": commit,
            "task_count": len(tasks), "plan_sha256": controller.digest(controller.PLAN)}


def _final_model_id(port: int, train_run_id: str) -> str:
    models = ops.api(port, "/api/models").get("models") or []
    matches = [item for item in models
               if (item.get("provenance") or {}).get("run_id") == train_run_id
               and (item.get("provenance") or {}).get("env_step") == 400000
               and str(item.get("path", "")).rstrip("/").endswith("/" + train_run_id + "/final")]
    if len(matches) != 1:
        raise RuntimeError(f"{port} exact 400k final model for {train_run_id} absent/ambiguous")
    return matches[0]["id"]


def advance_common_evaluation() -> dict:
    """Append retained final-model evaluations and same-route final B tasks.

    The root scientific audit writes only a retained-arms receipt. All model
    IDs, scene IDs and task outputs are resolved from completed workbench API
    records; no run ID or result path is manually entered into the plan.
    """
    plan, old_sha = controller.load_plan()
    if plan["revision"] != 1:
        raise ValueError("common evaluation advance requires the pilot-phase plan")
    if not controller.STATE.is_file():
        raise RuntimeError("controller has no verified pilot/scene receipts")
    state = json.loads(controller.STATE.read_text(encoding="utf-8"))
    receipts = state.get("tasks") or {}
    decision_file = ops.LOCAL / "controller/gates/retained_arms.json"
    if not decision_file.is_file():
        raise RuntimeError("missing source-backed scientific retention decision")
    decision = json.loads(decision_file.read_text(encoding="utf-8"))
    audit_path = Path(decision["audit_path"])
    if not audit_path.is_file() or controller.digest(audit_path) != decision["audit_sha256"]:
        raise RuntimeError("retained-arms decision audit SHA differs")
    retained = decision.get("retained_arms")
    if not isinstance(retained, list) or len(set(retained)) != len(retained) or not set(retained) <= {"C0", "C1", "I0"}:
        raise ValueError("retained arms outside authorized first round")
    if "C1" in retained and "C0" not in retained:
        raise ValueError("C1 attribution requires a matched C0 final model")
    for port in ops.SSH_PORTS:
        if receipts.get(f"import_final_{port}", {}).get("state") != "completed":
            raise RuntimeError("880-scene final import incomplete on both instances")
    for arm in retained:
        port = 20424 if arm == "C1" else 20418
        pilot = receipts.get(f"{arm}_pilot_{port}") or {}
        if pilot.get("state") != "completed" or not pilot.get("run_id"):
            raise RuntimeError(f"{arm} retained without completed same-ID full training")
        if receipts.get(f"approve_{arm}_pilot", {}).get("state") != "completed":
            raise RuntimeError(f"{arm} missing saved pilot scientific approval")

    p17_configs = json.loads(Path("D:/paper/outputs/verification/s2p19_research_20261007/p17_sca_configs.json").read_text(encoding="utf-8"))
    stochastic_B = p17_configs["V1"]["config_power_refinement"]
    greedy_B = p17_configs["B1"]["config_power_refinement"]
    if (stochastic_B["methods"] != ["none", "sca", "sca_si_lp"] or
            greedy_B["methods"] != stochastic_B["methods"] or
            stochastic_B["repeats"] != 3 or greedy_B["repeats"] != 1):
        raise ValueError("saved P17 common B route differs")
    appended = []
    appended.append(task("V2_final_direct_20418", "common_evaluation", 20418,
        "constructive_evaluate",
        {"experiment_id": "S2-P19", "model_id": ref("import_V2_model_20418", "model_id"),
         "scene_id": ref("import_final_20418", "scene_id"),
         "action_mode": "stochastic", "action_seed": 53001, "repeats": 3,
         "baselines": []},
        "final_direct_model_2640", ["import_V2_model_20418", "import_final_20418"]))
    appended.append(task("V2_final_B_20418", "refinement", 20418, "refine",
        {"source_run_id": ref("V2_final_direct_20418", "run_id"),
         "source_group": "model", "partition": "test", "config": stochastic_B},
        "final_B_one_stochastic_arm_2640", ["V2_final_direct_20418"]))
    appended.append(task("baseline_test_20418", "common_evaluation", 20418, "improvement",
        {"mode": "baselines", "partition": "test", "i0_config": ops.api(20418, "/api/s2p19/catalog")["defaults"]["I0"],
         "scene_id": ref("import_final_20418", "scene_id"),
         "methods": ["greedy", "rule", "random"], "repeats": 1, "action_seed": 53001},
        "final_direct_baselines_4400", ["import_final_20418"]))
    for group in ("greedy", "rule", "random"):
        controls = stochastic_B if group == "random" else greedy_B
        disk_class = ("final_B_one_stochastic_arm_2640" if group == "random"
                      else "final_B_one_deterministic_arm_880")
        appended.append(task(f"baseline_{group}_B_20418", "refinement", 20418, "refine",
            {"source_run_id": ref("baseline_test_20418", "run_id"),
             "source_group": group, "partition": "test", "config": controls},
            disk_class, ["baseline_test_20418"]))
    for arm in retained:
        port = 20424 if arm == "C1" else 20418
        pilot = f"{arm}_pilot_{port}"
        final_scene = f"import_final_{port}"
        if arm in ("C0", "C1"):
            model_id = _final_model_id(port, receipts[pilot]["run_id"])
            body = {"experiment_id": "S2-P19", "model_id": model_id,
                    "scene_id": ref(final_scene, "scene_id"),
                    "action_mode": "stochastic", "action_seed": 53001, "repeats": 3,
                    "baselines": []}
            kind = "constructive_evaluate"
        else:
            body = {"mode": "evaluate", "partition": "test",
                    "scene_id": ref(final_scene, "scene_id"),
                    "model_run_id": ref(pilot, "run_id"),
                    "action_seed": 53001, "repeats": 3}
            kind = "improvement"
        eval_id = f"{arm}_final_direct_{port}"
        appended.append(task(eval_id, "common_evaluation", port, kind, body,
                             "final_direct_model_2640",
                             [pilot, final_scene]))
        appended.append(task(f"{arm}_final_B_{port}", "refinement", port, "refine",
            {"source_run_id": ref(eval_id, "run_id"), "source_group": "model",
             "partition": "test", "config": stochastic_B},
            "final_B_one_stochastic_arm_2640", [eval_id]))
    appended.append({"id": "formal_report_and_archive", "phase": "report_archive",
                     "port": 20418, "kind": "milestone",
                     "requires": [row["id"] for row in appended if row["phase"] == "refinement"],
                     "receipt_file": str(ops.LOCAL / "controller/gates/formal_report_archive_receipt.json")})
    new_plan = {**plan, "revision": 2, "parent_plan_sha256": old_sha,
                "terminal": False, "retention_audit_path": str(audit_path),
                "retention_audit_sha256": decision["audit_sha256"],
                "tasks": [*plan["tasks"], *appended]}
    controller.write_json(controller.PLAN, new_plan)
    controller.load_plan()
    return {"status": "appended_common_evaluation", "retained_arms": retained,
            "new_task_count": len(appended), "plan_sha256": controller.digest(controller.PLAN),
            "unplanned_routes": [],
            "note": "Retained arms plus frozen V2 and the 1/1/3 common baselines are scheduled; report/archive milestone remains."}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("bootstrap", "advance-common-evaluation"))
    args = parser.parse_args()
    result = bootstrap() if args.action == "bootstrap" else advance_common_evaluation()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
