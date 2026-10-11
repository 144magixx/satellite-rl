"""Resume an existing ND12 local DAG without re-registering or rewriting data.

Operator replaces the old local client after preserving its state. Remote
workers and the scientific run plan are untouched. Both existing OS locks are
retained; this entry cannot run beside the previous launcher/controller.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

from scripts.stage1_nondrl12 import ops
from scripts.stage1_nondrl12 import controller as original_controller
from scripts.stage1_nondrl12.controller import stable_sha, validate_plan
from scripts.stage1_nondrl12 import controller_pipeline
from scripts.stage1_nondrl12 import placement_transport
from scripts.stage1_nondrl12 import execution_sources
from satellite_rl.workbench import nondrl_source_repair
from satellite_rl.coverage.nondrl import source_lineage
from satellite_rl.coverage.nondrl.runner import worker_lock
from satellite_rl.training.run_monitor import atomic_json
from satellite_rl.workbench.runtime import process_start_time
from satellite_rl.workbench.storage import now


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def load_resume_inputs(plan_path, directory, commit):
    """Read original bytes, proving both canonical state bindings and roles."""
    paths = {"plan": Path(plan_path), "registry": Path(directory) / "dataset_registry.json",
             "controller_state": Path(directory) / "state.json", "launcher": Path(directory) / "launcher.json"}
    raw = {key: path.read_bytes() for key, path in paths.items()}
    values = {key: json.loads(value.decode("utf-8-sig")) for key, value in raw.items()}
    plan, registry, state, launcher = (values[k] for k in ("plan", "registry", "controller_state", "launcher"))
    validate_plan(plan, registry)
    if (plan["scientific_commit"] != commit or registry["scientific_commit"] != commit or
            launcher.get("scientific_commit") != commit or state.get("experiment_id") != "S1-ND12" or
            state["registry_sha256"] != stable_sha(registry) or state["plan_sha256"] != stable_sha(plan)):
        raise ValueError("existing state/registry/plan scientific binding differs")
    if launcher.get("plan_sha256") != sha_bytes(raw["plan"]):
        raise ValueError("original launcher plan byte SHA differs")
    return values, raw, {key: sha_bytes(value) for key, value in raw.items()}


def verify_registered_roles(registry, transport=ops):
    for port in original_controller.PORTS:
        records = {row["id"]: row for row in transport.api(port, "/api/coverage/nondrl/datasets")["datasets"]}
        for role, expected in registry["lanes"][str(port)].items():
            actual = records.get(expected["id"])
            if (actual is None or actual.get("schema_version") != 7 or actual.get("partition_role") != role or
                    any(actual.get(k) != expected[k] for k in ("id", "sha256", "count", "path")) or
                    actual.get("partition_manifest_sha256") != registry["partition_manifest_sha256"]):
                raise ValueError(f"existing {port}/{role} registered data differs")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--directory", type=Path, default=ops.LOCAL / "controller")
    parser.add_argument("--placement-file", type=Path)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args(argv)
    directory = args.directory.resolve()
    # Do not call prepare/register. Existing data registry bytes are immutable.
    with worker_lock(directory / "launcher_lock"), worker_lock(directory):
        values, raw, hashes = load_resume_inputs(args.plan, directory, args.commit)
        placement_path = args.placement_file or directory / "execution_placement.json"
        transport = placement_transport.Transport.from_file(placement_path)
        verify_registered_roles(values["registry"], transport)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_pipeline_" + str(os.getpid())
        history = directory / "history" / stamp
        history.mkdir(parents=True)
        for key, payload in raw.items():
            if key in ("controller_state","launcher") and len(payload)>1<<20:
                source=directory/("state.json" if key=="controller_state" else "launcher.json")
                ref=execution_sources.preserved_payload_ref(source,hashes[key],len(payload))
                if ref is None:raise ValueError("large previous coordinator bytes need explicit verified preservation")
                atomic_json(history/(key+".before.ref.json"),ref)
            else:
                (history / (key + ".before.json")).write_bytes(payload)
        modules = [Path(__file__), Path(controller_pipeline.__file__), Path(original_controller.__file__),
                   Path(placement_transport.__file__), Path(ops.__file__),Path(execution_sources.__file__),
                   Path(nondrl_source_repair.__file__),Path(source_lineage.__file__)]
        policy_path=directory/"source_policy.json"
        if policy_path.is_file():
            policy=json.loads(policy_path.read_bytes())
            contract_path=Path(policy["local_contract_path"])
            contract=execution_sources.contract_bytes(contract_path.read_bytes(),policy["contract_sha256"])
            modules.extend((policy_path,contract_path))
            for receipt in contract["validation_receipts"]:
                path=Path(policy.get("validation_path_map",{}).get(receipt["path"],receipt.get("local_path",receipt["path"])))
                if sha_bytes(path.read_bytes())!=receipt["sha256"]:raise ValueError("repair validation byte SHA differs")
                modules.append(path)
        disposition_path=directory/"bounded_dispositions.json"
        if disposition_path.is_file():modules.append(disposition_path)
        guards = {str(path): sha_bytes(path.read_bytes()) for path in modules}
        atomic_json(history / "transition.json", {"from_launcher_pid": values["launcher"].get("pid"),
                    "previous_byte_sha256": hashes, "new_module_sha256": guards,
                    "scientific_commit": args.commit, "registry_reused_without_register": True})
        launcher = {**values["launcher"], "pid": os.getpid(), "process_start_time": process_start_time(os.getpid()),
                    "phase": "running_workbench_dag", "client_type": "route_readiness_pipeline_v1",
                    "resumed_utc": now(), "resume_history": str(history), "module_sha256": guards,
                    "registry_file_sha256": hashes["registry"], "placement_file": str(transport.path),
                    "placement_sha256": transport.mapping_sha256, "logical_to_physical": transport.mapping}
        execution_sources.compact_state_history(directory/"launcher.json",launcher)
        # Previous failure bytes were preserved in the transition history.
        # Do not display them as the state of this new verified client.
        for stale in ("error", "traceback", "controller_attention"):
            launcher.pop(stale, None)
        def save(**fields):
            launcher.update(observed_utc=now(), **fields)
            atomic_json(directory / "launcher.json", launcher)
        controller = controller_pipeline.PipelineController(directory, values["plan"], values["registry"], transport)
        save(controller_phase=controller.state["phase"], jobs=len(controller.state["jobs"]))
        while True:
            try:
                transport.verify_unchanged()
                if (sha_bytes(args.plan.read_bytes()) != hashes["plan"] or
                        sha_bytes((directory / "dataset_registry.json").read_bytes()) != hashes["registry"] or
                        any(sha_bytes(Path(path).read_bytes()) != sha for path, sha in guards.items())):
                    raise RuntimeError("pipeline client source, plan or original registry bytes changed")
                if (directory / "launcher.stop.request").exists():
                    raise RuntimeError("explicit launcher stop requested")
                current = controller.tick()
                save(controller_phase=current["phase"], jobs=len(current["jobs"]),
                     route_attention=current.get("route_attention", {}), pipeline_routes=current.get("pipeline_routes", {}))
                print(json.dumps({"at": launcher["observed_utc"], "controller_phase": current["phase"],
                                  "jobs": len(current["jobs"]), "route_attention": list(current.get("route_attention", {}))}), flush=True)
                if current["phase"] in ("report_pending", "attention_required", "stopped"):
                    save(phase=current["phase"], controller_attention=current.get("attention"))
                    return
            except Exception as error:
                save(phase="attention_required", error=repr(error), traceback=traceback.format_exc())
                return
            if args.once:
                return
            time.sleep(20)


if __name__ == "__main__":
    main()
