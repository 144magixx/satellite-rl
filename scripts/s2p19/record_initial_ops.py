"""Record verified read-only S2-P19 operational preflight in run_manifest."""

from __future__ import annotations

import json
from pathlib import Path

from . import disk_guard, ops
from .ops_manifest import merge_operations


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    remote = ops.LOCAL / "preflight/remote_probe.json"
    inventory = ops.LOCAL / "preflight/data_inventory.json"
    deployment = ops.LOCAL / "preflight/deployment_plan.json"
    p18 = Path("D:/paper/outputs/verification/s2p19_research_20261007/p18_final_gate_reaudit.json")
    v2_pin = ops.LOCAL / "preflight/p18_v2_model.json"
    probes = read(remote)
    data = read(inventory)
    deploy = read(deployment)
    final = read(p18)
    v2 = read(v2_pin)
    patch = {
        "remote_preflight": {
            "status": "read_only_verified", "receipt": str(remote),
            "instances": {
                port: {"observed_utc": item.get("observed_utc"),
                       "dqn_executable": item.get("executable"),
                       "torch": item.get("torch"),
                       "gpu": item.get("nvidia_smi", {}).get("stdout"),
                       "new_api_port_state": item.get("ports", {}).get("8900" if port == "20418" else "8901")}
                for port, item in probes["instances"].items()}},
        "data_inventory": {"status": "source_records_verified_new_final_archives_absent",
                           "receipt": str(inventory), "observed_utc": data["observed_utc"],
                           "planned_final_dates": ["2024-02-01", "2024-02-02"],
                           "official_release_assets": [
                               {"date_utc": "2024-02-01", "asset_id": 149304672,
                                "expected_bytes": 1784412160},
                               {"date_utc": "2024-02-02", "asset_id": 149522532,
                                "expected_bytes": 1714810880}]},
        "p18_saved_final_reaudit": {"status": "read_only_completed", "receipt": str(p18),
            "V2_G1": final["roles"]["V2"]["G1_pass_descriptive"],
            "V2_G2": final["roles"]["V2"]["G2_pass_descriptive"],
            "no_new_evaluation": True},
        "historical_V2_model": {"status": "pinned_read_only_for_future_workbench_import",
                                "receipt": str(v2_pin), "source_path": v2["source_dir"],
                                "policy_sha256": v2["files"]["policy.pt"]["sha256"],
                                "source_run_id": v2["provenance"]["run_id"],
                                "source_env_step": v2["provenance"]["env_step"]},
        "deployment": {"status": "plan_only_waiting_clean_scientific_commit",
                       "receipt": str(deployment),
                       "targets": {port: {"api_port": item["api_port"],
                                          "port_busy": item["api_port_busy"],
                                          "repo_exists": item["repo_exists"]}
                                   for port, item in deploy["instances"].items()}},
        "disk_guard": {"status": "enabled_in_controller_not_yet_running",
                       "reserve_bytes": disk_guard.RESERVE_BYTES,
                       "forecast_complete_B_bytes_each": 10 * 1024 ** 2,
                       "receipt_20418": str(ops.LOCAL / "preflight/disk_20418.json"),
                       "receipt_20424": str(ops.LOCAL / "preflight/disk_20424.json")},
        "controller": {"status": "implemented_not_started_waiting_frozen_plan",
                       "plan_path": str(ops.LOCAL / "controller/plan.json"),
                       "state_path": str(ops.LOCAL / "controller/state.json")},
    }
    result = merge_operations(patch)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
