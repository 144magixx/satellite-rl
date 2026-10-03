"""Bounded Jan12 aircraft-scene X4 CUDA/CPU and A2-resume smoke check.

This is isolated code verification: one frozen real scene, one X4 action, and
one environment-state roundtrip. It creates no workbench or training task.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

from satellite_rl.coverage.balancing import BalanceStart, prepare_common_a2
from satellite_rl.coverage.balancing_env import BalanceBatchEnv
from satellite_rl.coverage.config import balance_default_config, config_hash
from satellite_rl.coverage.scenes import CoverageDataset, load_dataset


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--start", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-gpu", default="4090 D")
    args = parser.parse_args()
    repo = args.repo.resolve()
    import satellite_rl

    package = Path(satellite_rl.__file__).resolve()
    if (package != repo / "src/satellite_rl/__init__.py" or
            not torch.cuda.is_available() or torch.version.cuda != "12.1" or
            args.expected_gpu not in torch.cuda.get_device_name(0)):
        raise RuntimeError("Frozen S1-02 package or DQN CUDA differs")
    torch.set_num_threads(1)
    config = balance_default_config()
    config["coverage"]["balance"].update(route="X4", action_menu="X4")
    config["coverage"]["method"] = "local_search"
    dataset = load_dataset(args.source, "cluster", config)
    if len(dataset) != 1:
        raise RuntimeError("Smoke input must contain exactly one Jan12 aircraft scene")
    scene = dataset.scenes[0]
    foreign_record = json.loads(args.start.read_text(encoding="utf-8"))
    foreign_start = None
    foreign_note = "same_initial_centers_sha"
    try:
        foreign_start = BalanceStart.from_record(scene, foreign_record)
    except ValueError as error:
        if "original cluster-center SHA mismatch" not in str(error):
            raise
        foreign_note = "different_platform_initial_centers_sha; content_compared_after_remote_recompute"
    if (scene.metadata.get("source_dates") != ["2024-01-12"] or
            len(scene.aircraft_ids) < 100):
        raise RuntimeError("Jan12 aircraft source provenance differs")
    recomputed = prepare_common_a2(scene, config["coverage"], device="cpu")
    if not isinstance(recomputed, BalanceStart):
        raise RuntimeError("Frozen Jan12 A2 preparation did not produce a valid start")
    remote_record = recomputed.to_record()
    remote_path = args.output.with_name("real_scene_remote_a2.json")
    remote_path.parent.mkdir(parents=True, exist_ok=True)
    remote_path.write_text(json.dumps(remote_record, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    start = BalanceStart.from_record(scene, json.loads(remote_path.read_text(encoding="utf-8")))
    if foreign_start is not None:
        prior_centers, prior_assignment, prior_loads = (
            foreign_start.centers, foreign_start.associations, foreign_start.beam_loads_bps)
        prior_regions = foreign_start.regions
    else:
        prior_centers = np.asarray(foreign_record["centers"], dtype=np.float64)
        prior_assignment = np.asarray(foreign_record["associations"], dtype=np.int64)
        prior_loads = np.asarray(foreign_record["beam_loads_bps"], dtype=np.float64)
        prior_regions = foreign_record["regions"]
    if (not np.allclose(start.centers, prior_centers, rtol=0, atol=1e-10) or
            not np.array_equal(start.associations, prior_assignment) or
            not np.allclose(start.beam_loads_bps, prior_loads, rtol=1e-12, atol=1e-5) or
            start.regions != prior_regions):
        raise RuntimeError("Remote Jan12 A2 content differs from prior staged cache")
    collection = CoverageDataset([scene], {"initialization": "cluster"})
    environments = {}
    for device in ("cpu", "cuda"):
        env = BalanceBatchEnv(collection, config["coverage"], [remote_record],
                              num_envs=1, device=device, auto_reset=False)
        observation, mask = env.reset(scene_indices=[0])
        environments[device] = (env, observation, mask)
    cpu_mask = environments["cpu"][2][0].cpu()
    cuda_mask = environments["cuda"][2][0].cpu()
    if not torch.equal(cpu_mask, cuda_mask):
        raise RuntimeError("X4 legal action mask differs between CPU and CUDA")
    pair_choices = torch.nonzero(cpu_mask[5:], as_tuple=False).flatten() + 5
    if not pair_choices.numel():
        raise RuntimeError("Real Jan12 scene has no legal X4 paired-beam action")
    action = int(pair_choices[0])
    outcomes = {}
    for device, (env, observation, mask) in environments.items():
        _, reward, done, info = env.step(torch.tensor([action], device=device))
        outcomes[device] = {"env": env, "reward": float(reward[0].item()),
                            "done": bool(done[0].item()), "info": info}
    cpu, cuda = outcomes["cpu"]["env"], outcomes["cuda"]["env"]
    cpu_info, cuda_info = outcomes["cpu"]["info"], outcomes["cuda"]["info"]
    max_center = float(torch.max(torch.abs(cpu.centers - cuda.centers.cpu())).item())
    max_load = float(torch.max(torch.abs(cpu.loads - cuda.loads.cpu())).item())
    if (max_center > 1e-10 or max_load > 1e-5 or
            not torch.equal(cpu.assignment, cuda.assignment.cpu()) or
            not torch.equal(cpu_info["actual_action"].cpu(), cuda_info["actual_action"].cpu()) or
            not torch.equal(cpu_info["action_accepted"].cpu(), cuda_info["action_accepted"].cpu()) or
            not torch.equal(cpu_info["rejection_reason_code"].cpu(),
                            cuda_info["rejection_reason_code"].cpu()) or
            not np.isclose(outcomes["cpu"]["reward"], outcomes["cuda"]["reward"],
                           rtol=1e-9, atol=1e-6)):
        raise RuntimeError("Real Jan12 X4 action differs between CPU and CUDA")
    twin = BalanceBatchEnv(collection, config["coverage"], [remote_record],
                           num_envs=1, device="cuda", auto_reset=False)
    twin.load_state_dict(cuda.state_dict())
    resume_load_difference = float(torch.max(torch.abs(twin.loads - cuda.loads)).item())
    exact_tensor_fields = ("scene_index", "step_count", "beam_processes",
                           "assignment_attempts", "candidate_evaluations",
                           "done_mask", "initial_target", "stop_reason_code",
                           "centers", "assignment", "best_centers",
                           "best_associations", "best_decision_steps",
                           "best_beam_processes", "best_assignment_attempts",
                           "best_candidate_evaluations")
    exact_fields = {name: bool(torch.equal(getattr(twin, name), getattr(cuda, name)))
                    for name in exact_tensor_fields}
    rng_exact = twin.rng.bit_generator.state == cuda.rng.bit_generator.state
    if (not all(exact_fields.values()) or not rng_exact or
            not torch.allclose(twin.loads, cuda.loads, rtol=1e-12, atol=1e-6)):
        raise RuntimeError("Real Jan12 X4 environment resume state differs")
    result = {"status": "bounded_real_scene_cuda_check_passed",
              "checked_utc": datetime.now(timezone.utc).isoformat(),
              "source_path": str(args.source), "source_sha256": sha(args.source),
              "source_date_utc": "2024-01-12", "aircraft_count": len(scene.aircraft_ids),
              "scene_id": scene.scene_id, "source_hash": scene.source_hash,
              "a2_foreign_record_path": str(args.start),
              "a2_foreign_record_sha256": sha(args.start),
              "a2_foreign_start_sha256": foreign_record["start_sha256"],
              "a2_foreign_platform_note": foreign_note,
              "a2_remote_record_path": str(remote_path),
              "a2_remote_record_sha256": sha(remote_path),
              "a2_remote_start_sha256": remote_record["start_sha256"],
              "a2_recomputed_content_matches": True,
              "action_menu": "X4", "legal_pair_actions": int(pair_choices.numel()),
              "selected_pair_action": action, "beam_processes": int(cuda.beam_processes[0].item()),
              "actual_action": int(cuda_info["actual_action"][0].item()),
              "action_accepted": bool(cuda_info["action_accepted"][0].item()),
              "rejection_reason_code": int(cuda_info["rejection_reason_code"][0].item()),
              "center_max_abs_diff_cpu_cuda": max_center,
              "load_max_abs_diff_cpu_cuda_bps": max_load,
              "assignment_exact_cpu_cuda": True,
              "resume_center_assignment_exact": True,
              "resume_exact_fields": exact_fields,
              "resume_rng_exact": rng_exact,
              "resume_load_max_abs_diff_bps": resume_load_difference,
              "resume_load_tolerance": {"atol_bps": 1e-6, "rtol": 1e-12},
              "resume_state_verified": True,
              "config_hash": config_hash(config),
              "python": sys.executable, "torch": torch.__version__,
              "torch_cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
              "package_file": str(package),
              "scope": "One real Jan12 scene and one X4 paired action; not training or model evaluation."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "action": action,
                      "aircraft_count": len(scene.aircraft_ids),
                      "load_max_abs_diff_bps": max_load}, ensure_ascii=False))


if __name__ == "__main__":
    main()
