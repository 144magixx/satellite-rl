"""Submit the approved S2-P18 three-arm mechanism pilot to both instances.

The driver starts from the frozen workbench catalog defaults, applies one
explicit override block per arm using the field names the implementation
actually exposes (``state.features``, ``reward.shaping``, ``training.gamma_schedule``,
``network.readout_mode`` ...), asks the workbench to validate the result, then
submits one training job per arm through the workbench API. It never writes to
the repository.

Every override key is checked back against the validated configuration so a key
that the backend silently drops is reported instead of being trusted.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ops18  # noqa: E402

LOCAL = ops18.LOCAL
SEED = 51001
STEPS_PILOT = 100000
STEPS_FULL = 400000
SCENE_NAMES = {
    "train": "scenes_20261004_133711_dec915dc",
    "validation": "scenes_20261004_133835_f0592bee",
    "test": "scenes_20261004_133838_d7476a54",
}


def scene_records(port: int) -> dict:
    """Resolve each instance's own imported record IDs for the shared frozen inputs."""
    available = {item.get("name"): item for item in ops18.api(port, "/scenes", timeout=60).get("scenes", [])}
    resolved = {}
    for role, name in SCENE_NAMES.items():
        item = available.get(name)
        if item is None:
            raise RuntimeError(f"{port} 尚未导入 {role} 场景（{name}）")
        if int(item.get("count", 0)) != EXPECTED_COUNT[role]:
            raise RuntimeError(f"{port} {role} 场景数不符：{item.get('count')} != {EXPECTED_COUNT[role]}")
        resolved[role] = {"id": item.get("id"), "name": name, "count": item.get("count")}
    return resolved


EXPECTED_COUNT = {"train": 8064, "validation": 144, "test": 440}

# Approved 2026-10-06 placement: both cards busy with training only, so no
# non-learning timing runs concurrently on either host.
ARM_PLACEMENT = {"V2": 20418, "G0": 20418, "V3": 20424}

S2P18_FEATURES = [
    "beam", "power", "interference", "occupancy", "beta", "progress",
    "remaining_demand", "beam_count_normalized", "remaining_beams_normalized",
    "power_pressure", "group", "group_remaining_obligation", "remaining_slot_limit",
]

# The natural per-beam layout exposes exactly the fields of its context level,
# in a fixed order; "full" adds the global, per-beam and per-group blocks.
NATURAL_FULL_FEATURES = [
    "beam", "power", "interference", "occupancy",
    "natural_global_context", "natural_beam_context", "natural_resource_context",
]

BASE_OVERRIDES = {
    "algorithm": "sac",
    "environment": {
        "profile": "europe_v1",
        "beam_count_max": 200,
        "power_budget_w": 3000,
        "action_mode": "legacy_clip",
        "action_mask": "joint_feasible",
        "power_levels_w": [5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
    },
    "training": {
        "seed": SEED,
        "scene_order": "variable_beams",
        "scene_order_seed": SEED,
        "beam_count_mode": "mixed_four_buckets",
        "beam_count_min": 50,
        "beam_count_max": 200,
        "held_out_beam_counts": [60, 90, 140, 180],
        "batch_size": 256,
        "replay_capacity": 20000,
        "warmup_steps": 2000,
        "checkpoint_interval": 10000,
        "update_every": 1,
        "gradient_steps": 1,
        "tau": 0.005,
        "actor_lr": 1e-5,
        "critic_lr": 1e-4,
        "gradient_clip": 10,
        "alpha_mode": "fixed",
        "fixed_alpha": 0.01,
        "device": "cuda",
    },
    "evaluation": {
        "validation_count": 144,
        "test_count": 0,
        "validation_seed": 53001,
        "interval_steps": 20000,
        "extra_steps": [0],
        "repeats": 1,
        "action_mode": "stochastic",
    },
}

# Field names follow src/satellite_rl/workbench/config.py after the S2-P18
# implementation: state profile s2p18 adds the three observation scalars;
# reward.shaping carries the new reward gains with zero defaults; gamma_schedule
# carries the length-aware discount; the network block carries the per-beam
# context switch, the attention readout and the size conditioning.
ARM_OVERRIDES = {
    "V2": {
        # means A (throughput + spectrum gain) with B/C/E/H already folded into
        # the shared s2p18 state profile and the progress-stratified replay.
        # Network layout stays the frozen variable-beam layout; attention pooling
        # and size conditioning are only allowed under the natural per-beam
        # layout, so this arm keeps first_token / none.
        "state": {"context_level": "none", "demand_visibility": "known",
                  "features": list(S2P18_FEATURES)},
        "network": {"kind": "transformer", "transformer_layout": "tsac_14_variable_context",
                    "embedding_dim": 128, "num_heads": 8, "num_layers": 2,
                    "feedforward_dim": 512, "activation": "relu", "dropout": 0,
                    "context_beam_count": "fixed_100", "readout_mode": "first_token",
                    "beam_count_conditioning": "none"},
        "reward": {
            "preset": "legacy_repaired_throughput",
            "terms": [{"id": "si_delta", "weight": 1, "scale": 1, "timing": "step"}],
            "shaping": {"w_throughput": 1.0, "w_spectrum": 0.5, "w_shape": 0.0,
                        "throughput_reference": "mean_demand"},
        },
        "training": {
            "progress_strata": 4,
            "gamma": 0.99,
            "gamma_schedule": {"mode": "none", "default": 0.99, "by_beam_count": []},
        },
    },
    "V3": {
        # same observation as V2; only the credit assignment changes (D + F).
        "state": {"context_level": "none", "demand_visibility": "known",
                  "features": list(S2P18_FEATURES)},
        "network": {"kind": "transformer", "transformer_layout": "tsac_14_variable_context",
                    "embedding_dim": 128, "num_heads": 8, "num_layers": 2,
                    "feedforward_dim": 512, "activation": "relu", "dropout": 0,
                    "context_beam_count": "fixed_100", "readout_mode": "first_token",
                    "beam_count_conditioning": "none"},
        "reward": {
            "preset": "legacy_repaired",
            "terms": [{"id": "si_delta", "weight": 1, "scale": 1, "timing": "step"}],
            "shaping": {"w_throughput": 0, "w_spectrum": 0, "w_shape": 0.05,
                        "throughput_reference": "mean_demand"},
        },
        "training": {
            "progress_strata": 4,
            "gamma": 0.99,
            "gamma_schedule": {"mode": "by_beam_count", "default": 0.999,
                               "by_beam_count": [{"max_beam_count": 100, "gamma": 0.99},
                                                 {"max_beam_count": 200, "gamma": 0.999}]},
        },
    },
    "G0": {
        # per-beam set encoding: the natural per-beam layout is mandatory here
        # (actual-count beam context, attention pooling readout, size
        # conditioning); reward stays the frozen original. The natural layout
        # requires the exact field order of the requested context level, so the
        # level's own field list is written explicitly instead of inheriting the
        # preset's "none"-level list.
        "state": {"context_level": "full", "demand_visibility": "known",
                  "features": list(NATURAL_FULL_FEATURES)},
        "network": {"kind": "transformer", "transformer_layout": "tsac_natural_context",
                    "embedding_dim": 128, "num_heads": 8, "num_layers": 2,
                    "feedforward_dim": 512, "activation": "relu", "dropout": 0,
                    "context_beam_count": "actual", "readout_mode": "attention_pooling",
                    "beam_count_conditioning": "film"},
        "reward": {
            "preset": "legacy_repaired",
            "terms": [{"id": "si_delta", "weight": 1, "scale": 1, "timing": "step"}],
            "shaping": {"w_throughput": 0, "w_spectrum": 0, "w_shape": 0,
                        "throughput_reference": "mean_demand"},
        },
        "training": {
            "progress_strata": 4,
            "gamma": 0.99,
            "gamma_schedule": {"mode": "none", "default": 0.99, "by_beam_count": []},
        },
    },
}


def deep_update(target: dict, patch: dict) -> dict:
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            deep_update(target[key], value)
        else:
            target[key] = value
    return target


def get_path(config: dict, path: str):
    node = config
    for piece in path.split("."):
        if not isinstance(node, dict) or piece not in node:
            return None
        node = node[piece]
    return node


def verify_overrides(config: dict, patch: dict, prefix: str = "") -> list:
    """Report override keys the validated config did not carry through."""
    missing = []
    for key, value in patch.items():
        path = f"{prefix}{key}"
        if isinstance(value, dict):
            missing.extend(verify_overrides(config, value, prefix=path + "."))
            continue
        actual = get_path(config, path)
        if actual is None:
            missing.append(path)
        elif isinstance(value, list) and isinstance(actual, list):
            if len(actual) != len(value):
                missing.append(f"{path} (len {len(actual)} != {len(value)})")
        elif actual != value:
            missing.append(f"{path} ({actual!r} != {value!r})")
    return missing


def build_config(port: int, arm: str, steps: int) -> dict:
    catalog = ops18.api(port, "/catalog")
    defaults = catalog.get("defaults")
    if not isinstance(defaults, dict):
        raise RuntimeError("工作台目录缺少 defaults")
    config = deepcopy(defaults)
    config["name"] = f"S2-P18 {arm} 机制验证（{steps}步）"
    deep_update(config, BASE_OVERRIDES)
    overrides = ARM_OVERRIDES[arm]
    deep_update(config, overrides)
    config["training"]["total_steps"] = steps
    return config


def submit(port: int, arm: str, steps: int, *, execute: bool, tag: str = "") -> dict:
    # Request IDs are idempotent in the workbench: resubmitting the same ID returns
    # the existing job. A retry after a fixed startup failure therefore needs its own
    # tag, and the failed attempt stays in the records instead of being overwritten.
    suffix = f"-{tag}" if tag else ""
    request_id = f"s2p18-{arm.lower()}-pilot-{steps}-51001{suffix}"
    records = scene_records(port)
    config = build_config(port, arm, steps)
    # The arm identity is written into the frozen configuration: the validator
    # only permits the registered per-arm differences when the config declares
    # which arm it is, and the hash then changes with the arm.
    config["experiment"] = {"id": "S2-P18", "arm": arm}
    validated = ops18.api(port, "/validate", config, method="POST").get("config")
    if not isinstance(validated, dict):
        raise RuntimeError("工作台未返回校验后的配置")
    missing = verify_overrides(validated, ARM_OVERRIDES[arm])
    report = {"port": port, "arm": arm, "request_id": request_id, "steps": steps,
              "scene_records": records,
              "overrides_not_carried": missing,
              "state_dim": (validated.get("state") or {}).get("state_dim"),
              "config_hash": (validated.get("metadata") or {}).get("config_hash"),
              "config": validated}
    if missing:
        raise RuntimeError(f"以下配置键未被后台接受：{missing}")
    body = {"config": validated, "request_id": request_id,
            "scene_id": records["train"]["id"],
            "validation_scene_id": records["validation"]["id"]}
    if not execute:
        return {"status": "plan_only", **report}
    job = ops18.api(port, "/runs", body, method="POST", timeout=300)
    return {"status": "submitted", "job": job, **report}


def arm_runs_like(port: int, arm: str) -> list:
    """List this arm's runs, skipping superseded failed attempts."""
    runs = ops18.api(port, "/runs", timeout=60).get("runs", [])
    out = []
    for item in runs:
        request = str(item.get("request_id", ""))
        if not request.startswith(f"s2p18-{arm.lower()}-"):
            continue
        if "retry" in request and "retry3" not in request:
            continue
        detail = ops18.api(port, "/runs/" + item["run_id"], timeout=60)
        status = detail.get("status") or {}
        out.append({"run_id": item["run_id"], "request_id": request,
                    "kind": item.get("kind"),
                    "stage": status.get("stage") or item.get("stage"),
                    "env_step": status.get("env_step"), "directory": detail.get("path")})
    return out


def submit_evaluation(port: int, arm: str, *, execute: bool, tag: str = "") -> dict:
    """Submit the 440-scene common assessment for one arm.

    The server requires three stochastic action samples with seed 53001 for
    variable-beam assessments and rejects deterministic evaluation, so the call
    never relaxes those constraints.
    """
    records = scene_records(port)
    models = ops18.api(port, "/models", timeout=120).get("models", [])
    completed = [run for run in arm_runs_like(port, arm)
                 if run["kind"] == "train" and run["stage"] == "completed"]
    if not completed:
        raise RuntimeError(f"{arm} 尚无已完成的训练任务，不能提交考核")
    run = completed[0]
    candidates = [item for item in models if run["run_id"] in (item.get("path") or "")]
    if not candidates:
        raise RuntimeError(f"{arm} 的完成模型尚未登记到模型库")
    model = candidates[0]
    suffix = f"-{tag}" if tag else ""
    body = {"model_id": model["id"], "scene_id": records["test"]["id"],
            "experiment_id": "S2-P18", "action_mode": "stochastic",
            "repeats": 3, "action_seed": 53001,
            "baselines": ["fixed", "greedy"],
            "request_id": f"s2p18-{arm.lower()}-jan29-eval{suffix}"}
    plan = {"port": port, "arm": arm, "run_id": run["run_id"],
            "model_id": model["id"], "model_path": model.get("path"),
            "test_scenes": records["test"], "request_id": body["request_id"]}
    if not execute:
        return {"status": "plan_only", **plan}
    job = ops18.api(port, "/evaluations", body, method="POST", timeout=600)
    return {"status": "submitted", "job": job, **plan}


def submit_sca(port: int, arm: str, *, execute: bool, tag: str = "") -> dict:
    """Submit the unified SCA refinement for one arm's completed assessment.

    The server requires the frozen variable-beam method set, three stochastic
    action samples with seed 53001 and the total-power budget route, and it binds
    the refinement to a completed assessment through ``source_run_id``.
    """
    completed = [run for run in arm_runs_like(port, arm)
                 if run["kind"] == "evaluate" and run["stage"] == "completed"]
    if not completed:
        raise RuntimeError(f"{arm} 尚无已完成的440景考核，SCA 缺少冻结初解来源")
    assessment = completed[0]
    controls = {"methods": ["none", "sca", "sca_si_lp"], "power_budget_mode": "total",
                "action_mode": "stochastic", "repeats": 3, "action_seed": 53001}
    suffix = f"-{tag}" if tag else ""
    body = {"source_run_id": assessment["run_id"], "source_group": "model",
            "config": controls,
            "request_id": f"s2p18-{arm.lower()}-sca{suffix}"}
    plan = {"port": port, "arm": arm, "source_run_id": assessment["run_id"],
            "source_step": assessment.get("env_step"), "controls": controls,
            "request_id": body["request_id"]}
    if not execute:
        available = ops18.api(port, "/power-refinement/initial-sources", timeout=120)
        plan["initial_sources"] = [row for row in available.get("sources", [])
                                   if row.get("run_id") == assessment["run_id"]]
        return {"status": "plan_only", **plan}
    job = ops18.api(port, "/power-refinement/runs", body, method="POST", timeout=900)
    return {"status": "submitted", "job": job, **plan}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=tuple(ARM_OVERRIDES), required=True)
    parser.add_argument("--steps", type=int, choices=(STEPS_PILOT, STEPS_FULL),
                        default=STEPS_PILOT)
    parser.add_argument("--port", type=int, choices=ops18.PORTS)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--tag", default="", help="重试时使用的请求标记，避免幂等返回旧任务")
    parser.add_argument("--evaluate", action="store_true", help="提交440景共同考核而不是训练")
    parser.add_argument("--sca", action="store_true", help="提交统一SCA细调（需先有完成的考核）")
    args = parser.parse_args()
    port = args.port or ARM_PLACEMENT[args.arm]
    if args.sca:
        result = submit_sca(port, args.arm, execute=args.execute, tag=args.tag)
    elif args.evaluate:
        result = submit_evaluation(port, args.arm, execute=args.execute, tag=args.tag)
    else:
        result = submit(port, args.arm, args.steps, execute=args.execute, tag=args.tag)
    receipt = LOCAL / "preflight" / f"submit_{args.arm}_{args.steps}_{port}.json"
    ops18.write_json(receipt, result)
    print(json.dumps({key: value for key, value in result.items() if key != "config"},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
