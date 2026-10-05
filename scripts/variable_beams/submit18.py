"""Submit the approved S2-P18 three-arm mechanism pilot to both instances.

The script is a thin driver: it starts from the frozen workbench catalog
defaults, applies one explicit override block per arm, asks the workbench to
validate the resulting configuration, then submits one training job per arm
through the workbench API. It never writes to the repository.
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
SCENE_RECORDS = {
    "train": "scenes_20261004_133711_dec915dc",
    "validation": "scenes_20261004_133835_f0592bee",
    "test": "scenes_20261004_133838_d7476a54",
}

# Arm assignment approved 2026-10-06: both cards busy, non-learning timing
# stays exclusive because only training is submitted in this phase.
ARM_PLACEMENT = {"V2": 20418, "G0": 20418, "V3": 20424}

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

ARM_OVERRIDES = {
    "V2": {
        # reward gains a throughput and a spectrum term; observation gains the
        # corrected power reference, the reuse-group phase and the remaining
        # reservation duty; replay adds progress-stratified sampling.
        "reward": {
            "preset": "legacy_repaired_throughput",
            "terms": [{"id": "si_delta", "weight": 1, "scale": 1, "timing": "step"}],
        },
        "training": {
            "progress_stratified_sampling": True,
            "progress_strata": 4,
            "gamma": 0.99,
            "cross_beam_shaping_weight": 0.0,
        },
    },
    "V3": {
        # same state as V2, learning mechanism differs instead of the reward
        # main terms: length-aware discount and potential-based cross-beam
        # shaping.
        "reward": {
            "preset": "legacy_repaired",
            "terms": [{"id": "si_delta", "weight": 1, "scale": 1, "timing": "step"}],
        },
        "training": {
            "discount_profile": "by_beam_count",
            "gamma": 0.99,
            "gamma_long": 0.999,
            "gamma_threshold_beams": 100,
            "cross_beam_shaping_weight": 0.05,
        },
    },
    "G0": {
        # per-beam set encoding: open the beam context, switch the readout to
        # attention pooling and condition on N.
        "network_preset": "tsac_natural_context",
        "state": {"context_level": "full", "demand_visibility": "known"},
        "reward": {
            "preset": "legacy_repaired",
            "terms": [{"id": "si_delta", "weight": 1, "scale": 1, "timing": "step"}],
        },
        "training": {
            "gamma": 0.99,
            "readout_mode": "attention_pool",
            "size_conditioning": "film",
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


def catalog_defaults(port: int) -> dict:
    catalog = ops18.api(port, "/catalog")
    defaults = catalog.get("defaults")
    if not isinstance(defaults, dict):
        raise RuntimeError("工作台目录缺少 defaults")
    presets = {item.get("id"): item for item in catalog.get("network_presets", [])}
    return {"catalog": catalog, "defaults": defaults, "presets": presets}


def build_config(port: int, arm: str, steps: int) -> dict:
    material = catalog_defaults(port)
    overrides = ARM_OVERRIDES[arm]
    config = deepcopy(material["defaults"])
    config["name"] = f"S2-P18 {arm} 机制验证（{steps}步）"
    deep_update(config, BASE_OVERRIDES)
    preset_id = overrides.get("network_preset")
    if preset_id:
        preset = material["presets"].get(preset_id)
        if preset is None:
            raise RuntimeError(f"工作台目录缺少网络预设 {preset_id}")
        deep_update(config["network"], preset["network"])
        if preset.get("state_features"):
            config["state"]["features"] = list(preset["state_features"])
    deep_update(config, {key: value for key, value in overrides.items()
                         if key not in ("network_preset",)})
    config["training"]["total_steps"] = steps
    if "gamma" not in overrides.get("training", {}):
        config["training"]["gamma"] = 0.99
    return config


def validate(port: int, config: dict) -> dict:
    result = ops18.api(port, "/validate", config, method="POST")
    validated = result.get("config")
    if not isinstance(validated, dict):
        raise RuntimeError("工作台未返回校验后的配置")
    return validated


def submit(port: int, arm: str, steps: int, *, execute: bool) -> dict:
    request_id = f"s2p18-{arm.lower()}-pilot-51001"
    config = build_config(port, arm, steps)
    validated = validate(port, config)
    scenes = {role: {"id": record} for role, record in SCENE_RECORDS.items()}
    body = {"config": validated, "request_id": request_id,
            "scenes": {"train": SCENE_RECORDS["train"],
                       "validation": SCENE_RECORDS["validation"]},
            "scenes_by_role": scenes}
    plan = {"port": port, "arm": arm, "request_id": request_id, "steps": steps,
            "config": validated}
    if not execute:
        return {"status": "plan_only", **plan}
    job = ops18.api(port, "/runs", body, method="POST", timeout=300)
    return {"status": "submitted", "job": job, **plan}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=tuple(ARM_OVERRIDES), required=True)
    parser.add_argument("--steps", type=int, choices=(STEPS_PILOT, STEPS_FULL),
                        default=STEPS_PILOT)
    parser.add_argument("--port", type=int, choices=ops18.PORTS)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--dump-local", action="store_true",
                        help="只在本机打印计划，不连远端")
    args = parser.parse_args()
    port = args.port or ARM_PLACEMENT[args.arm]
    if args.dump_local:
        print(json.dumps({"arm": args.arm, "port": port, "steps": args.steps,
                          "overrides": ARM_OVERRIDES[args.arm]},
                         ensure_ascii=False, indent=2))
        return
    result = submit(port, args.arm, args.steps, execute=args.execute)
    receipt = LOCAL / "preflight" / f"submit_{args.arm}_{args.steps}_{port}.json"
    ops18.write_json(receipt, result)
    print(json.dumps({key: value for key, value in result.items() if key != "config"},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
