"""Check saved, same-scene four-color and random baseline evaluations."""

from __future__ import annotations

from collections import defaultdict
import json
import math
from pathlib import Path
import statistics

import numpy as np

from satellite_rl.training import experiment_runner as runner


POLICIES = ("four_color", "random")
FIELDS = ("si_sum", "si_mean", "sgm_mean", "throughput_bps",
          "total_power_w", "served_beams", "power_exceeded", "completed",
          "occupancy_fraction", "zero_slot_rate", "allocated_slots",
          "spectral_efficiency", "interference_beams", "in_range_interference_beams")


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def load_extended_baselines(plan, frozen_root, existing, directory):
    """Return the prior verified baselines plus two newly evaluated policies.

    Each random-policy scene is averaged across its three action seeds before
    comparison with a trained model or calculation of scene-to-scene spread.
    """
    directory = Path(directory)
    config_path = directory / "config_and_summary.json"
    records_path = directory / "per_scene_results.jsonl"
    config = _json(config_path)
    scenes = plan["test_scenes"]
    expected_scenes = set(scenes)
    seeds = tuple(int(value) for value in plan["action_seeds"])
    if (config.get("training_commit") != plan["source"]["git_commit"]
            or Path(config.get("training_root", "")).resolve() != Path(frozen_root).resolve()
            or config.get("scene_ids") != sorted(int(scene) for scene in expected_scenes)
            or config.get("scene_count") != 100
            or config.get("rollout_count") != 400
            or tuple(config.get("action_seeds", ())) != seeds
            or set(config.get("policies", {})) != set(POLICIES)
            or set(config.get("summary", {})) != set(POLICIES)
            or set(config.get("scene_sha256", {})) != expected_scenes
            or Path(config.get("per_scene_results_path", "")).resolve()
               != records_path.resolve()
            or runner.sha256_file(records_path) != config.get("per_scene_results_sha256")
            or not config.get("evaluation_commit")):
        raise RuntimeError("四色或随机基线的训练源码、场景或评价范围不符")
    for scene, reference in scenes.items():
        saved = config["scene_sha256"][scene]
        if (Path(saved["path"]).resolve() != Path(reference["path"]).resolve()
                or saved["sha256"] != reference["sha256"]):
            raise RuntimeError(f"补充基线配置中的场景来源不符：{scene}")
    source_paths = []
    for source, expected in config["source_sha256"].items():
        path = Path(source)
        if not path.is_file() or runner.sha256_file(path) != expected:
            raise RuntimeError(f"补充基线源码校验值不符：{path}")
        source_paths.append(path)
    evaluator = Path(config["evaluation_script"])
    if (not evaluator.is_file() or
            runner.sha256_file(evaluator) != config["evaluation_script_sha256"]):
        raise RuntimeError("补充基线评价脚本或校验值不符")
    source_paths.append(evaluator)
    observed = defaultdict(dict)
    trace_paths = {}
    row_count = 0
    with records_path.open(encoding="utf-8-sig") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            policy, scene = row["policy"], str(row["scene_id"])
            seed = row["action_seed"]
            if (policy not in POLICIES or scene not in expected_scenes or
                    (seed is not None and (type(seed) is not int or seed not in seeds)) or
                    (policy == "four_color" and seed is not None) or
                    (policy == "random" and seed is None)):
                raise RuntimeError("补充基线含未知策略、场景或动作种子")
            key = (scene, seed)
            if key in observed[policy]:
                raise RuntimeError(f"补充基线存在重复场景与动作种子：{policy}/{key}")
            reference = scenes[scene]
            if (Path(row["scene_path"]).resolve() != Path(reference["path"]).resolve()
                    or row["scene_sha256"] != reference["sha256"]
                    or row["total_beams"] != 100):
                raise RuntimeError("补充基线没有使用本轮保存的同一100场景")
            rates = np.asarray(row["final_rate_bps_per_beam"], dtype=np.float64)
            needs = np.asarray(row["required_rate_bps_per_beam"], dtype=np.float64)
            si = np.asarray(row["satisfaction_per_beam"], dtype=np.float64)
            sgm = np.asarray(row["sgm_per_beam"], dtype=np.float64)
            if (any(vector.shape != (100,) or not np.isfinite(vector).all()
                    for vector in (rates, needs, si, sgm))
                    or np.any(needs <= 0) or np.any(rates < 0)
                    or not np.allclose(si, np.minimum(rates / needs, 1.0),
                                       rtol=1e-10, atol=1e-8)
                    or not math.isclose(float(si.sum()), float(row["si_sum"]),
                                        rel_tol=0, abs_tol=1e-7)
                    or not math.isclose(float(si.mean()), float(row["si_mean"]),
                                        rel_tol=0, abs_tol=1e-9)
                    or not math.isclose(float(sgm.mean()), float(row["sgm_mean"]),
                                        rel_tol=0, abs_tol=1e-9)
                    or not math.isclose(float(rates.sum()), float(row["throughput_bps"]),
                                        rel_tol=1e-10, abs_tol=1e-5)):
                raise RuntimeError(f"补充基线终态逐束服务与汇总不一致：{policy}/{key}")
            for field in FIELDS:
                if not math.isfinite(float(row[field])):
                    raise RuntimeError(f"补充基线指标不是有限数：{policy}/{key}/{field}")
            if len(row["requested_actual_history"]) != int(row["served_beams"]):
                raise RuntimeError("补充基线逐步记录与处理束数不一致")
            trace_path = Path(row["trace_path"])
            digest = row["trace_sha256"]
            if (not trace_path.is_file() or runner.sha256_file(trace_path) != digest or
                    (trace_path in trace_paths and trace_paths[trace_path] != digest)):
                raise RuntimeError(f"补充基线逐步日志缺失或校验值不符：{trace_path}")
            trace_paths[trace_path] = digest
            observed[policy][key] = row
            row_count += 1
    if row_count != 400:
        raise RuntimeError(f"补充基线不是100条四色及300条随机考核：{row_count}")
    if ({scene for scene, _ in observed["four_color"]} != expected_scenes or
            set(observed["four_color"]) != {(scene, None) for scene in expected_scenes} or
            set(observed["random"]) != {(scene, seed) for scene in expected_scenes
                                        for seed in seeds}):
        raise RuntimeError("补充基线缺少共同场景或随机动作种子")
    by_policy = dict(existing["by_policy"])
    by_policy["four_color"] = {
        scene: observed["four_color"][(scene, None)] for scene in expected_scenes
    }
    by_policy["random"] = {
        scene: {field: statistics.mean(observed["random"][(scene, seed)][field]
                                       for seed in seeds) for field in FIELDS}
        for scene in expected_scenes
    }
    for policy in POLICIES:
        for field in FIELDS:
            values = [float(by_policy[policy][scene][field])
                      for scene in sorted(expected_scenes)]
            saved = config["summary"][policy][field]
            if (not math.isclose(statistics.mean(values), float(saved["mean"]),
                                 rel_tol=1e-10, abs_tol=1e-7) or
                    not math.isclose(statistics.stdev(values), float(saved["std"]),
                                     rel_tol=1e-10, abs_tol=1e-7)):
                raise RuntimeError(f"补充基线汇总与逐场景原始值不一致：{policy}/{field}")
    merged = dict(existing)
    merged_config = dict(existing["config"])
    merged_config["policies"] = {**existing["config"]["policies"], **config["policies"]}
    merged_config["summary"] = {**existing["config"]["summary"], **config["summary"]}
    merged["config"] = merged_config
    merged["by_policy"] = by_policy
    merged["extended"] = {
        "config_path": str(config_path), "config_sha256": runner.sha256_file(config_path),
        "records_path": str(records_path), "records_sha256": runner.sha256_file(records_path),
        "trace_path_sha256": {str(path): digest for path, digest in trace_paths.items()},
        "source_paths": sorted(str(path) for path in set(source_paths)),
        "evaluation_commit": config["evaluation_commit"],
    }
    return merged
