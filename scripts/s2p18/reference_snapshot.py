"""S2-P18 数值与哈希参照快照工具（只读，不训练）。

该工具被冻结前先在旧实现上运行一次并保存参照文件，改动后再运行一次做逐位对照。
它只做前向、掩码构造和配置哈希，不启动任何训练或评价任务。

用法：
    python scripts/s2p18/reference_snapshot.py --output <参照.json>
    python scripts/s2p18/reference_snapshot.py --compare <参照.json>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from satellite_rl.envs.satellite import BeamInfo  # noqa: E402
from satellite_rl.state_layouts import VARIABLE_BEAM_FEATURES  # noqa: E402
from satellite_rl.workbench.checkpoints import config_hash  # noqa: E402
from satellite_rl.workbench.config import config_summary, default_config, validate_config  # noqa: E402
from satellite_rl.workbench.environment import WorkbenchEnv  # noqa: E402


def _scene(count, profile, seed=51001):
    np.random.seed(seed)
    source = BeamInfo(profile=profile, count=count)
    scene = {key: np.array(value, copy=True) for key, value in source.beam_info.items()}
    scene["original_indices"] = np.array(source.shuffle_indices, copy=True)
    return scene


def _hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def _finite_floats(state):
    return [float(value) for value in np.asarray(state, dtype=np.float64).reshape(-1)]


def legacy_variable_config():
    """V0/V1 已冻结的 210 维可变束数配置（旧奖励、旧状态）。"""
    config = default_config()
    config["environment"].update(profile="europe_v1", beam_count_max=200, action_mask="joint_feasible")
    config["network"].update(kind="transformer", transformer_layout="tsac_14_variable_context",
                             embedding_dim=128, num_heads=8, num_layers=2,
                             feedforward_dim=512, dropout=0)
    config["state"]["features"] = list(VARIABLE_BEAM_FEATURES)
    return validate_config(config)


def legacy_207_config():
    config = default_config()
    config["environment"].update(profile="europe_v1", beam_count_max=200, action_mask="joint_feasible")
    config["network"].update(kind="transformer", transformer_layout="tsac_14_context",
                             embedding_dim=128, num_heads=8, num_layers=2,
                             feedforward_dim=512, dropout=0)
    config["state"]["features"] = list(VARIABLE_BEAM_FEATURES[:7])
    return validate_config(config)


def china_default_config():
    return validate_config(default_config())


def natural_none_config():
    config = default_config()
    config["environment"].update(profile="europe_v1")
    config["network"].update(kind="transformer", transformer_layout="tsac_natural_context",
                             embedding_dim=128, num_heads=8, num_layers=2,
                             feedforward_dim=512, dropout=0)
    config["state"].update(context_level="none", features=["beam", "power", "interference", "occupancy"])
    config["training"].update(scene_order="shuffle_each_cycle")
    return validate_config(config)


def _episode_snapshot(config, scene, turns):
    env = WorkbenchEnv(config)
    state = env.reset(scene=scene)
    rows = []
    rewards = []
    for turn in range(turns):
        mask = env.action_mask()
        requested = (turn * 7 % 100, turn % 5, turn % 10)
        if mask is not None:
            found = None
            for freq in range(100):
                for slot in range(10):
                    if mask["frequency_slots"][freq, slot]:
                        found = (freq, slot, 9)
                        break
                if found is not None:
                    break
            if found is None:
                raise RuntimeError("掩码在快照步数内没有合法动作")
            requested = found
        state, reward, done, info = env.step(requested)
        rewards.append(float(reward))
        rows.append({"state": _finite_floats(state),
                     "reward": float(reward),
                     "info_occupy": int(info["occupy"]),
                     "info_out": int(info["out"]),
                     "info_actual_slots": int(info["actual_action"]["slots"]),
                     "info_actual_freq": int(info["actual_action"]["freq"]),
                     "info_actual_power": float(info["actual_action"]["power_w"]),
                     "mask_pairs": (int(mask["frequency_slots"].sum()) if mask else None),
                     "reward_terms": {key: float(value["weighted"])
                                      for key, value in (info.get("reward_terms") or {}).items()}})
        if done:
            break
    metrics = env.metrics()
    return {"states": rows, "rewards": rewards,
             "metrics": {key: value for key, value in metrics.items()
                         if isinstance(value, (int, float, bool)) or value is None}}


def build_snapshot():
    snapshot = {}
    leagues = {
        "variable_210_100": (legacy_variable_config(), 100, 12),
        "variable_210_150": (legacy_variable_config(), 150, 8),
        "legacy_207_100": (legacy_207_config(), 100, 12),
        "natural_none_100": (natural_none_config(), 100, 12),
    }
    for name, (config, count, turns) in leagues.items():
        scene = _scene(count, config["environment"]["profile"])
        summary = config_summary(config)
        snapshot[name] = {
            "config_hash": config_hash(config),
            "state_dim": summary["state_dim"],
            "network_token_count": summary["network_token_count"],
            "state_layout": summary["state_layout"],
            "episode": _episode_snapshot(config, scene, turns),
        }
    snapshot["config_hashes"] = {
        "china_default": config_hash(china_default_config()),
        "europe_variable_210": config_hash(legacy_variable_config()),
        "europe_variable_207": config_hash(legacy_207_config()),
        "europe_natural_none": config_hash(natural_none_config()),
    }
    snapshot["snapshot_sha256"] = _hash(snapshot)
    return snapshot


def compare(reference, current):
    problems = []
    if reference["snapshot_sha256"] != current["snapshot_sha256"]:
        problems.append("总快照哈希不同")
    for name in sorted(set(reference) | set(current)):
        if name == "snapshot_sha256":
            continue
        want, got = reference.get(name), current.get(name)
        if want == got:
            continue
        if isinstance(want, dict) and isinstance(got, dict):
            for key in sorted(set(want) | set(got)):
                if want.get(key) != got.get(key):
                    problems.append(f"{name}.{key}: {want.get(key)!r} != {got.get(key)!r}")
        else:
            problems.append(f"{name}: 不一致")
    return problems


def _state_differences(want, got):
    """定位逐位不同的状态分量，供“修正2数值对照表”直接引用。"""
    rows = []
    want_states = want.get("states", [])
    got_states = got.get("states", [])
    for step, (left, right) in enumerate(zip(want_states, got_states)):
        for index, (a, b) in enumerate(zip(left["state"], right["state"])):
            if a != b:
                rows.append((step, index, a, b))
    return rows


def numeric_report(reference, current):
    """打印每个场景的状态分量差异表（只用于人工核对，不参与通过/失败判定）。"""
    for name in sorted(reference):
        if not isinstance(reference[name], dict) or "episode" not in reference[name]:
            continue
        rows = _state_differences(reference[name]["episode"], current[name]["episode"])
        width = len(reference[name]["episode"]["states"][0]["state"]) if reference[name]["episode"]["states"] else 0
        print(f"[{name}] 状态宽 {width}，逐位不同分量数 {len(rows)}")
        for step, index, a, b in rows[:8]:
            print(f"    step={step} 分量#{index}: 修正前={a!r} 修正后={b!r}")
        want_rewards = reference[name]["episode"]["rewards"]
        got_rewards = current[name]["episode"]["rewards"]
        same = want_rewards == got_rewards
        print(f"    奖励序列逐位相同: {same}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--compare", type=Path)
    parser.add_argument("--numeric-report", action="store_true",
                        help="同时打印逐分量的数值差异表")
    args = parser.parse_args(argv)
    snapshot = build_snapshot()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"已写出参照: {args.output}")
    if args.compare:
        reference = json.loads(args.compare.read_text(encoding="utf-8"))
        if args.numeric_report:
            numeric_report(reference, snapshot)
        problems = compare(reference, snapshot)
        if problems:
            print(f"逐位对照不一致项数: {len(problems)}")
            for row in problems[:20]:
                print(" -", row[:200])
            return 1
        print(f"逐位对照一致: {args.compare}")
    if not args.output and not args.compare:
        print(json.dumps({key: value for key, value in snapshot.items()
                          if key != "config_hashes"}, ensure_ascii=False)[:400])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
