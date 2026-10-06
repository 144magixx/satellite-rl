"""S2-P18 三臂配置端到端核验（不训练）。

用途：
1. 构造 V2/V3/G0 三臂配置，走服务端 `validate_config` 与 `config_hash`；
2. 打印“实际生效”的关键配置片段，证明配置里的键没有被后台静默丢弃；
3. 各做一次前向、掩码构造与一次参数更新冒烟（可选 CUDA）；
4. 另跑一遍旧 100 束逐位对照（默认配置）。

用法：
    python scripts/s2p18/verify_arms.py [--cuda] [--out <json>]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from satellite_rl.agents.workbench_algorithms import build_agent  # noqa: E402
from satellite_rl.envs.satellite import BeamInfo  # noqa: E402
from satellite_rl.state_layouts import (S2P18_VARIABLE_FEATURES,  # noqa: E402
                                        natural_shapes_actual_count)
from satellite_rl.workbench.checkpoints import config_hash  # noqa: E402
from satellite_rl.workbench.config import config_summary, default_config, validate_config  # noqa: E402
from satellite_rl.workbench.environment import WorkbenchEnv  # noqa: E402


S2P18_EXTRA_FEATURES = ["group", "group_remaining_obligation", "remaining_slot_limit"]


def scene_of(count, seed=51001):
    np.random.seed(seed)
    source = BeamInfo(profile="europe_v1", count=count)
    scene = {key: np.array(value, copy=True) for key, value in source.beam_info.items()}
    scene["original_indices"] = np.array(source.shuffle_indices, copy=True)
    return scene


def _variable_base():
    config = default_config()
    config["environment"].update(profile="europe_v1", beam_count_max=200,
                                  action_mask="joint_feasible")
    config["training"].update(
        seed=51001, scene_order="variable_beams", beam_count_mode="mixed_four_buckets",
        beam_count_min=50, beam_count_max=200, held_out_beam_counts=[60, 90, 140, 180],
        total_steps=400000, batch_size=256, replay_capacity=20000, warmup_steps=2000,
        update_every=1, gradient_steps=1, alpha_mode="fixed", fixed_alpha=0.01,
        gamma=0.99, tau=0.005, actor_lr=1e-5, critic_lr=1e-4, gradient_clip=10,
        checkpoint_interval=10000)
    config["evaluation"].update(validation_count=144, validation_seed=53001,
                                interval_steps=20000, extra_steps=[0], test_count=0,
                                action_mode="stochastic")
    return config


def arm_configs():
    """按前端 preset 的同一组键构造三臂（等价于页面“套用 S2-P18 预设”）。"""
    arms = {}
    v2 = _variable_base()
    v2["network"].update(kind="transformer", transformer_layout="tsac_14_variable_context",
                         embedding_dim=128, num_heads=8, num_layers=2, feedforward_dim=512,
                         dropout=0, context_beam_count="fixed_100", readout_mode="first_token",
                         beam_count_conditioning="none")
    v2["state"]["features"] = list(S2P18_VARIABLE_FEATURES)
    v2["reward"]["preset"] = "legacy_repaired_throughput"
    v2["reward"]["shaping"] = dict(w_throughput=50, w_spectrum=1, w_shape=0,
                                   throughput_reference="mean_demand")
    v2["training"]["progress_strata"] = 4
    arms["V2"] = v2

    v3 = _variable_base()
    v3["network"].update(kind="transformer", transformer_layout="tsac_14_variable_context",
                         embedding_dim=128, num_heads=8, num_layers=2, feedforward_dim=512,
                         dropout=0, context_beam_count="fixed_100", readout_mode="first_token",
                         beam_count_conditioning="none")
    v3["state"]["features"] = list(S2P18_VARIABLE_FEATURES)
    v3["reward"]["preset"] = "legacy_repaired"
    v3["reward"]["shaping"] = dict(w_throughput=0, w_spectrum=0, w_shape=0.5,
                                   throughput_reference="mean_demand")
    v3["training"]["gamma_schedule"] = dict(mode="by_beam_count", default=0.995,
                                            by_beam_count=[dict(max_beam_count=100, gamma=0.99)])
    arms["V3"] = v3

    g0 = _variable_base()
    g0["environment"]["beam_count_max"] = 120
    g0["training"].update(beam_count_max=120, beam_count_mode="none",
                          scene_order="shuffle_each_cycle")
    g0["network"].update(kind="transformer", transformer_layout="tsac_natural_context",
                         embedding_dim=128, num_heads=8, num_layers=2, feedforward_dim=512,
                         dropout=0, context_beam_count="actual",
                         readout_mode="attention_pooling", beam_count_conditioning="film")
    g0["state"].update(context_level="layout",
                       features=["beam", "power", "interference", "occupancy",
                                 "natural_global_context", "natural_beam_context",
                                 "natural_resource_context"])
    g0["reward"]["preset"] = "legacy_repaired"
    arms["G0"] = g0
    return arms


def effective_fragment(config):
    """回读“实际生效”的关键键，用于证明配置没有被静默丢弃。"""
    return {
        "state.features": list(config["state"]["features"]),
        "state.context_level": config["state"]["context_level"],
        "reward.preset": config["reward"]["preset"],
        "reward.shaping": dict(config["reward"]["shaping"]),
        "training.gamma": config["training"]["gamma"],
        "training.gamma_schedule": dict(config["training"]["gamma_schedule"]),
        "training.progress_strata": config["training"]["progress_strata"],
        "network.kind": config["network"]["kind"],
        "network.transformer_layout": config["network"]["transformer_layout"],
        "network.context_beam_count": config["network"]["context_beam_count"],
        "network.readout_mode": config["network"]["readout_mode"],
        "network.beam_count_conditioning": config["network"]["beam_count_conditioning"],
        "environment.action_mask": config["environment"]["action_mask"],
        "state_dim": config_summary(config)["state_dim"],
        "network_token_count": config_summary(config)["network_token_count"],
        "state_version": config_summary(config)["state_version"],
        "effective_gamma": config_summary(config)["effective_gamma"],
        "reward_shaping_summary": config_summary(config)["reward_shaping"],
        "config_hash": config_hash(config),
    }


def smoke(config, count, device="cpu", update=False):
    """前向、掩码与前向一致性冒烟（不训练）。"""
    import torch

    summary = config_summary(config)
    env = WorkbenchEnv(config)
    state = env.reset(scene=scene_of(count))
    mask = env.action_mask()
    agent = build_agent(config, summary["state_dim"], tuple(summary["action_dims"]), device)
    with torch.no_grad():
        action = agent.act(state, deterministic=True, action_mask=mask)
    next_state, reward, done, info = env.step(action)
    row = {"beam_count": count, "state_width": int(state.shape[0]),
           "expected_width": env.state_width(count),
           "mask_pairs": int(mask["frequency_slots"].sum()),
           "action": list(action), "reward": float(reward),
           "actual_slots": int(info["actual_action"]["slots"]),
           "reward_terms": sorted((info.get("reward_terms") or {}).keys())}
    if update:
        # 单次参数更新冒烟：只用这批转移，不写任何实验目录。进度分层采样要求每个
        # (数量桶 × 进度段) 都有成员，因此这里显式覆盖 replay 配置里的桶/段。
        replay = getattr(agent, "replay", None)
        seed_state = np.asarray(state, dtype=np.float32).reshape(-1)
        seed_mask = env.action_mask()
        if replay is not None and getattr(replay, "bucket_indices", None):
            buckets = list(replay.bucket_indices)
            per_cell = 4
            strata = replay.progress_strata if replay.segment_indices is not None else 1
            for bucket in buckets:
                for segment in range(strata):
                    for _ in range(per_cell):
                        replay.add(seed_state, (0, 0, 0), 0.0, seed_state, False,
                                   seed_mask, seed_mask, bucket=bucket,
                                   progress_segment=(segment if replay.segment_indices is not None
                                                     else None))
            row["replay_seeded"] = {"buckets": buckets, "progress_strata": strata,
                                    "per_cell": per_cell,
                                    "stratum_sizes": ({key: [len(cell) for cell in value]
                                                       for key, value in replay.segment_indices.items()}
                                                      if replay.segment_indices is not None else None),
                                    "width": int(seed_state.shape[0])}
        elif replay is not None:
            # legacy 回放（bucket 模式 none）：按批大小补足转移后做一次更新。
            for _ in range(int(agent.train["batch_size"])):
                replay.add(seed_state, (0, 0, 0), 0.0, seed_state, False,
                           seed_mask, seed_mask)
            row["replay_seeded"] = {"mode": "none", "added": int(agent.train["batch_size"]),
                                    "width": int(seed_state.shape[0])}
        agent.train["warmup_steps"] = 0
        diagnostics = agent.update()
        row["update"] = {"update_step": agent.update_step,
                         "actor_loss": diagnostics.get("actor_loss"),
                         "critic_1_loss": diagnostics.get("critic_1_loss")}
        row["replay_sampled_by_bucket"] = dict(agent.replay.sampled_by_bucket)
        if agent.replay.segment_indices is not None:
            row["replay_sampled_by_stratum"] = dict(agent.replay.sampled_by_stratum)
            row["replay_replacement_draws_by_stratum"] = dict(
                agent.replay.replacement_draws_by_stratum)
    return row


def legacy_bitwise(count=100, turns=8):
    """旧 100 束逐位对照：默认（旧）配置下的状态与奖励序列指纹。"""
    import hashlib

    config = _variable_base()
    config["network"].update(kind="transformer", transformer_layout="tsac_14_variable_context",
                             embedding_dim=128, num_heads=8, num_layers=2, feedforward_dim=512,
                             dropout=0)
    config["state"]["features"] = ["beam", "power", "interference", "occupancy", "beta",
                                   "progress", "remaining_demand", "beam_count_normalized",
                                   "remaining_beams_normalized", "power_pressure"]
    config["reward"]["preset"] = "legacy_repaired"
    config["evaluation"]["action_mode"] = "stochastic"
    config["training"]["checkpoint_interval"] = 10000
    # 旧 100 束锚点用冻结的 V0/V1 配置（40 万步、每 20000 步验证），只做前几位对照。
    config["training"]["total_steps"] = 400000
    config["evaluation"].update(validation_count=144, validation_seed=53001,
                                interval_steps=20000, extra_steps=[0], test_count=0,
                                action_mode="stochastic")
    env = WorkbenchEnv(validate_config(config))
    state = env.reset(scene=scene_of(count))
    digest = hashlib.sha256()
    rewards = []
    digest.update(np.asarray(state, dtype=np.float64).tobytes())
    for _ in range(turns):
        mask = env.action_mask()
        freq, slot = int(np.nonzero(mask["frequency_slots"])[0][0]), int(
            np.nonzero(mask["frequency_slots"])[1][0])
        power = int(np.nonzero(mask["power"])[0][-1])
        state, reward, done, _ = env.step((freq, slot, power))
        rewards.append(float(reward))
        digest.update(np.asarray(state, dtype=np.float64).tobytes())
        if done:
            break
    return {"beam_count": count, "turns": len(rewards), "rewards": rewards,
            "state_reward_sha256": digest.hexdigest(),
            "state_dim": int(state.shape[0]),
            "reward_preset": config["reward"]["preset"],
            "config_hash": config_hash(validate_config(config))}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda", action="store_true", help="在 CUDA 上做前向与单次更新冒烟")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    report = {"arms": {}, "legacy": legacy_bitwise()}
    for name, raw in arm_configs().items():
        try:
            config = validate_config(raw)
        except ValueError as error:
            report["arms"][name] = {"error": str(error)}
            print(f"[{name}] 配置被拒绝: {error}")
            continue
        summary = config_summary(config)
        device = "cuda" if args.cuda else "cpu"
        count = 100 if name != "G0" else 120
        report["arms"][name] = {
            "effective": effective_fragment(config),
            "summary_state_dim": summary["state_dim"],
            "warnings": summary["warnings"],
            "smoke": smoke(config, count, device=device, update=True),
        }
    text = json.dumps(report, ensure_ascii=False, indent=2, default=str)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        print(f"已写出核验报告: {args.out}")
    for name, row in report["arms"].items():
        if "effective" not in row:
            print(f"[{name}] 未通过校验: {row.get('error')}")
            continue
        eff = row["effective"]
        print(f"[{name}] state_dim={eff['state_dim']} tokens={eff['network_token_count']} "
              f"preset={eff['reward.preset']} progress_strata={eff['training.progress_strata']} "
              f"gamma_schedule={eff['training.gamma_schedule']['mode']} "
              f"readout={eff['network.readout_mode']} cond={eff['network.beam_count_conditioning']}")
        print("      smoke:", {key: value for key, value in row["smoke"].items()
                               if key not in ("reward_terms",)})
    print("[legacy]", report["legacy"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
