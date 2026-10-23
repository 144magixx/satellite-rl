"""I0 网络结构对照：训练（只存网络权重的轻量断点）与按 S2-FIX 口径评价。

训练循环与 s2fix/i0.py 相同（同一 SearchEnv、奖励、SAC 更新、热身与回合上限），区别只在
网络由配置选择，以及断点不再带 2 万条回放（原 L1 断点每个约 700 MB）。
"""
from collections import Counter
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from satellite_rl.sac_improvement.agent import ImprovementSAC, pack_actions, _distribution
from satellite_rl.sac_improvement.networks import build_network, parameter_count
from .search_env import SearchEnv, ACTION_WIDTH, configuration


def make_config(seed, steps, device, **network):
    return configuration(seed=seed, total_steps=steps, max_episodes=max(1, steps // 32),
                         device=device, **network)


def save_weights(agent, config, path, **extra):
    torch.save(dict(config=config, actor=agent.actor.state_dict(), critic1=agent.critic1.state_dict(),
                    critic2=agent.critic2.state_dict(), **extra), path)


def train(records, output_dir, seed, steps, device, **network):
    from .data import load_input
    cfg = make_config(seed, steps, device, **network)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = random.Random(seed)
    agent, env = ImprovementSAC(cfg), SearchEnv(cfg)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "network.json").write_text(json.dumps(dict(
        config=cfg, actor_parameters=parameter_count(agent.actor)), indent=1), encoding="utf8")
    actions, episodes = Counter(), 0
    returns, gains, episode_return = [], [], 0.
    state = env.reset(load_input(rng.choice(records)))
    features, mask, identifiers = pack_actions(env)
    started = time.perf_counter()
    for step in range(1, steps + 1):
        if step <= cfg["warmup_steps"]:
            action = rng.choice(np.flatnonzero(mask).tolist())
        else:
            action = agent.choose(state, features, mask)
        next_state, reward, done, info = env.step(identifiers[action])
        next_features, next_mask, next_ids = pack_actions(env)
        agent.replay.add(state, features, mask, identifiers, action, reward,
                         next_state, next_features, next_mask, done)
        actions[info["kind"]] += 1
        episode_return += reward
        if step > cfg["warmup_steps"] and step % cfg["update_every"] == 0:
            for _ in range(cfg["gradient_steps"]):
                agent.update()
        state, features, mask, identifiers = next_state, next_features, next_mask, next_ids
        if done:
            episodes += 1
            returns.append(episode_return)
            gains.append(env.best_eval.W - env.base_eval.W)
            episode_return = 0.
            if episodes < cfg["max_episodes"] and step < steps:
                state = env.reset(load_input(rng.choice(records)))
                features, mask, identifiers = pack_actions(env)
        last = step == steps or episodes >= cfg["max_episodes"]
        if step % cfg["checkpoint_interval"] == 0 or last:
            record = dict(step=step, episodes=episodes, seconds=time.perf_counter() - started,
                          episode_return=float(np.mean(returns)) if returns else episode_return,
                          best_W_gain=float(np.mean(gains)) if gains else env.best_eval.W - env.base_eval.W,
                          action_proportions={k: v / sum(actions.values()) for k, v in actions.items()})
            with (output_dir / "curve.jsonl").open("a", encoding="utf8") as stream:
                stream.write(json.dumps(record) + "\n")
            save_weights(agent, cfg, output_dir / f"weights_{step}.pt", step=step, episodes=episodes)
            returns.clear()
            gains.clear()
            actions.clear()
        if last:
            save_weights(agent, cfg, output_dir / "final.pt", step=step, episodes=episodes)
            break
    return output_dir / "final.pt"


def load_actor(path, device="cpu"):
    saved = torch.load(path, map_location=device, weights_only=False)
    config = dict(saved["config"], device=device)
    actor = build_network(config, ACTION_WIDTH).to(device)
    actor.load_state_dict(saved["actor"])
    return actor.eval(), config


@torch.no_grad()
def run_policy(stage_input, actor, config, action_seed, start=None):
    """同 s2fix/i0.policy：按策略概率抽样（非贪婪），32 步或主动停止，输出过程中最好的方案。"""
    torch.manual_seed(action_seed)
    env = SearchEnv(dict(config, start=start or config["start"]))
    state = env.reset(stage_input)
    device = next(actor.parameters()).device
    while not env.done:
        features, mask, identifiers = pack_actions(env)
        logits = actor(torch.as_tensor(state[None], dtype=torch.float32, device=device),
                       torch.as_tensor(features[None], device=device))
        probability, _, _ = _distribution(logits, torch.as_tensor(mask[None], device=device))
        action = int(torch.multinomial(probability[0], 1).item())
        state, _, _, _ = env.step(identifiers[action])
    return env.allocation(), env.probe_count
