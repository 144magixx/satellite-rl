"""L1 candidate SAC, using the original I0 network and SAC update."""
from collections import Counter
from functools import lru_cache
import json
from pathlib import Path
import random
import numpy as np
import torch
from satellite_rl.sac_improvement.agent import ImprovementSAC, pack_actions
from .search_env import SearchEnv, configuration


def train(records, output_dir, seed=51001, steps=200000, device="cpu", config=None):
    from .data import load_input
    cfg = configuration(**dict(config or {}, seed=seed, total_steps=steps, device=device))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = random.Random(seed)
    agent, env = ImprovementSAC(cfg), SearchEnv(cfg)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    actions, episodes = Counter(), 0
    returns, gains, episode_return = [], [], 0.
    state = env.reset(load_input(rng.choice(records)))
    features, mask, identifiers = pack_actions(env)
    final = output_dir / "final.pt"
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
        if step % cfg["checkpoint_interval"] == 0 or step == steps or episodes >= cfg["max_episodes"]:
            record = dict(step=step, episodes=episodes,
                          episode_return=float(np.mean(returns)) if returns else episode_return,
                          best_W_gain=float(np.mean(gains)) if gains else env.best_eval.W-env.base_eval.W,
                          action_proportions={key:value/sum(actions.values()) for key,value in actions.items()})
            with (output_dir / "curve.jsonl").open("a", encoding="utf8") as stream:
                stream.write(json.dumps(record) + "\n")
            payload = dict(config=cfg, step=step, episodes=episodes, agent=agent.state_dict())
            torch.save(payload, output_dir / f"checkpoint_{step}.pt")
            torch.save(payload, final)
            returns.clear()
            gains.clear()
            actions.clear()
        if episodes >= cfg["max_episodes"]:
            break
    return final


@lru_cache(maxsize=None)
def load_policy(checkpoint, device="cpu"):
    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = dict(saved["config"], device=device)
    agent = ImprovementSAC(cfg)
    agent.load_state_dict(saved["agent"])
    agent.actor.eval()
    return agent


def policy(stage_input, checkpoint, seed=51001, return_info=False, device="cpu"):
    agent = load_policy(checkpoint, device)
    torch.manual_seed(seed)
    env = SearchEnv(agent.config)
    state = env.reset(stage_input)
    while not env.done:
        features, mask, identifiers = pack_actions(env)
        action = agent.choose(state, features, mask)
        state, _, _, _ = env.step(identifiers[action])
    result = env.allocation()
    return (result, dict(probes=env.probe_count, decisions=env.decision_count)) if return_info else result
