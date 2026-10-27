"""第四轮：贪心行为克隆。在训练场景上让容量感知贪心走完整景、记下（动作前观测, 贪心动作），
再让 SAC3 的演员最小化 −log π(贪心动作)。只训演员；之后照常做 SAC。"""
import torch

from .agents import to_store
from .env import BatchEnv, ItemSampler
from .rules import GreedyRule


@torch.no_grad()
def collect_greedy_demos(bank, config, items, options, device, n, seed):
    env = BatchEnv(bank, config, options, device, ItemSampler(bank, config, items, True, seed))
    rule = GreedyRule(env)
    parts, actions, count = [], [], 0
    while count < n:
        obs = env.observe()
        active = torch.nonzero(env.active).flatten()
        action = rule.act(obs)
        stored = to_store(obs)
        parts.append({k: v[active] for k, v in stored.items()})
        actions.append(action[active])
        count += len(active)
        env.step(action)
    return dict(obs={k: torch.cat([p[k] for p in parts])[:n] for k in parts[0]}, action=torch.cat(actions)[:n])
