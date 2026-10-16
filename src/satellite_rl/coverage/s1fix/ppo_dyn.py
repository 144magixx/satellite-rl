"""小型CPU PPO，按完整回合计算GAE，最终检查点用于评价。"""
from pathlib import Path
from time import perf_counter
import csv

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from .dynamic import DynamicEpisode
from .io import write_rows


class ActorCritic(nn.Module):
    def __init__(self, observation_dim=16):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(observation_dim, 64), nn.ReLU(),
                                  nn.Linear(64, 64), nn.ReLU())
        self.actor = nn.Linear(64, 5)
        self.critic = nn.Linear(64, 1)

    def forward(self, observation):
        hidden = self.body(observation)
        return self.actor(hidden), self.critic(hidden).squeeze(-1)


def advantages(rewards, values, gamma=.99, gae_lambda=.95):
    """输入为一个终止回合；末状态价值0，不跨回合传播优势。"""
    result = np.zeros(len(rewards), dtype=np.float32)
    carry = 0.
    for i in range(len(rewards) - 1, -1, -1):
        following = values[i + 1] if i + 1 < len(values) else 0.
        delta = rewards[i] + gamma * following - values[i]
        carry = delta + gamma * gae_lambda * carry
        result[i] = carry
    return result, result + np.asarray(values, dtype=np.float32)


def update(model, optimizer, episodes, epochs=4, clip=.2, entropy_coefficient=.01):
    observations = torch.as_tensor(np.concatenate([e["observations"] for e in episodes]), dtype=torch.float32)
    actions = torch.as_tensor(np.concatenate([e["actions"] for e in episodes]), dtype=torch.long)
    old_logprob = torch.as_tensor(np.concatenate([e["logprobs"] for e in episodes]), dtype=torch.float32)
    advantage = torch.as_tensor(np.concatenate([e["advantages"] for e in episodes]), dtype=torch.float32)
    returns = torch.as_tensor(np.concatenate([e["returns"] for e in episodes]), dtype=torch.float32)
    advantage = (advantage - advantage.mean()) / (advantage.std(unbiased=False) + 1e-8)
    for _ in range(epochs):
        logits, values = model(observations)
        distribution = Categorical(logits=logits)
        ratio = (distribution.log_prob(actions) - old_logprob).exp()
        policy_loss = -torch.minimum(ratio * advantage, ratio.clamp(1 - clip, 1 + clip) * advantage).mean()
        value_loss = .5 * (values - returns).square().mean()
        loss = policy_loss + value_loss - entropy_coefficient * distribution.entropy().mean()
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), .5)
        optimizer.step()
    return float(loss.detach())


def collect_episode(model, episode):
    observations, actions, logprobs, values, rewards, rows = [], [], [], [], [], []
    while episode.index < len(episode.frames):
        observation = episode.observation()
        decision = perf_counter()
        with torch.no_grad():
            logits, value = model(torch.as_tensor(observation))
            distribution = Categorical(logits=logits)
            action = distribution.sample()
            logprob = distribution.log_prob(action)
        episode.decision_seconds += perf_counter() - decision
        reward, _, row = episode.step(int(action))
        observations.append(observation)
        actions.append(int(action))
        logprobs.append(float(logprob))
        values.append(float(value))
        rewards.append(reward)
        rows.append(row)
    gae, returns = advantages(rewards, values)
    return dict(observations=np.asarray(observations), actions=np.asarray(actions),
                logprobs=np.asarray(logprobs), advantages=gae, returns=returns), rewards, rows


def run_name(lam, seed, forecast=True):
    return f"{'B4' if forecast else 'B4_no_forecast'}_lam{lam:g}_seed{seed}"


def train(sequences, cache, output_dir, lam=.2, seed=51001, forecast=True,
          episodes=1200, episodes_per_update=16, epochs=4):
    torch.set_num_threads(1)
    if not 1 <= episodes <= 1200:
        raise ValueError("episodes必须在1到1200之间，较小值仅用于冒烟")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = ActorCritic(16 if forecast else 15)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    name = run_name(lam, seed, forecast)
    model_dir = Path(output_dir) / "B" / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    curve_path = Path(output_dir) / "B" / "curves" / f"{name}.csv"
    batch, curves = [], []

    def save(path, episode):
        torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                        episode=episode, lam=lam, seed=seed, forecast=forecast,
                        observation_dim=16 if forecast else 15, torch_rng=torch.get_rng_state(),
                        numpy_rng=rng.bit_generator.state), path)

    for episode_index in range(1, episodes + 1):
        sequence = sequences[int(rng.integers(len(sequences)))]
        trajectory, rewards, rows = collect_episode(model, DynamicEpisode(sequence, cache, lam, forecast))
        batch.append(trajectory)
        if len(batch) == episodes_per_update or episode_index == episodes:
            update(model, optimizer, batch, epochs)
            batch = []
        actions = np.bincount(trajectory["actions"], minlength=5) / len(rows)
        curve = dict(run=name, episode=episode_index, **{"return": float(np.sum(rewards))},
                     mean_gap=float(np.mean([r["peak"] - r["LB"] for r in rows])),
                     mean_churn=float(np.mean([r["churn"] for r in rows])))
        curve.update({f"action_{i}": float(fraction) for i, fraction in enumerate(actions)})
        curves.append(curve)
        if episode_index % 100 == 0:
            save(model_dir / f"{name}_{episode_index:04d}.pt", episode_index)
            write_rows(curve_path, curves, append=True)
            curves = []
            print(f"{name}: {episode_index}/{episodes}", flush=True)
    write_rows(curve_path, curves, append=True)
    final = model_dir / f"{name}_final.pt"
    save(final, episodes)
    return final


def merge_curves(output_dir):
    """训练任务各写自己的曲线；全部训练结束后由评价主进程合并。"""
    rows = []
    for path in sorted((Path(output_dir) / "B" / "curves").glob("*.csv")):
        with path.open(encoding="utf-8", newline="") as stream:
            rows.extend(csv.DictReader(stream))
    write_rows(Path(output_dir) / "B" / "train_curves.csv", rows)


def load_policy(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = ActorCritic(checkpoint["observation_dim"])
    model.load_state_dict(checkpoint["model"])
    model.eval()

    def argmax(observation):
        with torch.no_grad():
            logits, _ = model(torch.as_tensor(observation))
            return int(logits.argmax())
    return argmax, checkpoint
