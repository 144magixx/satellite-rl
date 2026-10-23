"""W32 P01-M1 SAC with G1 groups and the shared v2 physical model.

分组方式 grouping：greedy＝容量感知贪心定组（S2-FIX 的 L2，默认）；order8＝处理顺序（需求从大到小）%8，
即原版 T-SAC 与分散摆放规则用的组号（报告21的 X2）；choose＝每束由策略自己选组（X4，见 tsac_choose）。
"""
from collections import deque
import json
from pathlib import Path
import random
from time import perf_counter

import numpy as np
import torch

from satellite_rl.power30.agent import Power30Agent, build_config
from satellite_rl.power30.environment import Power30Env
from .common import seed_all, write_json
from .greedy import greedy_allocation

GROUPINGS = ('greedy', 'order8', 'choose')


def make_env(stage_input, grouping='greedy'):
    if grouping == 'choose':
        from .tsac_choose import GroupChoiceEnv
        return GroupChoiceEnv(stage_input)
    if grouping == 'greedy':
        groups, _, _, _ = greedy_allocation(stage_input)
        stage_input.group_by_beam_id = dict(zip(stage_input.layout['beam_ids'], map(int, groups)))
    else:
        stage_input.group_by_beam_id = {bid: k % 8 for k, bid in enumerate(stage_input.processing_index_to_beam_id)}
    return Power30Env(stage_input, 'P01', 'M1')


def grouping_of(config):
    return config.get('s2fix_grouping', 'greedy')


def make_agent(config, device=None):
    if grouping_of(config) == 'choose':
        from .tsac_choose import GroupChoiceAgent
        return GroupChoiceAgent(config, device=device)
    return Power30Agent(config, device=device)


def config_for(seed=51001, encoder='transformer', steps=150000, device='cpu', network=None, grouping='greedy'):
    """``network`` overrides embedding_dim/num_heads/num_layers/feedforward_dim; None keeps L2."""
    config = build_config('P01', 'M1', device=device, seed=seed)
    config.update(s2fix=True, encoder=encoder, version='s2fix_tsac_v1')
    if network:
        config['network'].update(network)
        config['network_overrides'] = dict(network)
    if grouping != 'greedy':
        config['s2fix_grouping'] = grouping
    config['training']['total_steps'] = steps
    return config


def save_policy(agent, config, path, step):
    torch.save(dict(config=config, step=step, update_step=agent.update_step,
                    **{name: getattr(agent, name).state_dict()
                       for name in ('actor', 'critic1', 'critic2')}), path)


def load_policy(path, device='cpu'):
    saved = torch.load(path, map_location=device, weights_only=False)
    config = saved['config']
    config['training']['device'] = device
    agent = make_agent(config, device=device)
    for name in ('actor', 'critic1', 'critic2'):
        getattr(agent, name).load_state_dict(saved[name])
        getattr(agent, name).eval()
    return agent


def rollout(stage_input, agent, seed):
    seed_all(seed)
    env = make_env(stage_input, grouping_of(agent.config))
    state, done = env.reset(), False
    while not done:
        action = agent.act(state, deterministic=False, action_mask=env.action_mask())
        state, _, done, _ = env.step(action)
    return env.groups.copy(), env.starts.copy(), env.counts.copy(), env.powers.copy()


def train(records, output_dir, *, seed=51001, encoder='transformer', steps=150000,
          device='cpu', network=None, save_every=10000, grouping='greedy'):
    from .data import load_input
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_all(seed)
    config = config_for(seed, encoder, steps, device, network, grouping)
    agent = make_agent(config)
    write_json(output_dir / 'network.json', dict(network=config['network'], encoder=encoder, grouping=grouping,
                                                 parameters=agent.parameter_counts()))
    scene_rng, order = random.Random(seed), []
    env, state, episode_return, episode = None, None, 0., 0
    recent, curve = deque(maxlen=100), []
    act_seconds = update_seconds = 0.
    started = perf_counter()
    with (output_dir / 'episodes.jsonl').open('w', encoding='utf-8') as stream:
        for step in range(1, steps + 1):
            if env is None:
                if not order:
                    order = list(range(len(records)))
                    scene_rng.shuffle(order)
                record = records[order.pop()]
                env = make_env(load_input(record), grouping)
                state, episode_return = env.reset(), 0.
            mask = env.action_mask()
            clock = perf_counter()
            action = agent.act(state, env_step=step-1, action_mask=mask)
            act_seconds += perf_counter() - clock
            next_state, reward, done, _ = env.step(action)
            agent.observe(state, action, reward, next_state, done, action_mask=mask,
                          next_action_mask=None if done else env.action_mask())
            if agent.ready(step, done):
                clock = perf_counter()
                agent.update()
                update_seconds += perf_counter() - clock
            state, episode_return = next_state, episode_return + reward
            if done:
                episode += 1
                recent.append(episode_return)
                stream.write(json.dumps(dict(step=step, episode=episode,
                    case_id=record['case_id'], episode_return=episode_return,
                    metrics=env.metrics(), beams=env.beam_results())) + '\n')
                stream.flush()
                env = None
            if step % 10000 == 0 or step == steps:
                curve.append(dict(step=step, updates=agent.update_step, episodes=episode,
                    mean_return=float(np.mean(recent)) if recent else episode_return,
                    seconds=perf_counter()-started, act_seconds=act_seconds,
                    update_seconds=update_seconds))
                write_json(output_dir / 'curve.json', curve)
                if step % save_every == 0:
                    save_policy(agent, config, output_dir / f'policy_{step}.pt', step)
                print(json.dumps(curve[-1]), flush=True)
    save_policy(agent, config, output_dir / 'model.pt', steps)
    return curve[-1]
