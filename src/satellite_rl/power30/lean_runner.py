"""W31: unchanged W30 science, periodic checkpoints and compact episode output."""
import argparse
import csv
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

from .agent import Power30Agent, build_config
from .environment import Power30Env
from .runner import _default_loader, isolated_rng, seed_all

NODES = (50000, 100000, 150000)
EPISODE_FIELDS = ('episode', 'env_step', 'case_id', 'return', 'SI', 'SGM',
                  'power_w', 'N', 'seconds', 'complete')
EVAL_FIELDS = ('mother', 'case_id', 'repeat', 'action_seed', 'env_step', 'SI',
               'SGM', 'power_w', 'N', 'constraints_ok', 'seconds')


def atomic_torch(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    torch.save(value, tmp)
    os.replace(tmp, path)


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(tmp, path)


def save_beams(path, episodes):
    """Ragged episodes represented by offsets; no object arrays/pickle needed."""
    rows = [beam for episode in episodes for beam in episode['beams']]
    arrays = {key: np.asarray([r[key] for r in rows]) for key in
              ('beam_id', 'demand_bps', 'capacity_bps', 'SI', 'SGM', 'start', 'count', 'power_w')}
    arrays['served_bps'] = np.minimum(arrays['capacity_bps'], arrays['demand_bps'])
    arrays['offsets'] = np.cumsum([0] + [len(e['beams']) for e in episodes])
    for key in ('episode', 'env_step', 'case_id', 'complete'):
        arrays[key] = np.asarray([e[key] for e in episodes])
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(tmp, path)


def read_cases(data_dir):
    data_dir = Path(data_dir)
    return tuple(json.loads((data_dir / (role + '_runtime.json')).read_text(encoding='utf-8'))['rows']
                 for role in ('train', 'development'))


def evaluate(agent, records, menu, group, directory, node, *, loader=_default_loader, repeats=3):
    """Same paired stochastic seed map and complete terminal metrics as W30."""
    directory = Path(directory)
    episodes = []
    with (directory / f'eval_{node}.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=EVAL_FIELDS)
        writer.writeheader()
        with isolated_rng(274000):
            for mother, record in enumerate(records):
                for repeat in range(repeats):
                    action_seed = 274000 + mother + repeat * 1000000
                    seed_all(action_seed)
                    started = time.perf_counter()
                    env = Power30Env(loader(record), menu, group)
                    state = env.reset()
                    done = False
                    while not done:
                        action = agent.act(state, deterministic=False, action_mask=env.action_mask())
                        state, _, done, _ = env.step(action)
                    # Export validates actual allocation, including resource constraints.
                    env.export_allocation()
                    metrics = env.metrics()
                    writer.writerow(dict(mother=mother, case_id=record['case_id'], repeat=repeat,
                        action_seed=action_seed, env_step=node, SI=metrics['SI_mean'], SGM=metrics['SGM'],
                        power_w=metrics['total_power_w'], N=env.N, constraints_ok=True,
                        seconds=time.perf_counter() - started))
                    episodes.append(dict(episode=mother * repeats + repeat, env_step=node,
                        case_id=record['case_id'], complete=True, beams=env.beam_results()))
    save_beams(directory / f'eval_{node}_beams.npz', episodes)


def train(config, records, directory, *, loader=_default_loader, until_step=None):
    """until_step is a local smoke-test pause; production uses the frozen 150k cap."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    seed_all(config['training']['seed'])
    agent = Power30Agent(config)
    order_rng = random.Random(config['training']['scene_order_seed'])
    progress = dict(env_step=0, episode=0, position=0, order=[], episode_return=0.,
                    episode_seconds=0., wall_seconds=0.)
    current = None
    env = state = None
    pending = []
    chunk = 0
    csv_offset = 0
    resume = directory / 'resume.pt'
    if resume.exists():
        saved = torch.load(resume, map_location='cpu', weights_only=False)
        if saved['config'] != config:
            raise ValueError('Resume configuration differs')
        progress = saved['progress']
        current = saved['current']
        if current is not None:
            env = Power30Env(loader(records[current]), config['menu'], config['group'])
            env.load_state_dict(saved['environment'])
            state = saved['observation']
        order_rng.setstate(saved['scene_rng'])
        pending, chunk, csv_offset = saved['pending'], saved['chunk'], saved['csv_offset']
        # Restore last so construction/restore work cannot shift action RNG.
        agent.load_state_dict(saved['agent'])
        logging.info('Resumed env_step=%s update_step=%s', progress['env_step'], agent.update_step)
    episode_path = directory / 'episodes.csv'
    if episode_path.exists():
        with episode_path.open('r+b') as stream:
            stream.truncate(csv_offset)
    # Discard only outputs beyond the saved training cursor after an interrupted run.
    for path in directory.glob('beams_*.npz'):
        if int(path.stem.split('_')[1]) >= chunk:
            path.unlink()
    session_start = last_progress = time.perf_counter()
    start_step = progress['env_step']
    base_wall = progress['wall_seconds']
    target = min(config['training']['total_steps'], until_step or config['training']['total_steps'])
    with episode_path.open('a', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=EPISODE_FIELDS)
        if stream.tell() == 0:
            writer.writeheader()

        def notify(stage):
            elapsed = time.perf_counter() - session_start
            value = dict(stage=stage, env_step=progress['env_step'], update_step=agent.update_step,
                         episode=progress['episode'], steps_per_second=(progress['env_step']-start_step)/max(elapsed, 1e-9),
                         wall_seconds=base_wall + elapsed, latest_time=datetime.now(timezone.utc).isoformat())
            atomic_json(directory / 'progress.json', value)
            return value

        def save():
            stream.flush()
            progress['wall_seconds'] = base_wall + time.perf_counter() - session_start
            atomic_torch(resume, dict(config=config, agent=agent.state_dict(), progress=progress,
                scene_rng=order_rng.getstate(), current=current,
                environment=env.state_dict() if env is not None else None, observation=state,
                pending=pending, chunk=chunk, csv_offset=stream.tell()))

        def terminal(complete):
            nonlocal chunk, pending
            metrics = env.metrics()
            writer.writerow(dict(episode=progress['episode'], env_step=progress['env_step'],
                case_id=records[current]['case_id'], **{'return': progress['episode_return']},
                SI=metrics['SI_mean'], SGM=metrics['SGM'], power_w=metrics['total_power_w'],
                N=env.N, seconds=progress['episode_seconds'], complete=complete))
            pending.append(dict(episode=progress['episode'], env_step=progress['env_step'],
                case_id=records[current]['case_id'], complete=complete, beams=env.beam_results()))
            stream.flush()
            if len(pending) == 500:
                save_beams(directory / f'beams_{chunk:04d}.npz', pending)
                chunk += 1
                pending = []

        notify('training')
        while progress['env_step'] < target:
            started = time.perf_counter()
            if env is None:
                if progress['position'] == len(progress['order']):
                    progress['order'] = list(range(len(records)))
                    order_rng.shuffle(progress['order'])
                    progress['position'] = 0
                current = progress['order'][progress['position']]
                env = Power30Env(loader(records[current]), config['menu'], config['group'])
                state = env.reset()
                progress['episode_return'] = progress['episode_seconds'] = 0.
            mask = env.action_mask()
            action = agent.act(state, action_mask=mask, env_step=progress['env_step'])
            next_state, reward, done, _ = env.step(action)
            agent.observe(state, action, reward, next_state, done, mask, None if done else env.action_mask())
            progress['env_step'] += 1
            progress['episode_return'] += reward
            if agent.ready(progress['env_step'], done):
                agent.update()
            state = next_state
            progress['episode_seconds'] += time.perf_counter() - started
            if done:
                terminal(True)
                progress['episode'] += 1
                progress['position'] += 1
                env = state = current = None
            if progress['env_step'] in NODES:
                atomic_torch(directory / f"policy_{progress['env_step']}.pt",
                    dict(**agent.policy_state(), config=config, env_step=progress['env_step']))
            if progress['env_step'] % 10000 == 0:
                save()
            if time.perf_counter() - last_progress >= 30:
                notify('training')
                last_progress = time.perf_counter()
        save()
        # Partial budget-terminal episode remains explicit; its unserved beams have SI=0.
        if target == config['training']['total_steps']:
            if env is not None:
                terminal(False)
            if pending:
                save_beams(directory / f'beams_{chunk:04d}.npz', pending)
        result = notify('trained' if target == config['training']['total_steps'] else 'smoke_paused')
    return agent, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--menu', choices=('P00', 'P10', 'P01', 'P11'), required=True)
    parser.add_argument('--group', choices=('M0', 'M1'), required=True)
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--threads', type=int, default=3)
    args = parser.parse_args()
    directory = Path(args.run_dir)
    directory.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=directory / 'worker.log', level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')
    try:
        if (directory / 'DONE').exists():
            return
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(1)
        config = build_config(args.menu, args.group, args.device, seed=51001)
        records, dev = read_cases(args.data_dir)
        agent, progress = train(config, records, directory)
        progress['stage'] = 'evaluation'
        atomic_json(directory / 'progress.json', progress)
        for node in NODES:
            policy = torch.load(directory / f'policy_{node}.pt', map_location=agent.device, weights_only=False)
            for name in ('actor', 'critic1', 'critic2'):
                getattr(agent, name).load_state_dict(policy[name])
            evaluate(agent, dev, args.menu, args.group, directory, node)
        progress.update(stage='completed', latest_time=datetime.now(timezone.utc).isoformat())
        atomic_json(directory / 'progress.json', progress)
        (directory / 'DONE').write_text(progress['latest_time'] + '\n', encoding='utf-8')
    except Exception:
        logging.exception('Worker failed')
        raise


if __name__ == '__main__':
    main()
