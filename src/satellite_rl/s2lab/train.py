"""训练循环：B 个场景并行采样，完整断点（网络、优化器、回放、随机状态、步数），按 run 目录自动续跑，
--max-hours 到点先存后退，SIGTERM/SIGINT 也先存后退。运行目录：
  config.json  checkpoint.pt  policy_best.pt  policy_final.pt  curve.jsonl  dev.jsonl  status.json  train.log
"""
from collections import deque
from datetime import datetime
import json
import os
import random
import signal
import socket
import time
from pathlib import Path

import numpy as np
import torch

from .agents import make_agent
from .env import BatchEnv, EnvOptions, ItemSampler, SceneBank, DEFAULT_POWER_LEVELS
from .evaluate import evaluate_agent, summarize
from .scenes import read_config, read_splits

BEAM_DIM = 20


def now():
    return datetime.now().astimezone().isoformat(timespec='seconds')


def write_json(path, value):
    tmp = Path(str(path) + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=1, default=float), encoding='utf-8')
    os.replace(tmp, path)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def env_options(run):
    return EnvOptions(batch_size=run.get('batch_envs', 64), mask_mode=run.get('mask_mode', 'full'),
                      group_mode=run.get('group_mode', 'free'), reward_mode=run.get('reward_mode', 'paper'),
                      order=run.get('order', 'demand'), power_levels=tuple(run.get('power_levels', DEFAULT_POWER_LEVELS)),
                      beam_tokens=run.get('encoder') == 'tf_beams')


def agent_config(run):
    cfg = dict(run.get('agent', {}))
    cfg.update(encoder=run['encoder'], size=run.get('size'), pointer=run.get('pointer', False), batch_envs=run.get('batch_envs', 64))
    return cfg


class Status:
    def __init__(self, run_dir, run):
        self.path = Path(run_dir) / 'status.json'
        self.data = json.loads(self.path.read_text(encoding='utf-8')) if self.path.exists() else {}
        for key, value in dict(run_id=run['run_id'], state='new', step=0, total_steps=run['total_steps'], elapsed_hours=0.,
                               created_at=now(), sessions=[]).items():
            self.data.setdefault(key, value)
        self.data.update(host=socket.gethostname(), pid=os.getpid(), lane=run.get('lane', ''), error=None)

    def update(self, **kw):
        self.data.update(kw, updated_at=now())
        write_json(self.path, self.data)


def train(run_dir, dataset_dir, run, device, max_hours=None, log=print):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / 'config.json', run)
    device = torch.device(device)
    config = read_config(dataset_dir)
    splits = read_splits(dataset_dir)
    physics = config.physics_obj()
    opts = env_options(run)
    status = Status(run_dir, run)
    status.update(state='starting', device=str(device), dataset=str(dataset_dir), max_hours=max_hours)
    seed_all(run['seed'])
    log(f"[{now()}] run {run['run_id']} loading scenes …")
    bank_train = SceneBank(dataset_dir, [it['key'] for it in splits['train']], physics, device)
    bank_dev = SceneBank(dataset_dir, [it['key'] for it in splits['dev']], physics, device, nmax=bank_train.nmax)
    sampler = ItemSampler(bank_train, config, splits['train'], True, run['seed'])
    env = BatchEnv(bank_train, config, opts, device, sampler)
    agent = make_agent(run['algo'], agent_config(run), len(opts.power_levels), device, BEAM_DIM)
    log(f"parameters: {agent.parameter_count():,}")
    step, episodes, elapsed_before, best = 0, 0, 0., -1e9
    ckpt_path = run_dir / 'checkpoint.pt'
    if ckpt_path.exists():
        saved = torch.load(ckpt_path, map_location=device, weights_only=False)
        agent.load_state_dict(saved['agent'])
        if hasattr(agent, 'replay') and saved.get('replay') is not None:
            agent.replay.load_state_dict(saved['replay'])
        step, episodes, elapsed_before, best = saved['step'], saved['episodes'], saved['elapsed'], saved['best']
        torch.set_rng_state(saved['rng']['torch'].cpu())
        if device.type == 'cuda' and saved['rng']['cuda'] is not None:
            torch.cuda.set_rng_state(saved['rng']['cuda'].cpu())
        np.random.set_state(saved['rng']['numpy'])
        random.setstate(saved['rng']['python'])
        sampler.rng.bit_generator.state = saved['rng']['sampler']
        log(f"resumed at step {step} (elapsed {elapsed_before / 3600:.2f} h)")
    total = run['total_steps']
    stop = {'flag': False, 'reason': ''}

    def handler(signum, frame):
        stop['flag'], stop['reason'] = True, f'signal {signum}'
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, handler)
    started = time.time()
    deadline = started + max_hours * 3600 if max_hours else None
    last_ckpt = started
    ckpt_every = run.get('ckpt_minutes', 20) * 60
    log_every, dev_every = run.get('log_every', 2000), run.get('dev_every', 10000)
    next_log = (step // log_every + 1) * log_every
    next_dev = (step // dev_every + 1) * dev_every
    recent = deque(maxlen=200)
    reward_recent = deque(maxlen=200)
    losses = {}
    status.data['sessions'].append(dict(started_at=now(), from_step=step))
    status.update(state='running', step=step)

    def save_checkpoint(final=False):
        payload = dict(agent=agent.state_dict(), replay=agent.replay.state_dict() if hasattr(agent, 'replay') else None,
                       step=step, episodes=episodes, elapsed=elapsed_before + time.time() - started, best=best,
                       rng=dict(torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state() if device.type == 'cuda' else None,
                                numpy=np.random.get_state(), python=random.getstate(), sampler=sampler.rng.bit_generator.state))
        tmp = run_dir / 'checkpoint.pt.tmp'
        torch.save(payload, tmp)
        os.replace(tmp, ckpt_path)
        torch.save(agent.policy_state(), run_dir / 'policy_last.pt')

    def dev_eval():
        nonlocal best
        agent.main_module().eval()
        results = evaluate_agent(agent, bank_dev, config, splits['dev'], opts, device)
        agent.main_module().train()
        summary = summarize(results)['all']
        score = summary['SGM'] + summary['W']
        record = dict(step=step, episodes=episodes, score=score, **{k: summary[k] for k in (
            'W', 'SI', 'SGM', 'worst10', 'power_w', 'slots', 'copol_overlap', 'interference_loss', 'mean_in_db', 'adj_ratio_pooled', 'unserved')})
        with (run_dir / 'dev.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record) + '\n')
        torch.save(agent.policy_state(), run_dir / f'policy_{step}.pt')
        if score > best:
            best = score
            torch.save(agent.policy_state(), run_dir / 'policy_best.pt')
        status.update(last_dev=record, best_dev=best)
        log(f"[{now()}] dev step={step} W={summary['W']:.4f} SGM={summary['SGM']:.4f} P={summary['power_w']:.0f} adj={summary['adj_ratio_pooled']:.2f} score={score:.4f} best={best:.4f}")
        return record

    obs = env.observe()
    t_log = time.time()
    step_log = step
    while step < total and not stop['flag']:
        active = env.active.clone()
        action = agent.act(obs, explore=True)
        reward, done = env.step(action)
        next_obs = env.observe()
        agent.observe(obs, action, reward, done, next_obs, active)
        obs = next_obs
        step += int(active.sum())
        reward_recent.append(float(reward[active].mean()) if active.any() else 0.)
        info = agent.update(step)
        if info:
            losses = info
        if env.finished:
            recent.extend(env.finished)
            episodes += len(env.finished)
            env.finished = []
        if step >= next_log:
            next_log += log_every
            sps = (step - step_log) / max(1e-9, time.time() - t_log)
            t_log, step_log = time.time(), step
            agg = {k: float(np.mean([r[k] for r in recent])) for k in ('W', 'SI', 'SGM', 'power_w', 'slots', 'objective', 'interference_loss')} if recent else {}
            record = dict(step=step, episodes=episodes, sps=sps, reward=float(np.mean(reward_recent)) if reward_recent else 0.,
                          elapsed_hours=(elapsed_before + time.time() - started) / 3600, **agg, **{f'loss_{k}': v for k, v in losses.items()})
            with (run_dir / 'curve.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(record) + '\n')
            status.update(state='running', step=step, episodes=episodes, env_steps=agent.env_steps, updates=agent.updates,
                          elapsed_hours=record['elapsed_hours'], sps=sps, last_train=agg)
            log(f"[{now()}] step={step} ep={episodes} sps={sps:.0f} r={record['reward']:.3f} " +
                ' '.join(f'{k}={v:.3f}' for k, v in agg.items()) + ' ' + ' '.join(f'{k}={v:.3g}' for k, v in losses.items()))
        if step >= next_dev:
            next_dev += dev_every
            dev_eval()
        if time.time() - last_ckpt > ckpt_every:
            save_checkpoint()
            last_ckpt = time.time()
            status.update(checkpoint_at=now(), step=step)
        if deadline and time.time() > deadline:
            stop['flag'], stop['reason'] = True, 'max_hours'
    elapsed_total = elapsed_before + time.time() - started
    if step >= total and not stop['flag']:
        dev_eval()
        torch.save(agent.policy_state(), run_dir / 'policy_final.pt')
        save_checkpoint(final=True)
        status.update(state='done', step=step, episodes=episodes, elapsed_hours=elapsed_total / 3600, finished_at=now())
        log(f"[{now()}] done: {step} steps, {elapsed_total / 3600:.2f} h")
        try:
            from .plots import plot_training, setup_chinese_font
            setup_chinese_font()
            figure = plot_training(run_dir, title=f"{run['run_id']}（{run.get('label', '')}，seed {run['seed']}）")
            log(f'training figure: {figure}')
        except Exception as error:      # 画图失败不影响训练结果
            log(f'training figure failed: {error!r}')
        return 'done'
    save_checkpoint()
    status.update(state='paused', step=step, episodes=episodes, elapsed_hours=elapsed_total / 3600, paused_at=now(), pause_reason=stop['reason'])
    log(f"[{now()}] paused at step {step} ({stop['reason']}); checkpoint saved")
    return 'paused'
