"""实验矩阵：主表、无掩码副表、三组消融（参数量、令牌划分/指针、组自由度）与奖励对照；队列与状态汇总。"""
from datetime import datetime
import json
from pathlib import Path

TOTAL_STEPS = 200000
SEEDS = (1, 2, 3)
# 网络主体与 SAC 核心按历史最好版直出 T-SAC（S10→W26→W31/W32 P01-M1→S2-FIX L2，t0_width/agent.py）：
# 宽128/8头/2层、FFN 512/ReLU/后置LayerNorm；温度固定0.01、γ0.99、τ0.005、批256、热身2000、梯度裁剪10；
# actor lr 1e-5、critic lr 1e-4、回放 20000、每环境步更新 1 次。所有 SAC 运行共用，只留令牌/组/步数差异。
MAIN_SIZE = dict(d=128, layers=2, heads=8, ff=512, activation='relu', norm_first=False)
SAC_CORE = dict(actor_lr=1e-5, critic_lr=1e-4, replay=20000, utd=1.0, batch=256, warmup=2000, alpha=.01, gamma=.99, tau=.005)
PPO_CORE = dict(lr=3e-4, rollout=32, epochs=4, minibatch=512, clip=.2, ent_coef=.01)
DQN_CORE = dict(lr=1e-4, replay=20000, utd=1.0, batch=256, warmup=2000, eps_steps=100000, target_every=1000)


def base_run(run_id, algo, encoder, seed, group, **kw):
    core = dict(sac=SAC_CORE, ppo=PPO_CORE, dqn=DQN_CORE)[algo]
    run = dict(run_id=run_id, group=group, algo=algo, encoder=encoder, size=dict(MAIN_SIZE), pointer=False,
               mask_mode='full', group_mode='free', reward_mode='paper', seed=seed, total_steps=TOTAL_STEPS,
               batch_envs=64, dev_every=10000, log_every=2000, ckpt_minutes=20, agent=dict(core))
    run.update(kw)
    return run


def build_matrix():
    runs = []
    algos = dict(tsac=('sac', 'tf_blocks'), cnnsac=('sac', 'cnn'), mlpsac=('sac', 'mlp'), ppo=('ppo', 'tf_blocks'), dqn=('dqn', 'tf_blocks'))
    # 主表：全掩码，三种子
    for name, (algo, enc) in algos.items():
        for s in SEEDS:
            runs.append(base_run(f'main-{name}-s{s}', algo, enc, s, 'main', label=name))
    # 主表：历史最好版 T-SAC 原样（本组 10 块令牌、贪心定组、老超参）
    for s in SEEDS:
        runs.append(base_run(f'main-tsaclegacy-s{s}', 'sac', 'tf_legacy', s, 'main', label='tsac_legacy', group_mode='greedy'))
    # 无掩码副表：单种子
    for name, (algo, enc) in algos.items():
        runs.append(base_run(f'nomask-{name}-s1', algo, enc, 1, 'nomask', label=name, mask_mode='none'))
    # 消融 A：参数量（T-SAC，单种子；d128L2 即主表）
    for d, layers in ((32, 1), (64, 2), (128, 4), (128, 8), (256, 2), (256, 4), (512, 4)):
        heads = 4 if d <= 64 else 8
        runs.append(base_run(f'size-d{d}L{layers}-s1', 'sac', 'tf_blocks', 1, 'size', label=f'd{d}L{layers}',
                             size=dict(MAIN_SIZE, d=d, layers=layers, heads=heads, ff=4 * d)))
    # 消融 B：令牌划分（每束一令牌）与指针头，三种子
    for s in SEEDS:
        runs.append(base_run(f'token-beams-s{s}', 'sac', 'tf_beams', s, 'token', label='beam_tokens'))
        runs.append(base_run(f'token-pointer-s{s}', 'sac', 'tf_blocks', s, 'token', label='pointer', pointer=True))
    # 消融 C：组的自由度，三种子
    for s in SEEDS:
        runs.append(base_run(f'group-mod8-s{s}', 'sac', 'tf_blocks', s, 'group', label='mod8', group_mode='mod8'))
        runs.append(base_run(f'group-greedy-s{s}', 'sac', 'tf_blocks', s, 'group', label='greedy_group', group_mode='greedy'))
    # 奖励对照：delta，三种子
    for s in SEEDS:
        runs.append(base_run(f'reward-delta-s{s}', 'sac', 'tf_blocks', s, 'reward', label='delta', reward_mode='delta'))
    # 追加（用户 10-24 13:29 待讨论事项 3、4，10-24 15:40 排入队列）：历史版＋delta 奖励；PPO/DQN 的 MLP 版
    for s in SEEDS:
        runs.append(base_run(f'extra-legacydelta-s{s}', 'sac', 'tf_legacy', s, 'extra', label='legacy_delta', group_mode='greedy', reward_mode='delta'))
        runs.append(base_run(f'extra-ppomlp-s{s}', 'ppo', 'mlp', s, 'extra', label='ppo_mlp'))
        runs.append(base_run(f'extra-dqnmlp-s{s}', 'dqn', 'mlp', s, 'extra', label='dqn_mlp'))
    return runs


# 队列顺序：先主表 T-SAC 与奖励对照（决定口径），再主表其余，再消融，再无掩码，最后追加组
PRIORITY = ('main-tsac-', 'main-tsaclegacy', 'reward-delta', 'main-cnnsac', 'main-mlpsac', 'main-ppo', 'main-dqn', 'token', 'group', 'size', 'nomask', 'extra')


def ordered_run_ids(runs):
    def rank(run):
        for i, prefix in enumerate(PRIORITY):
            if run['run_id'].startswith(prefix):
                return (i, run['seed'], run['run_id'])
        return (len(PRIORITY), run['seed'], run['run_id'])
    return [r['run_id'] for r in sorted(runs, key=rank)]


def write_matrix(runs_dir):
    runs_dir = Path(runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)
    runs = build_matrix()
    (runs_dir / 'matrix.json').write_text(json.dumps(runs, ensure_ascii=False, indent=1), encoding='utf-8')
    (runs_dir / 'queue.json').write_text(json.dumps(dict(order=ordered_run_ids(runs), created_at=datetime.now().isoformat()), indent=1), encoding='utf-8')
    return runs


def read_matrix(runs_dir):
    return json.loads((Path(runs_dir) / 'matrix.json').read_text(encoding='utf-8'))


def read_queue(runs_dir):
    return json.loads((Path(runs_dir) / 'queue.json').read_text(encoding='utf-8'))['order']


def run_status(runs_dir, run_id):
    path = Path(runs_dir) / run_id / 'status.json'
    if not path.exists():
        return dict(run_id=run_id, state='new', step=0)
    return json.loads(path.read_text(encoding='utf-8'))


def heartbeat_stale(status, minutes=15):
    updated = status.get('updated_at')
    if not updated:
        return True
    then = datetime.fromisoformat(updated)
    return (datetime.now(then.tzinfo) - then).total_seconds() > minutes * 60


def claimable(status):
    """failed 不自动重试：看过 train.log 后把 status.json 的 state 改回 paused 即可重新排队。"""
    state = status.get('state', 'new')
    if state in ('new', 'paused'):
        return True
    if state in ('running', 'starting', 'claimed') and heartbeat_stale(status):
        return True
    return False


def collect(runs_dir):
    runs = read_matrix(runs_dir)
    rows = []
    for run in runs:
        st = run_status(runs_dir, run['run_id'])
        rows.append(dict(run_id=run['run_id'], group=run['group'], label=run.get('label', ''), seed=run['seed'],
                         state=st.get('state', 'new'), step=st.get('step', 0), total=run['total_steps'],
                         elapsed_hours=st.get('elapsed_hours', 0.), host=st.get('host', ''), lane=st.get('lane', ''),
                         updated_at=st.get('updated_at', ''), best_dev=st.get('best_dev'), last_dev=st.get('last_dev'),
                         stale=heartbeat_stale(st) if st.get('state') in ('running', 'starting', 'claimed') else False,
                         pause_reason=st.get('pause_reason', '')))
    return rows


def render_state_markdown(rows, dataset_note='', extra_sections=''):
    done = sum(r['state'] == 'done' for r in rows)
    running = [r for r in rows if r['state'] in ('running', 'starting', 'claimed') and not r['stale']]
    lines = [f"# S2-LAB 接手状态（自动生成 {datetime.now().isoformat(timespec='minutes')}）", '',
             f"总运行 {len(rows)}：完成 {done}，运行中 {len(running)}，暂停 {sum(r['state'] == 'paused' for r in rows)}，"
             f"未开始 {sum(r['state'] == 'new' for r in rows)}，失败 {sum(r['state'] == 'failed' for r in rows)}，"
             f"心跳过期 {sum(r['stale'] for r in rows)}。", '']
    if dataset_note:
        lines += [dataset_note, '']
    lines += ['| 运行 | 组 | 变体 | 种子 | 状态 | 步数 | 用时h | 卡 | 最近开发集 (W/SGM) | 最好分 | 更新时间 |', '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for r in rows:
        ld = r['last_dev'] or {}
        dev = f"{ld['W']:.3f}/{ld['SGM']:.3f}@{ld['step']}" if ld else ''
        state = r['state'] + ('(过期)' if r['stale'] else '')
        best = f"{r['best_dev']:.3f}" if isinstance(r['best_dev'], (int, float)) else ''
        lines.append(f"| {r['run_id']} | {r['group']} | {r['label']} | {r['seed']} | {state} | {r['step']}/{r['total']} | {r['elapsed_hours']:.2f} | {r['lane']} | {dev} | {best} | {r['updated_at'][:16]} |")
    if extra_sections:
        lines += ['', extra_sections]
    return '\n'.join(lines) + '\n'
