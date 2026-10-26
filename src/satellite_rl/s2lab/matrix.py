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


def build_matrix_round2():
    """第二轮：修复入池状态与每束令牌耦合后，第一轮 54 个全部重跑，并加待做清单第 1—6 项。
    SAC 默认 double_q='joint'（历史做法）；检查点按开发集 SGM 选（policy_best_sgm.pt）。"""
    runs = build_matrix()
    for r in runs:
        if r['algo'] == 'sac':
            r['agent']['double_q'] = 'joint'
        if r['group'] == 'nomask':
            r['illegal_penalty'] = 2.                       # 第 5 项：显式非法动作惩罚
    legacy_delta = dict(group_mode='greedy', reward_mode='delta')
    for s in SEEDS:
        # 第 1 项：原样 14 令牌
        runs.append(base_run(f't14-in-paper-s{s}', 'sac', 'tsac14', s, 'tsac14', label='t14_in_paper', group_mode='greedy',
                             size=dict(MAIN_SIZE, intf_mode='in')))
        runs.append(base_run(f't14-in-delta-s{s}', 'sac', 'tsac14', s, 'tsac14', label='t14_in_delta', group_mode='greedy', reward_mode='delta',
                             size=dict(MAIN_SIZE, intf_mode='in')))
        runs.append(base_run(f't14-dbw-paper-s{s}', 'sac', 'tsac14', s, 'tsac14', label='t14_dbw_paper', group_mode='greedy',
                             size=dict(MAIN_SIZE, intf_mode='dbw')))
        # 第 3 项：躲干扰（都在"历史版 12 令牌＋delta"上）
        runs.append(base_run(f'avoid-rule-s{s}', 'sac', 'tf_legacy', s, 'avoid', label='avoid_rule', start_mode='rule', **legacy_delta))
        runs.append(base_run(f'avoid-topk-s{s}', 'sac', 'tf_legacy', s, 'avoid', label='avoid_topk', start_mode='topk', start_topk=8, **legacy_delta))
        runs.append(base_run(f'avoid-intf-s{s}', 'sac', 'tf_legacy', s, 'avoid', label='avoid_intf', group_mode='greedy', reward_mode='delta_intf'))
        runs.append(base_run(f'avoid-long-s{s}', 'sac', 'tf_legacy', s, 'avoid', label='avoid_long400k', total_steps=400000, **legacy_delta))
        # 第 6 项：功率档 0.5—50 W 步长 0.5
        runs.append(base_run(f'power100-s{s}', 'sac', 'tf_legacy', s, 'power', label='power100',
                             power_levels=[round(.5 * i, 1) for i in range(1, 101)], **legacy_delta))
    # 第 4 项：CNN/MLP-SAC 排除超参因素（单种子）
    runs.append(base_run('fix-cnn-ln-s1', 'sac', 'cnn', 1, 'fix', label='cnn_layernorm', size=dict(MAIN_SIZE, layernorm=True)))
    runs.append(base_run('fix-mlp-ln-s1', 'sac', 'mlp', 1, 'fix', label='mlp_layernorm', size=dict(MAIN_SIZE, layernorm=True)))
    runs.append(base_run('fix-cnn-auto-s1', 'sac', 'cnn', 1, 'fix', label='cnn_autoalpha', agent=dict(SAC_CORE, auto_alpha=True)))
    runs.append(base_run('fix-mlp-auto-s1', 'sac', 'mlp', 1, 'fix', label='mlp_autoalpha', agent=dict(SAC_CORE, auto_alpha=True)))
    # 双 Q 取法对照（第一轮做法，单种子）
    runs.append(base_run('dq-perhead-tsac-s1', 'sac', 'tf_blocks', 1, 'doubleq', label='tsac_perhead', agent=dict(SAC_CORE, double_q='per_head')))
    runs.append(base_run('dq-perhead-legacy-s1', 'sac', 'tf_legacy', 1, 'doubleq', label='legacy_perhead', group_mode='greedy',
                         agent=dict(SAC_CORE, double_q='per_head')))
    # 卡有富余：参数量补种子 2、3（d512L4 单个约 17 h 除外），放队尾，空闲才跑
    for d, layers in ((32, 1), (64, 2), (128, 4), (128, 8), (256, 2), (256, 4)):
        heads = 4 if d <= 64 else 8
        for s in (2, 3):
            runs.append(base_run(f'size-d{d}L{layers}-s{s}', 'sac', 'tf_blocks', s, 'size', label=f'd{d}L{layers}',
                                 size=dict(MAIN_SIZE, d=d, layers=layers, heads=heads, ff=4 * d)))
    for r in runs:
        if r['algo'] == 'sac':
            r['agent'].setdefault('double_q', 'joint')
    return runs


# 第二轮按账号分队列：A（u2025140002）跑主表、奖励、追加、躲干扰、功率档、令牌；B（u2024018003）跑其余。慢的放前面。
ROUND2_ACCOUNT_B = ('size-', 'group-', 'nomask-', 't14-', 'fix-', 'dq-')
ROUND2_PRIORITY = ('size-d512', 'avoid-long', 'token-beams', 'size-d128L8', 'main-tsac-', 'main-tsaclegacy', 'extra-legacydelta', 'reward-delta',
                   'avoid-', 'power100', 't14-', 'size-d256L4', 'group-', 'token-', 'size-', 'main-cnnsac', 'main-dqn', 'extra-dqnmlp', 'dq-',
                   'main-mlpsac', 'fix-', 'nomask-', 'main-ppo', 'extra-ppomlp')


def ordered_round2(runs, account):
    def rank(run):
        if run['group'] == 'size' and run['seed'] > 1:        # 补种子放队尾
            return (len(ROUND2_PRIORITY) + 1, run['seed'], run['run_id'])
        for i, prefix in enumerate(ROUND2_PRIORITY):
            if run['run_id'].startswith(prefix):
                return (i, run['seed'], run['run_id'])
        return (len(ROUND2_PRIORITY), run['seed'], run['run_id'])
    mine = [r for r in runs if r['run_id'].startswith(ROUND2_ACCOUNT_B) == (account == 'B')]
    return [r['run_id'] for r in sorted(mine, key=rank)]


def write_matrix_round2(runs_dir, account):
    runs_dir = Path(runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)
    runs = build_matrix_round2()
    (runs_dir / 'matrix.json').write_text(json.dumps(runs, ensure_ascii=False, indent=1), encoding='utf-8')
    (runs_dir / 'queue.json').write_text(json.dumps(dict(order=ordered_round2(runs, account), account=account,
                                                         created_at=datetime.now().isoformat()), indent=1), encoding='utf-8')
    return runs


def r3_run(name, seed, group, label, parent, agent=None, size=None, algo='sac3', **kw):
    """第三轮运行：父组 R0＝14 令牌 T-SAC＋delta（第二轮 t14-in-delta 配置，算法用 SAC3 重新实现）。
    agent/size 为相对 R0 的改动；parent 记录父组，报告按父组做同种子配对。"""
    core = dict(sac3=dict(SAC_CORE, policy='indep', double_q='joint', power='discrete', q_power='plain', pi_power='count'),
                ppo=dict(PPO_CORE), dqn=dict(DQN_CORE))[algo]
    run = dict(run_id=f'{name}-s{seed}', group=group, label=label, parent=parent, algo=algo, encoder='tsac14',
               size=dict(MAIN_SIZE, intf_mode='in'), pointer=False, mask_mode='full', group_mode='greedy', reward_mode='delta',
               seed=seed, total_steps=TOTAL_STEPS, batch_envs=64, dev_every=10000, log_every=2000, ckpt_minutes=20, agent=core)
    run['agent'] = dict(core, **(agent or {}))
    run['size'] = dict(run['size'], **(size or {}))
    run.update(kw)
    return run


# 第三轮配置：名称 → (组, 标签, 父组, 智能体改动, 编码器改动, 其它)
ROUND3 = {
    'r3-r0': ('base', 'R0', None, {}, {}, {}),
    'r3-hist': ('train', '历史双Q', 'r3-r0', dict(double_q='hist'), {}, {}),
    'r3-legal': ('action', 'L 合法落位', 'r3-r0', dict(policy='legal'), {}, {}),
    'r3-cond': ('action', 'A 槽数条件化', 'r3-legal', dict(policy='cond'), {}, {}),
    'r3-condtop8': ('action', "B′ 低干扰前8", 'r3-cond', dict(policy='cond', topk=8), {}, {}),
    'r3-p1': ('power', 'P1 评论家功率项看槽数', 'r3-cond', dict(policy='cond', q_power='count'), {}, {}),
    'r3-p2': ('power', 'P2 连续功率', 'r3-p1', dict(policy='cond', q_power='count', power='continuous', pi_power='count'), {}, {}),
    'r3-p3': ('power', 'P3 连续功率看起点', 'r3-p2', dict(policy='cond', q_power='count_start', power='continuous', pi_power='count_start'), {}, {}),
    'r3-wide': ('train', '加宽 d256', 'r3-r0', {}, dict(d=256, ff=1024), {}),
    'r3-replay': ('train', '回放池10万', 'r3-r0', dict(replay=100000), {}, {}),
    'r3-lr': ('train', '演员学习率3e-5', 'r3-r0', dict(actor_lr=3e-5), {}, {}),
    'r3-auto': ('train', '自动温度', 'r3-r0', dict(auto_alpha=True, alpha_lr=3e-4, alpha_kappa=.3), {}, {}),
    'r3-f-pneed': ('state', '＋所需功率', 'r3-r0', {}, dict(feats=['pneed']), {}),
    'r3-f-share': ('state', '＋功率份额', 'r3-r0', {}, dict(feats=['share']), {}),
    'r3-f-outint': ('state', '＋出向干扰', 'r3-r0', {}, dict(feats=['outint']), {}),
    'r3-f-snr': ('state', '＋信噪比', 'r3-r0', {}, dict(feats=['snr']), {}),
    'r3-f-sens': ('state', '＋邻束敏感度', 'r3-r0', {}, dict(feats=['sens']), {}),
    'r3-f-future': ('state', '＋后续邻居', 'r3-r0', {}, dict(feats=['future']), {}),
    'r3-f-gspec': ('state', '＋全局频谱', 'r3-r0', {}, dict(feats=['gspec']), {}),
    'r3-f-all': ('state', '＋全部补充', 'r3-r0', {}, dict(feats=['pneed', 'share', 'outint', 'snr', 'sens', 'future', 'gspec']), {}),
    'r3-x-nointf': ('state', '−干扰输入', 'r3-r0', {}, dict(intf_zero=True), {}),
    'r3-x-noglobal': ('state', '−全局令牌', 'r3-r0', {}, dict(no_global=True), {}),
    'r3-ppo': ('algo', 'PPO（14令牌）', 'r3-r0', None, {}, dict(algo='ppo')),
    'r3-dqn': ('algo', 'DQN（14令牌）', 'r3-r0', None, {}, dict(algo='dqn')),
    'r3-mlpsac': ('algo', 'MLP-SAC（同字段）', 'r3-r0', {}, {}, dict(encoder='mlp14')),
}
ROUND3_ACCOUNT_B = ('r3-p3', 'r3-hist', 'r3-cond', 'r3-p1', 'r3-f-pneed', 'r3-f-outint', 'r3-f-sens', 'r3-f-gspec', 'r3-x-nointf',
                    'r3-replay', 'r3-auto', 'r3-dqn', 'r3-ppo')
ROUND3_FIRST = ('r3-wide', 'r3-p3', 'r3-p2', 'r3-f-all', 'r3-r0', 'r3-hist', 'r3-legal', 'r3-cond')


def build_matrix_round3():
    runs = []
    for name, (group, label, parent, agent, size, other) in ROUND3.items():
        for s in SEEDS:
            other = dict(other)
            algo = other.pop('algo', 'sac3')
            runs.append(r3_run(name, s, group, label, parent, agent, size, algo=algo, **other))
    return runs


def ordered_round3(runs, account):
    def rank(run):
        name = run['run_id'].rsplit('-s', 1)[0]
        first = ROUND3_FIRST.index(name) if name in ROUND3_FIRST else len(ROUND3_FIRST)
        return (first, run['seed'], list(ROUND3).index(name))
    mine = [r for r in runs if (r['run_id'].rsplit('-s', 1)[0] in ROUND3_ACCOUNT_B) == (account == 'B')]
    return [r['run_id'] for r in sorted(mine, key=rank)]


def write_matrix_round3(runs_dir, account, extra_runs=()):
    runs_dir = Path(runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)
    runs = build_matrix_round3() + list(extra_runs)
    (runs_dir / 'matrix.json').write_text(json.dumps(runs, ensure_ascii=False, indent=1), encoding='utf-8')
    order = ordered_round3([r for r in runs if r['run_id'].startswith('r3-') and r['run_id'].rsplit('-s', 1)[0] in ROUND3], account)
    order += [r['run_id'] for r in extra_runs if r.get('account', 'A') == account]
    (runs_dir / 'queue.json').write_text(json.dumps(dict(order=order, account=account, created_at=datetime.now().isoformat()), indent=1),
                                         encoding='utf-8')
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
