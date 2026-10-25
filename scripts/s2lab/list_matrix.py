"""把实验矩阵（全部训练运行）渲染成清单文档 docs/Claude维护/S2-LAB/03_训练清单.md。"""
from collections import Counter
from pathlib import Path

from _common import ROOT
from satellite_rl.s2lab.matrix import build_matrix, ordered_run_ids, SAC_CORE, PPO_CORE, DQN_CORE, MAIN_SIZE

GROUP_NAMES = dict(main='主表（全掩码，3 种子）', nomask='无掩码副表（1 种子）', size='消融A：参数量（1 种子）',
                   token='消融B：令牌划分与指针头（3 种子）', group='消融C：组的自由度（3 种子）', reward='奖励对照：delta（3 种子）',
                   extra='追加（待讨论事项 3、4）：历史版＋delta 奖励、PPO/DQN 的 MLP 版（3 种子）')
ENC = dict(tf_blocks='Transformer，80 块令牌（8 组×10 块）', tf_legacy='Transformer，本组 10 块＋全局＋标量共 12 令牌（历史 14 令牌版的近似）',
           tf_beams='Transformer，80 块令牌＋每束一令牌', cnn='一维卷积', mlp='全连接')
ALGO = dict(sac='SAC', ppo='PPO', dqn='DQN')
GM = dict(free='自己选组', mod8='顺序%8 定组', greedy='贪心定组')


def main():
    runs = build_matrix()
    order = {rid: i + 1 for i, rid in enumerate(ordered_run_ids(runs))}
    lines = ['# S2-LAB 训练清单（自动生成自 `s2lab/matrix.py`）', '',
             f'共 **{len(runs)}** 个训练运行，分 {len(GROUP_NAMES)} 组；另有不训练的规则（贪心、固定、随机）与文献方法（退火、遗传 32/320 次）在评价时运行。',
             '每个运行 20 万环境步（每步一个波束决策，64 景并行），每 1 万步评一次开发集（60 景），按 SGM+W 选 `policy_best.pt`。', '',
             '公共设定：网络主体 ' + ', '.join(f'{k}={v}' for k, v in MAIN_SIZE.items()) + '；',
             'SAC 核心 ' + ', '.join(f'{k}={v}' for k, v in SAC_CORE.items()) + '；',
             'PPO ' + ', '.join(f'{k}={v}' for k, v in PPO_CORE.items()) + '；DQN ' + ', '.join(f'{k}={v}' for k, v in DQN_CORE.items()) + '。', '']
    counts = Counter(r['group'] for r in runs)
    lines += ['| 组 | 运行数 | 说明 |', '| --- | ---: | --- |']
    for g, name in GROUP_NAMES.items():
        lines.append(f'| {g} | {counts[g]} | {name} |')
    lines += ['', '## 逐个运行', '', '| 队列序 | run_id | 组 | 算法 | 编码器 | 规模 | 指针 | 掩码 | 选组 | 奖励 | 种子 |',
              '| ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: |']
    for r in sorted(runs, key=lambda r: order[r['run_id']]):
        size = r['size']
        lines.append(f"| {order[r['run_id']]} | {r['run_id']} | {r['group']} | {ALGO[r['algo']]} | {ENC[r['encoder']]} | d{size['d']} L{size['layers']} H{size['heads']} ff{size.get('ff', 4 * size['d'])} | "
                     f"{'是' if r.get('pointer') else ''} | {'全掩码' if r['mask_mode'] == 'full' else '无掩码'} | {GM[r['group_mode']]} | {r['reward_mode']} | {r['seed']} |")
    lines += ['', '## 评价时另跑（不训练）', '', '| 名称 | 说明 |', '| --- | --- |',
              '| rule-greedy | 容量感知贪心：按需求估槽数、功率按需求比例且不超所需、允许组内干扰最小的空闲段 |',
              '| rule-fixed | 固定 4 槽、固定功率档（预算/150）、组轮转、组内首个空闲段 |',
              '| rule-random | 合法动作中均匀随机 |',
              '| meta-sa32 / meta-sa320 | 模拟退火（Cocco 2018），贪心起点，32 / 320 次真实评价 |',
              '| meta-ga32 / meta-ga320 | 遗传算法（Paris 2019），贪心起点＋变异体种群，32 / 320 次真实评价 |', '']
    doc = ROOT / 'docs' / 'Claude维护' / 'S2-LAB' / '03_训练清单.md'
    doc.write_text('\n'.join(lines), encoding='utf-8')
    print(doc, len(runs))


if __name__ == '__main__':
    main()
