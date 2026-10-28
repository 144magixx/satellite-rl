"""从 numbers.json 生成报告用的 Markdown 表（避免手抄出错）。"""
import json

n = json.load(open('D:/paper/outputs/experiments/s3lab_20261029/report/numbers.json', encoding='utf-8'))
T = ['ptr', 'greedy', 'fixed', 'random']
TZ = {'random': '随机', 'fixed': '固定规则', 'greedy': '容量感知贪心', 'ptr': '块指针头 T-SAC'}
M = ['sca', 'aravanis', 'wmmse', 'slsqp', 'target_cap', 'choi_chan', 'proportional', 'maxmin', 'equal']
MZ = {'none': '不细调', 'sca': '**SCA（本研究）**', 'aravanis': 'Aravanis 两段式', 'choi_chan': 'Choi-Chan 注水',
      'proportional': '按需求比例', 'equal': '等功率', 'target_cap': '目标容量迭代', 'maxmin': '最大最小公平',
      'wmmse': 'WMMSE', 'slsqp': 'SLSQP'}
out = []
for s, sz in (('old', '旧确认集 287 景'), ('new', '新确认集 300 景')):
    out.append(f'\n**表：{sz}，各方法把 W 提高了多少（SGM 不降；括号内为 95% 区间）**\n')
    out.append('| 方法 | ' + ' | '.join(TZ[t] for t in T) + ' | 单景中位耗时 |')
    out.append('| --- | ' + ' | '.join('---:' for _ in T) + ' | ---: |')
    out.append('| 细调前 W | ' + ' | '.join(f"{n['main'][f'{s}|{t}|none']['a_W'][0]:.3f}" for t in T) + ' | — |')
    for m in M:
        cells = []
        for t in T:
            r = n['main'][f'{s}|{t}|{m}']['d_W']
            cells.append(f'+{r[0]:.3f}（{r[1]:+.3f}～{r[2]:+.3f}）' if m == 'sca' else f'{r[0]:+.3f}')
        tm = sorted(n['main'][f'{s}|{t}|{m}']['t_median'] for t in T)
        out.append(f'| {MZ[m]} | ' + ' | '.join(cells) + f' | {tm[0]:.2f}—{tm[-1]:.2f} 秒 |')
    out.append(f'\n**表：{sz}，SCA 细调前后（W / 最差 10% 飞机 / SGM / 总功率）**\n')
    out.append('| 第二阶段方案 | W | 最差 10% | SGM | 功率 W |')
    out.append('| --- | --- | --- | --- | --- |')
    for t in T:
        a, b = n['main'][f'{s}|{t}|none'], n['main'][f'{s}|{t}|sca']
        out.append(f"| {TZ[t]} | {a['a_W'][0]:.3f} → {b['a_W'][0]:.3f} | {a['a_worst10'][0]:.3f} → {b['a_worst10'][0]:.3f} | "
                   f"{a['a_SGM'][0]:.3f} → {b['a_SGM'][0]:.3f} | {a['a_power_w'][0]:.0f} → {b['a_power_w'][0]:.0f} |")
    out.append(f'\n**表：{sz}，同一细调方法下相邻两档的差（W / 最差 10% / SGM；括号内为 95% 区间）**\n')
    out.append('| 细调方法 | 块指针头 − 贪心 | 贪心 − 固定 | 固定 − 随机 |')
    out.append('| --- | --- | --- | --- |')
    for m in ['none', 'sca', 'aravanis', 'wmmse', 'slsqp']:
        cells = []
        for pair in ('ptr-greedy', 'greedy-fixed', 'fixed-random'):
            g = n['gaps'][f'{s}|{m}|{pair}']
            cells.append(' / '.join(f"{g[c][0]:+.3f}（{g[c][1]:+.3f}～{g[c][2]:+.3f}）" for c in ('a_W', 'a_worst10', 'a_SGM')))
        out.append(f"| {MZ[m]} | " + ' | '.join(cells) + ' |')
    out.append(f'\n**表：{sz}，SCA 后分负载档的 W 与两种上界**\n')
    out.append('| 方案 | 档 | 细调前 W | SCA 后 W | 无干扰 50 W 上界 | 缺槽上界 | 缺槽需求占比 | 链路/50 W 限制需求占比 |')
    out.append('| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |')
    for t in T:
        for st, sz2 in (('ample', '宽裕'), ('marginal', '勉强'), ('short', '不够')):
            a, b, bd = n['strata'][f'{s}|{t}|none|{st}'], n['strata'][f'{s}|{t}|sca|{st}'], n['bounds'][f'{s}|{t}|{st}']
            out.append(f"| {TZ[t]} | {sz2} | {a['a_W']:.3f} | {b['a_W']:.3f} | {bd['W_U']:.3f} | {bd['W_H']:.3f} | {bd['slot_short_demand']:.1%} | {bd['link_short_demand']:.1%} |")
open('D:/paper/outputs/experiments/s3lab_20261029/report/tables.md', 'w', encoding='utf-8').write('\n'.join(out))
print('\n'.join(out))
