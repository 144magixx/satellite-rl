"""第二轮各组是否收敛：看开发集 SGM（60 景，每 1 万步）末段走势与训练场景 SGM 末段斜率。

判据（每个运行）：
  末段提升 = 最后 5 次开发集评价均值 − 再往前 5 次均值（约 16—20 万步 对 11—15 万步）
  末段斜率 = 最后 8 次开发集评价对步数的线性斜率（每 10 万步的 SGM 变化）
  最好点位置 = 开发集 SGM 最高的那次评价在第几万步
  训练场景斜率 = curve.jsonl 最后 30% 的训练场景 SGM 线性斜率（每 10 万步）
组内取种子均值。末段提升 < 0.01 且斜率 < 0.05/10万步 记为"已平"，否则"还在涨"或"在掉"。
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path('D:/paper/outputs/experiments/s2lab_round2_20261025')


def runs():
    out = {}
    for acc in ('A', 'B'):
        for d in (ROOT / f'account_{acc}' / 'runs2').iterdir():
            if d.is_dir() and not d.name.startswith('_') and (d / 'dev.jsonl').exists():
                out[d.name] = d
    return out


def read(path):
    return [json.loads(l) for l in path.read_text(encoding='utf-8').splitlines() if l.strip()]


def stats(d):
    dev = read(d / 'dev.jsonl')
    steps = np.array([r['step'] for r in dev], float)
    sgm = np.array([r['SGM'] for r in dev])
    last, prev = sgm[-5:].mean(), sgm[-10:-5].mean()
    k = min(8, len(sgm))
    slope = np.polyfit(steps[-k:] / 1e5, sgm[-k:], 1)[0]
    curve = [r for r in read(d / 'curve.jsonl') if 'SGM' in r]
    cs = np.array([r['step'] for r in curve], float)
    cv = np.array([r['SGM'] for r in curve])
    tail = cs >= cs.max() * .7
    tslope = np.polyfit(cs[tail] / 1e5, cv[tail], 1)[0]
    return dict(final=sgm[-1], last5=last, gain=last - prev, slope=slope, best_step=steps[sgm.argmax()] / 1e4, best=sgm.max(),
                train_slope=tslope, total=steps.max())


def main():
    rs = runs()
    groups = {}
    for name in rs:
        base = name.rsplit('-s', 1)[0]
        groups.setdefault(base, []).append(name)
    rows = []
    for g, members in sorted(groups.items()):
        st = [stats(rs[m]) for m in sorted(members)]
        mean = {k: float(np.mean([s[k] for s in st])) for k in st[0]}
        gain, slope = mean['gain'], mean['slope']
        verdict = '已平' if abs(gain) < .01 and abs(slope) < .05 else ('还在涨' if gain > 0 or slope > 0 else '在掉')
        rows.append((g, len(st), mean, verdict, [round(s['gain'], 3) for s in st]))
    print(f"{'组':28s} n  末5均值 末段提升 斜率/10万 训练斜率/10万 最好点(万步) 判定  各种子末段提升")
    for g, n, m, v, seeds in rows:
        print(f"{g:28s} {n} {m['last5']:.3f} {m['gain']:+.3f} {m['slope']:+.3f} {m['train_slope']:+.3f} {m['best_step']:5.1f}/{m['total']/1e4:.0f} {v} {seeds}")
    # 40 万步那组在 20 万步时与之后的对比
    long = [rs[f'avoid-long-s{s}'] for s in (1, 2, 3)]
    for d in long:
        dev = read(d / 'dev.jsonl')
        at = {r['step'] // 10000: r['SGM'] for r in dev}
        w = lambda a, b: np.mean([at[i] for i in range(a, b + 1) if i in at])
        print(d.name, f"16—20万 {w(16, 20):.3f}  26—30万 {w(26, 30):.3f}  36—40万 {w(36, 40):.3f}")


if __name__ == '__main__':
    main()
