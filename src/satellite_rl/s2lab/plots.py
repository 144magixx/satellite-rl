"""训练过程图：奖励、损失、SGM、SI（训练窗口均值做滑动平滑，开发集点叠加）。每个运行结束时自动生成，也可批量重画。"""
import json
from pathlib import Path

import numpy as np

_ZH = {'on': True}      # 远端无中文字体时切英文标签


def T(zh, en):
    return zh if _ZH['on'] else en


def _read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def smooth(values, window):
    values = np.asarray(values, float)
    if len(values) < 2 or window <= 1:
        return values
    out = np.full(len(values), np.nan)
    for i in range(len(values)):
        lo = max(0, i - window + 1)
        chunk = values[lo:i + 1]
        if np.isfinite(chunk).any():
            out[i] = np.nanmean(chunk)
    return out


def plot_training(run_dir, window=5, title=None):
    """生成 <run_dir>/training.png；返回路径，matplotlib 不可用时返回 None。"""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    run_dir = Path(run_dir)
    curve = _read_jsonl(run_dir / 'curve.jsonl')
    dev = _read_jsonl(run_dir / 'dev.jsonl')
    if not curve:
        return None
    steps = np.asarray([c['step'] for c in curve])
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5))
    ax = axes[0, 0]
    reward = [c.get('reward', np.nan) for c in curve]
    ax.plot(steps, reward, color='tab:gray', alpha=.35, lw=1, label=T('每2000步均值', 'per-2000-step mean'))
    ax.plot(steps, smooth(reward, window), color='tab:blue', lw=2, label=T(f'滑动平均({window})', f'moving avg ({window})'))
    ax.set_title(T('训练奖励（每步）', 'training reward per step')); ax.legend(fontsize=8)
    ax = axes[0, 1]
    loss_keys = [k for k in curve[-1] if k.startswith('loss_') and k not in ('loss_entropy', 'loss_q', 'loss_eps')]
    for k in loss_keys:
        vals = [c.get(k, np.nan) for c in curve]
        ax.plot(steps, smooth(vals, window), lw=1.8, label=k[5:])
    ax.set_title(T('损失（滑动平均）', 'losses (moving avg)')); ax.set_yscale('symlog'); ax.legend(fontsize=8)
    for ax, key, name in ((axes[1, 0], 'SGM', T('SGM（直出）', 'SGM (direct)')), (axes[1, 1], 'SI', T('波束平均满足率 SI（直出）', 'beam-mean SI (direct)'))):
        vals = [c.get(key, np.nan) for c in curve]
        if np.isfinite(vals).any():
            ax.plot(steps, vals, color='tab:gray', alpha=.35, lw=1, label=T('训练场景（最近200回合）', 'train scenes (last 200 ep.)'))
            ax.plot(steps, smooth(vals, window), color='tab:green', lw=2, label=T(f'滑动平均({window})', f'moving avg ({window})'))
        if key == 'SI' and not np.isfinite(vals).any():
            wv = [c.get('W', np.nan) for c in curve]
            if np.isfinite(wv).any():
                ax.plot(steps, smooth(wv, window), color='tab:olive', lw=2, label=T('训练 W（旧日志无SI）', 'train W (old log has no SI)'))
        if dev:
            ax.plot([d['step'] for d in dev], [d.get(key, np.nan) for d in dev], 'o-', color='tab:red', ms=4, lw=1.2, label=T('开发集', 'dev set'))
        ax.set_title(name); ax.legend(fontsize=8)
    for ax in axes.flat:
        ax.set_xlabel(T('环境步', 'env steps')); ax.grid(alpha=.3)
    fig.suptitle(title or run_dir.name)
    fig.tight_layout()
    path = run_dir / 'training.png'
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_group_dev(run_dirs, out_path, title=None, reference=None):
    """多个运行的开发集 SGM / W / SI 叠加比较；reference={名称: {SGM,W,SI}} 画成水平参考线（规则在同一开发集的直出值）。"""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    palette = plt.rcParams['axes.prop_cycle'].by_key()['color']
    styles = ['-', '--', ':', '-.']
    colors, seen = {}, set()
    for run_dir in run_dirs:
        dev = _read_jsonl(Path(run_dir) / 'dev.jsonl')
        if not dev:
            continue
        name = Path(run_dir).name
        base, _, seed = name.rpartition('-s')
        base = base or name
        colors.setdefault(base, palette[len(colors) % len(palette)])
        style = styles[(int(seed) - 1) % len(styles)] if seed.isdigit() else '-'
        steps = [d['step'] for d in dev]
        for ax, key in zip(axes, ('SGM', 'W', 'SI')):
            ax.plot(steps, [d.get(key, np.nan) for d in dev], style, marker='o', ms=2.5, lw=1.3, color=colors[base],
                    label=base if base not in seen else None)
        seen.add(base)
    for ax, key in zip(axes, ('SGM', 'W', 'SI')):
        for i, (name, vals) in enumerate((reference or {}).items()):
            if key in vals and np.isfinite(vals[key]):
                ax.axhline(vals[key], color='k', lw=1, ls=['-', '--', ':'][i % 3], alpha=.6, label=name if key == 'SGM' else None)
        ax.set_title(T(f'开发集 {key}', f'dev {key}')); ax.set_xlabel(T('环境步', 'env steps')); ax.grid(alpha=.3)
    axes[0].legend(fontsize=7)
    fig.suptitle(title or T('开发集曲线', 'dev curves'))
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def setup_chinese_font():
    try:
        import matplotlib.pyplot as plt
        from matplotlib import font_manager
    except ImportError:
        return
    names = {f.name for f in font_manager.fontManager.ttflist}
    for name in ('Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC', 'WenQuanYi Zen Hei', 'Source Han Sans SC'):
        if name in names:
            plt.rcParams['font.family'] = name
            break
    else:
        _ZH['on'] = False
    plt.rcParams['axes.unicode_minus'] = False
