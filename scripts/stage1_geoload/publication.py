"""Read-only figures from report.py compact tables; never audits or runs workers.

Incomplete production reports are rejected. Synthetic QA requires an explicit
fixture marker and output directory called synthetic_qa. No inferential claims.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

PROFILES = ('base', 'one_hotspot', 'two_hotspots', 'edge_heavy')
LABELS = {'MLM': 'MLM候选似然排序', 'rule': '恒等顺序实现对照',
          'random': '随机候选排序', 'JOINT': '联合重分基础顺序'}
TABLES = ('per_case', 'per_mother', 'timing', 'candidate', 'learningcurve')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def number(row, key):
    value = row.get(key)
    if value in ('', None):
        return None
    value = float(value)
    if not math.isfinite(value):
        raise ValueError('nonfinite compact value: ' + key)
    return value


def load(source, output, synthetic=False):
    source, output = Path(source).resolve(), Path(output).resolve()
    summary = json.loads((source / 'summary.json').read_text(encoding='utf-8-sig'))
    if summary.get('group') != 'S1-G16':
        raise ValueError('wrong experiment group')
    if synthetic:
        if summary.get('synthetic_fixture') is not True or output.name != 'synthetic_qa':
            raise ValueError('synthetic fixture marker and synthetic_qa output required')
    elif summary.get('status') != 'formal_ready' or summary.get('formal_ready') is not True or summary.get('issues'):
        raise ValueError('production report is not formal_ready; no figures written')
    if source == output or source in output.parents:
        raise ValueError('output must be separate from report input')
    tables = {}
    for name in TABLES:
        with (source / (name + '.csv')).open(encoding='utf-8-sig', newline='') as stream:
            tables[name] = list(csv.DictReader(stream))
    validation = source / 'validation_cases.csv'
    if validation.exists():
        with validation.open(encoding='utf-8-sig', newline='') as stream:
            tables['validation_cases'] = list(csv.DictReader(stream))
    return summary, tables


def paired(rows, metric, baseline='JOINT'):
    """Descriptive same-case, same-N differences; no row-wise uncertainty."""
    selected = {(r['case_id'], r['method']): r for r in rows}
    if len(selected) != len(rows):
        raise ValueError('duplicate case/method in common test')
    result = []
    for row in rows:
        other = selected.get((row['case_id'], baseline))
        if row['method'] == baseline or other is None:
            continue
        a, b = number(row, metric), number(other, metric)
        if a is None or b is None or number(row, 'N') != number(other, 'N'):
            continue
        result.append(dict(method=row['method'], case_id=row['case_id'],
                           mother_id=row['mother_id'], demand_profile=row['demand_profile'],
                           metric=metric, difference=a-b, baseline=baseline))
    return result


def build(source, output, *, synthetic=False):
    summary, tables = load(source, output, synthetic)
    # Imports occur only after authorization/closure checks; no worker dependency.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    import numpy as np

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    font = 'Microsoft YaHei' if any(f.name == 'Microsoft YaHei' for f in font_manager.fontManager.ttflist) else 'DejaVu Sans'
    test = [r for r in tables['per_case'] if r.get('task', '').startswith('common_test_')]
    completed = [r for r in test if r.get('outcome') == 'completed']
    methods = sorted({r['method'] for r in test})
    palette = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#000000', '#8C6100', '#56B4E9', '#6D6D6D', '#654EA3']
    markers = ['o', 's', '^', 'D', 'v', 'P', 'X', '*', '+']
    outputs = []
    missing = []

    def save(fig, name, rows, description):
        fig.suptitle(('SYNTHETIC QA — ' if synthetic else 'S1-G16 — ') + description, fontsize=11)
        fig.text(.02, .01, '单seed51001；四profile共享母场景；描述性原始点；参考非全局最优', fontsize=8)
        for suffix in ('png', 'pdf'):
            target = output / (name + '.' + suffix)
            fig.savefig(target, dpi=300, facecolor='white')
            outputs.append(target)
        target = output / (name + '.csv')
        keys = list(dict.fromkeys(k for row in rows for k in row))
        with target.open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=keys)
            writer.writeheader(); writer.writerows(rows)
        outputs.append(target)
        plt.close(fig)

    def empty(ax, text):
        ax.text(.5, .5, text, transform=ax.transAxes, ha='center', va='center', wrap=True)
        ax.set_axis_off()

    with plt.rc_context({'font.family': font, 'pdf.fonttype': 42, 'axes.unicode_minus': False,
                         'font.size': 9, 'figure.constrained_layout.h_pad': .15}):
        # Every observation remains visible, including extreme values; no clipping.
        fig, axes = plt.subplots(3, 2, figsize=(12, 12), layout='constrained')
        pair_rows = []
        for i, metric in enumerate(('P', 'T', 'CV')):
            for j, method in enumerate(methods):
                for p, profile in enumerate(PROFILES):
                    values = [number(r, metric) for r in completed if r['method']==method and r['demand_profile']==profile]
                    values = [v for v in values if v is not None]
                    axes[i, 0].scatter([p+(j-(len(methods)-1)/2)*.07]*len(values), values,
                        color=palette[j % len(palette)], marker=markers[j % len(markers)], s=14,
                        alpha=.7, label=LABELS.get(method, method) if p==0 else None)
            pairs = paired(completed, metric); pair_rows.extend(pairs)
            for j, method in enumerate(methods):
                vals=[r['difference'] for r in pairs if r['method']==method]
                axes[i, 1].scatter([j]*len(vals), vals, color=palette[j % len(palette)], marker=markers[j % len(markers)], s=15, alpha=.7)
            axes[i, 0].set(xticks=range(4), xticklabels=PROFILES, ylabel=metric+' (无量纲)')
            axes[i, 1].set(xticks=range(len(methods)), xticklabels=[LABELS.get(m,m) for m in methods], ylabel=metric+' − JOINT (同景同N)')
            axes[i, 1].axhline(0, color='gray', linewidth=.8)
            axes[i, 1].tick_params(axis='x', rotation=55)
        if methods: axes[0,0].legend(fontsize=7, ncol=2)
        save(fig, '01_same_N', completed+pair_rows, '同N负载与配对差；点为景，母为独立单位')

        fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout='constrained')
        counts=[]
        date_rows=[]
        for p, profile in enumerate(PROFILES):
            rows=[r for r in test if r['demand_profile']==profile]
            mothers={r['mother_id'] for r in rows}; dates={r.get('date') for r in rows}
            counts.append(dict(demand_profile=profile, mothers=len(mothers), dates=';'.join(sorted(d for d in dates if d)), method_rows=len(rows)))
            for date in sorted(d for d in dates if d):
                date_rows.append(dict(demand_profile=profile,date=date,mothers=len({r['mother_id'] for r in rows if r.get('date')==date})))
        dates=sorted({r['date'] for r in date_rows});bottom=[0]*4
        for j,date in enumerate(dates):
            values=[sum(r['mothers'] for r in date_rows if r['date']==date and r['demand_profile']==profile) for profile in PROFILES]
            axes[0].bar(range(4),values,bottom=bottom,color=palette[j % len(palette)],hatch=['','//','xx','..'][j % 4],label=date)
            bottom=[a+b for a,b in zip(bottom,values)]
        if dates:axes[0].legend(fontsize=7)
        axes[0].set(xticks=range(4), xticklabels=PROFILES, ylabel='实际母场景数（共享母）', ylim=(0, max([r['mothers'] for r in counts]+[40])*1.15))
        empty(axes[1], '输入定义：热点半径0.5°、倍率4；边缘前5%、倍率8\n总请求守恒，位置不变。\n逐机输入分布未由compact CSV导出；不推造空间图。')
        missing.append('input spatial distribution not available in compact schema')
        save(fig, '02_input_profiles', counts+date_rows, '输入profile与实际母分母/日期分层；计划test40母')

        variable=[r for r in tables['per_case'] if r.get('method')=='VARIABLE']
        fig, axes=plt.subplots(1, 3, figsize=(13, 5), layout='constrained')
        for i, metric in enumerate(('maxload', 'top10_mean', 'CV')):
            for j, profile in enumerate(PROFILES):
                rows=[r for r in variable if r.get('demand_profile')==profile and r.get('outcome')=='completed' and number(r,'N') is not None and number(r,metric) is not None]
                axes[i].scatter([number(r,'N') for r in rows], [number(r,metric) for r in rows], marker=markers[j], color=palette[j], s=16, alpha=.7, label=profile)
            axes[i].set(xlabel='实际非空束数N', ylabel=metric+(' (原请求单位)' if metric!='CV' else ' (无量纲)'))
        failed=sum(r.get('outcome')!='completed' for r in variable)
        if variable: axes[0].legend(fontsize=7)
        else: empty(axes[0], '变量N表无记录；不补0')
        save(fig, '03_variable_N', variable, f'变量N原单位；{failed}/{len(variable)}记录未完成或本预算未找到；不跨N排名')

        fig, axes=plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
        axes=axes.flatten()
        learning=tables['learningcurve']
        if learning:
            axes[0].plot([number(r,'update') for r in learning], [number(r,'loss') for r in learning], color='#0072B2', linewidth=.7)
            axes[0].set(xlabel='真实提交更新', ylabel='原始loss（无平滑）')
        else: empty(axes[0], '教师门槛未通过：未训练' if summary.get('gate_passed') is False else '无真实训练记录')
        validation=tables.get('validation_cases', [])
        for i, metric in enumerate(('P','T','CV'),1):
            if validation:
                for j, profile in enumerate(PROFILES):
                    rows=[r for r in validation if r.get('demand_profile')==profile and number(r,metric) is not None]
                    axes[i].scatter([number(r,'node') for r in rows], [number(r,metric) for r in rows], marker=markers[j],color=palette[j],s=13,alpha=.7,label=profile)
                axes[i].set(xlabel='真实验证节点（仅训练诊断）', ylabel=metric+' (无量纲)',xticks=[2500,5000,7500,10000])
            else:
                empty(axes[i], '未训练，无验证节点' if summary.get('gate_passed') is False else 'compact CSV无真实节点数据；不补曲线')
        if validation: axes[1].legend(fontsize=7)
        else: missing.append('validation node metrics unavailable; no invented curve')
        save(fig, '04_learning', learning+validation, 'MLM完整候选排序模型；不生成布局或自主选N')

        fig, axes=plt.subplots(2, 2, figsize=(12, 9), layout='constrained')
        times=[r for r in tables['timing'] if r.get('task','').startswith('common_test_')]
        candidates=[r for r in tables['candidate'] if r.get('task','').startswith('common_test_')]
        for j, method in enumerate(methods):
            for metric, marker in [('initialization_seconds','o'),('online_seconds','s'),('model_cold_load_seconds','^')]:
                values=[number(r,metric) for r in times if r['method']==method]; values=[v for v in values if v is not None]
                axes[0,0].scatter([j]*len(values), values, marker=marker, color=palette[j % len(palette)], s=15, alpha=.7, label=metric if j==0 else None)
            denom=sum(r['method']==method for r in test)
            success=sum(r['method']==method and r.get('outcome')=='completed' for r in test)
            axes[0,1].bar(j, success/denom if denom else np.nan, color=palette[j % len(palette)])
            axes[0,1].text(j, .04, f'{success}/{denom}', ha='center', fontsize=7, rotation=90)
            for key, marker in [('known_completed_candidates','o'),('accepted','s'),('completed_late','^')]:
                values=[number(r,key) for r in candidates if r['method']==method]; values=[v for v in values if v is not None]
                axes[1,0].scatter([j]*len(values), values, marker=marker, color=palette[j % len(palette)], s=15, alpha=.7, label=key if j==0 else None)
        for ax in axes.flat:
            ax.set(xticks=range(len(methods)), xticklabels=[LABELS.get(m,m) for m in methods]);ax.tick_params(axis='x',rotation=55)
        axes[0,0].set(ylabel='分段实际秒');axes[0,1].set(ylabel='完整可行发布率（全部预定记录）',ylim=(0,1.05))
        axes[1,0].set(ylabel='真实完成候选数/接受/late')
        if methods: axes[0,0].legend(fontsize=7); axes[1,0].legend(fontsize=7)
        if not times:empty(axes[0,0], '无common test分段时间数据；不补零')
        if not candidates:empty(axes[1,0], '无common test候选账；不补零')
        empty(axes[1,1], '退步轨迹未在compact表导出；不造零退步。\n记录候选几何来自worker；独立复算限最终/参考布局。\n未知中断开销不置零；观测时间不证明速度优势。')
        missing.append('per-candidate deterioration trajectory absent from compact CSV')
        save(fig, '05_costs', times+candidates+test, '算法配置观测成本；恒等包装不算独立算法族')

    manifest=dict(group='S1-G16', synthetic=synthetic, formal_ready=summary.get('formal_ready'),
        source_summary_sha256=sha(Path(source)/'summary.json'),
        sources={name+'.csv':sha(Path(source)/(name+'.csv')) for name in TABLES},
        artifacts={p.name:sha(p) for p in outputs}, font=font, matplotlib=matplotlib.__version__,
        transformations=['common_test task prefix filter', 'completed metrics only; failures retained in denominators',
                         'same-case same-N differences vs JOINT; no bootstrap', 'raw points; no smoothing; no imputation'],
        unavailable=missing, unit='request unit as stored; compact schema does not assert Mbps',
        independent_unit='mother; four profiles dependent; dates preserved in figure CSV',
        reference='budget-limited feasible reference, not global optimum',
        journal_requirements='not specified; provisional general export',
        candidate_geometry_limitation=summary.get('candidate_geometry_limitation'), single_training_seed=51001)
    if (Path(source)/'validation_cases.csv').exists():
        manifest['sources']['validation_cases.csv']=sha(Path(source)/'validation_cases.csv')
    (output/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2),encoding='utf-8')
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', required=True);parser.add_argument('--output', required=True)
    parser.add_argument('--synthetic-qa', action='store_true')
    args=parser.parse_args()
    build(args.report,args.output,synthetic=args.synthetic_qa)


if __name__=='__main__':
    main()
