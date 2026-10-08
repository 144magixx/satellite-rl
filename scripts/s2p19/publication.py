"""Recheck completed S2-P19 tables and draw publication figures; no new evaluation."""
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import statistics as st

ROOT = Path('D:/paper/outputs/experiments/sac_routes_s2p19_20261007')
DATA = ROOT / 'analysis/final_report_material'
OUT = ROOT / 'analysis/publication'
ROLES = ['greedy', 'rule', 'random', 'V2', 'C0', 'C1', 'I0']
LABELS = ['贪心', '规则局部搜索', '随机局部搜索', 'V2 历史模型', 'C0 目标对齐SAC', 'C1 示范回放SAC', 'I0 多步改进SAC']
COLORS = ['#606060', '#c67c00', '#48a9a6', '#9275b5', '#cf6e7e', '#3973ac', '#c22b23']


def load_csv(name):
    with (DATA / name).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def main():
    summary = json.loads((DATA / 'summary.json').read_text(encoding='utf-8'))
    assert summary['formal_ready'] and not summary['missing']
    scenes, actions = load_csv('per_scene.csv'), load_csv('per_action.csv')
    assert len(scenes) == 12320 and len(actions) == 29920
    key = lambda x: (x['role'], x['stage'], x['mother_scene_id'], int(x['beam_count']), float(x['load_multiplier']))
    indexed = {key(x): x for x in scenes}
    assert len(indexed) == len(scenes)
    grouped = defaultdict(list)
    for x in actions:
        assert x['resource_legal'] == 'True' and x['beam_audit_available'] == 'True'
        grouped[key(x)].append(x)
    for k, rows in grouped.items():
        expected = 1 if k[0] in ('greedy', 'rule') else 3
        assert len(rows) == expected and len({x['repeat_index'] for x in rows}) == expected
        for metric in ['si', 'sgm', 'basic_met_count', 'power_w', 'rho_p05', 'rho_below_08_fraction']:
            assert abs(st.mean(float(x[metric]) for x in rows) - float(indexed[k][metric])) < 1e-11
    result = {'scope': 'same-scene mean recomputation from complete per-action tables; no policy/physics/optimizer rerun',
              'all_29920_resource_records_legal': True, 'all_12320_repeat_means_rechecked': True,
              'input_sha256': {name: digest(DATA / name) for name in ['summary.json', 'per_action.csv', 'per_scene.csv', 'by_n_load.csv']},
              'aggregate': {}, 'pairs': {}, 'validation_nodes': {}, 'figures': []}
    for role in ROLES:
        for stage in ['direct', 'final_B']:
            rows = [x for x in scenes if x['role'] == role and x['stage'] == stage]
            assert len(rows) == 880
            vals = {k: st.mean(float(x[k]) for x in rows) for k in ['si', 'sgm', 'basic_met_count', 'power_w', 'slots_total', 'rho_p05', 'rho_below_08_fraction']}
            vals.update({'largeN_si': st.mean(float(x['si']) for x in rows if int(x['beam_count']) >= 125),
                         'N200_si': st.mean(float(x['si']) for x in rows if int(x['beam_count']) == 200),
                         'basic_fraction': st.mean(float(x['basic_met_count']) / int(x['beam_count']) for x in rows),
                         'solver_fallback_counts': dict(Counter(x['fallback_used'] for x in actions if x['role'] == role and x['stage'] == stage))})
            result['aggregate'][role + '/' + stage] = vals
    for name, pair in summary['comparisons'].items():
        left, right, stage = pair['left'], pair['right'], pair['stage']
        rows = [x for x in scenes if x['role'] == left and x['stage'] == stage]
        deltas = {k: [] for k in ['si', 'sgm', 'basic_met_count']}
        negative = []
        mother_times = {}
        for row in rows:
            other = indexed[(right, stage, row['mother_scene_id'], int(row['beam_count']), float(row['load_multiplier']))]
            mother_times[row['mother_scene_id']] = row['selected_time_utc']
            for k in deltas:
                deltas[k].append(float(row[k]) - float(other[k]))
            if float(row['si']) - float(other['si']) < -1e-8:
                negative.append({k: row[k] for k in ['mother_scene_id', 'selected_time_utc', 'beam_count', 'load_multiplier']})
                negative[-1]['delta_si'] = float(row['si']) - float(other['si'])
        assert abs(st.mean(deltas['si']) - pair['mean_delta']) < 1e-12
        chronological = sorted(mother_times, key=lambda m: (mother_times[m], m)) == sorted(mother_times)
        assert chronological, 'time-block order requires explicit correction'
        result['pairs'][name] = {k: v for k, v in pair.items() if k != 'per_mother'}
        result['pairs'][name].update({'mother_ID_order_matches_chronological': chronological,
            'positive_mothers': sum(v > 1e-8 for v in pair['per_mother'].values()),
            'metrics': {k: {'mean': st.mean(v), 'min': min(v), 'positive_scenes': sum(x > 1e-8 for x in v),
                            'negative_scenes': sum(x < -1e-8 for x in v), 'ties': sum(abs(x) <= 1e-8 for x in v)} for k, v in deltas.items()},
            'negative_si_scene_ids': negative})
    for arm in ['C0', 'C1', 'I0']:
        nodes = defaultdict(list)
        for row in summary['training_curves']:
            if row['arm'] == arm and row['metric'] == 'validation_si':
                nodes[row['env_step']].append(row['value'])
        result['validation_nodes'][arm] = {str(k): {'mean_si': st.mean(v), 'saved_rows': len(v)} for k, v in sorted(nodes.items())}
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.sans-serif': ['Microsoft YaHei', 'SimHei'], 'axes.unicode_minus': False, 'font.size': 11})
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'timing_summary.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['role', 'stage', 'unit', 'segment', 'n', 'missing', 'mean', 'median', 'p95'])
        writer.writeheader()
        for unit, source in [('single_action_seconds', summary['timing']), ('same_scene_repeat_sum_seconds', summary['scene_repeat_timing'])]:
            for key_, segments in source.items():
                role, stage = key_.split('/')
                for segment, values in segments.items():
                    writer.writerow({'role': role, 'stage': stage, 'unit': unit, 'segment': segment, **values})
    for metric, title, factor in [('si', '全束平均需求满足率 SI（%）', 100), ('sgm', '供需匹配程度 SGM', 1), ('rho_below_08_fraction', 'C/D < 0.8 的波束比例（%）', 100)]:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.8), sharex=True)
        for ax, stage, heading in zip(axes, ['direct', 'final_B'], ['直接分配', '相同完整功率细调 B 后']):
            for role, label, color in zip(ROLES, LABELS, COLORS):
                rows = [x for x in scenes if x['role'] == role and x['stage'] == stage]
                ns = sorted({int(x['beam_count']) for x in rows})
                ys = [factor * st.mean(float(x[metric]) for x in rows if int(x['beam_count']) == n) for n in ns]
                ax.plot(ns, ys, marker='o', markersize=3, color=color, label=label, linewidth=2.4 if role == 'I0' else 1.4)
            ax.set(title=heading, xlabel='目标波束数 N', ylabel=title)
            ax.grid(alpha=.2)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=4, frameon=False)
        fig.suptitle('40母场景 / 880衍生场景；重复动作先取均值；训练种子51001', fontsize=12)
        fig.tight_layout(rect=(0, .12, 1, .94))
        path = OUT / (metric + '_by_n_panels.png'); fig.savefig(path, dpi=180); plt.close(fig)
        result['figures'].append(str(path))
    fig, ax = plt.subplots(figsize=(10, 5))
    pairs = [(base, stage) for base in ['greedy', 'rule', 'random'] for stage in ['direct', 'final_B']]
    for i, (base, stage) in enumerate(pairs):
        pair = summary['comparisons']['I0_vs_' + base + '_' + stage]
        v, low, high = [100 * pair[k] for k in ['mean_delta', 'ci95_low', 'ci95_high']]
        ax.errorbar(v, i, xerr=[[v-low], [high-v]], fmt='o', capsize=4, color='#c22b23' if stage == 'final_B' else '#3973ac')
    ax.set_yticks(range(len(pairs)), [LABELS[ROLES.index(b)] + (' · 直接' if s == 'direct' else ' · 最终B') for b, s in pairs])
    ax.axvline(0, color='grey', ls='--');ax.invert_yaxis();ax.grid(axis='x', alpha=.2)
    ax.set(xlabel='I0需求满足率增量（百分点）；40母配对bootstrap 95%区间', title='I0对贪心与同菜单非学习搜索的质量差异')
    fig.tight_layout();path=OUT/'i0_paired_intervals.png';fig.savefig(path,dpi=180);plt.close(fig);result['figures'].append(str(path))
    (OUT/'verified_statistics.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'complete_scene_means_verified': len(scenes), 'figures': result['figures']}))


if __name__ == '__main__':
    main()
