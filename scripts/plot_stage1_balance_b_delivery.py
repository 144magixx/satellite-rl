"""Plot audited S1-02 B comparisons with explicit retained/raw and timing scope."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from audit_stage1_balance import sha
from report_stage1_balance import _matplotlib


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    delivery = root / 'analysis/B_delivery'
    source = root / 'analysis/B_report/report_manifest.json'
    manifest = json.loads(source.read_text(encoding='utf-8'))
    supplement_path = delivery / 'supplement/supplement.json'
    supplement = json.loads(supplement_path.read_text(encoding='utf-8'))
    assert supplement['scope'] == 'full_six_method_archived_result_supplement'
    audit_path = Path(supplement['audit_path'])
    assert sha(audit_path) == supplement['audit_sha256']
    assert json.loads(audit_path.read_text(encoding='utf-8'))['passed']
    input_files = [source, supplement_path, audit_path]
    for record in supplement['results'].values():
        path = Path(record['path'])
        assert sha(path) == record['sha256'], path
    def read_csv(name):
        path = delivery / 'supplement' / name
        input_files.append(path)
        with path.open(encoding='utf-8-sig', newline='') as handle:
            return list(csv.DictReader(handle))
    raw_best = read_csv('per_scene_PPO_raw_vs_best.csv')
    paired = read_csv('per_scene_PPO_vs_same_menu_search.csv')
    timing = read_csv('per_scene_recorded_time.csv')
    assert len(raw_best) == len(paired) == 300 and len(timing) == 600
    menus = ['X0', 'X1', 'X4']
    labels = ['X0 四向固定步长', 'X1 八方向', 'X4 相邻双束']
    stats = manifest['summaries']
    def rows_for(rows, menu):
        result = sorted((row for row in rows if row['menu'] == menu), key=lambda row: row['scene_id'])
        assert len(result) == len({row['scene_id'] for row in result}) == 100
        return result
    raw = {menu: rows_for(raw_best, menu) for menu in menus}
    matches = {menu: rows_for(paired, menu) for menu in menus}
    for menu in menus:
        assert [r['scene_id'] for r in raw[menu]] == [r['scene_id'] for r in matches[menu]]
        assert all(row['raw_state_recorded'] == 'True' for row in raw[menu])
        assert np.isclose(np.mean([float(r['best_subject_cv']) for r in raw[menu]]),
                          stats['PPO_'+menu]['subject_cv_mean'])
        for kind, key in [('ppo', 'PPO_'), ('search', 'search_')]:
            values = [float(row['recorded_total_pipeline_seconds']) for row in timing
                      if row['method'] == menu+'_'+kind]
            assert len(values) == 100
            assert np.isclose(np.mean(values), stats[key+menu]['full_pipeline_seconds_mean'])
    plt, font = _matplotlib()
    figures = delivery / 'figures'
    figures.mkdir(parents=True, exist_ok=True)
    produced = []
    def save(fig, name):
        path = figures / (name+'.png')
        fig.savefig(path, dpi=160)
        plt.close(fig)
        produced.append({'path': str(path), 'sha256': sha(path)})
    colors = ['#187C92', '#D87822']
    x = np.arange(3)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    settings = [('full_coverage_legal_scenes', '双100%覆盖且合法 / 场景数', 100, None),
                ('subject_cv_mean', '主体CV均值（越低越均衡）', .65, .2),
                ('subject_in_band_beams_mean', '合理负载波束数均值 / 100', 100, 90),
                ('full_pipeline_seconds_mean', '已记录分段合计均值 / 秒', 82, None)]
    for ax, (field, title, maximum, target) in zip(axes.flat, settings):
        for index, prefix in enumerate(['PPO_', 'search_']):
            values = [stats[prefix+menu][field] for menu in menus]
            bars = ax.bar(x+(index-.5)*.34, values, width=.32, color=colors[index],
                          label='A2＋PPO（保留最好布局）' if index == 0 else 'A2＋同菜单搜索')
            for bar, value in zip(bars, values):
                text = f'{value:.4f}' if field == 'subject_cv_mean' else f'{value:.2f}'
                ax.text(bar.get_x()+bar.get_width()/2, value+maximum*.018, text,
                        ha='center', fontsize=9)
        if target is not None:
            ax.axhline(target, color='#555555', linestyle='--', linewidth=1)
            ax.text(2.43, target, f'目标 {target:g}', ha='right', va='bottom', fontsize=9)
        ax.set_xticks(x, labels, fontsize=9)
        ax.set_ylim(0, maximum)
        ax.set_ylabel(title)
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
    axes[0, 0].legend(loc='lower left', fontsize=9)
    fig.suptitle('S1-02 B：同100场景、同A2起点；PPO与同菜单搜索\n'
                 '计时不含此前输入导入/初始聚类生成，无缺项补零', fontsize=13)
    save(fig, 'six_method_overview')

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for ax, metric, title, ymax, target in [
        (axes[0], 'subject_cv', '主体CV均值', .65, .2),
        (axes[1], 'subject_in_band_beams', '合理负载波束数均值 / 100', 100, 90)]:
        for index, state in enumerate(['raw', 'best']):
            values = [np.mean([float(row[state+'_'+metric]) for row in raw[menu]]) for menu in menus]
            bars = ax.bar(x+(index-.5)*.34, values, width=.32,
                          color=['#738294', '#187C92'][index],
                          label='PPO原终态' if index == 0 else 'PPO保留最好布局')
            for bar, value in zip(bars, values):
                ax.text(bar.get_x()+bar.get_width()/2, value+ymax*.018,
                        f'{value:.4f}' if metric == 'subject_cv' else f'{value:.2f}',
                        ha='center', fontsize=9)
        ax.axhline(target, color='#555555', linestyle='--', linewidth=1)
        ax.set_xticks(x, labels, fontsize=9)
        ax.set_ylim(0, ymax)
        ax.set_ylabel(title)
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
    axes[0].legend(loc='lower left', fontsize=9)
    fig.suptitle('同一PPO轨迹的两种结果：每菜单均为同100场景配对\n'
                 '“最好”按冻结目标排序，不等于每个场景CV最小', fontsize=13)
    save(fig, 'raw_vs_retained')

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, constrained_layout=True)
    for menu, label in zip(menus, labels):
        cv_delta = [float(row['PPO_minus_search_subject_cv']) for row in matches[menu]]
        time_delta = [float(row['PPO_minus_search_recorded_total_seconds']) for row in matches[menu]]
        axes[0].scatter(np.arange(1,101), cv_delta, s=15, alpha=.7, label=label)
        axes[1].scatter(np.arange(1,101), time_delta, s=15, alpha=.7, label=label)
    axes[0].set_ylabel('主体CV差（负值表示PPO更低）')
    axes[1].set_ylabel('已记录分段合计时间差 / 秒')
    axes[1].set_xlabel('共同场景序号（按场景ID排序，无平滑）')
    for ax in axes:
        ax.axhline(0, color='#555555', linewidth=1)
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=9, ncol=3)
    fig.suptitle('同菜单配对：A2＋PPO保留布局 − A2＋同菜单搜索', fontsize=13)
    save(fig, 'paired_menu_differences')
    receipt = {'scope': 'Audited six-method final delivery plots; no new simulation',
               'inputs': {str(path): sha(path) for path in input_files},
               'figures': produced, 'font': font, 'script_sha256': sha(Path(__file__)),
               'raw_scenes_per_menu': 100, 'retained_summary_and_timing_checked': True,
               'independent_scenes': 100, 'same_scene_raw_best_paired': True}
    (delivery/'plot_receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'figures':len(produced),'receipt':str(delivery/'plot_receipt.json')}))


if __name__ == '__main__':
    main()
