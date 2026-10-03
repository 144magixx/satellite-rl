"""Plot recorded S1-02 A timing segments without claiming unrecorded import time."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from report_stage1_balance import _matplotlib


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest_path = args.root / 'analysis/A_report/report_manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    labels = ['A0 C1基础布局', 'A1 C1＋修补',
              'A2 C1＋修补＋归属', 'A3 C1＋修补＋归属＋四向调位']
    keys = ['c1', 'edge_preprocess', 'repair', 'assignment', 'adjustment', 'final_verification']
    names = ['C1局部搜索', '边缘资格识别', '覆盖修补', '用户归属', '四向调位', '最终核验']
    stage_means, totals, sources = [], [], {}
    for label in ('A0', 'A1', 'A2', 'A3'):
        source = manifest['sources'][label]
        path = args.root / 'jobs' / Path(source['path']).parent.name / 'results.json'
        raw = path.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == source['sha256'], path
        rows = json.loads(raw)['scenes']
        assert len(rows) == 100
        rows.sort(key=lambda row: row['scene_id'])
        stages = [row['balance_diagnostics']['stage_seconds'] for row in rows]
        # A stage absent from a method is not executed, unlike a missing recorded time.
        sums = [sum(stage.values()) for stage in stages]
        recorded = [row['balance_diagnostics']['total_pipeline_seconds'] for row in rows]
        assert np.allclose(sums, recorded, rtol=1e-12, atol=1e-10)
        assert np.isclose(np.mean(recorded), manifest['summaries'][label]['full_pipeline_seconds_mean'])
        stage_means.append([np.mean([stage.get(key, 0.0) for stage in stages]) for key in keys])
        totals.append(recorded)
        sources[label] = {'path': str(path), 'sha256': source['sha256']}
    plt, font = _matplotlib()
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), constrained_layout=True)
    values = np.asarray(stage_means)
    bottom = np.zeros(4)
    for i, name in enumerate(names):
        axes[0].bar(np.arange(4), values[:, i], bottom=bottom, label=name)
        bottom += values[:, i]
    axes[0].set_xticks(np.arange(4), labels, fontsize=10)
    axes[0].set_ylabel('已记录分段时间均值 / 秒')
    axes[0].legend(ncol=3, loc='upper left', fontsize=9)
    axes[0].set_ylim(0, max(bottom) * 1.3)
    for i, total in enumerate(bottom):
        axes[0].text(i, total + .12, f'{total:.3f}', ha='center')
    for label, values in zip(labels, totals):
        axes[1].plot(np.arange(100), values, marker='.', markersize=4, linewidth=.6, label=label)
    axes[1].set_ylabel('已记录分段合计 / 秒')
    axes[1].set_xlabel('共同场景序号（按场景ID排序，未平滑）')
    axes[1].legend(ncol=2, fontsize=9)
    axes[1].grid(alpha=.2)
    fig.suptitle('S1-02 A：公共步骤实测分段复用；未记录的输入导入/初始聚类时间不补零', fontsize=12)
    args.output.mkdir(parents=True, exist_ok=True)
    target = args.output / 'recorded_timing.png'
    fig.savefig(target, dpi=150)
    plt.close(fig)
    receipt = {'sources': sources, 'timing_keys': keys, 'stage_means_seconds': stage_means,
               'scene_count_per_method': 100, 'source_sums_verified': True,
               'plot_font': font, 'figure_sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
               'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               'scope': 'Recorded segment sum only; unavailable input-import/clustering timings are not zero-filled.'}
    (args.output / 'timing_plot_receipt.json').write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(target)


if __name__ == '__main__':
    main()
