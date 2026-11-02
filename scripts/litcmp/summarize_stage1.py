"""Average seeds within scenes and compare feasible pairs with S1-FIX."""
import argparse
from pathlib import Path
import pandas as pd
from satellite_rl.s2fix.common import run_manifest


def summarize(lit, ref):
    keys = ['scene_set', 'date', 'frame_time', 'N_budget']
    means = ['N_used', 'peak', 'top10', 'CV', 'Omega_sr', 'width_mean',
             'width_min', 'width_max', 'spacing_violations', 't_total']
    rate = lit.groupby(keys + ['arm']).feasible.mean().reset_index()
    scenes = lit[lit.feasible].groupby(keys + ['arm'])[means].mean().reset_index()
    rows = []
    for (scene_set, budget, arm), group in rate.groupby(['scene_set', 'N_budget', 'arm']):
        selected = scenes[(scenes.scene_set == scene_set) & (scenes.N_budget == budget) & (scenes.arm == arm)]
        row = dict(scene_set=scene_set, N_budget=budget, arm=arm,
                   feasible_rate=group.feasible.mean(), **selected[means].mean().to_dict())
        for method in ('FLOW-AO', 'FLOW-AO-no-spacing', 'INIT'):
            reference = ref[(ref.arm == method) & ref.feasible & (ref.effort == 1)]
            reference = reference.groupby(keys, as_index=False)['peak'].mean()
            paired = selected.merge(reference, on=keys, suffixes=('_lit', '_ref'))
            row[f'{method}_paired_scenes'] = len(paired)
            row[f'{method}_peak_ratio_lit_over_ref'] = (paired.peak_lit / paired.peak_ref).mean()
            row[f'{method}_not_higher_fraction'] = (paired.peak_ref <= paired.peak_lit).mean()
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--lit', nargs='+', required=True)
    p.add_argument('--ref', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    with run_manifest(Path(args.output).parent, args):
        summarize(pd.concat([pd.read_csv(path) for path in args.lit]),
                  pd.read_csv(args.ref)).to_csv(args.output, index=False)


if __name__ == '__main__':
    main()
