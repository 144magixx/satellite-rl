"""Average repetitions and seeds within each scene before paired comparison."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from satellite_rl.s2fix.common import run_manifest


def paired_interval(frame, rng):
    days = [part['difference'].to_numpy() for _,part in frame.groupby('date')]
    samples = []
    for _ in range(2000):
        chosen = rng.integers(len(days),size=len(days))
        samples.append(np.concatenate([rng.choice(days[i],size=len(days[i]),replace=True)
                                        for i in chosen]).mean())
    return np.quantile(samples,[.025,.975])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', nargs='+', required=True)
    parser.add_argument('--output', required=True, help='Output directory')
    args = parser.parse_args()
    with run_manifest(args.output,args):
        directory = Path(args.output)
        frame = pd.concat([pd.read_csv(p) for p in args.input],ignore_index=True)
        columns = [c for c in frame if c.startswith(('before_','after_')) or c.endswith('_seconds')]
        keys = ['set','scene_id','date','method']
        per_seed = frame.groupby(keys+['seed'])[columns].mean().reset_index()
        scenes = per_seed.groupby(keys)[columns].mean().reset_index()
        scenes.to_csv(directory/'scene_means.csv',index=False)
        summary = scenes.groupby(['set','method'])[columns].mean()
        summary['scenes'] = scenes.groupby(['set','method']).size()
        summary.to_csv(directory/'summary.csv')
        rng, rows = np.random.default_rng(51001), []
        for dataset, group in scenes.groupby('set'):
            for reference in ('G1','S1','S2'):
                base = group[group.method==reference]
                for method, arm in group.groupby('method'):
                    if method == reference:
                        continue
                    paired = arm.merge(base,on=['set','scene_id','date'],suffixes=('_arm','_reference'))
                    if paired.empty:
                        continue
                    for metric in columns:
                        paired['difference'] = paired[metric+'_arm']-paired[metric+'_reference']
                        lower,upper = paired_interval(paired,rng)
                        rows.append(dict(set=dataset,method=method,reference=reference,metric=metric,
                            scenes=len(paired),difference=paired.difference.mean(),ci95_low=lower,ci95_high=upper))
        pd.DataFrame(rows).to_csv(directory/'paired_differences.csv',index=False)


if __name__ == '__main__':
    main()
