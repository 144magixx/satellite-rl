"""Check revision-one stacking on the largest development scene and a night scene."""
import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from satellite_rl.s2fix.common import OUTPUT_ROOT
from satellite_rl.s2fix.data import build_scene, save_record, source_cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', default=str(OUTPUT_ROOT / 'stacking_revision1'))
    args = parser.parse_args()
    destination = Path(args.output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    cases = source_cases('development')
    day = max(cases, key=lambda case: len(case['aircraft_ids']))
    night = min(cases, key=lambda case: len(case['aircraft_ids']))
    checks = []
    for label, original in [('day', day), ('night', night)]:
        case, layout, check = build_scene(original, 86400000, 150)
        stacks = np.asarray(layout['stack_id'])
        centers = np.asarray(layout['centres_lat_lon_deg'])
        np.testing.assert_array_equal(case['aircraft_ids'], original['aircraft_ids'])
        np.testing.assert_array_equal(case['lat_lon_deg'], original['lat_lon_deg'])
        np.testing.assert_array_equal(case['request_bps_int64'], np.full(check['n'], 86400000))
        for stack in np.unique(stacks[stacks >= 0]):
            cluster = centers[stacks == stack]
            np.testing.assert_array_equal(cluster, np.repeat(cluster[:1], len(cluster), axis=0))
        check['UTC'] = original['UTC']
        check['stack_sizes'] = dict(Counter(int(stack) for stack in stacks if stack >= 0))
        check['stack_centers_lat_lon_deg'] = {
            str(stack): centers[np.flatnonzero(stacks == stack)[0]].tolist()
            for stack in np.unique(stacks[stacks >= 0])}
        save_record(destination / f'{label}_case.json', case)
        save_record(destination / f'{label}_layout.json', layout)
        save_record(destination / f'{label}_check.json', check)
        checks.append(check)
        print(check, flush=True)
    save_record(destination / 'numbers.json', checks)


if __name__ == '__main__':
    main()
