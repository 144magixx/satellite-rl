"""Check smallest/largest real development scenes and random packing limits."""
import argparse
import json
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from algorithm import random_allocation,InitializedSearchEnv
from evaluate import check
from satellite_rl.s2fix.data import read_records,load_input

p=argparse.ArgumentParser();p.add_argument('--data-dir',required=True);args=p.parse_args()
for n in (1,89,150,200):
    stage=SimpleNamespace(layout={'request_bps_by_beam_int64':np.arange(1,n+1)*43200000})
    with patch('algorithm.isolated_slot_counts',return_value=np.full(n,10,int)):
        a=random_allocation(stage,51001);b=random_allocation(stage,51001)
    for x,y in zip(a,b):np.testing.assert_array_equal(x,y)
    env=InitializedSearchEnv()
    assert env._legal(a[1],a[2],a[0])
    assert a[3].sum()<=3000+1e-7 and a[3].max()<=50+1e-7
    print('packing',n,'ok',flush=True)
records=read_records(args.data_dir,'dev_d432_n150')
sizes=[len(load_input(r).case['aircraft_ids']) for r in records]
for index in (int(np.argmin(sizes)),int(np.argmax(sizes))):
    check(records[index],index,51001)
    print(json.dumps(dict(scene=index,aircraft=sizes[index],greedy_equivalence=True,
                         pure_has_no_greedy_call=True)),flush=True)
