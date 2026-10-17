"""Run only specification section 11: two development scenes and short training."""
import argparse
import csv
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

from satellite_rl.s2fix.common import OUTPUT_ROOT, run_manifest, write_json
from satellite_rl.s2fix.data import build_set, read_records
from satellite_rl.s2fix.experiments import evaluate_scene
from satellite_rl.s2fix.i0 import train as train_i0
from satellite_rl.s2fix.tsac import train as train_tsac

PROBE = '''
import numpy as np
from satellite_rl.power30.simple_data import load_input
from satellite_rl.power30.environment import Power30Env
import sys
env = Power30Env(load_input(dict(case_id=sys.argv[1],role='development',
    case_path=sys.argv[2],layout_path=sys.argv[3])), 'P01','M1')
rng=np.random.default_rng(51001)
states,rewards=[],[]
state,done=env.reset(),False
while not done:
    states.append(state)
    mask=env.action_mask()
    f,k=rng.choice(np.argwhere(mask['frequency_slots']))
    p=rng.choice(np.flatnonzero(mask['power']))
    state,reward,done,_=env.step((int(f),int(k),int(p)))
    rewards.append(reward)
np.savez(sys.argv[4],states=states,rewards=rewards,rates=env.rates,
         starts=env.starts,counts=env.counts,powers=env.powers)
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', default=str(OUTPUT_ROOT/'smoke'))
    parser.add_argument('--baseline-src', required=True)
    parser.add_argument('--v1-data', default='D:/paper-w32/outputs/experiments/s2_w32_20261017/local/sample_new/dev_n100/objects')
    args = parser.parse_args()
    torch.set_num_threads(1)
    directory = Path(args.output_dir)
    with run_manifest(directory,args):
        checks = build_set('dev_d432_n150',directory/'data',workers=2,limit=2)
        records = read_records(directory/'data','dev_d432_n150')
        (directory/'evaluation'/'beams').mkdir(parents=True,exist_ok=True)
        rows=[]
        for i, record in enumerate(records):
            rows.extend(evaluate_scene((i,record,['G1','R','S1'],{},str(directory/'evaluation'),'cpu',1)))
            print('evaluated',i,flush=True)
        with (directory/'evaluation'/'results.csv').open('w',newline='',encoding='utf-8') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
            writer.writeheader();writer.writerows(rows)
        i0_path = train_i0(records,directory/'L1',steps=200,device='cpu')
        i0_saved = torch.load(i0_path,map_location='cpu',weights_only=False)
        tsac = train_tsac(records,directory/'L2',steps=300,device='cpu')
        v1root = Path(args.v1_data)
        from satellite_rl.shared_interface.content import _decode
        import json
        case_id = _decode(json.loads((v1root/'0000_case.wire').read_text(encoding='utf-8')))['case_id']
        for label, source in [('before',args.baseline_src),('after',str(Path(__file__).resolve().parents[2]/'src'))]:
            env=dict(os.environ,PYTHONPATH=source)
            subprocess.run([sys.executable,'-c',PROBE,case_id,str(v1root/'0000_case.wire'),
                str(v1root/'0000_layout.wire'),str(directory/f'v1_{label}.npz')],env=env,check=True)
        before,after=np.load(directory/'v1_before.npz'),np.load(directory/'v1_after.npz')
        regression={key:bool(np.array_equal(before[key],after[key])) for key in before.files}
        assert all(regression.values()), regression
        result=dict(flow=checks, evaluation_rows=len(rows), L1_steps=i0_saved['step'],
                    L1_updates=i0_saved['agent']['updates'], L2=tsac,v1_bitwise_equal=regression)
        write_json(directory/'numbers.json',result)
        print(result,flush=True)


if __name__ == '__main__':
    main()
