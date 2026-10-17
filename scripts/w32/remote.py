"""One-shot W32 file transfer, launch and observations; credentials from environment."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import shlex
import subprocess
import tarfile
import tempfile
import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)
import paramiko

ROOT='/home/u2025140002/second_stage_satellite_rl/s2_w32'
OLD='/home/u2025140002/second_stage_satellite_rl/s2_w30/shared_data/input596'
PYTHON='/home/u2025140002/second_stage_satellite_rl/envs/DQN/bin/python'
PREP=ROOT+'/source/preparation'
LOCAL=Path('D:/paper/outputs/experiments/s2_w32_20261017/codex')
RESULT=LOCAL/'result.json'


def connect(port):
    client=paramiko.SSHClient();client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect('10.160.17.102',port=port,username='u2025140002',
                   password=os.environ['BUPT_PASSWORD'],timeout=20)
    return client


def run(client,command):
    _,out,err=client.exec_command(command,timeout=120)
    output,error=out.read().decode(),err.read().decode()
    if out.channel.recv_exit_status():raise RuntimeError(error or output)
    return output


def update(**fields):
    value=json.loads(RESULT.read_text(encoding='utf-8')) if RESULT.exists() else {}
    value.update(fields)
    RESULT.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


def prepare():
    c=connect(20627)
    try:
        run(c,f'mkdir -p {PREP} {ROOT}/original_confirmation/objects {ROOT}/evals/regression_old_interface')
        with tempfile.TemporaryDirectory() as temp:
            archive=Path(temp)/'source.tar'
            with tarfile.open(archive,'w') as tar:
                for folder in ('src','scripts/w32'):
                    for path in Path(folder).rglob('*'):
                        if path.is_file() and '__pycache__' not in path.parts:
                            tar.add(path,arcname=path.as_posix())
            with c.open_sftp() as sftp:sftp.put(str(archive),ROOT+'/prepare_source.tar')
            run(c,f'tar -xf {ROOT}/prepare_source.tar -C {PREP}')
        inputs={}
        with c.open_sftp() as sftp:
            for role in ('train','development'):
                with sftp.open(OLD+'/'+role+'_runtime.json') as stream:rows=json.load(stream)['rows']
                inputs[role]=[dict(case_id=r['case_id'],role=r['role'],case_path=r['case_ref']['path'],
                    layout_path=r['layout_ref']['path'],layout_content_SHA=r['layout_content_SHA']) for r in rows]
            local=Path('D:/paper/outputs/experiments/s2_width26_20261014/collected/inputs')
            manifest=json.loads((local/'test_published.json').read_text(encoding='utf-8'))
            inputs['test']=[]
            for row in manifest['rows']:
                paths={}
                for kind in ('case','layout'):
                    sha=row[kind+'_ref']['sha256'];path=local/'objects'/sha
                    remote=ROOT+'/original_confirmation/objects/'+sha
                    sftp.put(str(path),remote);paths[kind+'_path']=remote
                inputs['test'].append(dict(case_id=row['case_id'],role='test',
                    layout_content_SHA=row['layout_content_SHA'],**paths))
            with sftp.open(ROOT+'/original_inputs.json','w') as stream:stream.write(json.dumps(inputs))
        print('prepared inputs', {k:len(v) for k,v in inputs.items()},flush=True)
    finally:c.close()


def regression():
    c=connect(20653)
    try:
        command=f'env PYTHONPATH={PREP}/src OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=1 {PYTHON} {PREP}/scripts/w32/regressions.py --inputs {ROOT}/original_inputs.json --old-data {OLD} --old-run /home/u2025140002/second_stage_satellite_rl/s2_w31/runs/P00-M0 --output-dir {ROOT}/evals/regression_old_interface'
        print(run(c,f'nohup setsid {command} > {ROOT}/regression.log 2>&1 < /dev/null & echo $!'),flush=True)
    finally:c.close()


def generate():
    def lane(pair):
        shard,port=pair;c=connect(port)
        try:
            command=f'env PYTHONPATH={PREP}/src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 {PYTHON} {PREP}/scripts/w32/build_data.py --inputs {ROOT}/original_inputs.json --output-dir {ROOT}/data --workers 12 --shard {shard} --shards 3'
            return port,run(c,f'nohup setsid {command} > {ROOT}/data_{shard}.log 2>&1 < /dev/null & echo $!')
        finally:c.close()
    with ThreadPoolExecutor(3) as pool:
        for result in pool.map(lane,enumerate((20627,20653,20655))):print(result,flush=True)


def merge():
    c=connect(20627)
    try:
        print(run(c,f'PYTHONPATH={PREP}/src OPENBLAS_NUM_THREADS=1 {PYTHON} {PREP}/scripts/w32/build_data.py --inputs {ROOT}/original_inputs.json --output-dir {ROOT}/data --merge --shards 3'),flush=True)
        with c.open_sftp() as sftp:
            for name in ('data_checks.csv','layout_stats.csv'):
                sftp.get(ROOT+'/data/'+name,str(LOCAL/name))
    finally:c.close()


def deploy():
    commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    source=ROOT+'/source/'+commit
    c=connect(20627)
    try:
        run(c,f'mkdir -p {source}')
        with tempfile.TemporaryDirectory() as temp:
            archive=Path(temp)/'source.tar'
            subprocess.run(['git','-c','core.autocrlf=false','archive','--format=tar','-o',str(archive),commit,
                            'src','scripts/w32','pyproject.toml'],check=True)
            with c.open_sftp() as sftp:sftp.put(str(archive),ROOT+'/source_upload.tar')
        print(run(c,f'tar -xf {ROOT}/source_upload.tar -C {source}; bash -n {source}/scripts/w32/launch_lane.sh; bash -n {source}/scripts/w32/after_training.sh'),flush=True)
    finally:c.close()
    def lane(port):
        c=connect(port)
        try:return port,run(c,f'bash {source}/scripts/w32/launch_lane.sh {source} {PYTHON} {port}')
        finally:c.close()
    groups={}
    with ThreadPoolExecutor(3) as pool:
        for port,output in pool.map(lane,(20627,20653,20655)):
            print(port,output,flush=True)
            for line in output.splitlines():
                arm,pid,_,directory=line.split()
                groups[arm]=dict(pid=int(pid),port=port,output_dir=directory)
    update(status='training_started',commit=commit,remote_source=source,changes_committed=True,
           groups=groups,launched_at=datetime.now(timezone.utc).isoformat(),blocker=None)


def observe():
    value=json.loads(RESULT.read_text(encoding='utf-8'))
    def lane(port):
        c=connect(port)
        try:
            gpu=run(c,'nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,utilization.memory --format=csv,noheader,nounits').strip()
            rows={}
            with c.open_sftp() as sftp:
                for arm,info in value['groups'].items():
                    if info['port']!=port:continue
                    with sftp.open(info['output_dir']+'/progress.json') as stream:rows[arm]=json.load(stream)
                    rows[arm]['process_command']=run(c,f"ps -p {info['pid']} -o args=").strip()
            return str(port),gpu,rows
        finally:c.close()
    gpus={};now=datetime.now(timezone.utc)
    with ThreadPoolExecutor(3) as pool:
        for port,gpu,rows in pool.map(lane,(20627,20653,20655)):
            used,total,util,memutil=map(int,gpu.split(','))
            gpus[port]=dict(memory_used_MiB=used,memory_total_MiB=total,gpu_utilization_percent=util,memory_utilization_percent=memutil)
            for arm,progress in rows.items():
                value['groups'][arm].update(progress)
                speed=progress['steps_per_second']
                if speed>0:value['groups'][arm]['estimated_training_completion']=(now+timedelta(seconds=(150000-progress['env_step'])/speed)).isoformat()
    update(groups=value['groups'],gpu_observation=gpus,observed_at=now.isoformat(),
           seconds_since_launch=(now-datetime.fromisoformat(value['launched_at'])).total_seconds())
    print(json.dumps(dict(gpus=gpus,groups=value['groups']),indent=2),flush=True)


def after():
    value=json.loads(RESULT.read_text(encoding='utf-8'));source=value['remote_source']
    watchers={}
    for port in (20627,20653,20655):
        c=connect(port)
        try:
            arms=' '.join(arm for arm,info in value['groups'].items() if info['port']==port)
            watchers[str(port)]=int(run(c,f'nohup setsid bash {source}/scripts/w32/after_training.sh {source} {PYTHON} watch {port} 3 {arms} >> {ROOT}/after_training.log 2>&1 < /dev/null & echo $!').strip())
        finally:c.close()
    c=connect(20627)
    try:pid=int(run(c,f'nohup setsid bash {source}/scripts/w32/after_training.sh {source} {PYTHON} > {ROOT}/after_training_main.log 2>&1 < /dev/null & echo $!').strip())
    finally:c.close()
    update(after_training_pid=pid,after_training_watchers=watchers,after_training_deployed=True)
    print(pid,watchers,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('prepare','regression','generate','merge','deploy','observe','after'))
    globals()[parser.parse_args().action]()
