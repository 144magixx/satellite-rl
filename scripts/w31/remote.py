"""Deploy W31 once, or take a read-only progress/GPU snapshot."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile

import paramiko

ROOT = '/home/u2025140002/second_stage_satellite_rl/s2_w31'
PYTHON = '/home/u2025140002/second_stage_satellite_rl/envs/DQN/bin/python'
RESULT = Path('D:/paper/outputs/experiments/s2_w31_20261017/codex/task1_result.json')


def connect(port):
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect('10.160.17.102', port=port, username='u2025140002',
                   password=os.environ['BUPT_PASSWORD'], timeout=20)
    return client


def run(client, command):
    _, out, err = client.exec_command(command, timeout=120)
    output, error = out.read().decode(), err.read().decode()
    if out.channel.recv_exit_status():
        raise RuntimeError(error or output)
    return output


def deploy():
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    source = ROOT + '/source/' + commit
    with tempfile.TemporaryDirectory() as temp:
        archive = Path(temp) / 'source.tar'
        subprocess.run(['git', '-c', 'core.autocrlf=false', 'archive', '--format=tar', '-o', str(archive), commit,
                        'src', 'scripts/w31', 'pyproject.toml'], check=True)
        client = connect(20627)
        try:
            run(client, f'mkdir -p {source}')
            with client.open_sftp() as sftp:
                sftp.put(str(archive), ROOT + '/source_upload.tar')
            run(client, f'tar -xf {ROOT}/source_upload.tar -C {source}; rm {ROOT}/source_upload.tar')
        finally:
            client.close()

    def lane(port):
        client = connect(port)
        try:
            # This is the only interpreter/CUDA requirement for the new lane.
            cuda = run(client, PYTHON + ' -c ' + shlex.quote(
                'import torch; assert torch.cuda.is_available(); print(torch.__version__)')).strip()
            output = run(client, f'bash {source}/scripts/w31/launch_lane.sh {source} {PYTHON} {port}')
            return port, cuda, output
        finally:
            client.close()

    groups = {}
    with ThreadPoolExecutor(2) as pool:
        for port, cuda, output in pool.map(lane, (20627, 20653)):
            print(port, cuda, output, flush=True)
            for line in output.splitlines():
                arm, pid, _, directory = line.split()
                groups[arm] = dict(pid=int(pid), port=port, output_dir=directory)
    value = dict(commit=commit, remote_source=source, interpreter=PYTHON,
        launched_at=datetime.now(timezone.utc).isoformat(), groups=groups,
        local_self_check=dict(arms=8, steps_per_arm=317, explicit_updates_per_arm=3,
            exact_resume=True, final_policy_saved=True, evaluation_matches_W30=True,
            power_menus_valid=True, frozen_configurations_match=True,
            scientific_results=False), issues=[
                'py310 lacks torch; stock torch2.1.1 lacks tuple-dimension Tensor.any used by frozen science. Selected existing shared DQN torch2.3.1+cu121 environment on both ports; failed workers stopped at env_step=0.',
                'Disabled Git archive autocrlf conversion after shell rejected CRLF; failed deployment attempts made no scientific updates.'])
    RESULT.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def observe():
    value = json.loads(RESULT.read_text(encoding='utf-8'))

    def lane(port):
        client = connect(port)
        try:
            gpu = run(client, 'nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,utilization.memory --format=csv,noheader,nounits').strip()
            processes = run(client, 'nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits').strip()
            with client.open_sftp() as sftp:
                rows = {}
                for arm, group in value['groups'].items():
                    if group['port'] != port:
                        continue
                    with sftp.open(group['output_dir'] + '/progress.json') as stream:
                        rows[arm] = json.load(stream)
                    run(client, 'kill -0 ' + str(group['pid']))
            return port, dict(gpu=gpu, processes=processes), rows
        finally:
            client.close()

    previous = {arm: data.get('env_step') for arm, data in value['groups'].items()}
    gpus = {}
    now = datetime.now(timezone.utc)
    with ThreadPoolExecutor(2) as pool:
        for port, gpu, rows in pool.map(lane, (20627, 20653)):
            gpus[str(port)] = gpu
            for arm, progress in rows.items():
                value['groups'][arm].update(progress)
                old = previous[arm]
                if old is not None:
                    value['groups'][arm]['step_increase_since_previous_observation'] = progress['env_step'] - old
                speed = progress['steps_per_second']
                # Approximate training ETA; evaluation time has not been measured yet.
                if speed > 0:
                    value['groups'][arm]['estimated_training_completion'] = (now + timedelta(
                        seconds=(150000-progress['env_step'])/speed)).isoformat()
    value['gpu_observation'] = gpus
    value['observed_at'] = now.isoformat()
    value['seconds_since_launch'] = (now-datetime.fromisoformat(value['launched_at'])).total_seconds()
    value['estimated_completion_note'] = 'Measured training throughput extrapolation; evaluation time additional.'
    RESULT.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(value, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('deploy', 'observe'))
    args = parser.parse_args()
    deploy() if args.action == 'deploy' else observe()
