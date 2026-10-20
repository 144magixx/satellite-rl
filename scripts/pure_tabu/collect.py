"""Persistently collect completed evaluation outputs without restarting jobs."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time
import paramiko

REMOTE='/home/u2025140002/s2_pure_tabu_20261021'
DATA='/home/u2025140002/s2fix_20261018/data'
SETS=['conf_d432_n150','dev_d432_n150','dev_d216_n150','dev_d864_n150','dev_d432_n200']


def client():
    c=paramiko.SSHClient();c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect('10.160.17.102',port=20627,username='u2025140002',
              password=os.environ['BUPT_PASSWORD'],allow_agent=False,look_for_keys=False,timeout=20)
    c.get_transport().set_keepalive(20)
    return c


def pull_tree(sftp,remote,local):
    local.mkdir(parents=True,exist_ok=True)
    for entry in sftp.listdir_attr(remote):
        import stat
        if stat.S_ISDIR(entry.st_mode):pull_tree(sftp,remote+'/'+entry.filename,local/entry.filename)
        else:sftp.get(remote+'/'+entry.filename,str(local/entry.filename))


def command(c,cmd):
    _,out,err=c.exec_command(cmd);text=out.read().decode();error=err.read().decode()
    if out.channel.recv_exit_status():raise RuntimeError(error)
    return text


def main():
    p=argparse.ArgumentParser();p.add_argument('--output-dir',required=True);a=p.parse_args()
    local=Path(a.output_dir);local.mkdir(parents=True,exist_ok=True)
    conf_pulled=(local/'confirmation_ready.json').exists()
    while True:
        c=None
        try:
            c=client();s=c.open_sftp()
            with s.open(REMOTE+'/eval/status.json') as stream:status=json.load(stream)
            (local/'remote_status.json').write_text(json.dumps(status),encoding='utf-8')
            print(json.dumps(status),flush=True)
            try:
                s.stat(REMOTE+'/eval/conf_d432_n150/results.csv');conf_ready=True
            except FileNotFoundError:conf_ready=False
            (local/'eval').mkdir(exist_ok=True)
            s.get(REMOTE+'/eval/run_manifest.json',str(local/'eval'/'run_manifest.json'))
            if not conf_pulled and status['completed']>=704 and conf_ready:
                pull_tree(s,REMOTE+'/eval/conf_d432_n150',local/'eval'/'conf_d432_n150')
                pull_tree(s,DATA+'/conf_d432_n150',local/'data'/'conf_d432_n150')
                (local/'confirmation_ready.json').write_text(json.dumps(status),encoding='utf-8')
                conf_pulled=True
                print('CONFIRMATION_READY',flush=True)
            try:
                s.stat(REMOTE+'/eval/completed.json');done=True
            except FileNotFoundError:done=False
            if done:
                archive=REMOTE+'/results.tar.gz'
                data_members=' '.join('data/'+dataset for dataset in SETS)
                command(c,f'tar -czf {archive} -C {REMOTE} eval run.log serial_diagnostic -C /home/u2025140002/s2fix_20261018 {data_members}')
                s.get(archive,str(local/'results.tar.gz'))
                with tarfile.open(local/'results.tar.gz') as tar:
                    for member in tar.getmembers():
                        if not (local/member.name).resolve().is_relative_to(local.resolve()):
                            raise ValueError('archive path outside experiment')
                    tar.extractall(local)
                s.close();c.close();c=None
                for script in ('summarize.py','verify.py'):
                    subprocess.run([sys.executable,str(Path(__file__).with_name(script)),str(local/'eval')],check=True)
                (local/'collected.json').write_text(json.dumps(status),encoding='utf-8')
                print('ALL_COLLECTED_AND_VERIFIED',flush=True)
                return
            s.close();c.close();c=None
        except Exception as exc:print(repr(exc),flush=True)
        finally:
            if c is not None:c.close()
        time.sleep(60)


if __name__=='__main__':main()
