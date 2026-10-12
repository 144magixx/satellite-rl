"""Freeze control source, then launch one detached bounded-round controller."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile
from satellite_rl.training.run_monitor import atomic_json
from satellite_rl.workbench.runtime import pid_alive,process_start_time
from . import ops


def launch(commit,spec_path,state_root,python=sys.executable):
    source=Path(__file__).resolve().parents[2];spec_path=Path(spec_path).resolve();state_root=Path(state_root).resolve()
    actual=subprocess.run(['git','-C',str(source),'rev-parse','HEAD'],capture_output=True,text=True,check=True).stdout.strip()
    dirty=subprocess.run(['git','-C',str(source),'status','--porcelain','--','src','scripts'],capture_output=True,text=True,check=True).stdout.strip()
    if actual!=commit or dirty:raise ValueError('control launch requires clean explicit frozen source commit')
    spec=json.loads(spec_path.read_text(encoding='utf-8'))
    if spec.get('scientific_commit')!=commit:raise ValueError('spec/control scientific epoch differs')
    state_root.mkdir(parents=True,exist_ok=True);registry=state_root/'launch.json'
    if registry.exists():
        old=json.loads(registry.read_text(encoding='utf-8'))
        if pid_alive(old['pid'],old['process_start_time']):
            if old['spec_sha256']!=hashlib.sha256(spec_path.read_bytes()).hexdigest():raise ValueError('active controller owns another spec')
            return old
    if shutil.disk_usage(state_root).free < 8*(1<<30):raise ValueError('control archive volume lacks8GiB reserve')
    snapshot=ops.LOCAL/'control_sources'/commit;snapshot.mkdir(parents=True,exist_ok=True)
    manifest_path=snapshot/'control_manifest.json'
    if not manifest_path.is_file():
        data=subprocess.run(['git','-C',str(source),'archive','--format=zip',commit,'src','scripts','pyproject.toml'],
                            capture_output=True,check=True).stdout
        with zipfile.ZipFile(io.BytesIO(data)) as package:
            for entry in package.infolist():
                target=(snapshot/entry.filename).resolve()
                if not target.is_relative_to(snapshot.resolve()):raise ValueError('control archive path escapes')
            package.extractall(snapshot)
        manifest={'commit':commit,'archive_sha256':hashlib.sha256(data).hexdigest(),
                  'files':{p.relative_to(snapshot).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in snapshot.rglob('*') if p.is_file()}}
        atomic_json(manifest_path,manifest)
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    for relative,expected in manifest['files'].items():
        if hashlib.sha256((snapshot/relative).read_bytes()).hexdigest()!=expected:raise ValueError('frozen control source changed')
    frozen_spec=state_root/'spec.json';data=spec_path.read_bytes()
    if frozen_spec.exists() and frozen_spec.read_bytes()!=data:raise ValueError('state root already owns a different spec')
    frozen_spec.write_bytes(data)
    env=os.environ.copy();env['PYTHONPATH']=str(snapshot/'src');env['PYTHONDONTWRITEBYTECODE']='1'
    cmd=[str(python),'-B','-u','-m','scripts.stage1_ai_search.controller','--spec',str(frozen_spec),'--state-root',str(state_root)]
    flags=subprocess.CREATE_NO_WINDOW|subprocess.CREATE_NEW_PROCESS_GROUP if os.name=='nt' else 0
    with (state_root/'controller.stdout.log').open('ab',buffering=0) as out,(state_root/'controller.stderr.log').open('ab',buffering=0) as err:
        child=subprocess.Popen(cmd,cwd=snapshot,env=env,stdin=subprocess.DEVNULL,stdout=out,stderr=err,
            close_fds=True,start_new_session=os.name!='nt',creationflags=flags)
    value={'group':'S1-AI17','pid':child.pid,'process_start_time':process_start_time(child.pid),'command':cmd,
           'cwd':str(snapshot),'source_commit':commit,'manifest_sha256':hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
           'spec_sha256':hashlib.sha256(data).hexdigest(),'source_immutable':True,'science_direct_calls':0}
    atomic_json(registry,value);return value


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--commit',required=True)
    parser.add_argument('--spec',required=True);parser.add_argument('--state-root',required=True);parser.add_argument('--python',default=sys.executable)
    args=parser.parse_args();print(json.dumps(launch(args.commit,args.spec,args.state_root,args.python),ensure_ascii=False))
