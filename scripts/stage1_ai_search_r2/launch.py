"""Launch one finite R2 controller from an immutable Git source snapshot."""
import argparse,hashlib,io,json,os,shutil,subprocess,sys,zipfile
from pathlib import Path
from satellite_rl.training.run_monitor import atomic_json
from satellite_rl.workbench.runtime import pid_alive,process_start_time
from satellite_rl.workbench.geoload_runner import exclusive_lock
from . import ops
from .deploy import SOURCE,git
from .plan_builder import digest


def launch(commit,spec_path,state_root,python=sys.executable):
    spec_path=Path(spec_path).resolve();raw=spec_path.read_bytes();spec=json.loads(raw)
    if git('rev-parse','HEAD')!=commit or git('status','--porcelain','--','src','scripts','tests','pyproject.toml'):
        raise ValueError('control launch requires clean explicit root-frozen commit')
    if spec.get('group')!=ops.GROUP or spec.get('scientific_commit')!=commit or spec.get('automatic_extension')is not False:
        raise ValueError('finite no-training R2 spec required')
    review=spec['root_review'];review_value=json.loads(Path(review['path']).read_text(encoding='utf-8'))
    if digest(review['path'])!=review['sha256'] or review_value.get('passed')is not True or review_value.get('plan_sha256')!=spec['plan']['sha256']:
        raise ValueError('root plan review changed')
    root=Path(state_root).resolve();root.mkdir(parents=True,exist_ok=True);registry=root/'launch.json'
    with exclusive_lock(root/'launch.lock',blocking=True):
        if registry.exists():
            old=json.loads(registry.read_text(encoding='utf-8'));state=json.loads((root/'state.json').read_text(encoding='utf-8')) if (root/'state.json').exists() else {}
            if old['spec_sha256']!=hashlib.sha256(raw).hexdigest() or old['source_commit']!=commit:raise ValueError('state already owns another controller epoch/spec')
            if pid_alive(old['pid'],old['process_start_time']) or state.get('phase') in ('completed','attention_required'):
                return old
            raise ValueError('original controller exited without known terminal state; preserve receipt and recover explicitly')
        if (root/'launch_started.json').exists():raise ValueError('unknown launch outcome: original intent retained; audit before restart')
        if shutil.disk_usage(root).free<8*(1<<30):raise ValueError('8GiB control disk reserve')
        snapshot=ops.LOCAL/'control_sources'/commit;snapshot.mkdir(parents=True,exist_ok=True);manifest_path=snapshot/'control_manifest.json'
        if not manifest_path.exists():
            data=subprocess.run(['git','-C',str(SOURCE),'archive','--format=zip',commit,'src','scripts','pyproject.toml'],capture_output=True,check=True).stdout
            with zipfile.ZipFile(io.BytesIO(data)) as bundle:
                if any(not (snapshot/x.filename).resolve().is_relative_to(snapshot.resolve()) for x in bundle.infolist()):raise ValueError('snapshot path escapes')
                bundle.extractall(snapshot)
            atomic_json(manifest_path,{'commit':commit,'archive_sha256':hashlib.sha256(data).hexdigest(),
                'files':{p.relative_to(snapshot).as_posix():digest(p) for p in snapshot.rglob('*') if p.is_file()}})
        manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
        if manifest['commit']!=commit or any(digest(snapshot/p)!=sha for p,sha in manifest['files'].items()):raise ValueError('frozen control source changed')
        frozen_spec=root/'spec.json'
        if frozen_spec.exists() and frozen_spec.read_bytes()!=raw:raise ValueError('state frozen spec differs')
        frozen_spec.write_bytes(raw)
        cmd=[str(python),'-B','-u','-m','scripts.stage1_ai_search_r2.controller','--spec',str(frozen_spec),'--state-root',str(root)]
        atomic_json(root/'launch_started.json',{'command':cmd,'cwd':str(snapshot),'source_commit':commit,'spec_sha256':hashlib.sha256(raw).hexdigest()})
        env=os.environ.copy();env.update(PYTHONPATH=str(snapshot/'src'),PYTHONDONTWRITEBYTECODE='1')
        flags=subprocess.CREATE_NO_WINDOW|subprocess.CREATE_NEW_PROCESS_GROUP if os.name=='nt' else 0
        with (root/'controller.stdout.log').open('ab',buffering=0) as out,(root/'controller.stderr.log').open('ab',buffering=0) as err:
            child=subprocess.Popen(cmd,cwd=snapshot,env=env,stdin=subprocess.DEVNULL,stdout=out,stderr=err,
                close_fds=True,start_new_session=os.name!='nt',creationflags=flags)
        receipt={'group':ops.GROUP,'pid':child.pid,'process_start_time':process_start_time(child.pid),'command':cmd,'cwd':str(snapshot),
            'source_commit':commit,'manifest_sha256':digest(manifest_path),'spec_sha256':hashlib.sha256(raw).hexdigest(),'source_immutable':True,'science_direct_calls':0}
        atomic_json(registry,receipt);return receipt


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('commit','spec','state-root'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--python',default=sys.executable);args=parser.parse_args()
    print(json.dumps(launch(args.commit,args.spec,args.state_root,args.python)))
