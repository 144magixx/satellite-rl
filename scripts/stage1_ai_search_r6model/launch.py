"""Launch one persistent controller from a verified immutable Git export."""
import argparse,hashlib,json,os,subprocess,sys,zipfile
from pathlib import Path
from satellite_rl.training.run_monitor import atomic_json
from satellite_rl.workbench.geoload_runner import exclusive_lock
from satellite_rl.workbench.runtime import pid_alive,process_start_time
from .deploy import git,SOURCE
from .plan_builder import digest

def launch(spec_path,state_root):
 spec_path=Path(spec_path).resolve();state=Path(state_root).resolve();state.mkdir(parents=True,exist_ok=True);spec=json.loads(spec_path.read_text('utf-8'));commit=spec['scientific_commit'];record=state/'controller_launch.json'
 with exclusive_lock(state/'launch.lock'):
  if record.exists():
   old=json.loads(record.read_text('utf-8'))
   if old['spec_sha256']!=digest(spec_path)or old['source_commit']!=commit:raise ValueError('other controller owns launch')
   if pid_alive(old['pid'],old['process_start_time']):return old
   raise ValueError('original controller exited; retain record and use a new explicitly reviewed attempt')
  if git('rev-parse','HEAD')!=commit or git('status','--porcelain','--','src','scripts','tests','.gitattributes'):raise ValueError('clean frozen source required')
  root=state/'control_sources'/commit;bundle=state/'control_source.zip'
  subprocess.run(['git','-C',str(SOURCE),'archive','--format=zip','--output='+str(bundle),commit],check=True)
  if root.exists():raise ValueError('unregistered prior control export; preserve and audit')
  with zipfile.ZipFile(bundle)as z:z.extractall(root)
  manifest={p.relative_to(root).as_posix():digest(p)for p in root.rglob('*')if p.is_file()};atomic_json(state/'control_source_manifest.json',{'commit':commit,'files':manifest,'bundle_sha256':digest(bundle)})
  log=(state/'controller.log').open('ab',buffering=0);env=os.environ.copy();env.update(PYTHONPATH=str(root/'src'),PYTHONDONTWRITEBYTECODE='1');command=[sys.executable,'-B','-u','-m','scripts.stage1_ai_search_r6model.controller','--spec',str(spec_path),'--state-root',str(state)]
  child=subprocess.Popen(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt'else 0,start_new_session=os.name!='nt');birth=process_start_time(child.pid);r={'pid':child.pid,'process_start_time':birth,'command':command,'cwd':str(root),'source_commit':commit,'source_manifest_sha256':digest(state/'control_source_manifest.json'),'spec_sha256':digest(spec_path)};atomic_json(record,r);return r
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--spec',required=True);p.add_argument('--state-root',required=True);a=p.parse_args();print(json.dumps(launch(a.spec,a.state_root)))