"""Read-only 20424 workbench/controller snapshot with a local manifest mirror."""
from __future__ import annotations

import argparse
import json

from ops_common import (ARCHIVE, DQN, LOCAL, OPS, REPO, WORKBENCH_PORT,
                        require_commit, remote_python, utc_now, write_json)


def observe(commit: str) -> dict:
    code = f'''
import datetime,hashlib,json,pathlib,subprocess,urllib.request
repo=pathlib.Path({REPO!r});archive=pathlib.Path({ARCHIVE!r});ops=pathlib.Path({OPS!r})
manifest_path=ops/'controller_manifest.json';launch_path=ops/'controller_launch.json';server_path=ops/'workbench_8781_launch.json'
def birth(pid):
 if type(pid) is not int:return None
 try:
  raw=(pathlib.Path('/proc')/str(pid)/'stat').read_text()
  return int(raw[raw.rfind(')')+2:].split()[19])
 except (OSError,ValueError,IndexError):return None
def api(path):
 request=urllib.request.Request(f'http://127.0.0.1:{WORKBENCH_PORT}/api'+path,
  headers={{'Cache-Control':'no-cache'}})
 with urllib.request.urlopen(request,timeout=8) as response:return json.load(response)
manifest=json.loads(manifest_path.read_text()) if manifest_path.is_file() else None
launch=json.loads(launch_path.read_text()) if launch_path.is_file() else None
server=json.loads(server_path.read_text()) if server_path.is_file() else None
out={{'checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'target':'u2024110224@10.160.17.102:20424','manifest':manifest,
 'manifest_sha256':hashlib.sha256(manifest_path.read_bytes()).hexdigest() if manifest else None,
 'controller_launch':launch,'controller_birth_matches':birth(launch['pid'])==launch['start_ticks'] if launch else None,
 'server_launch':server,'server_birth_matches':birth(server['pid'])==server['start_ticks'] if server else None,
 'jobs':{{}}}}
try:
 system=api('/system')
 out['system']={{k:system.get(k) for k in ('source_root','archive_root','python','package_file','cuda','cuda_build','gpu_name')}}
 if manifest:
  for key,item in manifest['jobs'].items():
   if not item.get('run_id'):continue
   detail=api('/runs/'+item['run_id']);job=detail['job'];status=detail['status']
   out['jobs'][key]={{'run_id':item['run_id'],'request_id':job.get('request_id'),
    'stage':status.get('stage'),'env_step':status.get('env_step'),
    'budget_env_steps':status.get('budget_env_steps'),'worker_pid':status.get('pid'),
    'worker_birth_ticks':birth(status.get('pid'))}}
except Exception as exc:out['api_error']=type(exc).__name__+': '+str(exc)[:400]
head=subprocess.run(['git','-C',str(repo),'rev-parse','HEAD'],capture_output=True,text=True)
out['repo_head']=head.stdout.strip() if head.returncode==0 else None
out['commit_matches']=out['repo_head']=={commit!r}
print(json.dumps(out,ensure_ascii=False))
'''
    result = remote_python(code, timeout=90)
    if result["manifest"] is not None:
        write_json(LOCAL / "controller_manifest.json", result["manifest"])
    path = LOCAL / "controller_observation.json"
    write_json(path, result)
    return {"receipt": str(path), "checked_utc": result["checked_utc"],
            "phase": (result["manifest"] or {}).get("phase"),
            "controller_birth_matches": result["controller_birth_matches"],
            "server_birth_matches": result["server_birth_matches"],
            "jobs": result["jobs"], "api_error": result.get("api_error")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--commit", required=True)
    print(json.dumps(observe(require_commit(parser.parse_args().commit)), ensure_ascii=False))


if __name__ == "__main__":
    main()
