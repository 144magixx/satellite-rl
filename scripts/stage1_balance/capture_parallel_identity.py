"""Capture bounded, read-only S1-02 identities before a scheduler handoff."""

from __future__ import annotations

from datetime import datetime
import json

from remote_sources import LOCAL, REMOTE, connect, run_code


CODE = r'''
import datetime,hashlib,json,os,pathlib,socket,subprocess,urllib.request
root=pathlib.Path(__REMOTE__)
ops=root/'ops'
ids={'server':65103,'controller':73769,'X1':138079}
def proc(pid):
 p=pathlib.Path('/proc')/str(pid)
 if not p.exists():return {'pid':pid,'exists':False}
 raw=(p/'stat').read_text();parts=raw[raw.rfind(')')+2:].split()
 argv=[v.decode('utf-8','replace') for v in (p/'cmdline').read_bytes().split(bytes([0])) if v]
 out={'pid':pid,'exists':True,'state':parts[0],'ppid':int(parts[1]),
      'start_ticks':int(parts[19]),'argv':argv,
      'wchan':(p/'wchan').read_text().strip(),'cwd':str((p/'cwd').resolve()),
      'exe':str((p/'exe').resolve()),'pgid':os.getpgid(pid),'sid':os.getsid(pid)}
 return out
def sha(path):
 path=pathlib.Path(path)
 return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
def read(path):
 path=pathlib.Path(path)
 return json.loads(path.read_text()) if path.is_file() else None
def job(run_id):
 p=root/'outputs/workbench/jobs'/run_id
 return {'run_id':run_id,'job':read(p/'job.json'),'status':read(p/'status.json'),
         'launch':read(p/'launch.json'),'config_sha256':sha(p/'config.json')}
with urllib.request.urlopen('http://127.0.0.1:8892/api/system',timeout=12) as r:
 system=json.load(r)
gpu=subprocess.run(['nvidia-smi','--query-gpu=uuid,name,memory.total,memory.used,utilization.gpu',
                    '--format=csv,noheader'],capture_output=True,text=True,timeout=12)
route=read(ops/'current_controller_route.json')
plan=ops/'controller_plan_recovery.json'
analysis=ops/'analysis_override/57ceffac9162578e0c924f551466d408a80fc527'
out={'observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'processes':{name:proc(pid) for name,pid in ids.items()},
 'route':route,'plan_sha256':sha(plan),'controller_state_sha256':sha(ops/'controller_state_recovery.json'),
 'controller_state':read(ops/'controller_state_recovery.json'),
 'old_service_receipt':read(ops/'workbench_recovery_launch.json'),
 'new_service_pointer':read(ops/'current_workbench_service.json'),
 'analysis_sha256':{name:sha(analysis/name) for name in
                    ('audit_stage1_balance.py','report_stage1_balance.py','select_stage1_menus.py')},
 'repo_head':subprocess.run(['git','-C',str(root/'repo'),'rev-parse','HEAD'],
                            capture_output=True,text=True).stdout.strip(),
 'system':system,'gpu':gpu.stdout.strip(),
 'X1':job('train_ppo_20261004_070900_2918897c'),
 'X4':job('train_ppo_20261004_070900_44af22cb')}
print(json.dumps(out,ensure_ascii=False))
'''


def main() -> None:
    code = CODE.replace("__REMOTE__", repr(REMOTE))
    client = connect()
    try:
        result = run_code(client, code)
    finally:
        client.close()
    stamp = datetime.fromisoformat(result["observed_utc"]).strftime("%Y%m%dT%H%M%SZ")
    path = LOCAL / "preflight" / ("parallel_switch_before_" + stamp + ".json")
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = {"path": str(path), "observed_utc": result["observed_utc"],
               "processes": {key: {item: val.get(item) for item in
                            ("exists", "state", "start_ticks", "wchan", "sid")}
                            for key, val in result["processes"].items()},
               "X1_stage": (result["X1"]["status"] or {}).get("stage"),
               "X1_step": (result["X1"]["status"] or {}).get("env_step"),
               "X4_stage": (result["X4"]["status"] or {}).get("stage"),
               "gpu": result["gpu"]}
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
