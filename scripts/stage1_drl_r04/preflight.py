"""Read-only four-instance resource and source probe for S1-D12A."""
from __future__ import annotations

from datetime import datetime, timezone
import json

from scripts.stage1_drl_r04.ops import API_PORTS, LOCAL, SSH_PORTS, instance, remote_python


REMOTE = r'''
import datetime, getpass, json, os, pathlib, socket, subprocess, sys
def call(argv, timeout=15):
 try:
  p=subprocess.run(argv,capture_output=True,text=True,timeout=timeout)
  return {"code":p.returncode,"stdout":p.stdout[-20000:],"stderr":p.stderr[-2000:]}
 except Exception as e: return {"error":type(e).__name__+": "+str(e)}
def port_owner(port):
 s=socket.socket()
 try:
  s.bind(('127.0.0.1',port)); return 'free'
 except OSError as e: return 'occupied: '+str(e)
 finally: s.close()
record={"utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
 "host":socket.gethostname(),"user":getpass.getuser(),"executable":sys.executable,
 "prefix":sys.prefix,"cpu_count":os.cpu_count(),
 "dqn_exists":pathlib.Path('/home/u2024110224/first_stage_satellite_rl/envs/DQN/bin/python').is_file(),
 "gpu":call(['nvidia-smi','--query-gpu=index,name,uuid,memory.total,memory.used,utilization.gpu','--format=csv,noheader']),
 "gpu_processes":call(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader']),
 "processes":call(['ps','-eo','pid,lstart,args','--sort=-start_time']),
 "ports":{str(p):port_owner(p) for p in [8920,8921,8922,8923]},
 "listen":call(['ss','-ltnp']),
 "disk":call(['df','-hT','/home/u2024110224']),
 "quota":call(['quota','-s'],timeout=5),
 "mount":call(['findmnt','-T','/home/u2024110224','-o','SOURCE,FSTYPE,TARGET,OPTIONS']),
 "existing_lane":str(pathlib.Path('/home/u2024110224/first_stage_satellite_rl/s1_d12a').exists())}
try:
 import torch, satellite_rl
 record['torch']={"version":torch.__version__,"cuda_build":torch.version.cuda,
  "cuda_available":torch.cuda.is_available(),
  "name":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
  "satellite_rl_path":list(satellite_rl.__path__)}
except Exception as e: record['torch_error']=type(e).__name__+': '+str(e)
print(json.dumps(record,ensure_ascii=False))
'''


def main() -> None:
    rows = {}
    for port in SSH_PORTS:
        try:
            result = remote_python(REMOTE, port, timeout=65)
            rows[str(port)] = {"target": instance(port)["target"],
                               "planned_api_port": API_PORTS[port],
                               "transport_exit_code": result["exit_code"],
                               "probe": json.loads(result["stdout"].splitlines()[-1])
                               if result["exit_code"] == 0 else None,
                               "stderr": result["stderr"][-2000:]}
        except Exception as exc:
            rows[str(port)] = {"target": instance(port)["target"],
                               "error": type(exc).__name__ + ": " + str(exc)}
    output = {"experiment_id": "S1-D12A", "observed_utc": datetime.now(timezone.utc).isoformat(),
              "read_only": True, "instances": rows}
    destination = LOCAL / "preflight" / "remote_probe.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"path": str(destination), "summary": {
        port: {"error": row.get("error"),
               "device": (row.get("probe") or {}).get("torch", {}).get("name"),
               "api_port": (row.get("probe") or {}).get("ports", {}).get(str(API_PORTS[int(port)]))}
        for port, row in rows.items()}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
