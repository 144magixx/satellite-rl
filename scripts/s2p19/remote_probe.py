"""Read-only identity/resource/port probe for the two approved SSH instances.

Writes no credential or remote file. Uses S2-P18's pinned transport only as a
connection implementation; it does not touch that experiment's tasks.
"""

from __future__ import annotations

import json
from pathlib import Path

from scripts.variable_beams import ops18


OUTPUT = Path("D:/paper/outputs/experiments/sac_routes_s2p19_20261007/preflight/remote_probe.json")
REMOTE = r'''
import datetime,json,os,pathlib,socket,subprocess,sys
def run(args):
 try:
  p=subprocess.run(args,text=True,capture_output=True,timeout=12)
  return {"exit_code":p.returncode,"stdout":p.stdout[:12000],"stderr":p.stderr[:1200]}
 except Exception as exc: return {"error":type(exc).__name__+": "+str(exc)[:200]}
def port_state(port):
 s=socket.socket();s.settimeout(.25)
 try: return "listening" if s.connect_ex(("127.0.0.1",port))==0 else "free"
 finally:s.close()
try:
 import torch,satellite_rl
 torch_info={"version":torch.__version__,"cuda_build":torch.version.cuda,
  "cuda_available":torch.cuda.is_available(),"gpu_count":torch.cuda.device_count(),
  "gpu_name":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
  "package_file":str(pathlib.Path(satellite_rl.__file__).resolve())}
except Exception as exc: torch_info={"error":type(exc).__name__+": "+str(exc)[:200]}
processes=run(["ps","-eo","pid,lstart,args"])
if "stdout" in processes:
 lines=processes["stdout"].splitlines()
 processes["stdout"]="\n".join([line for line in lines if any(key in line for key in
  ("workbench", "controller", "train_sac", "evaluate_sac", "refine_sac"))][:60])
print(json.dumps({"observed_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
 "hostname":socket.gethostname(),"executable":sys.executable,"python_version":sys.version.split()[0],
 "torch":torch_info,"nvidia_smi":run(["nvidia-smi","--query-gpu=uuid,name,memory.total,memory.used,utilization.gpu","--format=csv,noheader,nounits"]),
 "ports":{str(p):port_state(p) for p in (8781,8896,8897,8898,8899,8900,8901)},
 "processes":processes,
 "archive_mount_free_bytes":__import__('shutil').disk_usage('/home/u2024110224').free,
 "s2p19_root_exists":pathlib.Path('/home/u2024110224/second_stage_satellite_rl/sac_routes_s2p19').exists()},ensure_ascii=False))
'''


def main() -> None:
    payload = {"probe_kind": "read_only", "target": "u2024110224@10.160.17.102",
               "instances": {}}
    for port in (20418, 20424):
        try:
            result = ops18.remote_python(REMOTE, port, timeout=45)
            if result["exit_code"]:
                payload["instances"][str(port)] = {
                    "error": result["stderr"][-1000:], "exit_code": result["exit_code"]}
            else:
                payload["instances"][str(port)] = json.loads(result["stdout"].splitlines()[-1])
        except Exception as exc:
            payload["instances"][str(port)] = {"error": type(exc).__name__ + ": " + str(exc)[:200]}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for port, detail in payload["instances"].items():
        print(port, {key: detail.get(key) for key in ("observed_utc", "executable", "torch", "ports", "error")})


if __name__ == "__main__":
    main()
