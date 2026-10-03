"""Stop only the verified idle 77d S1-02 server before recovery deployment."""

from __future__ import annotations

from datetime import datetime, timezone
import json

from remote_sources import LOCAL, REMOTE, connect, run_code


OLD_COMMIT = "77d18aac49e36e0b4490d82689230d9aa9bbee52"


def main() -> None:
    code = r'''
import datetime,json,os,pathlib,signal,socket,subprocess,time
root=pathlib.Path(__REMOTE__);ops=root/'ops';old=__COMMIT__
launch=json.loads((ops/'workbench_launch.json').read_text())
state=json.loads((ops/'controller_state.json').read_text())
controller=json.loads((ops/'controller_launch.json').read_text())
if (launch['commit']!=old or launch['port']!=8892 or
    launch['archive_root']!=str(root) or state['phase']!='needs_attention'):
 raise RuntimeError('Initial source/service/controller state differs')
prior=pathlib.Path('/proc')/str(controller['pid'])
if prior.exists():
 stat=(prior/'stat').read_text();ticks=int(stat[stat.rfind(')')+2:].split()[19])
 command=(prior/'cmdline').read_bytes().decode('utf-8','replace')
 if ticks==controller['start_ticks'] and 'controller.py' in command:
  raise RuntimeError('Original failed controller still running')
active={'queued','starting','initializing','warmup','training','validating',
        'running','evaluating','preparing'}
jobs=[]
for path in (root/'outputs/workbench/jobs').glob('*/status.json'):
 value=json.loads(path.read_text());jobs.append({'run_id':path.parent.name,'stage':value.get('stage')})
if any(item['stage'] in active for item in jobs):
 raise RuntimeError('Old workbench still owns active/queued work')
gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory',
 '--format=csv,noheader'],text=True).strip()
if gpu:raise RuntimeError('GPU compute process is still active on 20418')
proc=pathlib.Path('/proc')/str(launch['pid'])
if not proc.exists():raise RuntimeError('Original server PID absent; inspect before changing Git')
stat=(proc/'stat').read_text();ticks=int(stat[stat.rfind(')')+2:].split()[19])
args=[part.decode('utf-8','replace') for part in
      (proc/'cmdline').read_bytes().split(bytes([0])) if part]
if (ticks!=launch['start_ticks'] or 'satellite_rl.workbench.server' not in args or
    '--port' not in args or args[args.index('--port')+1]!='8892' or
    (proc/'cwd').resolve()!=root/'repo'):
 raise RuntimeError('Original server PID/start/cmdline/cwd differs')
os.kill(launch['pid'],signal.SIGTERM)
for _ in range(50):
 time.sleep(.2)
 with socket.socket() as probe:
  probe.settimeout(.2)
  listening=probe.connect_ex(('127.0.0.1',8892))==0
 if not listening:break
else:raise RuntimeError('Original server did not release 8892 after SIGTERM')
receipt={'status':'verified_initial_workbench_stopped','stopped_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'old_commit':old,'pid':launch['pid'],'start_ticks':ticks,'port':8892,
 'controller_phase':state['phase'],'job_stages':jobs,'gpu_compute_apps':gpu,
 'old_launch_sha256':__import__('hashlib').sha256((ops/'workbench_launch.json').read_bytes()).hexdigest()}
target=ops/'workbench_initial_stop.json'
if target.exists():raise RuntimeError('Initial stop receipt exists')
target.write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(receipt,ensure_ascii=False))
'''
    code = code.replace("__REMOTE__", repr(REMOTE)).replace("__COMMIT__", repr(OLD_COMMIT))
    client = connect()
    try:
        receipt = run_code(client, code)
    finally:
        client.close()
    output = LOCAL / "preflight/workbench_initial_stop.json"
    output.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"receipt": str(output), "status": receipt["status"],
                      "checked_utc": datetime.now(timezone.utc).isoformat()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
