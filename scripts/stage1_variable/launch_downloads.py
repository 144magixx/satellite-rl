"""Stage frozen source contracts and start four resumable raw downloads."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path

from scripts.stage1_variable import ops


ASSIGNMENT = {20418: ("validation", "2024-02-03"),
              20424: ("test", "2024-02-04"),
              20473: ("test", "2024-02-05"),
              20474: ("sequence", "2024-02-06")}
FILES = ("release_api.json", "PREFERRED_RELEASES.txt",
         "LICENSE-ODbL-official.txt", "selection_contract.json",
         "source_preflight.json")
SCRIPT = Path(__file__).with_name("download_source.py")


def _remote(code, port, timeout=120):
    result = ops.remote_python(code, port, timeout=timeout)
    if result["exit_code"]:
        raise RuntimeError(f"{port}: {result['stderr'][-1000:]}")
    return json.loads(result["stdout"].splitlines()[-1])


def launch(port):
    role, day = ASSIGNMENT[port]
    local = ops.LOCAL / "source" / role / day
    pin = json.loads((local / "source_preflight.json").read_text(encoding="utf-8"))
    remote = f"{ops.BASE}/source/{day}"
    _remote(f"import json,pathlib;p=pathlib.Path({remote!r});p.mkdir(parents=True,exist_ok=True);print(json.dumps({{'path':str(p)}}))", port)
    checks = {}
    for name in FILES:
        target = f"{remote}/{name}"
        receipt = ops.upload(local / name, target, port)
        expected = hashlib.sha256((local / name).read_bytes()).hexdigest()
        if receipt["remote"] != expected:
            raise RuntimeError(f"{port}: source metadata SHA differs: {name}")
        checks[name] = expected
    script_path = remote + "/download_source.py"
    script_receipt = ops.upload(SCRIPT, script_path, port)
    if script_receipt["remote"] != hashlib.sha256(SCRIPT.read_bytes()).hexdigest():
        raise RuntimeError(f"{port}: downloader SHA differs")
    code = f'''
import json,pathlib,subprocess,datetime
root=pathlib.Path({remote!r});pin=json.loads((root/'source_preflight.json').read_text())
if pin['asset_id']!={pin['asset_id']} or pin['date_utc']!={day!r}:
 raise RuntimeError('frozen asset differs')
status=root/'download_status.json';receipt=root/'download_launch.json'
if status.exists() and json.loads(status.read_text()).get('status')=='completed':
 print(json.dumps({{'state':'already_completed','port':{port},'asset_id':pin['asset_id']}}));raise SystemExit
if receipt.exists():
 old=json.loads(receipt.read_text());p=pathlib.Path('/proc')/str(old.get('pid'))
 if p.exists():
  raw=(p/'stat').read_text();ticks=int(raw[raw.rfind(')')+2:].split()[19])
  if ticks==old.get('start_ticks'):
   print(json.dumps({{'state':'already_running','port':{port},'receipt':old}}));raise SystemExit
log=(root/'download.log').open('ab',buffering=0)
child=subprocess.Popen([{ops.DQN!r},'-B',{script_path!r},'--data-dir',str(root)],
 stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
 cwd=str(root),start_new_session=True,close_fds=True)
raw=(pathlib.Path('/proc')/str(child.pid)/'stat').read_text()
ticks=int(raw[raw.rfind(')')+2:].split()[19])
value={{'state':'launched','port':{port},'pid':child.pid,'start_ticks':ticks,
 'asset_id':pin['asset_id'],'expected_bytes':pin['expected_bytes'],
 'observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'source_preflight_sha256':{checks['source_preflight.json']!r},
 'downloader_sha256':{script_receipt['remote']!r}}}
receipt.write_text(json.dumps(value,indent=2)+'\\n')
print(json.dumps(value))
'''
    started = _remote(code, port)
    return {"port": port, "role": role, "date": day, "remote": remote,
            "asset_id": pin["asset_id"], "metadata_sha256": checks,
            "downloader_sha256": script_receipt["remote"], "launch": started}


def main():
    rows = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(launch, port): port for port in ASSIGNMENT}
        for future in as_completed(futures):
            port = futures[future]
            try:
                rows[str(port)] = future.result()
            except Exception as error:
                rows[str(port)] = {"error": repr(error)}
    target = ops.LOCAL / "preflight" / "source_download_launch.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
