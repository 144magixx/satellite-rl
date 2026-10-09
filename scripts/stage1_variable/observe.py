"""Read one bounded live snapshot of source downloads and S1-V10 workbenches."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json

from scripts.stage1_variable import ops
from scripts.stage1_variable.launch_downloads import ASSIGNMENT


def lane(port):
    role, day = ASSIGNMENT[port]
    source = f"{ops.BASE}/source/{day}"
    code = f'''
import json,pathlib,socket,datetime
root=pathlib.Path({source!r});status=root/'download_status.json';pin=root/'source_preflight.json'
p=json.loads(pin.read_text()) if pin.exists() else None
part=root/(p['archive_name']+'.part') if p else None
archive=root/p['archive_name'] if p else None
with socket.socket() as sock:
 sock.settimeout(2);api=sock.connect_ex(('127.0.0.1',{ops.API_PORTS[port]}))==0
print(json.dumps({{'observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'download':json.loads(status.read_text()) if status.exists() else None,
 'partial_bytes':part.stat().st_size if part and part.exists() else 0,
 'archive_bytes':archive.stat().st_size if archive and archive.exists() else 0,
 'api_listening':api}},ensure_ascii=False))
'''
    result = ops.remote_python(code, port, timeout=30)
    if result["exit_code"]:
        return {"role": role, "date": day, "error": result["stderr"][-500:]}
    value = json.loads(result["stdout"].splitlines()[-1])
    value.update(role=role, date=day)
    if value["api_listening"]:
        try:
            response = ops.api(port, "/api/coverage/runs", timeout=15)
            value["runs"] = [r for r in response.get("body", {}).get("runs", [])
                             if r.get("experiment_id") == "S1-V10"]
        except Exception as error:
            value["api_error"] = str(error)
    return value


def main():
    result = {"observed_utc": datetime.now(timezone.utc).isoformat(), "lanes": {}}
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(lane, p): p for p in ops.SSH_PORTS}
        for future in as_completed(futures):
            port = futures[future]
            try:
                result["lanes"][str(port)] = future.result()
            except Exception as error:
                result["lanes"][str(port)] = {"error": repr(error)}
    path = ops.LOCAL / "preflight" / "live_observation.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({p: {"date": row.get("date"), "download_state":
                           (row.get("download") or {}).get("status"),
                           "bytes": row.get("archive_bytes") or row.get("partial_bytes"),
                           "api_listening": row.get("api_listening"),
                           "error": row.get("error")}
                      for p, row in result["lanes"].items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
