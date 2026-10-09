"""Pinned, isolated four-lane transport for S1-D12A.

Only the approved workbench API may start formal jobs. Credentials stay in
the process environment and are never included in a receipt.
"""
from __future__ import annotations

from pathlib import Path
import json

from scripts.s2p19 import ops as transport


SSH_PORTS = (20418, 20424, 20473, 20474)
API_PORTS = {20418: 8920, 20424: 8921, 20473: 8922, 20474: 8923}
BASE = "/home/u2024110224/first_stage_satellite_rl/s1_d12a"
LOCAL = Path("D:/paper/outputs/experiments/first_stage_s1_d12a_20261010")
DQN = "/home/u2024110224/first_stage_satellite_rl/envs/DQN/bin/python"


def instance(port: int) -> dict:
    if port not in SSH_PORTS:
        raise ValueError("S1-D12A is restricted to the approved four instances")
    root = f"{BASE}/{port}"
    return {"ssh_port": port, "api_port": API_PORTS[port], "root": root,
            "repo": root + "/repo", "archive": root + "/archive",
            "ops": root + "/ops", "data": root + "/data",
            "target": f"u2024110224@10.160.17.102:{port}"}


def connect(port: int):
    instance(port)
    return transport.connect(port)


def remote_python(code: str, port: int, timeout: int = 120) -> dict:
    instance(port)
    return transport.remote_python(code, port, timeout=timeout)


def upload(path: Path, remote_path: str, port: int) -> dict:
    instance(port)
    return transport.upload(path, remote_path, port)


def api(port: int, path: str, body: dict | None = None,
        method: str | None = None, timeout: int = 60) -> dict:
    """Call the instance-local workbench with a caller-supplied request ID."""
    instance(port)
    import re
    if not re.fullmatch(r"/api/[A-Za-z0-9_/-]+", path):
        raise ValueError("invalid S1-D12A API path")
    method = method or ("POST" if body is not None else "GET")
    if method not in ("GET", "POST") or (method == "POST" and body is None) or (method == "GET" and body is not None):
        raise ValueError("invalid S1-D12A API method/body")
    payload = json.dumps(body, ensure_ascii=False) if body is not None else None
    code = f'''\
import json,urllib.error,urllib.request
data={payload!r}
request=urllib.request.Request('http://127.0.0.1:{API_PORTS[port]}{path}',
 data=None if data is None else data.encode('utf-8'), method={method!r},
 headers={{'Content-Type':'application/json','Cache-Control':'no-cache'}})
try:
 with urllib.request.urlopen(request,timeout={timeout}) as response:
  print(json.dumps({{'http_status':response.status,'body':json.load(response)}},ensure_ascii=False))
except urllib.error.HTTPError as error:
 print(json.dumps({{'http_status':error.code,'error':error.read().decode('utf-8','replace')[:3000]}},ensure_ascii=False))
'''
    response = remote_python(code, port, timeout=timeout + 30)
    if response["exit_code"]:
        raise RuntimeError(f"{port} workbench transport failed: {response['stderr'][-1000:]}")
    result = json.loads(response["stdout"].splitlines()[-1])
    if result["http_status"] >= 400:
        raise RuntimeError(f"{port} workbench HTTP {result['http_status']}: {result.get('error')}")
    return result["body"]
