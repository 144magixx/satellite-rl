"""Pinned four-lane SSH transport and read-only workbench observation for S1-V10."""
from __future__ import annotations

import json
from pathlib import Path

from scripts.s2p19 import ops as transport


SSH_PORTS = (20418, 20424, 20473, 20474)
API_PORTS = {20418: 8910, 20424: 8911, 20473: 8912, 20474: 8913}
BASE = "/home/u2024110224/first_stage_satellite_rl/s1_v10"
DQN = "/home/u2024110224/first_stage_satellite_rl/envs/DQN/bin/python"
LOCAL = Path("D:/paper/outputs/experiments/first_stage_s1_v10_20261009")


def instance(port: int) -> dict:
    if port not in SSH_PORTS:
        raise ValueError("S1-V10 permits only four approved SSH ports")
    root = f"{BASE}/{port}"
    return {"port": port, "api_port": API_PORTS[port], "root": root,
            "repo": root + "/repo", "archive": root, "ops": root + "/ops",
            "data": root + "/data",
            "target": f"u2024110224@10.160.17.102:{port}"}


def connect(port: int):
    instance(port)
    return transport.connect(port)


def remote_python(code: str, port: int, timeout: int = 120) -> dict:
    return transport.remote_python(code, port, timeout=timeout)


def upload(local: Path, remote: str, port: int) -> dict:
    return transport.upload(local, remote, port)


def api(port: int, endpoint: str, body: dict | None = None, *, timeout: int = 120):
    """Call only this lane's loopback API; caller supplies exact request IDs."""
    instance(port)
    if not endpoint.startswith("/api/") or ".." in endpoint:
        raise ValueError("invalid workbench API path")
    payload = json.dumps(body, ensure_ascii=False) if body is not None else None
    code = f'''
import json,urllib.error,urllib.request
data={payload!r}
request=urllib.request.Request('http://127.0.0.1:{API_PORTS[port]}{endpoint}',
 data=None if data is None else data.encode('utf-8'),
 method='GET' if data is None else 'POST',
 headers={{'Content-Type':'application/json','Cache-Control':'no-cache'}})
try:
 with urllib.request.urlopen(request,timeout={timeout}) as response:
  print(json.dumps({{'status':response.status,'body':json.load(response)}}))
except urllib.error.HTTPError as error:
 print(json.dumps({{'status':error.code,'error':error.read().decode('utf-8','replace')[:3000]}}))
'''
    result = remote_python(code, port, timeout=timeout + 30)
    if result["exit_code"]:
        raise RuntimeError(f"S1-V10 API failed: {result['stderr'][-800:]}")
    return json.loads(result["stdout"].splitlines()[-1])
