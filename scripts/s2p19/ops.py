"""Pinned SSH transport and isolated workbench API for S2-P19 only."""

from __future__ import annotations

import json
import base64
import hashlib
from pathlib import Path
import re

from scripts.variable_beams import ops18 as transport


EXPERIMENT = "S2-P19"
SSH_PORTS = (20418, 20424, 20473, 20474)
API_PORTS = {20418: 8900, 20424: 8901, 20473: 8904, 20474: 8905}
NEW_HOST_KEY = "a6ym3EfC35pPdkqWO8lg+4As2nLDI1eKu8cur4P7WDc"
KEEPALIVE_SECONDS = 30
REMOTE_BASE = "/home/u2024110224/second_stage_satellite_rl/sac_routes_s2p19"
LOCAL = Path("D:/paper/outputs/experiments/sac_routes_s2p19_20261007")
DQN = transport.DQN
PRIORITY_DEPS = ("/home/u2024110224/third_stage_satellite_rl/"
                 "s3_si_priority_20261003/shared_data/deps/cvxpy-1.6.5-clarabel-0.11.1")


def instance(port: int) -> dict:
    if port not in SSH_PORTS:
        raise ValueError("S2-P19 permits only its four authorized instances")
    root = f"{REMOTE_BASE}/{port}"
    return {"ssh_port": port, "api_port": API_PORTS[port], "root": root,
            "repo": root + "/repo", "archive": root + "/archive",
            "ops": root + "/ops", "data": root + "/data",
            "target": f"u2024110224@10.160.17.102:{port}"}


def connect(port: int):
    instance(port)
    if port in transport.PORTS:
        client = transport.connect(port)
    else:
        import paramiko

        class PinnedKey(paramiko.MissingHostKeyPolicy):
            def missing_host_key(self, client, hostname, key):
                actual = base64.b64encode(hashlib.sha256(key.asbytes()).digest()).decode().rstrip("=")
                if actual != NEW_HOST_KEY:
                    raise paramiko.SSHException("Unexpected S2-P19 SSH host key")

        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(PinnedKey())
        client.connect(transport.HOST, port=port, username=transport.USER,
                       password=transport._password(), allow_agent=False,
                       look_for_keys=False, timeout=15, banner_timeout=20,
                       auth_timeout=20)
    connection = client.get_transport()
    if connection is None:
        client.close()
        raise RuntimeError("S2-P19 SSH transport unavailable")
    try:
        connection.set_keepalive(KEEPALIVE_SECONDS)
    except Exception:
        client.close()
        raise
    return client


def remote_python(code: str, port: int, timeout: int = 120) -> dict:
    instance(port)
    payload = base64.b64encode(code.encode("utf-8")).decode("ascii")
    command = f'{DQN} -c "import base64;exec(base64.b64decode(\'{payload}\').decode(\'utf-8\'))"'
    client = connect(port)
    try:
        return transport.execute(client, command, timeout=timeout)
    finally:
        client.close()


def upload(path: Path, remote_path: str, port: int) -> dict:
    instance(port)
    import paramiko
    client = connect(port)
    try:
        with paramiko.SFTPClient.from_transport(client.get_transport()) as sftp:
            sftp.put(str(path), remote_path + ".part")
            sftp.posix_rename(remote_path + ".part", remote_path)
        result = transport.execute(client, f"sha256sum {remote_path}", timeout=300)
        return {"local_sha256": transport.sha256_file(path),
                "remote": result["stdout"].split()[0], "path": remote_path,
                "exit_code": result["exit_code"]}
    finally:
        client.close()


def api(port: int, path: str, body: dict | None = None,
        method: str = "GET", timeout: int = 60) -> dict:
    """Use the real instance-local workbench API; never call runner directly."""
    if port not in SSH_PORTS or not re.fullmatch(r"/api/[A-Za-z0-9_/-]+", path):
        raise ValueError("invalid S2-P19 API target")
    if method not in ("GET", "POST") or (method == "POST" and body is None):
        raise ValueError("invalid API method/body")
    payload = json.dumps(body, ensure_ascii=False) if body is not None else None
    code = f'''
import json,urllib.error,urllib.request
data={payload!r}
data=None if data is None else data.encode('utf-8')
request=urllib.request.Request('http://127.0.0.1:{API_PORTS[port]}{path}',data=data,
 method={method!r},headers={{'Content-Type':'application/json','Cache-Control':'no-cache'}})
try:
 with urllib.request.urlopen(request,timeout={timeout}) as response:
  print(json.dumps({{'http_status':response.status,'body':json.loads(response.read().decode('utf-8'))}},ensure_ascii=False))
except urllib.error.HTTPError as error:
 print(json.dumps({{'http_status':error.code,'error':error.read().decode('utf-8','replace')[:3000]}},ensure_ascii=False))
'''
    response = remote_python(code, port, timeout=timeout + 30)
    if response["exit_code"]:
        raise RuntimeError(f"{port} API {path} transport: " + response["stderr"][-1000:])
    parsed = json.loads(response["stdout"].splitlines()[-1])
    if parsed["http_status"] >= 400:
        raise RuntimeError(f"{port} API {path} HTTP {parsed['http_status']}: {parsed.get('error')}")
    return parsed["body"]
