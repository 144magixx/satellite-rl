"""Bounded SSH transport for the S2-P18 two-instance deployment.

Credentials are read only from the user-level BUPT_PASSWORD variable and are
never written to receipts or logs. Long scientific work belongs to the
workbench on each instance; this module only moves files, starts the two
isolated services and reads their APIs.
"""
from __future__ import annotations

import base64
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

HOST = "10.160.17.102"
USER = "u2024110224"
PORTS = (20418, 20424)
# S2-P17 used 8781; S1-03-R1 used 8896/8897 on the same shared host network
# namespace. S2-P18 keeps its own pair so no port is ever shared across
# experiments.
API_PORTS = {20418: 8898, 20424: 8899}
BASE = "/home/u2024110224/second_stage_satellite_rl/variable_beams_s2p18"
DQN = "/home/u2024110224/first_stage_satellite_rl/envs/DQN/bin/python"
SOURCE_DATA = "/home/u2024110224/second_stage_satellite_rl/variable_beams_20261004/20424/data/prepared"
LOCAL = Path("D:/paper/outputs/experiments/variable_beams_s2p18_20261006")
HOST_KEYS = {
    20418: "a6ym3EfC35pPdkqWO8lg+4As2nLDI1eKu8cur4P7WDc",
    20424: None,  # filled from the pinned TOFU record below
}
PIN_20424 = Path("D:/paper/outputs/verification/g11_remote_launch_20261001/hostkey_20424.json")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def instance(port: int) -> dict:
    if port not in PORTS:
        raise ValueError("S2-P18 transport only permits the two authorized SSH ports")
    root = f"{BASE}/{port}"
    return {
        "port": port,
        "root": root,
        "repo": f"{root}/repo",
        "archive": f"{root}/archive",
        "ops": f"{root}/ops",
        "api_port": API_PORTS[port],
        "target": f"{USER}@{HOST}:{port}",
    }


def _password() -> str:
    value = os.environ.get("BUPT_PASSWORD")
    if not value:
        raise RuntimeError("BUPT_PASSWORD is unavailable")
    return value


def _expected_host_key(port: int) -> str:
    if port == 20418:
        return HOST_KEYS[20418]
    item = json.loads(PIN_20424.read_text(encoding="utf-8"))
    if item.get("target") != f"{USER}@{HOST}:{port}":
        raise RuntimeError("20424 saved host-key target differs")
    return item["host_key_sha256"]


def connect(port: int):
    import paramiko

    expected = _expected_host_key(port)

    class PinnedKey(paramiko.MissingHostKeyPolicy):
        def missing_host_key(self, client, hostname, key):
            actual = base64.b64encode(hashlib.sha256(key.asbytes()).digest()).decode().rstrip("=")
            if actual != expected:
                raise paramiko.SSHException("Unexpected SSH host key")

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(PinnedKey())
    client.connect(HOST, port=port, username=USER, password=_password(),
                   allow_agent=False, look_for_keys=False, timeout=15,
                   banner_timeout=20, auth_timeout=20)
    return client


def execute(client, command: str, timeout: int = 120) -> dict:
    _, stdout, stderr = client.exec_command(command, timeout=timeout)
    out = stdout.read().decode("utf-8", "replace")
    err = stderr.read().decode("utf-8", "replace")
    code = stdout.channel.recv_exit_status()
    secret = _password()
    return {"exit_code": code, "stdout": out.replace(secret, "[REDACTED]"),
            "stderr": err.replace(secret, "[REDACTED]")}


def remote_python(code: str, port: int, timeout: int = 120) -> dict:
    """Run one python body on the instance with the DQN interpreter."""
    payload = base64.b64encode(code.encode("utf-8")).decode("ascii")
    command = (f"{DQN} -c \"import base64;exec(base64.b64decode('{payload}').decode('utf-8'))\"")
    client = connect(port)
    try:
        return execute(client, command, timeout=timeout)
    finally:
        client.close()


def upload(local: Path, remote: str, port: int) -> dict:
    """Upload a file atomically (write a .part copy, then rename)."""
    import paramiko

    local = Path(local)
    client = connect(port)
    try:
        with paramiko.SFTPClient.from_transport(client.get_transport()) as sftp:
            staging = remote + ".part"
            sftp.put(str(local), staging)
            sftp.posix_rename(staging, remote)
        digest = execute(client, f"sha256sum {remote}", timeout=300)
        return {"local_sha256": sha256_file(local), "remote": digest["stdout"].split()[0],
                "path": remote, "exit_code": digest["exit_code"]}
    finally:
        client.close()


def api(port: int, endpoint: str, body: dict | None = None, method: str = "GET",
        timeout: int = 120) -> dict:
    """Call the instance-local workbench API through a loopback python client.

    HTTP errors carry the workbench's JSON error body, which is what the caller
    needs to see; surface it instead of a bare urllib traceback.
    """
    api_port = API_PORTS[port]
    payload = json.dumps(body, ensure_ascii=False) if body is not None else None
    code = f'''
import json,urllib.error,urllib.request
data={payload!r}
data=None if data is None else data.encode("utf-8")
request=urllib.request.Request("http://127.0.0.1:{api_port}/api{endpoint}",data=data,
 method={method!r},headers={{"Content-Type":"application/json","Cache-Control":"no-cache"}})
try:
 with urllib.request.urlopen(request,timeout={timeout}) as response:
  print(response.read().decode("utf-8"))
except urllib.error.HTTPError as error:
 detail=error.read().decode("utf-8","replace")
 print(json.dumps({{"__http_error__":error.code,"detail":detail}},ensure_ascii=False))
'''
    result = remote_python(code, port, timeout=timeout + 30)
    if result["exit_code"]:
        raise RuntimeError(f"API {endpoint} failed: {result['stderr'][-800:]}")
    parsed = json.loads(result["stdout"].strip().splitlines()[-1])
    if isinstance(parsed, dict) and "__http_error__" in parsed:
        raise RuntimeError(f"API {endpoint} HTTP {parsed['__http_error__']}: {parsed['detail'][:900]}")
    return parsed
