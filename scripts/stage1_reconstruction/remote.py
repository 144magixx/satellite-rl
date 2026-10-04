"""Bounded SSH transport for the two explicitly authorized S1-03 instances.

Credentials are read only from BUPT_PASSWORD and never written to receipts.
Long scientific operations belong to the workbench, not this transport.
"""
from __future__ import annotations

import base64
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex


HOST = "10.160.17.102"
USER = "u2024110224"
PORTS = (20418, 20424)
HOST_KEY = "a6ym3EfC35pPdkqWO8lg+4As2nLDI1eKu8cur4P7WDc"
DQN = "/home/u2024110224/first_stage_satellite_rl/envs/DQN/bin/python"
REMOTE_BASE = "/home/u2024110224/first_stage_satellite_rl/s1_03"
LOCAL = Path("D:/paper/outputs/experiments/first_stage_s1_03_20261005")


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def connect(port):
    import paramiko
    if port not in PORTS:
        raise ValueError("S1-03 transport only permits the two authorized SSH ports")
    password = os.environ.get("BUPT_PASSWORD")
    if not password:
        raise RuntimeError("BUPT_PASSWORD is unavailable")
    class PinnedKey(paramiko.MissingHostKeyPolicy):
        def missing_host_key(self, client, hostname, key):
            actual = base64.b64encode(hashlib.sha256(key.asbytes()).digest()).decode().rstrip("=")
            if actual != HOST_KEY:
                raise paramiko.SSHException("Unexpected SSH host key")
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(PinnedKey())
    client.connect(HOST, port=port, username=USER, password=password,
                   allow_agent=False, look_for_keys=False, timeout=15,
                   banner_timeout=20, auth_timeout=20)
    client.get_transport().set_keepalive(20)
    return client


def run_code(client, code, *, timeout=60):
    payload = base64.b64encode(code.encode("utf-8")).decode("ascii")
    command = shlex.join([DQN, "-B", "-c", f"exec(__import__('base64').b64decode('{payload}'))"])
    _, stdout, stderr = client.exec_command(command, timeout=timeout)
    out = stdout.read().decode("utf-8")
    err = stderr.read().decode("utf-8")
    status = stdout.channel.recv_exit_status()
    if status:
        raise RuntimeError(f"Remote operation failed ({status}): {(err or out)[-3000:]}")
    return json.loads(out)


def upload_once(sftp, local, remote):
    data = Path(local).read_bytes()
    try:
        with sftp.open(remote, "rb") as stream:
            existing = stream.read()
    except FileNotFoundError:
        existing = None
    if existing is not None and existing != data:
        raise RuntimeError("Immutable remote file differs: " + remote)
    if existing is None:
        with sftp.open(remote, "wb") as stream:
            stream.write(data)
    with sftp.open(remote, "rb") as stream:
        if digest(stream.read()) != digest(data):
            raise RuntimeError("Remote SHA mismatch: " + remote)
    return {"path": remote, "sha256": digest(data), "bytes": len(data),
            "action": "verified_existing" if existing is not None else "uploaded_verified"}


def api(client, port, endpoint, body=None, *, timeout=120):
    if port not in PORTS:
        raise ValueError("Unknown instance")
    api_port = {20418: 8894, 20424: 8895}[port]
    code = f"""
import json, urllib.request, urllib.error
body={body!r}
request=urllib.request.Request('http://127.0.0.1:{api_port}/api/'+{endpoint!r},
 data=None if body is None else json.dumps(body,ensure_ascii=False).encode(),
 headers={{'Content-Type':'application/json'}})
try:
 with urllib.request.urlopen(request,timeout={timeout}) as response: result=json.load(response)
except urllib.error.HTTPError as error:
 raise RuntimeError(str(error.code)+': '+error.read().decode()[:3000])
print(json.dumps(result,ensure_ascii=False))
"""
    return run_code(client, code, timeout=timeout+10)
