"""ND12 logical-lane transport; preserve archives/API ports across GPU placement.

Current default lends physical20474 to another user-authorized experiment.
Logical20474 remains a fourth data shard, served by physical20473 at API8943.
All network/compute operations resolve placement before opening an SSH channel.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

from scripts.stage1_nondrl12 import ops

CURRENT = {"experiment_id": "S1-ND12", "schema_version": 1,
           "mapping": {"20418": 20418, "20424": 20424, "20473": 20473, "20474": 20473},
           "released_physical_ports": [20474]}


def validate_placement(value):
    if (not isinstance(value, dict) or value.get("experiment_id") != "S1-ND12" or
            value.get("schema_version") != 1 or set(value.get("mapping", {})) != {str(p) for p in ops.SSH_PORTS}):
        raise ValueError("placement must declare all four ND12 logical lanes")
    mapping = value["mapping"]
    released = value.get("released_physical_ports")
    if (not isinstance(released, list) or len(set(released)) != len(released) or
            any(type(p) is not int or p not in ops.SSH_PORTS for p in released) or
            any(type(p) is not int or p not in ops.SSH_PORTS for p in mapping.values())):
        raise ValueError("invalid physical executors or resource release")
    if any(mapping[str(p)] != p for p in ops.SSH_PORTS[:3]) or mapping["20474"] not in (20473, 20474):
        raise ValueError("placement only supports the authorized 20474-to20473 migration")
    if set(mapping.values()).intersection(released):
        raise ValueError("a released physical executor cannot receive ND12 requests")
    return value


class Transport:
    SSH_PORTS, API_PORTS, BASE, LOCAL, DQN = ops.SSH_PORTS, ops.API_PORTS, ops.BASE, ops.LOCAL, ops.DQN

    def __init__(self, placement=None, *, raw=None, path=None):
        self.placement = validate_placement(json.loads(json.dumps(CURRENT)) if placement is None else placement)
        self.mapping = {int(k): v for k, v in self.placement["mapping"].items()}
        self.path = Path(path).resolve() if path is not None else None
        self.raw = raw if raw is not None else json.dumps(self.placement, sort_keys=True).encode()
        self.mapping_sha256 = hashlib.sha256(self.raw).hexdigest()

    @classmethod
    def from_file(cls, path):
        source = Path(path).resolve()
        raw = source.read_bytes()
        return cls(json.loads(raw.decode("utf-8-sig")), raw=raw, path=source)

    def verify_unchanged(self):
        if self.path is not None and hashlib.sha256(self.path.read_bytes()).hexdigest() != self.mapping_sha256:
            raise RuntimeError("frozen executor placement bytes changed")

    def physical_port(self, logical_port):
        self.verify_unchanged()
        if type(logical_port) is not int or logical_port not in self.mapping:
            raise ValueError("unknown ND12 logical lane")
        return self.mapping[logical_port]

    def instance(self, logical_port):
        physical = self.physical_port(logical_port)
        return {**ops.instance(logical_port), "logical_port": logical_port, "physical_ssh_port": physical,
                "placement_sha256": self.mapping_sha256}

    def connect(self, logical_port):
        return ops.connect(self.physical_port(logical_port))

    def remote_python(self, code, logical_port, timeout=120):
        return ops.remote_python(code, self.physical_port(logical_port), timeout=timeout)

    def upload(self, path, remote_path, logical_port):
        # Keep the original logical archive/path, route only the SSH executor.
        return ops.upload(path, remote_path, self.physical_port(logical_port))

    def api(self, logical_port, path, body=None, method=None, timeout=60):
        self.physical_port(logical_port)
        if not re.fullmatch(r"/api/[A-Za-z0-9_/-]+", path):
            raise ValueError("invalid workbench API path")
        method = method or ("POST" if body is not None else "GET")
        if method not in ("GET", "POST") or ((method == "GET") != (body is None)):
            raise ValueError("invalid workbench method/body")
        payload = None if body is None else json.dumps(body, ensure_ascii=False)
        # API8943 belongs to logical74 even when its physical SSH executor is73.
        code = f'''import json,urllib.request,urllib.error
data={payload!r}
r=urllib.request.Request('http://127.0.0.1:{self.API_PORTS[logical_port]}{path}',data=None if data is None else data.encode(),method={method!r},headers={{'Content-Type':'application/json','Cache-Control':'no-cache'}})
try:
 with urllib.request.urlopen(r,timeout={timeout}) as response: print(json.dumps({{'status':response.status,'body':json.load(response)}}))
except urllib.error.HTTPError as e: print(json.dumps({{'status':e.code,'error':e.read().decode('utf-8','replace')[:2000]}}))
'''
        result = self.remote_python(code, logical_port, timeout=timeout + 30)
        if result["exit_code"]:
            raise RuntimeError(f"logical{logical_port}/physical{self.mapping[logical_port]} API transport: " + result["stderr"][-600:])
        value = json.loads(result["stdout"].splitlines()[-1])
        if value["status"] >= 400:
            raise RuntimeError(f"logical{logical_port} HTTP {value['status']}: " + value.get("error", ""))
        return value["body"]
