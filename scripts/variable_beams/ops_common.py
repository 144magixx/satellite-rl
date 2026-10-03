"""Single-target transport and contracts for the S2-P17 operations scripts.

Only the explicitly authorized 20424 endpoint is reachable through this module.
Nothing here starts a workbench or creates a formal task.
"""
from __future__ import annotations

import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re


TRANSPORT_ROOT = Path("D:/paper/outputs/verification/g11_remote_launch_20261001")

SSH_PORT = 20424
WORKBENCH_PORT = 8781
BASE = "/home/u2024110224/second_stage_satellite_rl/variable_beams_20261004/20424"
REPO = BASE + "/repo"
ARCHIVE = BASE + "/archive"
OPS = BASE + "/ops"
DQN = "/home/u2024110224/first_stage_satellite_rl/envs/DQN/bin/python"
PRIORITY_DEPS = "/home/u2024110224/third_stage_satellite_rl/s3_si_priority_20261003/shared_data/deps/cvxpy-1.6.5-clarabel-0.11.1"
LOCAL = Path("D:/paper/outputs/experiments/variable_beams_20261004/preflight")
TRAIN_SOURCE = BASE + "/data/prepared/train/record.json"
VALIDATION_SOURCE = BASE + "/data/prepared/validation_eligible12_v2/record.json"
TEST_SOURCE = BASE + "/data/prepared/test/record.json"
T0_SOURCE = "/home/u2024110224/second_stage_satellite_rl/g11/20418/archive/models/runs/workbench/train_sac_20261001_162126_33e89e95/final"
T0_POLICY_SHA = "ae095e8246d22b2716ba154315cee18584917d215580821cb53f9aeba78eb824"
T0_MODEL_SHA = "1d4055e40ba6314ad8753ae64c1321a30fff7d7c3b25429b86ffca92c5388dde"
REQUESTS = {
    "V0_train": "s2p17-v0-train-51001",
    "V1_train": "s2p17-v1-train-51001",
    "V0_eval": "s2p17-v0-jan29-eval",
    "V1_eval": "s2p17-v1-jan29-eval",
    "H0_eval": "s2p17-h0-jan29-eval",
    "V0_sca": "s2p17-v0-jan29-sca",
    "V1_sca": "s2p17-v1-jan29-sca",
    "H0_sca": "s2p17-h0-jan29-sca",
    "fixed_sca": "s2p17-fixed-jan29-sca",
    "greedy_sca": "s2p17-greedy-jan29-sca",
}
JOB_KEYS = tuple(REQUESTS)
SCA_SOURCES = {
    "V0_sca": ("V0_eval", "model"),
    "V1_sca": ("V1_eval", "model"),
    "H0_sca": ("H0_eval", "model"),
    "fixed_sca": ("V0_eval", "fixed"),
    "greedy_sca": ("V0_eval", "greedy"),
}
THREAD_ENV = {name: "1" for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                                      "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_commit(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{40}", value.lower()):
        raise ValueError("须提供40位正式科学提交")
    return value.lower()


def require_sha(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value.lower()):
        raise ValueError("须提供64位SHA-256")
    return value.lower()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".tmp")
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    pending.replace(path)


def load_scope(path: Path, expected_sha: str) -> dict:
    if sha256(path) != require_sha(expected_sha):
        raise ValueError("用户确认的训练范围合同SHA不一致")
    scope = json.loads(path.read_text(encoding="utf-8"))
    if (scope.get("experiment_id") != "S2-P17" or scope.get("status") != "user_confirmed"
            or scope.get("decision") not in ("eligible520", "replace576")
            or scope.get("mother_count") not in (520, 576)
            or scope.get("derived_scene_count") != scope.get("mother_count") * 14
            or not scope.get("user_message_reference")):
        raise ValueError("训练母景范围尚未由用户明确确认")
    if (scope["decision"] == "eligible520") != (scope["mother_count"] == 520):
        raise ValueError("母场景选择与批准数量矛盾")
    require_sha(scope.get("train_record_sha256"))
    return scope


def remote_python(code: str, *, timeout: int = 120) -> dict:
    connect, execute, _ = _transport()
    encoded = base64.b64encode(code.encode("utf-8")).decode("ascii")
    command = "python3 -B -c 'exec(__import__(\"base64\").b64decode(\"" + encoded + "\"))'"
    transport = connect(SSH_PORT)
    try:
        result = execute(transport, command, timeout=timeout)
    finally:
        transport.close()
    if result["exit_code"]:
        raise RuntimeError(result["stderr"][-1600:])
    return json.loads(result["stdout"])


def remote_file(local: Path, destination: str) -> None:
    connect, _, paramiko = _transport()
    transport = connect(SSH_PORT)
    try:
        with paramiko.SFTPClient.from_transport(transport) as sftp:
            sftp.put(str(local), destination)
    finally:
        transport.close()


def _transport():
    """Load the pinned SSH transport only from the local Windows orchestrator."""
    import sys

    if not TRANSPORT_ROOT.is_dir():
        raise RuntimeError("Pinned local SSH transport missing")
    if str(TRANSPORT_ROOT) not in sys.path:
        sys.path.insert(0, str(TRANSPORT_ROOT))
    from remote_transport import connect, execute, paramiko  # noqa: E402
    return connect, execute, paramiko


def api_code(body: str) -> str:
    """Place one Python body behind a strictly localhost workbench API helper."""
    return f'''
import json,urllib.request
def api(method,path,payload=None,timeout=120):
 data=None if payload is None else json.dumps(payload,ensure_ascii=False).encode('utf-8')
 request=urllib.request.Request('http://127.0.0.1:{WORKBENCH_PORT}/api'+path,
  data=data,method=method,headers={{'Content-Type':'application/json','Cache-Control':'no-cache'}})
 with urllib.request.urlopen(request,timeout=timeout) as response:return json.load(response)
{body}
'''
