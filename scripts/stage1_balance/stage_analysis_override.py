"""Stage only the reviewed audit tolerance fix for e22 result postprocessing."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

sys.path.insert(0, "C:/Users/Admin/.codex/worktrees/stage1-balance-actions/paper/scripts/stage1_balance")
from remote_sources import LOCAL, REMOTE, connect, upload_once  # noqa: E402
import remote_stage1  # noqa: E402


REPO = Path("C:/Users/Admin/.codex/worktrees/stage1-balance-actions/paper")
SCIENCE = "e22f449ccee11d525bce1c8011f297dc87bc86e1"
PLAN_SHA = "a9cebefc5b06eaf25f7459c7670b9bb11f9aaf218dfd90d5ecb0f987872f0bf7"
NAMES = ("audit_stage1_balance.py", "report_stage1_balance.py", "select_stage1_menus.py")


def source(revision: str, name: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(REPO), "show",
                                    revision + ":scripts/" + name])


def sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-commit", required=True)
    args = parser.parse_args()
    commit = args.analysis_commit.lower()
    if not re.fullmatch(r"[0-9a-f]{40}", commit) or commit == SCIENCE:
        raise ValueError("Independent analysis commit required")
    frozen = {name: source(commit, name) for name in NAMES}
    original = {name: source(SCIENCE, name) for name in NAMES}
    if (frozen[NAMES[0]] == original[NAMES[0]] or
            any(frozen[name] != original[name] for name in NAMES[1:])):
        raise RuntimeError("Analysis override must change audit only; report/select remain e22-identical")
    output = LOCAL / "preflight/analysis_override" / commit
    output.mkdir(parents=True, exist_ok=True)
    for name in NAMES:
        target = output / name
        if target.exists() and target.read_bytes() != frozen[name]:
            raise RuntimeError("Existing staged analysis script differs: " + name)
        target.write_bytes(frozen[name])
    patch = subprocess.check_output(["git", "-C", str(REPO), "diff", SCIENCE,
                                     commit, "--", "scripts/audit_stage1_balance.py"])
    contract = {"experiment_id": "S1-02", "status": "isolated_analysis_override",
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "science_commit": SCIENCE, "analysis_commit": commit,
                "plan_sha256": PLAN_SHA,
                "original_audit_sha256": sha(original[NAMES[0]]),
                "audit_patch_sha256": sha(patch),
                "files": {name: {"sha256": sha(frozen[name]), "bytes": len(frozen[name])}
                          for name in NAMES},
                "scope": "Audit movement numerical tolerance only; e22 workbench worker/source unchanged"}
    contract_file = output / "contract.json"
    if contract_file.exists():
        previous = json.loads(contract_file.read_text(encoding="utf-8"))
        contract["created_utc"] = previous.get("created_utc")
    encoded = (json.dumps(contract, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    if contract_file.exists() and contract_file.read_bytes() != encoded:
        raise RuntimeError("Existing analysis override contract differs")
    contract_file.write_bytes(encoded)
    remote_dir = REMOTE + "/ops/analysis_override/" + commit
    client = connect()
    try:
        remote_stage1.execute(client, "mkdir -p " + remote_dir)
        with client.open_sftp() as sftp:
            uploads = [upload_once(sftp, output / name, remote_dir + "/" + name)
                       for name in (*NAMES, "contract.json")]
            uploads.append(upload_once(
                sftp, LOCAL / "preflight/postprocess_override.py",
                REMOTE + "/ops/postprocess_override.py"))
            uploads.append(upload_once(
                sftp, LOCAL / "preflight/controller_recovery_attempt2.py",
                REMOTE + "/ops/controller_recovery_attempt2.py"))
    finally:
        client.close()
    receipt = {"status": "analysis_override_staged_sha_verified",
               "staged_utc": datetime.now(timezone.utc).isoformat(),
               "contract_sha256": sha(encoded), "contract": contract,
               "uploads": uploads}
    path = LOCAL / "preflight/analysis_override_stage_receipt.json"
    path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"receipt": str(path), "analysis_commit": commit,
                      "audit_sha256": contract["files"][NAMES[0]]["sha256"],
                      "contract_sha256": sha(encoded)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
