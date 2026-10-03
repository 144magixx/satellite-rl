"""Create the reviewable frozen S1-02 API plan after the science commit."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

from remote_sources import LOCAL, REMOTE


REPO = Path(__file__).resolve().parents[2]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--output", type=Path, default=LOCAL / "plan_frozen.json")
    args = parser.parse_args()
    commit = args.commit.lower()
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("Full frozen scientific commit required")
    head = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                                   text=True).strip()
    if head != commit:
        raise RuntimeError("Worktree HEAD differs from frozen science commit")
    sys.path.insert(0, str(REPO / "src"))
    from satellite_rl.coverage.config import balance_default_config, validate_config

    config = validate_config(balance_default_config())
    dates = {"train": ["2024-01-08", "2024-01-09"],
             "validation": ["2024-01-12"], "A_test": ["2024-01-28"],
             "B_test": ["2024-01-30"]}
    local_records = {
        "train": LOCAL / "development/data/source_train/record.json",
        "validation": LOCAL / "development/data/source_validation/record.json",
        "A_test": LOCAL / "A/data/staged_test/record.json",
        "B_test": LOCAL / "B/data/staged_test/record.json",
    }
    remote_records = {
        "train": REMOTE + "/data/source_train/record.json",
        "validation": REMOTE + "/data/source_validation/record.json",
        "A_test": REMOTE + "/data/A/staged_test/record.json",
        "B_test": REMOTE + "/data/B/staged_test/record.json",
    }
    records = {}
    for role, local in local_records.items():
        record = json.loads(local.read_text(encoding="utf-8"))
        if record.get("count") != (576 if role == "train" else 20 if role == "validation" else 100):
            raise RuntimeError("Scene count differs for " + role)
        if record.get("source", {}).get("source_dates") != dates[role]:
            raise RuntimeError("Scene UTC source date differs for " + role)
        records[role] = {"path": remote_records[role], "record_sha256": sha(local)}
    analysis = REMOTE + "/outputs/experiments/first_stage_s1_02_20261004/analysis"
    plan = {
        "experiment_id": "S1-02", "created_utc": datetime.now(timezone.utc).isoformat(),
        "execution_target": "u2024110224@10.160.17.102:20418",
        "remote_repo": REMOTE + "/repo", "source_commit": commit,
        "api_port": 8892, "base_config": config,
        "data_dates": dates, "input_records": records,
        "gates": {"A_report": analysis + "/A_report/gate.json",
                  "B_menu_selection": analysis + "/B_menu_selection/gate.json",
                  "B_report": analysis + "/B_report/gate.json"},
        "budgets": {"A_development": 80, "A_formal": 400,
                    "B_menu_screen": 100, "B_training_steps": 1228800,
                    "B_validation_results": 1200, "B_formal": 600,
                    "max_rounds": 10, "training_seed": 51001},
        "selection": "X0 plus two selected using Jan12 validation only; fixed-budget endpoint",
        "postprocess_scripts": ["audit_stage1_balance.py", "select_stage1_menus.py",
                                "report_stage1_balance.py"],
    }
    output = args.output.resolve()
    if output.is_file():
        prior = json.loads(output.read_text(encoding="utf-8"))
        plan["created_utc"] = prior.get("created_utc")
    encoded = (json.dumps(plan, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    if output.exists():
        if output.read_bytes() != encoded:
            raise RuntimeError("Existing frozen controller plan differs")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(encoded)
    print(json.dumps({"status": "controller_plan_frozen", "path": str(output),
                      "sha256": hashlib.sha256(encoded).hexdigest(),
                      "source_commit": commit}, ensure_ascii=False))


if __name__ == "__main__":
    main()
