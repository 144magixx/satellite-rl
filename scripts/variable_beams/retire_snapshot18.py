"""Retire a snapshot that only conflicts with the extension retarget.

The resumed job writes its first snapshot at exactly the pilot's final step, and
the existing file records the pilot configuration. The snapshot is a diagnostic
artefact (training recovery uses ``checkpoint.pt``), so the old file is renamed
rather than deleted, and both hashes are recorded in the receipt.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ops18  # noqa: E402

BODY = r'''
import hashlib,json,pathlib,shutil
directory=pathlib.Path(@DIR@)
step=@STEP@
suffix="@SUFFIX@"
source=directory/"learning_snapshots"/f"step_{step:09d}.pt"
out={"source":str(source),"exists":source.is_file()}
if source.is_file():
    out["source_sha256"]=hashlib.sha256(source.read_bytes()).hexdigest()
    target=source.with_name(source.stem+suffix+source.suffix)
    if target.exists():
        out["status"]="target_exists"
        out["target"]=str(target)
        out["target_sha256"]=hashlib.sha256(target.read_bytes()).hexdigest()
    else:
        shutil.move(str(source),str(target))
        out["status"]="retired"
        out["target"]=str(target)
        out["target_sha256"]=hashlib.sha256(target.read_bytes()).hexdigest()
    out["source_now_exists"]=source.is_file()
else:
    out["status"]="source_missing"
print(json.dumps(out,ensure_ascii=False))
'''


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, choices=ops18.PORTS, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--suffix", default="_pilot100k")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    detail = ops18.api(args.port, "/runs/" + args.run_id, timeout=60)
    directory = detail.get("path")
    if not directory:
        raise RuntimeError("缺少任务目录")
    body = (BODY.replace("@DIR@", repr(directory))
                .replace("@STEP@", str(args.step))
                .replace("@SUFFIX@", args.suffix))
    if not args.execute:
        print(json.dumps({"status": "plan_only", "directory": directory,
                          "step": args.step, "suffix": args.suffix},
                         ensure_ascii=False, indent=1))
        return
    result = ops18.remote_python(body, args.port, timeout=600)
    lines = [line for line in result["stdout"].strip().splitlines() if line.strip()]
    receipt = json.loads(lines[-1]) if lines else {"status": "no_output",
                                                   "stderr": result["stderr"][-400:]}
    ops18.write_json(ops18.LOCAL / "preflight"
                     / f"retire_snapshot_{args.run_id}_{args.step}.json",
                     {"checked_utc": ops18.utc_now(), "port": args.port,
                      "run_id": args.run_id, **receipt})
    print(json.dumps(receipt, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
