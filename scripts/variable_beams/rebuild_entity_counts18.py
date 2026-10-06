"""Repair a checkpoint's replay entity-count table so an extended job can resume.

Background: ``Replay.state_dict`` saves ``slot_entity_counts`` with one entry per ring
position, i.e. ``len(rows) + 1`` entries once the ring is saturated, while
``Replay.load_state_dict`` requires ``len(saved_counts) == len(rows)`` and rejects the
checkpoint otherwise. A pilot that ran to exactly its budget therefore cannot be
resumed from its own checkpoint without this one-off repair.

The repair trims the trailing entry, first proving that every remaining count still
matches its replay bucket's declared beam range, keeps a pristine copy of the
checkpoint, and records both hashes. Nothing but that one list length changes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ops18  # noqa: E402

BUCKET_RANGES = {"anchor_100": (100, 100), "lower": (50, 99),
                 "middle": (101, 149), "upper": (150, 200)}

BODY = r'''
import hashlib,json,pathlib,shutil,sys,traceback
directory=pathlib.Path(@DIR@)
execute=@EXECUTE@
ranges=json.loads(@RANGES@)
out={}
import torch
checkpoint=directory/"checkpoint.pt"
payload=torch.load(checkpoint,map_location="cpu",weights_only=False)
replay=(payload.get("agent") or {}).get("replay") or {}
rows=replay.get("rows") or []
buckets=replay.get("slot_buckets") or []
counts=replay.get("slot_entity_counts")
out.update({"rows":len(rows),"counts":None if counts is None else len(counts),
            "bucket_mode":replay.get("bucket_mode")})
if counts is None:
    out["status"]="no_counts_to_repair"
elif len(counts)==len(rows):
    out["status"]="already_consistent"
elif len(counts)!=len(rows)+1:
    out["status"]="unexpected_length_difference"
else:
    trimmed=[int(value) for value in counts[:len(rows)]]
    per_bucket={}
    for bucket,count in zip(buckets,trimmed):
        per_bucket.setdefault(bucket,[]).append(count)
    summary={}
    out_of_range=[]
    for bucket,values in per_bucket.items():
        low,high=ranges.get(bucket,(None,None))
        summary[bucket]={"n":len(values),"min":min(values),"max":max(values),
                         "declared":[low,high],
                         "in_range": bool(low is None or (min(values)>=low and max(values)<=high))}
        if low is not None and not summary[bucket]["in_range"]:
            out_of_range.append(bucket)
    out["bucket_summary"]=summary
    out["buckets_out_of_range"]=out_of_range
    if not execute:
        out["status"]="plan_only"
    elif out_of_range:
        out["status"]="rejected_bucket_mismatch"
    else:
        backup=directory/"checkpoint.pre_entityrepair.pt"
        if not backup.exists():
            shutil.copy2(checkpoint,backup)
        out["backup_sha256"]=hashlib.sha256(backup.read_bytes()).hexdigest()
        out["checkpoint_before_sha256"]=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        replay["slot_entity_counts"]=trimmed
        payload["agent"]["replay"]=replay
        torch.save(payload,checkpoint)
        verify=torch.load(checkpoint,map_location="cpu",weights_only=False)
        got=(verify.get("agent") or {}).get("replay",{}).get("slot_entity_counts") or []
        out["checkpoint_after_sha256"]=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        out["written"]=len(got)
        out["written_range"]=[min(got),max(got)] if got else None
        out["status"]="repaired" if len(got)==len(rows) else "failed_verify"
print(json.dumps(out,ensure_ascii=False,indent=1))
'''


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, choices=ops18.PORTS, required=True)
    parser.add_argument("--directory", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    body = (BODY.replace("@DIR@", repr(args.directory))
                .replace("@EXECUTE@", "True" if args.execute else "False")
                .replace("@RANGES@", repr(json.dumps(BUCKET_RANGES))))
    result = ops18.remote_python(body, args.port, timeout=1800)
    lines = [line for line in result["stdout"].strip().splitlines() if line.strip()]
    receipt = json.loads("\n".join(lines)) if lines else {"status": "no_output",
                                                          "stderr": result["stderr"][-400:]}
    name = Path(args.directory).name
    ops18.write_json(ops18.LOCAL / "preflight" / f"repair_entity_counts_{name}.json",
                     {"checked_utc": ops18.utc_now(), "port": args.port,
                      "directory": args.directory, **receipt})
    print(json.dumps(receipt, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
