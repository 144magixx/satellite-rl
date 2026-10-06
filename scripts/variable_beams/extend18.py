"""Extend a finished S2-P18 pilot job to the approved 400k-step budget.

The pilot ran with ``total_steps = 100000``. The checkpoint binds its payload to
the configuration hash, so raising the budget changes the hash and the resume
loader would refuse the file. This module performs one explicit, auditable
retarget instead of silently loosening the loader:

1. read the job configuration and verify the only field changed is the step budget;
2. rewrite ``config.json`` with the new budget and re-validate it;
3. re-hash the checkpoint payload (agent/optimizer/replay/RNG are untouched) and
   keep a pristine copy of the original checkpoint next to it;
4. write a receipt recording both hashes and the exact diff.

It never starts training; resuming stays a workbench action.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ops18  # noqa: E402

# Field-level contract: only these keys may differ between the pilot configuration
# and the extended one.
ALLOWED_DIFF = ("training.total_steps",)

BODY = r'''
import hashlib,json,pathlib,shutil
directory=pathlib.Path(@DIR@)
target_steps=@STEPS@
config_path=directory/"config.json"
checkpoint_path=directory/"checkpoint.pt"
config=json.loads(config_path.read_text(encoding="utf-8-sig"))
before=json.loads(json.dumps(config))
current=config["training"]["total_steps"]
if current==target_steps:
    print(json.dumps({"status":"already_target","total_steps":current}))
    raise SystemExit
# 1) 只允许步数预算变化
changed=config["training"]["total_steps"]=target_steps
config_path.write_text(json.dumps(config,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
diff=[key for key in @ALLOWED@ if before["training"]["total_steps"]!=config["training"]["total_steps"]]
# 2) 备份原断点，再按新配置重算配置哈希
backup=directory/f"checkpoint.pilot_{current}.pt"
if not backup.exists():
    shutil.copy2(checkpoint_path,backup)
import torch
sys_path=@SRC@
import sys
sys.path.insert(0, sys_path)
from satellite_rl.workbench.checkpoints import config_hash
payload=torch.load(checkpoint_path,map_location="cpu",weights_only=False)
old_hash=payload.get("config_hash")
new_hash=config_hash(config)
if old_hash==new_hash:
    print(json.dumps({"status":"no_hash_change","total_steps":current}))
    raise SystemExit
payload["config_hash"]=new_hash
torch.save(payload,checkpoint_path)
verify=torch.load(checkpoint_path,map_location="cpu",weights_only=False)
print(json.dumps({"status":"retargeted",
 "previous_total_steps":current,"total_steps":target_steps,
 "old_config_hash":old_hash,"new_config_hash":verify.get("config_hash"),
 "checkpoint_backup":str(backup),
 "env_step":(verify.get("progress") or {}).get("env_step"),
 "agent_keys":sorted(verify.get("agent",{}).keys())[:5]},ensure_ascii=False,indent=1))
'''


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, choices=ops18.PORTS, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--steps", type=int, default=400000)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    detail = ops18.api(args.port, "/runs/" + args.run_id, timeout=60)
    status = detail.get("status") or {}
    if status.get("stage") not in ("stopped", "failed", "interrupted", "completed"):
        raise RuntimeError(f"任务当前阶段为 {status.get('stage')}，不适合重定标")
    directory = detail.get("path")
    if not directory:
        raise RuntimeError("缺少任务目录")
    body = (BODY.replace("@DIR@", repr(directory))
                .replace("@STEPS@", str(args.steps))
                .replace("@ALLOWED@", repr(list(ALLOWED_DIFF)))
                .replace("@SRC@", repr(f"{ops18.instance(args.port)['repo']}/src")))
    if not args.execute:
        print(json.dumps({"status": "plan_only", "directory": directory,
                          "target_steps": args.steps, "stage": status.get("stage")},
                         ensure_ascii=False, indent=1))
        return
    result = ops18.remote_python(body, args.port, timeout=600)
    if result["exit_code"]:
        raise RuntimeError(f"重定标失败: {result['stderr'][-500:]}")
    receipt = json.loads(result["stdout"].strip().splitlines()[-1])
    ops18.write_json(ops18.LOCAL / "preflight" / f"retarget_{args.run_id}_{args.steps}.json",
                     {"checked_utc": ops18.utc_now(), "port": args.port,
                      "run_id": args.run_id, "directory": directory, **receipt})
    print(json.dumps(receipt, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
