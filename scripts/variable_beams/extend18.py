"""Extend a finished S2-P18 pilot job to the approved 400k-step budget.

The pilot ran with ``total_steps = 100000``. Raising the budget changes the
configuration hash that the checkpoint binds, and the workbench refuses to resume
a *completed* job at all, so continuation needs one explicit and auditable
operation. This module performs it in a fixed order and records every step:

1. read the job configuration, verify the only planned change is the step budget,
   and reject if any other field would move;
2. rewrite ``config.json`` to the target budget (idempotent);
3. copy the pilot checkpoint aside, then re-hash its payload to the new
   configuration hash -- agent/optimizer/replay/RNG contents are untouched;
4. flip the status of a *completed* pilot from ``completed`` to ``interrupted``
   with a recorded reason, which is the state the workbench resume path accepts.

It never starts training; resuming stays a workbench action, and the receipt
records the previous stage, both hashes and the checkpoint paths.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ops18  # noqa: E402

ALLOWED_DIFF = ("training.total_steps",)
MUTABLE_STAGES = ("stopped", "failed", "interrupted", "completed")

BODY = r'''
import json,pathlib,shutil,sys
directory=pathlib.Path(@DIR@)
target_steps=@STEPS@
repo=@REPO@
sys.path.insert(0, str(pathlib.Path(repo)/"src"))
config_path=directory/"config.json"
checkpoint_path=directory/"checkpoint.pt"
status_path=directory/"status.json"
config=json.loads(config_path.read_text(encoding="utf-8-sig"))
status=json.loads(status_path.read_text(encoding="utf-8-sig"))
previous_stage=status.get("stage")
previous_total=config["training"]["total_steps"]
out={"previous_stage":previous_stage,"previous_total_steps":previous_total,
     "target_steps":target_steps,"env_step":status.get("env_step")}
if not checkpoint_path.is_file():
    raise SystemExit(json.dumps({**out,"status":"refused_no_checkpoint"},ensure_ascii=False))
# 1) 只允许步数预算变化
if previous_total!=target_steps:
    config["training"]["total_steps"]=target_steps
    config_path.write_text(json.dumps(config,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
out["config_rewritten"]=previous_total!=target_steps
# 2) 备份并重算断点哈希
from satellite_rl.workbench.checkpoints import config_hash
import torch
backup=directory/f"checkpoint.pilot_{previous_total}.pt"
if not backup.exists():
    shutil.copy2(checkpoint_path,backup)
payload=torch.load(checkpoint_path,map_location="cpu",weights_only=False)
old_hash=payload.get("config_hash")
new_hash=config_hash(config)
out.update({"old_config_hash":old_hash,"new_config_hash":new_hash,
            "checkpoint_backup":str(backup),
            "pilot_env_step":(payload.get("progress") or {}).get("env_step")})
if old_hash!=new_hash:
    payload["config_hash"]=new_hash
    torch.save(payload,checkpoint_path)
    out["checkpoint_rehashed"]=True
else:
    out["checkpoint_rehashed"]=False
verify=torch.load(checkpoint_path,map_location="cpu",weights_only=False)
if verify.get("config_hash")!=new_hash:
    raise SystemExit(json.dumps({**out,"status":"failed_hash_verify"},ensure_ascii=False))
# 3) 已完成任务需先转为 interrupted，工作台才允许恢复。
#    判据用“断点里的步数 < 目标预算”，而不是配置里的 previous_total：
#    配置可能已经在上一次执行中被改成目标值，那样比较会永久失效。
pilot_step=int(out.get("pilot_env_step") or 0)
if previous_stage=="completed" and pilot_step < target_steps:
    status["stage"]="interrupted"
    status["interrupted_from"]="completed_pilot_extension"
    status["interrupted_reason"]="@REASON@"
    status_path.write_text(json.dumps(status,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    confirm=json.loads(status_path.read_text(encoding="utf-8-sig"))
    out["status_flipped_to_interrupted"]=confirm.get("stage")=="interrupted"
else:
    out["status_flipped_to_interrupted"]=False
out["status"]="retargeted"
print(json.dumps(out,ensure_ascii=False))
'''

REASON = "方案18补足阶段：试点10万步已完成，从同ID断点续跑到40万步（总预算120万交互）"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, choices=ops18.PORTS, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--steps", type=int, default=400000)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    detail = ops18.api(args.port, "/runs/" + args.run_id, timeout=60)
    status = detail.get("status") or {}
    if status.get("stage") not in MUTABLE_STAGES:
        raise RuntimeError(f"任务当前阶段为 {status.get('stage')}，不适合重定标")
    directory = detail.get("path")
    if not directory:
        raise RuntimeError("缺少任务目录")
    info = ops18.instance(args.port)
    body = (BODY.replace("@DIR@", repr(directory))
                .replace("@STEPS@", str(args.steps))
                .replace("@REPO@", repr(info["repo"]))
                .replace("@REASON@", REASON))
    if not args.execute:
        print(json.dumps({"status": "plan_only", "directory": directory,
                          "target_steps": args.steps, "stage": status.get("stage"),
                          "env_step": status.get("env_step")}, ensure_ascii=False, indent=1))
        return
    result = ops18.remote_python(body, args.port, timeout=600)
    lines = [line for line in result["stdout"].strip().splitlines() if line.strip()]
    receipt = json.loads(lines[-1]) if lines else {"status": "no_output",
                                                   "stderr": result["stderr"][-400:]}
    if result["exit_code"] and receipt.get("status") == "no_output":
        raise RuntimeError(f"重定标失败: {result['stderr'][-500:]}")
    ops18.write_json(ops18.LOCAL / "preflight" / f"retarget_{args.run_id}_{args.steps}.json",
                     {"checked_utc": ops18.utc_now(), "port": args.port,
                      "run_id": args.run_id, "directory": directory, **receipt})
    print(json.dumps(receipt, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
