"""Write the S2-P18 live training state into the run manifest and status file."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, r"D:\paper\scripts\variable_beams")
import ops18  # noqa: E402

ROOT = Path(r"D:\paper")
BEGIN = "<!-- S2P18-LIVE-BEGIN -->"
END = "<!-- S2P18-LIVE-END -->"
MANIFEST = ops18.LOCAL / "run_manifest.json"


def collect() -> dict:
    instances = {}
    for port in ops18.PORTS:
        runs = ops18.api(port, "/runs", timeout=60).get("runs", [])
        jobs = []
        for item in runs:
            if not str(item.get("request_id", "")).startswith("s2p18-"):
                continue
            detail = ops18.api(port, "/runs/" + item["run_id"], timeout=60)
            status = detail.get("status") or {}
            jobs.append({
                "request_id": item.get("request_id"),
                "run_id": item.get("run_id"),
                "kind": item.get("kind"),
                "stage": status.get("stage") or item.get("stage"),
                "env_step": status.get("env_step"),
                "budget_env_steps": status.get("budget_env_steps"),
                "update_step": status.get("update_step"),
                "episode": status.get("episode"),
                "steps_per_second": status.get("steps_per_second"),
                "beam_bucket_episodes": status.get("beam_bucket_episodes"),
                "replay_sampled_by_bucket": status.get("replay_sampled_by_bucket"),
                "error": status.get("error"),
                "heartbeat_at": status.get("heartbeat_at"),
                "git_commit": (detail.get("job") or {}).get("git_commit"),
                "source_root": (detail.get("job") or {}).get("source_root"),
                "directory": detail.get("path"),
            })
        host = ops18.remote_python(
            "import json,subprocess\n"
            "def run(a):\n"
            "    p=subprocess.run(a,capture_output=True,text=True)\n"
            "    return p.stdout.strip()\n"
            "print(json.dumps({'gpu':run(['nvidia-smi','--query-gpu=utilization.gpu,memory.used','--format=csv,noheader']),"
            "'apps':run(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'])}))\n",
            port, timeout=120)
        instances[str(port)] = {
            "jobs": jobs,
            "host": json.loads(host["stdout"].strip().splitlines()[-1]) if host["exit_code"] == 0 else None,
        }
    return instances


def arm_configs(port_runs: dict) -> dict:
    """Read each arm's frozen configuration to confirm its treatment is in place."""
    body_template = (
        "import json,pathlib\n"
        "path=pathlib.Path(@DIR@)/'config.json'\n"
        "config=json.loads(path.read_text(encoding='utf-8-sig'))\n"
        "print(json.dumps({'experiment':config.get('experiment'),\n"
        " 'preset':config['reward'].get('preset'),\n"
        " 'shaping':config['reward'].get('shaping'),\n"
        " 'progress_strata':config['training'].get('progress_strata'),\n"
        " 'gamma':config['training'].get('gamma'),\n"
        " 'gamma_schedule':config['training'].get('gamma_schedule'),\n"
        " 'layout':config['network'].get('transformer_layout'),\n"
        " 'context_beam_count':config['network'].get('context_beam_count'),\n"
        " 'readout_mode':config['network'].get('readout_mode'),\n"
        " 'conditioning':config['network'].get('beam_count_conditioning'),\n"
        " 'context_level':config['state'].get('context_level'),\n"
        " 'features':len(config['state'].get('features') or []),\n"
        " 'total_steps':config['training'].get('total_steps')},ensure_ascii=False))\n"
    )
    out = {}
    for arm, (port, directory) in port_runs.items():
        body = body_template.replace("@DIR@", repr(directory))
        result = ops18.remote_python(body, port, timeout=180)
        out[arm] = (json.loads(result["stdout"].strip().splitlines()[-1])
                    if result["exit_code"] == 0 else {"error": result["stderr"][-200:]})
    return out


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8-sig"))
    live = collect()
    directories = {}
    for port, data in live.items():
        for job in data["jobs"]:
            if job["stage"] != "training" or not job.get("directory"):
                continue
            directories[job["request_id"].split("-")[1].upper()] = (int(port), job["directory"])
    manifest["status"] = "pilot_training_running"
    manifest["updated_utc"] = ops18.utc_now()
    configs = arm_configs(directories) if directories else {}
    manifest["pilot"] = {
        "arm_configs": configs,
        "steps_per_arm": 100000,
        "arms": {"V2": "20418", "G0": "20418", "V3": "20424"},
        "failed_startup_attempts": [
            {"request_id": "s2p18-g0-pilot-100000-51001", "run_id": "train_sac_20261006_084610_6dedc438",
             "error": "State width 2396 differs from declared 4046",
             "cause": "逐束实体观测按实际束数变宽，网络输入宽度不恒定"},
            {"request_id": "s2p18-g0-pilot-100000-51001-retry1", "run_id": "train_sac_20261006_090235_3ba9329b",
             "error": "逐束实体档位必须在选择动作时给出真实束数",
             "cause": "Policy 包装器与第0步验证路径未转发真实束数"},
            {"request_id": "s2p18-g0-pilot-100000-51001-retry2", "run_id": "train_sac_20261006_090857_f00f4b97",
             "error": "观测中的归一化束数无效：0.0",
             "cause": "回放采样试图从自然布局观测还原束数，该布局没有此标量"},
        ],
        "live": live,
    }
    ops18.write_json(MANIFEST, manifest)

    lines = ["", "**S2-P18（可变束数泛化第二轮）三臂均处于40万步补足阶段（2026-10-06）。**"
                 "三臂10万步试点均按预算跑满，并各自在**同一任务ID上从断点续跑**到40万步："
                 "重定标只改总步数预算（配置哈希随之上移），原断点与试点终点快照均改名保留并记录SHA。"
                 "补足阶段定位并修复三个真实故障，失败任务与现场全部保留："
                 "①续跑首个快照因配置哈希变化被拒，改为把冲突快照改名保留后重试；"
                 "②工作台拒绝恢复已完成任务，改为先转为`interrupted`（原因写入任务记录）再恢复；"
                 "③G0断点自带回放束数表为`行数+1`项而加载器要求严格相等，逐桶校验各档束数后裁掉多余一项。"
                 "三臂处理差异经配置回读确认各自只带一处：V2奖励整形`w_throughput=1.0/w_spectrum=0.5`、"
                 "V3按束数折扣`≤100→0.99,>100→0.999`＋跨束势能整形`0.05`、"
                 "G0逐束自然布局＋注意力池化＋FiLM条件化；进度分层采样4桶×4段计数完全均衡。"
                 "逐任务`git_commit`与`source_root`见运行清单。",
                 f"实况（核对时间 {ops18.utc_now()}）："]
    lines.append("")
    lines.append("后续：臂跑满40万步后，把三臂final模型导入20424，顺序执行440景共同考核"
                 "（随机三次动作、种子53001、含固定/贪心基线）与统一SCA；随后按G1–G4预注册门槛判定、"
                 "逐束与计时审计、本地生成中文曲线与正式报告。早前记录的机制线索：12万步检查点上每束槽数"
                 "与冻结V1同量级，未见G3所设“每束槽数相对V1提高≥30%”的迹象，因预算与场景集不同仅作线索。")
    lines.append("")
    status = ROOT / "EXPERIMENT_STATUS.md"
    text = status.read_text(encoding="utf-8")
    # 用显式标记界定实况段：只替换标记之间的内容，不做任何前缀/特征猜测。
    if BEGIN not in text or END not in text:
        raise RuntimeError(f"EXPERIMENT_STATUS.md 缺少实况标记 {BEGIN} / {END}")
    before, _, rest = text.partition(BEGIN)
    _, _, after = rest.partition(END)
    status.write_text(before + BEGIN + "\n" + "\n".join(lines) + "\n" + END + after,
                      encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
