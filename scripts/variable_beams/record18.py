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

    lines = ["", "**S2-P18双卡并行试点训练运行中（2026-10-06）：**三臂各10万步已提交并在两卡并行执行，"
                 "20418 跑 V2 与 G0（同卡两 worker，服务 max_workers=2），20424 跑 V3。"
                 "冻结源码提交 `fb14e36`（G0 三次启动失败已定位并修复，失败任务全部保留）："
                 "①逐束实体观测曾按实际束数变宽导致网络输入宽度不恒定；②Policy 包装器与第0步验证未转发真实束数；"
                 "③回放采样试图从自然布局观测还原束数。修复后三者分别改为固定宽度补零＋有效位屏蔽、"
                 "全链路传递真实束数、束数随转移写入回放。"
                 "V2/V3 的运行快照为 `0070cb49`（其代码路径不涉及上述三处），G0 为 `fb14e36`，"
                 "逐任务 `git_commit` 与 `source_root` 已记录在运行清单。",
                 f"实况（核对时间 {ops18.utc_now()}）："]
    for port, data in live.items():
        for job in data["jobs"]:
            if job["stage"] == "training":
                lines.append(f"- {port} {job['request_id']}：{job['env_step']}/{job['budget_env_steps']} 步、"
                             f"更新 {job['update_step']}、{job['steps_per_second']:.2f} 步/秒")
        if data.get("host"):
            lines.append(f"- {port} GPU：{data['host']['gpu']}（compute apps: {data['host']['apps'] or '无'}）")
    lines.append("")
    lines.append("后续：跑满10万步后按 G1–G4 门槛判定（以 N=100/120/160 为主），通过臂补足到40万步（总上限120万交互）；"
                 "随后统一 SCA 共同评价、逐束与计时审计、中文正式报告。")
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
