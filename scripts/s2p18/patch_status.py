"""把 S2-P18 实现记录插入 EXPERIMENT_STATUS.md 顶部（幂等）。"""

from pathlib import Path

p = Path("EXPERIMENT_STATUS.md")
text = p.read_text(encoding="utf-8")
marker = "**S2-P18 三条实验臂代码已实现并本地验证（2026-10-06，开发分支 codex/s2p18-variable-beams-r2）：**"
if marker in text:
    print("已存在，跳过")
else:
    entry = (
        marker
        + "按方案18 第二版实现奖励吞吐/频谱项、power_pressure 归一化修正、组号与预留义务可见、"
        "进度分层采样、按束数折扣、跨束势能型奖励整形、逐束实体编码（任意 N＋注意力池化＋FiLM）"
        "以及实验编号 S2-P17/S2-P18 集合化；默认配置（旧奖励、210维状态、旧读出）逐位不变，"
        "100 束同场景同动作的 210/207/204 维状态与奖励序列与改动前完全一致，仅在 N=150 的 "
        "power_pressure 分量按新口径变化（修正而非新增信息）。**尚未创建训练、评价或工作台任务，"
        "未在 main 上改动，未启动任何训练。** 验证：`python -m unittest discover -s tests` 704 项、"
        "2 项失败（`test_variable_beam_ops` 的 8781 端口占用检查，改动前即在 main 上失败）、"
        "跳过 4；三臂 `/api/validate` 等价流程与 CUDA 前向/单次更新冒烟通过。"
        "实现报告见开发树内 `docs/第二阶段/方案/`（本轮交付说明）。\n\n"
    )
    lines = text.split("\n")
    # 标题行 + 空行之后插入
    assert lines[0].startswith("# 实验状态")
    lines.insert(2, entry)
    p.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print("已插入 S2-P18 状态条目")
