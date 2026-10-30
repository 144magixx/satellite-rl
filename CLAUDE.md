# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 先读什么

- **`AGENTS.md` 是本项目的约束原件**（用户逐条下达的工作规则、授权范围、暂停/继续记录、资源与空间管理、报告写法）。它的效力高于本文件；本文件只做代码与命令导航。两者冲突以 `AGENTS.md` 和用户最新指令为准。
- **`EXPERIMENT_STATUS.md`** 是各实验组的人工接手摘要（以 `<!-- XXX_START -->` 标记分段）。正在运行或状态不明的组，必须以远端工作台API、进程和运行清单 `outputs/experiments/<组>/run_manifest.json` 现场刷新，不能凭旧 `running`、残留PID或旧快照下结论。
- 根目录的 `job.json`、`status.json`、`checkpoint.pt`、`*.jsonl`、`worker.log` 等是2026-09-25一次失败任务误写的散件（已在 `.gitignore`），**不是**任何任务的状态来源。
- `README_CODEBASE.md` 有逐文件的来源（历史/拆分/新增）与依赖地图。

## Claude 在本项目的分工（2026-10-17起）

S2-W30 起，**面向用户的实验报告由 Claude Code 撰写**，要求通俗易懂；Codex（automation-2）负责训练、评价、故障接续和归档，不再写 docs 下的 W30 正式报告。Claude 负责盯进度、核原始数据并写报告。数据入口：`outputs/experiments/s2_power30_20261016/run_manifest.json` 与 `control/execution_handoff.json`。写报告、读数据不构成启动/恢复实验的授权。

### 写作与编码风格（用户 2026-10-17 明确要求，已同步写入 `AGENTS.md` 顶部，对 Codex 同样适用）

用户看不懂 Codex 写的方案和报告，原因是写法，不是内容。不要犯同样的错：

- **先说结论和它意味着什么**，再讲为什么。例："单步换动作挖不出收益（匹配度只涨0.002，门槛0.01），所以不往这个方向训练。"
- **不写防御性语言。** 不要在每个结论后堆"这不证明……也不表示……不能把……"。真正重要的限制集中写一次，一两句说清。
- **代号第一次出现就用一句人话解释**（A1、S10、母场景、续策等），能不用代号就不用。
- **数字只保留到能看出大小**，并给出对比（"不到门槛一半"），不写九位小数。
- **流程/授权/预算/暂停/谁核验这类管理信息不进正文**，需要的放附录或运行清单。
- **代码不做防御性编程**：不为不会发生的情况加层层校验、守卫、回退和断言；按正常路径写清楚，真出错让它报错。
- **不要例行计算 SHA/hash。** 只在确有需要时（如核对传输是否损坏）算一次，不把算哈希、写回执当作工作本身。

Claude 生成的文档统一放在 `docs/Claude维护/`，按两位编号续写，并更新该目录的 `00_说明.md` 表格与 `docs/00_目录.md`。

## 命令

```bash
python -m pip install -e .                       # 可编辑安装（src/ 布局，包名 satellite_rl）
python -m unittest discover -s tests -v          # 全部测试（约240个文件，标准库 unittest）
python -m unittest tests.test_agent_fixes -v     # 单个测试文件
python -m unittest tests.test_agent_fixes.SomeTestCase.test_name   # 单个用例
```

- 正式训练环境是 Conda `DQN`（Python 3.11、PyTorch 2.3.1+cu121）：`conda run -n DQN python ...`。启动前核实际解释器、`torch.cuda.is_available()` 和 `satellite_rl.__file__`（可编辑安装可能指向开发目录而非冻结快照）。不得静默退回 base 或 CPU。
- 验证时用 `SATELLITE_RL_OUTPUT_ROOT=<临时目录>` 隔离输出；输入数据和预训练权重仍从项目内读取（`paths.py` 统一解析）。
- 无 lint/格式化工具。可选依赖 `pip install -e .[power-priority]`（cvxpy/clarabel，第三阶段功率优化）。
- 不要为检查导入而启动完整训练；旧 `train_sac_*` 入口默认跑4000回合且无命令行参数。

## 实验如何真正运行

- **正式训练/长评价必须经"实验工作台"发起**（页面或其同一API），生成完整任务记录；不能直接跑训练脚本再把结果放进目录。工作台：`satellite_rl/workbench/server.py`（`satellite-workbench` 入口，`--port --archive-root --python --max-workers --no-scheduler`），本地启动 `powershell -NoProfile -File scripts/start_workbench.ps1`（默认 http://127.0.0.1:8765/），前端在 `workbench/static/`。
- 新增实验开关必须贯通：前端配置 → 服务端校验 → 冻结任务配置 → 模型/断点元数据 → 训练与评价真正使用。改动时评估前端是否需要同步。
- 计算默认在远端（具体主机/端口按 `AGENTS.md` 最新授权段，当前 W30 为 `u2025140002@10.160.17.102` 的指定端口），凭据只读环境变量 `BUPT_PASSWORD`。每组有独立远端目录、工作台端口、归档与锁；正式任务绑定冻结提交的源码快照，运行中不得热改。
- `scripts/` 下是各轮实验的编排/审计/出图脚本（`remote_stage1.py` 负责源码同步与SSH隧道）；`outputs/experiments/**` 里的脚本是单次核验产物，不是默认程序。

## 架构总览

研究分三阶段，代码按阶段分模块（均在 `src/satellite_rl/`）：

| 阶段 | 问题 | 主要模块 |
| --- | --- | --- |
| 第一阶段：覆盖规划 | 给全部飞机定波束（可变束数 N≤200、指向/宽度），唯一归属、需求守恒、负载均衡 | `coverage/`（环境、PPO、重构、`variable_*`、`nondrl/`、`geoload/`、`ai_search*/` 各轮AI搜索） |
| 第二阶段：频率槽与功率 | 逐束选频率起点、槽数、功率 | `envs/satellite.py`（主环境）、`agents/`（Transformer/CNN/MLP SAC、PPO、DQN）、`training/`、`sac_improvement/`、`fair_comparison/` |
| 第三阶段：功率细调 | 固定槽后连续功率优化（SCA等） | `power_refinement/` |
| 跨阶段接口 | 同一飞机ID/位置/请求、几何与链路 | `shared_interface/` |

- **依赖边界**：`agents/` 算法模块不得在运行时导入 `training/` 或 `envs/`（`TYPE_CHECKING` 除外）；入口用 `if __name__ == "__main__":`；路径一律经 `paths.py`。
- `training/sac_runner.py` 是五种 SAC 的共享训练循环；`training/experiment_runner.py` 是可复现对照入口（`--model --formula --reward-variant --seed --steps --run-id --resume`）。`agents/sac_math.py` 放共享温度与联合双Q期望计算。
- `workbench/` 体量最大：除服务/调度（`server.py`、`service.py`、`runner.py`、`recovery.py`、`storage.py`、`telemetry.py`）外，大量 `r4_*`…`r7_*`、`fair20_*`、`s2p19_*`、`coverage_*` 文件是各实验轮次的阶段控制、守卫和数据准备，按轮次冻结，一般只读不改。
- 关键领域约束（详见 `AGENTS.md`）：每机统一固定请求 21.6 Mbps；第二阶段逐束需求必须等于第一阶段归属飞机请求之和；SI（需求满足率）与 SGM（供需匹配程度）分开报告，不同奖励的回报不直接排名；终态指标覆盖全部目标波束含未服务束。

## 文档与报告

- `docs/` 按阶段分：`docs/第一阶段/`、`docs/第二阶段/`、`docs/第三阶段/`，阶段内再分 `方案/`、`实验报告/`、`实验规范/`；共用的在 `docs/工作台/`、`docs/工程文档/`、`docs/资料/`。新文档先看对应目录和 `docs/00_目录.md`，用该系列下一个两位编号（如 `23_主题.md`），并同步目录、`EXPERIMENT_STATUS.md` 与链接。
- 正式报告必须附从原始数据生成的曲线；图、数据、脚本放 `outputs/`，模型放 `models/runs/`（二者均被 git 忽略）。对比表写"编号＋实验组合"，不只写 R0/P0。比较需同时报各场景耗时（分段：初始解/细调/核验/总计）。
- 用语沿用论文/开题报告（波束、频率槽、功率、需求满足率、供需匹配程度、干扰），先讲结论和实际含义；用户不熟悉凸优化，避免以术语为主解释。
- 与项目相关的用户问答同步记到 `Q&A/<类别>/<主题>.md` 并更新 `Q&A/问答索引.md`。

## Git

- 主分支 `main`，无远程。一次提交一个目的，前缀 `fix:`/`docs:`/`refactor:` 等；只暂存本任务文件（工作区常有其它会话的未提交改动，不要顺手提交或还原）。
- 较大算法/环境改动先开分支；撤销用 `git revert`，不用 `reset --hard`/`clean -fd`。
