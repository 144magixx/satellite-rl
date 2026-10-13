# 卫星资源分配实验

**S2-L25待审核：**[可变布局与逐束宽度适配方案25](docs/第二阶段/方案/25_可变布局与逐束宽度的T-SAC适配方案.md)已提供通俗说明和有限预算，代码在独立分支完成接口与正确性核验，见[工程记录07](docs/工程文档/07_第二阶段可变布局适配核验.md)。当前不训练；审核批准后才核验远端资源并决定并行。

基于 PyTorch 的卫星波束频率、频率槽及功率分配研究，包含 Transformer/CNN/MLP SAC、两种 PPO、DQN 与启发式基线。

项目先完成结构迁移，随后修复了用户指定的五项问题。文件对应关系见 [迁移说明](docs/工程文档/01_项目迁移说明.md)，最新行为变化与验证结果见 [修复说明](docs/工程文档/02_修复说明.md)。

**新窗口接手：先读 [AGENTS.md](AGENTS.md) 和 [实验状态与接手说明](EXPERIMENT_STATUS.md)。** 状态文件按实验组汇总实际结果、比较条件、报告与权重入口、作废记录和待办；运行状态需按其中说明现场刷新。

**S1-G16已交付：**[报告11](docs/第一阶段/实验报告/11_几何约束负载重分组与MLM候选排序对照.md)比较纯第一阶段几何覆盖与请求负载。联合重分比09单机调整更有效，07更擅长上尾/整体均衡，MLM尚无明确额外收益；少束目标未找到、多束构造完成。5张曲线、完整数据/模型/源码与双端交付均已核验，见[运行清单](outputs/experiments/first_stage_geometry_load_20261012/run_manifest.json)。

## 安装

在项目根目录执行本地可编辑安装，使 Python 能定位 `src/` 中的代码：

```powershell
python -m pip install -e .
```

如果当前环境已安装依赖，可运行 `python -m pip install -e . --no-deps --no-build-isolation`。原结构迁移验证使用 Python 3.11.7 与 PyTorch 2.3.1 CPU；正式 CUDA 实验使用独立的 `DQN` Conda 环境（Python 3.11.11、PyTorch 2.3.1+cu121、NumPy 1.26.4、pandas 2.1.4、Matplotlib 3.8.0）。正式训练启动前仍须现场核对 CUDA 与源码导入路径。

## 目录

近期实际代码已分别整理为[第一阶段BC/PPO笔记本](docs/第一阶段/实验规范/01_BC与PPO训练代码.ipynb)、[第二阶段SAC三路线笔记本](docs/第二阶段/实验规范/01_SAC三路线训练代码.ipynb)、[第三阶段连续功率优化笔记本](docs/第三阶段/实验规范/01_连续功率优化代码.ipynb)，各附同名HTML代码阅读版。源码格取自对应实验冻结提交并配中文解释，默认执行仅装载定义和核配置，正式任务仍由工作台管理。

想先看懂最近做过什么实验，可从[三阶段近期实验导读（HTML）](docs/资料/08_三阶段近期实验导读.html)开始，另附[可执行Jupyter笔记本](docs/资料/08_三阶段近期实验导读.ipynb)。按覆盖、频率、功率分别解释实验目的、结果和限制，图表只读已有结果；2026-10-12整理，无新增训练或环境评价。

跨阶段研究复盘见[三阶段研究瓶颈与关键问题分析](docs/资料/04_三阶段研究瓶颈与关键问题分析.md)：按开题报告审视目标可达性、学习价值、功率细调收益及最终用户服务证据；属于研究分析，不新增实验。

2026-10-10的中期材料分为[导师讨论版](docs/资料/06_中期研究进展与路线调整_导师讨论版.md)与[正式内容版](docs/资料/07_HTS资源分配中期报告_正式版.md)，均附可编辑Word。06保留完整历史效果及覆盖索引；07按用户更正，先分析开题行文再按其学术论证方式重写，以场景、模型、方法和证据组织正文，历史明细引用06。保留失败与未执行范围，不新增科学计算。

文档按[第一阶段覆盖规划](docs/第一阶段/00_目录.md)和[第二阶段频率槽与功率分配](docs/第二阶段/00_目录.md)分开，阶段内按类别归档；共用资料与完整入口见[文档总目录](docs/00_目录.md)。各阶段与类别独立编号，第一阶段首个方案为01，第二阶段及共用资料保留已发布编号。

本机浏览器工作台可在项目根目录运行 `powershell -NoProfile -File scripts/start_workbench.ps1` 启动，默认地址为 `http://127.0.0.1:8765/`。支持状态与奖励组合、三算法/三骨干、真实网络结构、后台训练、全面监控和模型测试。详见[使用说明](docs/工作台/03_实验工作台使用说明.md)与[验收记录](docs/工作台/02_实验工作台验收记录.md)。新增网络的学习效果仍需正式对照验证。

现已增加独立欧洲环境选项、航空轨迹与合成需求导入、按环境与用途区分的训练/验证/考核场景，并冻结评价数据和权重。操作及295项验证结果见[欧洲航空环境使用与核验](docs/工作台/04_欧洲航空环境使用与核验.md)。欧洲正式训练及共同考核已完成，结果见[欧洲六组独立场景报告](docs/第二阶段/实验报告/07_欧洲六组独立场景对照.md)；最新范围和第一阶段方案状态以EXPERIMENT_STATUS为准。

第一阶段布局环境、PPO与基线的本地源码在 `src/satellite_rl/coverage/`；工作台提供规则、聚类和无基础布局纯PPO三种入口。计算在用户指定的远端4090D/DQN执行，通过SSH隧道访问 `http://127.0.0.1:8876/#coverage`。`scripts/remote_stage1.py`负责已提交源码同步与隧道，认证读取 `BUPT_PASSWORD`；`scripts/run_stage1_suite.py`只通过同一工作台API协调七组任务。布局覆盖与负载采用独立评价，不使用第二阶段T-SAC奖励。预算、数据划分和并行依据见[第一阶段技术方案03](docs/第一阶段/方案/03_远端4090D部署与七组并行实验技术方案.md)，实际任务及结果以[实验状态](EXPERIMENT_STATUS.md)为准。

第一阶段新研究见[方案10 可变波束数全覆盖与负载均衡调研](docs/第一阶段/方案/10_可变波束数全覆盖与负载均衡调研方案.md)：最多200束、全部飞机覆盖、低中高需求和12条有出处的路线。用户已授权[方案11 四实例验证](docs/第一阶段/方案/11_十二路线四实例验证执行方案.md)，已在四端隔离工作台执行，两PPO及全部考核已完成，结果见[正式报告08](docs/第一阶段/实验报告/08_可变波束全覆盖与负载均衡十二路线对照.md)：受测场景全部几何覆盖、实际38–111束，均衡/承载取舍和学习/时序局限保留；交付状态以实验状态为准，不与上述历史七组入口混用。

第一阶段后续按[方案12 R04引导的DRL波束覆盖优化](docs/第一阶段/方案/12_R04引导的DRL波束覆盖优化方案.md)扣住开题DRL主线：R04作初解、算子和强对照，DRL承担真实布局决策；奖励、指向宽度和动态验证分阶段推进。用户现已授权[方案13 A阶段四实例执行](docs/第一阶段/方案/13_DRL覆盖A阶段四实例执行方案.md)，两PPO各262144、六法1800条独立考核与严格审计已完成，见[正式报告09](docs/第一阶段/实验报告/09_R04引导的DRL覆盖A阶段对照.md)：全部几何覆盖、实际40–111束；PPO能改进初解，但未过预设均衡/效率门槛，BC后继续PPO未显示追加收益。七张曲线/分层图、逐母计时及37任务归档均提供入口；后续指向宽度与时序尚未实施。

**2026-10-12更新：上述DRL主线是方案12形成时的定位，现已放宽为纯AI或AI＋优化。** [方案15 掩码布局重构与约束优化](docs/第一阶段/方案/15_掩码布局重构与约束优化方案.md)给出飞机归属补全、候选筛选＋局部优化及同条件对照设计；无需追加PPO，全部新预算仅建议，尚未实施。

2026-10-12新增[第一阶段路线综合调研](docs/第一阶段/方案/14_波束覆盖优化可行路线综合调研.md)：覆盖聚类、整数/约束规划、列生成、局部重建、学习辅助搜索、生成式方法及动态规划外层，共32类做法、41项方法来源。正文采用简短路线表，详细文献和检索记录放附件；这是研究方向比较，未启动新训练或评价。

已按用户要求建立并实施[非DRL专题](docs/第一阶段/方案/非DRL专题/00_专题总览与共同比较约定.md)，12份设计分别说明具体算法、独立对照、预算和停止条件。MLM仅是其中一条；S1-ND12已完成数据准备与开发筛选，MLM的10000更新与5700条共同考核已完成，等价数据读取来源保留；20474临时借出，本组用其余三物理端，原始归档、全量统计与5图已核，[报告10](docs/第一阶段/实验报告/10_非DRL十二路线覆盖与负载均衡对照.md)给出完整对比；最终小包与原始分卷已完成双端核验，见[最终验收](outputs/experiments/first_stage_nondrl12_20261012/delivery/final_root_acceptance.json)。具体源码、预算及进度见[运行清单](outputs/experiments/first_stage_nondrl12_20261012/run_manifest.json)。

```text
src/satellite_rl/
  envs/         两套原仿真环境
  agents/       网络、动作选择、损失和参数更新
  training/     八个训练入口与共享训练循环
  baselines/    固定、贪心、随机、四色、图着色、拉格朗日
  evaluation/   指标比较、需求扫描、逐波束比较
  plotting/     速率柱状图与需求曲线
  coverage/     第一阶段布局环境、PPO、基础规则和独立覆盖评价
  workbench/    本地工作台、配置、场景、任务与前端
  paths.py      原路径到新位置的统一映射
data/           城市数据
models/
  pretrained/   五组原评估权重
  runs/         新训练权重
outputs/
  logs/         实验日志
  metrics/      指标数据
  figures/      训练与论文图表
  evaluation/   场景评估图
docs/
  第一阶段/     覆盖规划方案与后续验证
  第二阶段/     资源分配方案、实验报告与规范
  方案/         两阶段衔接方案
  工程文档/     共用工程说明与核验
  工作台/       共用操作与配置说明
  资料/         开题报告与原始参考资料
examples/       图着色草稿、随机数演示
tests/          结构、路径和权重兼容性检查
```

## 训练

```powershell
python -m satellite_rl.training.train_sac_transformer
python -m satellite_rl.training.train_sac_cnn
python -m satellite_rl.training.train_sac_mlp
python -m satellite_rl.training.train_sac_mlp_matched
python -m satellite_rl.training.train_sac_cnn_matched
python -m satellite_rl.training.train_ppo_grouped
python -m satellite_rl.training.train_ppo_direct
python -m satellite_rl.training.train_dqn
```

上述旧训练入口保留脚本内参数配置方式，默认 4000 回合；没有引入 `--episodes` 或 `--seed` 等命令行参数。五种 SAC 共用 `training/sac_runner.py` 的训练流程。原三个入口仍使用原参数；新增的 `sac_mlp_matched` 使用两层 512 维隐藏层，`sac_cnn_matched` 使用 128→128 的融合层，学习率、动作转换及其余训练设置沿用共享驱动。

新增两种变体的单个 Actor/Critic 分别为 429,176 和 447,176 个参数，接近 Transformer 的 417,272；原 MLP/CNN 版本独立保留。2026-09-24 已按用户收窄后的范围完成种子 51001 的六组正式训练与补充评价，只有一个训练种子，不能据此估计重训波动；结果见 [单种子六组报告](outputs/experiments/sac_comparison_20260924_r2/single_seed_51001/analysis/中文单种子六组报告.md)。训练变体权重和原预训练资产仍分别保存在各自目录。

PPO 参数位于对应训练入口及辅助模块，DQN 原常量保留在 `agents/dqn.py`。两套环境的行为差异、动作映射和默认值均保留。

## 评估与绘图

```powershell
python -m satellite_rl.evaluation.compare_metrics
python -m satellite_rl.evaluation.demand_sweep
python -m satellite_rl.evaluation.compare_beam_satisfaction
python -m satellite_rl.plotting.plot_rate_comparison
python -m satellite_rl.plotting.plot_demand_satisfaction
python -m satellite_rl.baselines.fixed
python -m satellite_rl.baselines.greedy
python -m satellite_rl.baselines.random_policy
```

评估使用 `models/pretrained/` 的原权重；训练结果写入 `models/runs/`，不会自动替换预训练资产。脚本中的算法选择列表保持原样，例如逐波束比较当前只启用 T-SAC。

`paths.py` 相对项目位置解析输入输出，可编辑安装后从其他工作目录也能定位资产。`SATELLITE_RL_OUTPUT_ROOT` 可将新日志、指标、图表和训练权重重定向到独立目录，输入数据及预训练权重仍从本项目读取，主要用于隔离验证。

部分算法构造函数仍按原行为初始化日志，日志与保存方法也保留在算法类中；本次仅拆分训练流程，没有重写类内部职责。导入包模块不会启动训练、评估或绘图。

## 验证

```powershell
python -m unittest discover -s tests -v
```

测试覆盖算法依赖边界、全包导入、跨工作目录路径解析，以及五组原权重的严格加载和前向计算。迁移前后代码核对、运行对照和已知问题详见 [迁移说明](docs/工程文档/01_项目迁移说明.md)。

测试按模块放在 `tests/test_*.py`；包括 PPO 终止目标、SAC Critic 保存、终止步资源登记、训练/评估功率一致性、权重失败中止、匹配规模变体，以及六组单种子报告和补充评价入口。PPO direct 的训练与评估统一使用 3–30 W；缺少或不兼容的权重会明确报错并停止评估。

## 完整 Python 文件地图

本项目各个 Python 源码、测试、迁入历史文件的对应关系、生成的验证辅助脚本及模块依赖见 [README_CODEBASE.md](README_CODEBASE.md)。该索引还标明哪些入口为已取消的 18 组历史工具，哪些文件用于当前六组单种子结果。
