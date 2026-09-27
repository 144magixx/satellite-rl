# 卫星资源分配实验

基于 PyTorch 的卫星波束频率、频率槽及功率分配研究，包含 Transformer/CNN/MLP SAC、两种 PPO、DQN 与启发式基线。

项目先完成结构迁移，随后修复了用户指定的五项问题。文件对应关系见 [迁移说明](docs/迁移说明.md)，最新行为变化与验证结果见 [修复说明](docs/修复说明.md)。

**新窗口接手：先读 [AGENTS.md](AGENTS.md) 和 [实验状态与接手说明](EXPERIMENT_STATUS.md)。** 状态文件按实验组汇总实际结果、比较条件、报告与权重入口、作废记录和待办；运行状态需按其中说明现场刷新。

## 安装

在项目根目录执行本地可编辑安装，使 Python 能定位 `src/` 中的代码：

```powershell
python -m pip install -e .
```

如果当前环境已安装依赖，可运行 `python -m pip install -e . --no-deps --no-build-isolation`。原结构迁移验证使用 Python 3.11.7 与 PyTorch 2.3.1 CPU；正式 CUDA 实验使用独立的 `DQN` Conda 环境（Python 3.11.11、PyTorch 2.3.1+cu121、NumPy 1.26.4、pandas 2.1.4、Matplotlib 3.8.0）。正式训练启动前仍须现场核对 CUDA 与源码导入路径。

## 目录

本机浏览器工作台可在项目根目录运行 `powershell -NoProfile -File scripts/start_workbench.ps1` 启动，默认地址为 `http://127.0.0.1:8765/`。支持状态与奖励组合、三算法/三骨干、真实网络结构、后台训练、全面监控和模型测试。详见[使用说明](docs/实验工作台使用说明.md)与[验收记录](docs/实验工作台验收记录.md)。新增网络的学习效果仍需正式对照验证。

```text
src/satellite_rl/
  envs/         两套原仿真环境
  agents/       网络、动作选择、损失和参数更新
  training/     八个训练入口与共享训练循环
  baselines/    固定、贪心、随机、四色、图着色、拉格朗日
  evaluation/   指标比较、需求扫描、逐波束比较
  plotting/     速率柱状图与需求曲线
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
docs/           原论文文档、迁移说明和核对报告
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

测试覆盖算法依赖边界、全包导入、跨工作目录路径解析，以及五组原权重的严格加载和前向计算。迁移前后代码核对、运行对照和已知问题详见 [迁移说明](docs/迁移说明.md)。

测试按模块放在 `tests/test_*.py`；包括 PPO 终止目标、SAC Critic 保存、终止步资源登记、训练/评估功率一致性、权重失败中止、匹配规模变体，以及六组单种子报告和补充评价入口。PPO direct 的训练与评估统一使用 3–30 W；缺少或不兼容的权重会明确报错并停止评估。

## 完整 Python 文件地图

本项目各个 Python 源码、测试、迁入历史文件的对应关系、生成的验证辅助脚本及模块依赖见 [README_CODEBASE.md](README_CODEBASE.md)。该索引还标明哪些入口为已取消的 18 组历史工具，哪些文件用于当前六组单种子结果。
