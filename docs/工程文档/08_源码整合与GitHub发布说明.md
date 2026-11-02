# 源码整合与 GitHub 发布说明

来源：codex。2026-11-03。

当前开发代码统一整理到 `main`，公开仓库为 [144magixx/satellite-rl](https://github.com/144magixx/satellite-rl)。除已有三阶段主线，本次补入了归档开发线中从未进入主线的 302 个源码、脚本与测试文件，以及 4 份配套文档；原版框架复现另保存一份独立环境快照。各开发线的原始历史继续由 `archive/*` 标签保留。

## 整合范围与来源

| 内容 | 当前源码入口 | 主要归档来源 |
| --- | --- | --- |
| 文献对照 | [litcmp](../../src/satellite_rl/litcmp/)、[配套脚本](../../scripts/litcmp/) | `archive/litcmp` |
| 改进式 SAC 网络、直出 T-SAC 分组选项与尺寸选项 | [sac_improvement](../../src/satellite_rl/sac_improvement/)、[s2fix](../../src/satellite_rl/s2fix/) | `archive/cc/tsac-groups`、`archive/cc/tsac-size-study`、`archive/cc/i0-network-study` |
| 原版框架复现 | [original_compare](../../src/satellite_rl/original_compare/) | `archive/cc/orig-framework-compare`；原环境取自 `45427b4` |
| 第一阶段后续搜索、模型准备、数据与审计工具 | [coverage](../../src/satellite_rl/coverage/)、[workbench](../../src/satellite_rl/workbench/)、[scripts](../../scripts/) | `archive/codex/stage1-r8-owner-guidance`、`archive/codex/stage1-ai22`、`archive/detached/stage1-r7-recovery` |
| 历史第二阶段布局、三算法对照与运行工具 | [layout_resource](../../src/satellite_rl/layout_resource/)、[t0_comparison](../../src/satellite_rl/t0_comparison/)、[workbench](../../src/satellite_rl/workbench/) | `archive/codex/stage2-layout-width-v25`、`archive/detached/stage2-t0-fair24`、`archive/codex/stage2-fair20-parallel`、`archive/codex/stage2-fair20-repair2` |
| 早期 SAC 因子对照 | [training](../../src/satellite_rl/training/) | `archive/detached/frozen-sac-factorial-20260925` |

这些缺失源码在主线历史中从未出现；已被主线移动或删除的旧文档没有重新恢复。公共文件只补齐新模块需要的接口，当前第一阶段、S2-LAB 与 S3-LAB 的科学计算实现继续保留。

原版复现默认使用独立的 [env_45427b4.py](../../src/satellite_rl/original_compare/env_45427b4.py)，避免默认绑定随主线变化的环境。该快照与原提交字节相同；7 个离轴角的方向图，以及固定种子 51001 下 100 束场景的位置、距离和需求，均与原提交逐值一致。

## 安装与验证

[打包配置](../../pyproject.toml)同时发行 `satellite_rl` 与其编排模块依赖的 `scripts` 命名空间，因此在仓库之外导入工作台模块也能找到所需合同与校验代码。可编辑安装与 wheel 构建均采用已有 DQN 环境，不重新安装第三方依赖：

```powershell
python -m pip install -e . --no-deps --no-build-isolation
python -m pip wheel . --no-deps --no-build-isolation --wheel-dir <临时目录>
```

本次使用 Python 3.11、PyTorch 2.3.1+cu121，确认包实际来自当前项目。针对性检查涵盖：全包隔离导入、跨目录路径、原预训练权重加载、第一阶段几何、T-SAC 网络尺寸与分组、实际小型参数更新、I0 网络、功率约束、文献方法，以及 S2-LAB/S3-LAB 容量和梯度核对。48 项中 45 项通过，另有 15 项历史公共接口与模拟测试通过。

3 项归档对照缺少本地资产，已明确跳过：原版复现实验的开发集与确认集分散规则逐景 JSONL，以及 L1 种子 51001 的 actor 文件。没有生成替代数据来让测试通过；全量历史测试套件未运行。独立复核发现的旧构造函数默认参数丢失、MLP 与自主分组组合不兼容已修正；默认网络与已支持组合保留原行为。

历史 F20 双端迁移等运维代码在此保留来源与接口，其完整服务路由仍应使用对应 `archive/*` 冻结版本；本次检查不表示所有旧实验服务都已在当前入口恢复。未恢复已暂停任务，也未进行新的正式训练或科学评价。

## GitHub 中包含的内容

源码、脚本、测试、版本化文档、已有输入数据与原预训练权重随仓库上传。实验原始结果、生成网页的数据、新训练权重与缓存继续按 [.gitignore](../../.gitignore) 保存在本地归档；文档指向 `outputs/`、`models/runs/` 的链接需要相应本地文件。

上传前检查了全部 Git 引用可达的历史对象：未发现凭据候选或超过 GitHub 100 MiB 限制的文件。凭证始终由本机 Git 凭证管理器提供，没有写入仓库。
