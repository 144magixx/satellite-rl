# 匹配规模 SAC 变体说明

实施日期：2026-09-24。

## 目的与范围

在原 Transformer、CNN、MLP SAC 之外，新增参数量接近当前 Transformer 的 MLP 与 CNN 变体，用于后续容量对照。`matched` 表示参数规模接近，不表示逐个参数相等，也不保证表达能力、计算量或性能相等。

本次保留原算法与两套环境文件。新模块沿用原文件的中文注释、四空格缩进、`ReplayBuffer / PolicyNet / ValueNet / SAC` 分组和接口。SAC 损失、目标计算、动作选择、软更新及优化器配置沿用各自原版本；仅新变体的网络结构、宽度及实验输出位置不同。

## 文件与结构

| 变体 | 算法模块 | 训练入口 | 网络配置 |
| --- | --- | --- | --- |
| MLP 匹配规模版 | `src/satellite_rl/agents/sac_mlp_matched.py` | `src/satellite_rl/training/train_sac_mlp_matched.py` | 204→512→512→三个动作/Q头，两个隐藏层均有 ReLU |
| CNN 匹配规模版 | `src/satellite_rl/agents/sac_cnn_matched.py` | `src/satellite_rl/training/train_sac_cnn_matched.py` | 原 2→16→32 卷积，展平后融合层 3204→128→128，再接三个动作/Q头 |

两个变体都同时应用于 Actor 与各个 Critic，输出维度仍为 100、10、10。MLP Critic 在新增版本中使用非线性；原 MLP Critic 的线性实现保留，因此两者是独立配置，不是把旧权重换名。

## 参数量

| 变体 | 单个 Actor | 单个 Critic | 直接优化参数 | 含目标网络训练总参数 |
| --- | ---: | ---: | ---: | ---: |
| Transformer（参照） | 417,272 | 417,272 | 1,251,817 | 2,086,361 |
| MLP 匹配规模版 | 429,176 | 429,176 | 1,287,529 | 2,145,881 |
| CNN 匹配规模版 | 447,176 | 447,176 | 1,341,529 | 2,235,881 |

新 MLP/CNN 的单网络参数分别比参照多约 2.9% 和 7.2%。直接优化参数为 Actor＋两个 Critic＋一个 `log_alpha` 标量；总参数另外包含两个目标 Critic。不含梯度、优化器状态或激活。

## 运行

在已可编辑安装本项目的 Python 环境中执行：

```powershell
python -m satellite_rl.training.train_sac_mlp_matched
python -m satellite_rl.training.train_sac_cnn_matched
```

两入口共用 `training/sac_runner.py`，分别传入 `n_hiddens=512`、`n_hiddens=128` 和对应实验名称。共享驱动新增可选关键字 `n_hiddens`、`experiment_name`，原入口不传时仍采用 128 和原输出目录。默认回合数仍为 4000，经验池 20,000、预热阈值 2,000、批次 256，学习率等其余参数未改变。

## 独立输出

将下表中的 `<variant>` 替换为 `sac_mlp_matched` 或 `sac_cnn_matched`：

| 内容 | 位置 |
| --- | --- |
| 模型 | `models/runs/<variant>/` |
| 日志 | `outputs/logs/<variant>/` |
| 指标 | `outputs/metrics/<variant>/` |
| 曲线 | `outputs/figures/<variant>/` |

每次保存包括 Actor、两个独立 Critic、两个目标 Critic及完整模型，共六个权重文件。输出使用现有 `paths.project_path()`，支持 `SATELLITE_RL_OUTPUT_ROOT` 隔离运行。

现有评估算法列表仍对应原模型及原预训练资产。新增变体没有完整训练权重，也没有自动加入默认评估列表。训练完成后应加载对应新类和权重进行对照，不能把旧 MLP/CNN 的参数文件直接加载到这些不同形状的网络中。

## 验证

执行 `python -B -m unittest discover -s tests -v`，25 项测试全部通过，其中新增 5 项：

1. Actor/Critic 参数量、批次前向形状、有限值及动作概率归一化。
2. 构造受控权重验证新 MLP Critic 确实具有非线性。
3. 两个新 SAC 各进行一次真实参数更新，核对 Actor、Critic、目标网络变化和有限值；保存后逐张量核对五个独立网络文件，并严格重载 Actor。
4. 新训练入口正确传递算法、宽度和实验名称；原三个入口仍使用原默认调用。
5. 两个实际入口各执行一个回合，验证日志、指标、曲线和六个模型文件均进入独立目录。

训练入口测试仅在测试进程内将回合数临时限制为 1，未修改生产默认值；保持原经验池预热阈值，单步更新由单独测试覆盖。测试产物写入临时目录，结束后清理。CPU、PyTorch 2.3.1；运行中原环境存在 NumPy 标量转换弃用提示，原 Transformer 存在 nested-tensor 性能提示，本次没有扩大范围修改。

另外核对了本次新增前后所有原算法与环境文件的 SHA-256，完全一致；新 SAC 类的数值计算方法与对应原版本 AST 一致，类中日志和保存方法仅使用新名称与路径。

本次验证说明新增实现及入口可以运行，不证明收敛、泛化性能或参数匹配后的优势。完整训练与多种子评估尚未进行。
