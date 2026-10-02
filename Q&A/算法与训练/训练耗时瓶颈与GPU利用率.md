# 训练耗时瓶颈与 GPU 利用率

回答日期：2026-09-24。

**同日后续状态更新：用户已授权修复 DQN，现已完成 CUDA 环境修复和验证，详情见文末“追问：先修好 DQN 的 CUDA 运行环境”。下面的 CPU 耗时与环境状态保留为修复前记录，不能再用于描述修复后的 DQN；base 仍未改变。**

**最新实施状态：用户随后授权“只做性能优化，原有框架不要动”，保守提速已落地，35 项测试及新旧数值对照通过。原型阶段的“未修改源码”保留为历史记录；当前状态见文末实施追问及 [性能优化说明](../../docs/工程文档/04_性能优化说明.md)。两项 SAC 学习公式修正及正式对照训练仍未启动。**

## 问题

当前项目的训练耗时瓶颈在哪里？是否因为太多计算在 CPU 上运行，导致 GPU 利用率不高？

## 结论

本机已核查的默认/base 和 `DQN` 环境均未启用 CUDA：两者都是 CPU 版 PyTorch。用 base 启动现有训练入口时，模型更新也会落在 CPU；DQN 另外存在 Python 版本不满足项目声明、未安装项目的问题。因此第一优先级是解决实际训练解释器与 CUDA 构建，而不是先把整个 NumPy 环境搬上 GPU。

短时 CPU 微基准显示，当前 SAC 学习阶段的主要耗时在网络更新：默认 20 个 PyTorch intra-op 线程时，Transformer SAC 一次 batch=256 更新约 223 ms，CNN 约 55 ms；独立环境采样平均约 3 ms/步。环境仿真确实存在重复 CPU 计算，但本次测量不支持它是当前 Transformer/CNN 总训练耗时的首要原因。

修好 CUDA 后，单环境串行执行、单状态推理、动作回传、Python 环境和日志可能限制 GPU 连续工作；这部分是代码结构支持的判断，尚无本机 CUDA 时间线测量。不能将 CPU 微基准比例当作 GPU 训练比例，也不能据此给出 GPU 加速倍数。

本次只进行了静态检查、临时模型微基准、环境采样和问答记录；没有修改算法、环境、训练设置、Conda 环境或既有权重，没有运行 4000 回合训练，也没有产出新训练模型。

## 实际运行环境

| 检查对象 | 实测结果 |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4070 SUPER，12 GB；驱动 591.86 |
| CPU | Intel Core i7-14700KF，20 核、28 逻辑处理器 |
| 默认/base Python | `C:\Users\Admin\anaconda3\python.exe`，Python 3.11.7 |
| base PyTorch | `2.3.1+cpu`；`torch.version.cuda is None`；`torch.cuda.is_available() == False` |
| DQN Python | `C:\Users\Admin\anaconda3\envs\DQN\python.exe`，Python 3.8.19 |
| DQN PyTorch | 版本字符串 `2.3.1`，但 CUDA 构建为 `None`，可用性为 `False`，仍为 CPU 构建 |
| DQN 项目导入 | `importlib.util.find_spec('satellite_rl')` 返回 `None`；项目声明要求 Python >=3.10 |
| CPU 微基准依赖 | NumPy 1.26.4、pandas 2.1.4、Matplotlib 3.8.0 |
| PyTorch 默认线程 | intra-op 20，inter-op 28 |

检查时看到一个使用 base 的 `evaluate_learning.py` 推理审计进程；它不是正式训练入口。`nvidia-smi` 当时有桌面图形进程及总体 GPU 活动，不能把总体利用率归给本项目。以上仅描述本次核查的两个解释器，不代表其他环境或历史训练机器的状态。

设备选择依据：[SAC runner](/D:/paper/src/satellite_rl/training/sac_runner.py:100) 根据 `torch.cuda.is_available()` 自动选择 CUDA 或 CPU。网络本身已有 `.to(device)`，[batch 张量](/D:/paper/src/satellite_rl/agents/sac_transformer.py:365) 也指定 device，因此没有发现整个 Actor/Critic 忘记迁移设备的问题。

## 短时性能采样

### SAC 网络与外围操作

采用明确指定 CPU 的临时模型诊断，不把 CPU 当作后续正式训练环境。随机种子 `20260924`；收集 3 个随机动作环境回合中的真实转移作为固定批次来源；batch=256；每种配置预热 2 次更新、计时 4 次更新。动作推理、回放采样、逐步日志各计时 30 次。回放池为 20,000 条，由所收集转移重复填充，仅用于测量采样开销。

以下均为均值，单位 ms；默认 intra-op=20：

| SAC 变体 | 一次参数更新 | 一次动作推理 | 一次回放采样 | 一次日志写入 |
| --- | ---: | ---: | ---: | ---: |
| Transformer | 223.23 | 2.60 | 0.24 | 0.33 |
| CNN | 55.18 | 0.91 | 0.24 | 0.49 |
| MLP | 18.53 | 0.63 | 0.22 | 0.29 |
| CNN matched | 53.17 | 1.12 | 0.26 | 0.32 |
| MLP matched | 39.36 | 0.79 | 0.30 | 0.48 |

日志均值包含偶发文件 I/O 抖动，不能理解为每次都低于 0.5 ms。固定批次、未训练模型和短样本也不代表收敛后完整训练吞吐。

作为线程敏感性诊断，还在同一临时进程内测了 1、4 个 intra-op 线程，未修改项目默认值：

| 变体 | 1 线程更新 ms | 4 线程更新 ms | 20 线程更新 ms |
| --- | ---: | ---: | ---: |
| Transformer | 766.03 | 274.08 | 223.23 |
| CNN | 146.87 | 64.82 | 55.18 |
| MLP | 7.41 | 8.51 | 18.53 |
| CNN matched | 112.10 | 47.59 | 53.17 |
| MLP matched | 49.86 | 22.71 | 39.36 |

不能把“统一设为单线程”当作所有模型的提速办法：小 MLP 在本次采样中受益，但 Transformer 批量更新显著变慢。线程数须针对实际设备、模型和工作负载测量。

一次独立的 4 线程 Transformer 更新 cProfile 用时约 275 ms，其中网络前向合计约 132 ms，反向引擎约 125 ms，优化器约 12 ms。嵌套累计时间不可重复相加。这里进一步支持瓶颈主要在网络计算，而非 CSV 或 replay。

### 环境计算

对两套环境分别运行 5 个固定种子回合（`20260924` 至 `20260928`），再以相同种子做独立 cProfile。动作遵循 SAC 类别范围：组 `beam_idx % 8`、频率 0–99、槽数 1–10、功率 5–50 W；这是性能诊断，不是模型训练或策略效果对照。

- 主环境 `satellite`：500 步，不带 profiler 的 `env.step` 合计 1.5204 s，即 **3.041 ms/步**；5 次构造合计 8.90 ms，reset 合计 0.12 ms。
- `satellite_variant_103`：500 步，`env.step` 合计 1.0447 s，即 **2.089 ms/步**。
- 同种子普通运行与 profiler 运行的步数、奖励和终态统计一致。

主环境 cProfile 的 `env.step` 累计 1.226 s，其中：

| 路径 | 累计时间 | 在该次 step profile 中的比例 |
| --- | ---: | ---: |
| `_calc_group_slot_interference` | 0.991 s | 约 80.8% |
| `interfere_power` | 0.060 s | 约 4.9% |
| `interfere_effect` | 0.038 s | 约 3.1% |
| `SatelliteFreqState.distance` | 0.022 s | 约 1.8% |

`get_combined_state` 累计约 0.999 s，包含 `_calc_group_slot_interference`，两者不能相加。variant 的干扰状态计算同样占该次 step profile 约八成。

这些是共享桌面上的短时测量，另有推理审计和诊断进程，系统负载与频率变化会影响绝对值。主环境普通运行甚至比 profiler 运行更慢，因此不能根据 3.041 与 2.089 ms 判定两套环境有稳定速度差距。网络与环境也不是同一条端到端训练时间线，不将分项相加后包装成精确训练占比。

## 代码层面的性能问题

### 1. SAC 有可避免的梯度图和反向计算

五种 SAC 采用相同模式，以 Transformer 为例：

- [动作推理](/D:/paper/src/satellite_rl/agents/sac_transformer.py:280) 没有 `no_grad()`，但结果只用于离散采样。临时包裹 `no_grad()` 的动作计时从约 2.60 ms 降到 2.29 ms（20 线程）；这是短时量级参考，不是整体训练加速承诺。
- [TD 目标计算](/D:/paper/src/satellite_rl/agents/sac_transformer.py:304) 为 Actor 和两个目标 Critic 建图，直到 [loss 使用目标](/D:/paper/src/satellite_rl/agents/sac_transformer.py:382) 才 detach。前面的建图和中间激活保留没有用于目标分支反传。
- [Actor loss 的 Q 计算](/D:/paper/src/satellite_rl/agents/sac_transformer.py:412) 未冻结或分离 Critic 输出，随后 `actor_loss.backward()` 会额外计算两个 Critic 的梯度。但接下来只更新 Actor，Critic 的这部分梯度到下一次更新又被清空。本实现是离散动作的期望 Q，Actor 可通过动作概率得到梯度；Q 值在 Actor 步中可视为常量。
- `log_alpha` 的创建没有指定 device，仍是 CPU 标量；Actor loss 还会为它生成随后清掉的梯度。应整理设备与梯度边界，但不能因一个标量就声称整个 GPU 更新落在 CPU 或一定报设备错误。

一次 SAC 更新有 8 次批量网络前向。上述建议主要去掉无用图和多余反向，不等于可以删掉所有这些前向。没有实测修改后的更新耗时。`no_grad()` 与 `eval()` 不同，不能顺手切换 Transformer 的 Dropout 行为并称作等价性能修复。

### 2. GPU 启用后，单环境串行流程仍可能留下等待时间

[训练循环](/D:/paper/src/satellite_rl/training/sac_runner.py:156) 依次执行：

`单状态动作推理 → CPU 环境 step → CSV → CPU replay 采样 → batch=256 更新`

没有多个环境并行收集，也没有批量动作推理。前 2000 条经验还不执行更新；即使以后启用 CUDA，该阶段也主要是短小推理与 CPU 环境工作。Transformer 实际为 13 个 token、宽度 128、2 层，单状态推理工作量较小。

[动作返回](/D:/paper/src/satellite_rl/agents/sac_transformer.py:297) 分别调用三个 `.cpu().item()`；CPU 环境需要等动作结果。Replay 的 deque、`random.sample`、`zip` 和 NumPy 拼批在 CPU，每次更新再创建 5 类设备张量。两份 float32 状态 batch 合计仅约 0.4 MiB，没有依据把 PCIe 带宽列为首要瓶颈；更需要关注串行等待与小操作调度。

逐步日志每步开关 CSV 文件，但本机采样中远小于 Transformer/CNN 更新。绘图、最终 npz 和模型保存发生在训练末尾，不是循环里的持续瓶颈。

### 3. 环境热点是干扰状态的重复计算

[状态拼接](/D:/paper/src/satellite_rl/envs/satellite.py:440) 每步调用 [_calc_group_slot_interference](/D:/paper/src/satellite_rl/envs/satellite.py:742)，扫描同极化历史波束并逐频率槽计算。角度、天线增益、路径损耗存在重复计算：其中天线增益依赖波束对而不随同一波束的槽号改变，却位于槽循环内。

优先候选是缓存每回合不变的几何量/波束对角度，预计算波束与频点的路径损耗，把槽号无关的增益移出内层循环，并用数组累加替代部分 Python 循环。须保留当前公式和状态语义，验证每一步的状态、奖励、done 与终态指标，而不是同时更改物理模型。

其他次级路径包括 [干扰影响](/D:/paper/src/satellite_rl/envs/satellite.py:709) 对 `history_cinr` 的嵌套扫描；每步重复两次满意度汇总；每回合重复读城市 CSV。距离缓存每非终止步计算一次 8×100 数组，当前状态拼接没有直接使用它，但已有测试检查该缓存行为，不能未经评估直接删除。

主环境被所有 SAC、DQN、PPO direct 使用；variant_103 用于 PPO grouped。这里使用 SAC 范围动作测两套环境，不能冒充 PPO grouped 实际策略下的耗时分布。

### 4. 其他算法的静态发现

- [DQN 更新](/D:/paper/src/satellite_rl/agents/dqn.py:97) 对相同 `s_` 连续做三次 `self.online(s_)`，分别取三个动作头，可以考虑一次前向后拆分。此次没有单独测 DQN 更新耗时。
- [PPO 的 GAE](/D:/paper/src/satellite_rl/agents/ppo_direct.py:15) 将 TD delta 转为 NumPy，在 CPU 循环计算，再移回 device；PPO 逐回合收集再更新，采样期 GPU 可能大部分时间没有批量计算。此次没有单独测 PPO 更新耗时。

## 建议顺序与未确认事项

1. 先使用户指定的 DQN 环境满足 Python/项目依赖，并安装可用的 CUDA PyTorch；核查实际解释器、CUDA 张量运算、模型参数设备和项目导入。本次没有安装或修改环境。
2. 清理 SAC 无效建图、Actor 步的多余 Critic 反向，以及 DQN 重复前向；按原实验语义验证。这些尚未实施。
3. 在 CUDA 可用后重新做完整循环短时 profile，分别记录动作推理、环境、replay、传输、更新、日志和吞吐；GPU 阶段需用 CUDA events 或在测量边界同步，不能仅靠异步调用返回时间。
4. 若届时 CPU 环境/采样成为主要等待来源，先处理已定位的几何和干扰重复计算、缓冲日志，再评估多环境并行和批量动作推理。改变 batch、采样/更新比例或异步采样会影响实验过程，需要明确记录。

GPU 利用率不必追求始终 100%；应优先比较每秒环境步数、每秒更新次数、固定交互/更新预算的总墙钟时间，以及实际学习效果。当前尚未测到本机 CUDA 训练，无法确认启用后的利用率、首要等待段或加速倍数。

## 可复现产物

- [网络微基准脚本](/D:/paper/outputs/verification/training_perf_20260924/profile_network.py)
- [网络原始计时与依赖信息](/D:/paper/outputs/verification/training_perf_20260924/network_results.json)
- [网络更新 cProfile](/D:/paper/outputs/verification/training_perf_20260924/network_update_profile.txt)
- [环境微基准脚本](/D:/paper/outputs/verification/training_perf_20260924/profile_env.py)
- [环境原始结果及逐函数统计](/D:/paper/outputs/verification/training_perf_20260924/profile_env_results.json)
- [主环境 step cProfile](/D:/paper/outputs/verification/training_perf_20260924/profile_env_satellite_step.txt)

诊断脚本内部通过 `SATELLITE_RL_OUTPUT_ROOT` 隔离日志，未加载或写入训练权重。网络脚本需在可导入当前项目且满足依赖的 Python 中执行；本次明确使用 base 作 CPU 诊断，不改变“后续正式训练在 GPU 条件验证通过后使用 DQN”的约定。

```powershell
& 'C:\Users\Admin\anaconda3\python.exe' outputs/verification/training_perf_20260924/profile_network.py
& 'C:\Users\Admin\anaconda3\python.exe' outputs/verification/training_perf_20260924/profile_env.py
```

本次未更改生产源码，无需重复运行全套算法/权重兼容测试；已核查诊断退出正常、结果 JSON 可读取、预期配置与样本数齐全，环境普通/profiler 同种子结果一致。

## 追问：先修好 DQN 的 CUDA 运行环境

实施及验证日期：2026-09-24。

### 用户授权与完成状态

用户明确要求“先修好 DQN 的 CUDA 运行环境”。此次已实际修复运行环境，不再是前文的只读诊断。正式环境名仍为 `DQN`，原环境另存目录备份；没有修改算法、环境仿真或训练配置，没有启动正式训练。

| 项目 | 修复后实测 |
| --- | --- |
| 正式解释器 | `C:/Users/Admin/anaconda3/envs/DQN/python.exe` |
| Python | 3.11.11 |
| PyTorch | 2.3.1+cu121 |
| CUDA 构建 / 可用性 | 12.1 / True |
| GPU | NVIDIA GeForce RTX 4070 SUPER，计算能力 8.9，12 GB |
| cuDNN | 8907，即 8.9.7 |
| NumPy / pandas / Matplotlib | 1.26.4 / 2.1.4 / 3.8.0 |
| 项目安装 | `satellite-rl 0.1.0`，可编辑安装指向 `D:/paper` |
| 依赖检查 | `pip check` 无依赖冲突 |

PyTorch 保留 2.3.1，选择其官方提供的 CUDA 12.1 Windows 构建，避免本次同时升级算法框架版本；配置依据为 [PyTorch 官方历史版本安装说明](https://pytorch.org/get-started/previous-versions/#v231)。未改 NVIDIA 驱动，也未安装系统级 CUDA Toolkit。

### 实际修复过程与备份

1. 保存原环境的 Conda YAML、显式包清单和 pip 清单。
2. 建立临时环境 `DQN_cuda_stage_20260924`，使用 Python 3.11.11、CUDA PyTorch 与项目所需依赖。大安装包下载后核对官方发布的 SHA256，再安装；核心科学计算依赖与此前 CPU 诊断保持一致。
3. 临时环境先通过 `pip check`、实际 CUDA 运算、项目导入、权重前向和参数更新检查。
4. 检测到其他任务使用旧 DQN 进行评估，等该进程退出后才切换。将原 DQN 目录原样移至 `C:/Users/Admin/anaconda3/envs/DQN_cpu_backup_20260924`，再通过 Conda 将临时环境迁移为正式 DQN。
5. 为新 DQN 重建 pip 包的 12 个命令行启动器，避免 Windows 启动器残留临时解释器路径。`conda run -n DQN pip --version` 与 `f2py -v` 检查正常。
6. 在最终路径重新完成 GPU 验证、项目现有测试和环境导出。临时 Conda 环境已不再存在。

旧环境备份保留 Python 3.8.19、原 CPU PyTorch 和其他历史包；备份解释器已验证仍可导入原 CPU torch。它主要用于回滚，原样备份中的部分工具启动器可能仍引用原路径，若检查其中包应显式使用备份 `python.exe -m ...`。新 DQN 按当前项目需要重建，没有迁入旧环境所有无关的 Gym、Notebook、OpenCV 等工具。base、RL、pythonProject 等环境未改动。

### 验证结果

验证种子为 `20260924`，日志均经 `SATELLITE_RL_OUTPUT_ROOT` 隔离。实际检查包括：

- 全包 **41 个模块**导入成功。
- 在 `cuda:0` 进行矩阵乘法，并与 CPU 数值参考及解析梯度对照，结果通过。
- 读取原 Transformer SAC、CNN SAC、MLP SAC、PPO direct、DQN 五类权重，严格加载并在 CUDA 上前向，结果有限且输出形状正确。
- 用真实环境产生的 8 条转移验证五种 SAC，各进行一次 batch=8 更新；Actor、两个 Critic 和两个目标 Critic 的参数均在 CUDA，更新后有限且实际变化。
- DQN 使用原 batch=256 设置进行一次 CUDA 更新；8 条真实转移重复填充只是兼容性验证，并非声称有 256 条独立样本。在线 Q 参数实际变化，loss 有限。
- `conda run --no-capture-output -n DQN python -m unittest discover -s tests -v`：**25 项全部通过**，约 7 秒。现有测试包含其自身的临时单回合输出检查，不是新训练实验。
- 最终 GPU 验证脚本状态 `passed`，实测解释器明确来自正式 DQN；`pip check` 无依赖冲突。

验证没有改变原实现中的 SAC CPU 温度标量，五种 SAC 的 GPU 更新已确认正常。运行仍会提示当前构建没有 Flash Attention、Transformer 未启用 nested tensor 快速路径，以及已有 NumPy 标量转换弃用警告；这些提示没有导致验证失败，也没有在本次混入性能或算法修改。后续可独立处理。

### 使用方法与后续范围

```powershell
conda activate DQN
python -c "import sys, torch, satellite_rl; print(sys.executable); print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

也可使用 `conda run -n DQN python ...`，避免误用仍为 CPU 构建的 base。根目录 `AGENTS.md` 已更新为修复后的状态，后续训练默认用 DQN，并继续在开跑前检查解释器和 CUDA。

本次证明了 CUDA 与项目兼容性，尚未重新测完整 GPU 训练吞吐，不能把此前 CPU 更新耗时直接当作修复后的速度，也没有给出训练加速倍数。

### 修复与验证产物

- [正式 DQN GPU 检查结果](/D:/paper/outputs/verification/dqn_cuda_repair_20260924/cuda_validation.json)
- [可复用 GPU 验证脚本](/D:/paper/outputs/verification/dqn_cuda_repair_20260924/verify_cuda.py)
- [25 项测试日志](/D:/paper/outputs/verification/dqn_cuda_repair_20260924/unittest_results.txt)
- [修复后 Conda 环境导出](/D:/paper/outputs/verification/dqn_cuda_repair_20260924/dqn_after.yml)
- [修复后 pip 版本清单](/D:/paper/outputs/verification/dqn_cuda_repair_20260924/dqn_after_pip.txt)
- [原 DQN 环境导出](/D:/paper/outputs/verification/dqn_cuda_repair_20260924/dqn_before.yml)

导出文件是版本记录；若异机重建，需要将 pip 的 `torch==2.3.1+cu121` 指向官方 CUDA 12.1 索引，并调整项目本地路径，不能仅在默认 PyPI 索引执行导出清单就认定会得到同一 CUDA 构建。

## 追问：SAC 的多余梯度计算具体在哪里

回答日期：2026-09-24。

### 问题与准确结论

用户问：“你说 SAC 有多余梯度计算，在哪？”

需把前面的概括分成两类：**Actor 更新阶段存在真实的多余反向传播；动作采样与 TD 目标主要是多余的计算图构建和中间结果保存。** TD 目标已在损失处 detach，不能说目标 Critic 也被 loss 反向更新了。

以下以当前 Transformer SAC 为例。CNN、MLP 和两个 matched 版本采用相同结构，也存在这些路径。公共训练循环没有在外部关闭梯度记录。

### 1. Actor 步额外计算两个 Critic 的梯度

[Actor 步的 Critic 前向](/D:/paper/src/satellite_rl/agents/sac_transformer.py:412) 为：

```python
q1_f, q1_s, q1_p = self.critic_1(states)
q2_f, q2_s, q2_p = self.critic_2(states)
```

它们参与 `exp_min_q`，进一步进入 [Actor 损失与反传](/D:/paper/src/satellite_rl/agents/sac_transformer.py:423)：

```python
actor_loss = torch.mean(-self.log_alpha.exp() * entropy - exp_min_q)
self.actor_optimizer.zero_grad()
actor_loss.backward()
self.actor_optimizer.step()
```

这两个 Critic 的参数仍启用梯度记录，输出没有 detach；因此 `actor_loss.backward()` 同时对 Actor 和两个 Critic 求梯度。但随后只执行 Actor 优化器，两个 Critic 不使用这次算出的梯度。它们此前在本次 update 的第 393–394 行已经完成正常优化，Actor 步产生的额外梯度到下一次 update 的 [第 389–390 行](/D:/paper/src/satellite_rl/agents/sac_transformer.py:389) 才被清掉。

本项目是离散 SAC：Q 网络只接收状态，给出各离散动作的 Q 值，Actor 的梯度可经动作概率与 Q 值的乘积计算。建议仅在 Actor 步把 Critic 的 Q 计算视为常量，保留策略概率和期望计算的梯度。不能把整个 Actor 步包进 `no_grad()`；连续动作 SAC 若需要通过 Q 对动作求导，也不能直接照搬整个 Critic 前向禁用梯度的做法。

### 2. Actor 步还额外计算温度参数的梯度

同一 [第 423 行](/D:/paper/src/satellite_rl/agents/sac_transformer.py:423) 的 `self.log_alpha.exp()` 没有 detach，所以 Actor loss 反传也会计算 `log_alpha` 的梯度。随后 [第 432 行](/D:/paper/src/satellite_rl/agents/sac_transformer.py:432) 立即将其清空，再用专门的 `alpha_loss` 重新反传并更新。Actor loss 对温度参数的这次梯度计算没有用于参数更新。

这是标量层面的冗余，不能与两套 Critic 的反向传播等量齐观。建议在 Actor loss 中把温度视为常量，保留专门的温度更新。

### 3. TD 目标先建图，使用时才切断

[第 376 行](/D:/paper/src/satellite_rl/agents/sac_transformer.py:376) 调用 `calc_target()`；[该函数](/D:/paper/src/satellite_rl/agents/sac_transformer.py:304) 内部前向经过 Actor 和两个目标 Critic，默认会建立计算图。

到 [第 382 行](/D:/paper/src/satellite_rl/agents/sac_transformer.py:382) 和第 387 行才在损失中使用 `td_target.detach()`。因此这些目标分支的图和中间结果没有用于 Critic loss 的反传。准确说是“多建了图”，不是“两个目标 Critic 又做了反向传播”。建议目标计算从一开始就放在 `no_grad()` 中；前向求目标值仍然需要执行。

### 4. 动作选择只采样，却记录了 Actor 的图

[take_action 第 285 行](/D:/paper/src/satellite_rl/agents/sac_transformer.py:285) 调用 Actor，没有 `no_grad()`。结果只用于 `Categorical.sample()`，最后返回三个 Python 整数，没有对这次前向调用 backward。

建议只对动作采样关闭梯度记录。`no_grad()` 不等于 `eval()`；后者会改变当前 Transformer 的 Dropout 行为，不能作为同一项等价优化顺带加入。

### 五种 SAC 的对应位置

下表为本次读取的当前文件行号；每格链接指向该路径的起始位置。

| 变体 | 采样 Actor 前向 | TD 目标函数 | Actor 步 Critic 前向 | Actor backward | 温度梯度清空 |
| --- | --- | --- | --- | --- | --- |
| Transformer | [285](/D:/paper/src/satellite_rl/agents/sac_transformer.py:285) | [304](/D:/paper/src/satellite_rl/agents/sac_transformer.py:304) | [412](/D:/paper/src/satellite_rl/agents/sac_transformer.py:412) | [426](/D:/paper/src/satellite_rl/agents/sac_transformer.py:426) | [432](/D:/paper/src/satellite_rl/agents/sac_transformer.py:432) |
| CNN | [252](/D:/paper/src/satellite_rl/agents/sac_cnn.py:252) | [271](/D:/paper/src/satellite_rl/agents/sac_cnn.py:271) | [379](/D:/paper/src/satellite_rl/agents/sac_cnn.py:379) | [393](/D:/paper/src/satellite_rl/agents/sac_cnn.py:393) | [399](/D:/paper/src/satellite_rl/agents/sac_cnn.py:399) |
| MLP | [207](/D:/paper/src/satellite_rl/agents/sac_mlp.py:207) | [226](/D:/paper/src/satellite_rl/agents/sac_mlp.py:226) | [334](/D:/paper/src/satellite_rl/agents/sac_mlp.py:334) | [348](/D:/paper/src/satellite_rl/agents/sac_mlp.py:348) | [354](/D:/paper/src/satellite_rl/agents/sac_mlp.py:354) |
| CNN matched | [253](/D:/paper/src/satellite_rl/agents/sac_cnn_matched.py:253) | [272](/D:/paper/src/satellite_rl/agents/sac_cnn_matched.py:272) | [380](/D:/paper/src/satellite_rl/agents/sac_cnn_matched.py:380) | [394](/D:/paper/src/satellite_rl/agents/sac_cnn_matched.py:394) | [400](/D:/paper/src/satellite_rl/agents/sac_cnn_matched.py:400) |
| MLP matched | [200](/D:/paper/src/satellite_rl/agents/sac_mlp_matched.py:200) | [219](/D:/paper/src/satellite_rl/agents/sac_mlp_matched.py:219) | [327](/D:/paper/src/satellite_rl/agents/sac_mlp_matched.py:327) | [341](/D:/paper/src/satellite_rl/agents/sac_mlp_matched.py:341) | [347](/D:/paper/src/satellite_rl/agents/sac_mlp_matched.py:347) |

### 原代码的一次实际梯度路径核查

使用已修复的 DQN 环境、Python 3.11.11、PyTorch 2.3.1+cu121、NumPy 1.26.4、RTX 4070 SUPER；随机种子 `20260924`。用真实环境的 8 条转移调用原始 Transformer `SAC.update()` 一次，在各网络首个参数上注册梯度钩子，在优化器上注册 step 后钩子。未改源文件和更新方法，未加载或保存训练权重，构造器日志隔离到 `outputs/verification/sac_gradient_paths_20260924`。

| 参数所属对象 | 一次 update 内梯度钩子触发次数 | 优化器 step 次数 |
| --- | ---: | ---: |
| Actor | 1 | 1 |
| Critic 1 | 2 | 1 |
| Critic 2 | 2 | 1 |
| 目标 Critic 1、2 | 各 0 | 无优化器；末尾进行软更新 |
| log_alpha | 2 | 1 |

两在线 Critic 和 log_alpha 的两次钩子均收到非零梯度。单独检查 `calc_target()` 输出时 `requires_grad=True`，而 detach 后为 False，与“先建图后切断”的静态分析一致。钩子检查针对所选参数，结合原代码调用顺序用于确认梯度路径，不是更新耗时或全网络 FLOPs 统计。

原始记录见 [gradient_paths.json](/D:/paper/outputs/verification/sac_gradient_paths_20260924/gradient_paths.json)。尚未修改上述实现或测量消除冗余后的加速幅度；此次仅回答具体位置并记录证据。

## 追问：正式对照前，代码还有哪些提速空间

回答日期：2026-09-24。

用户要求在开始训练前检查优化机会，认为目前运行偏慢。此次由两个子 agent 分别进行 CUDA 网络更新诊断、CPU 环境等价优化原型验证；主任务核对数据搬运、采样和日志路径。**正式对照实验仍待确认；此次只新增隔离诊断脚本与记录，没有修改生产算法、环境或训练配置。**

检查起点没有发现持续运行的 Python 训练进程，GPU 总体活动还包括桌面程序，因此不能将显卡利用率快照归因于某个正在训练的模型。以下结果来自已修复的 DQN/CUDA 环境短时采样，取代此前纯 CPU 微基准用于本轮 GPU 优化判断；不是历史长任务的完整时间线。

### 已测：GPU 训练进程的 CPU 线程数与无用梯度路径

DQN Python 3.11.11、PyTorch 2.3.1+cu121、RTX 4070 SUPER；batch=256，真实环境转移形成固定批次，种子 20260924，保持 train 模式及 Dropout。每种配置预热 4 次，再测 12 次更新；每次测量边界使用 `torch.cuda.synchronize()`。下表单位 ms，为均值：

| 模型 | 原版、默认 20 个 CPU intra-op 线程 | 原版、1 个线程 | 去无用梯度候选、1 个线程 |
| --- | ---: | ---: | ---: |
| Transformer SAC | 38.95 | 26.51 | 19.81 |
| CNN matched | 17.50 | 11.10 | 9.24 |
| MLP matched | 13.45 | 7.94 | 6.72 |

这里的线程数是 GPU 训练进程使用的 PyTorch CPU intra-op 线程数，不是限制 GPU 核心数。它与前面“完全在 CPU 上训练时 Transformer 多线程更快”是不同工作负载，不能混用结论。本机 GPU 工作负载下，更多 CPU 线程没有带来更高吞吐。

内存中的候选仅实施以下效率变化：

- 动作采样使用 `no_grad()`，保留 Dropout 和相同采样方式。
- TD 目标计算使用 `no_grad()`。
- Actor 步仅对两个 Critic 的前向关闭梯度，策略概率与 Q 的期望计算继续保留 Actor 梯度。
- Actor loss 中对温度值 detach，温度仍由原 alpha loss 更新。

候选保留原始负目标熵、逐分支 min-Q、前向次数及顺序，没有提前实施待比较的两项数学修正。

相同初始化和 RNG 下，三个架构单次更新后的五个网络参数及 alpha 均**逐位相等**，动作也相同。Transformer 测量阶段的 CUDA peak allocated 约从 **443 MiB 降到 183 MiB**；这是 PyTorch 已分配显存峰值，不是整卡全部显存占用。

另用仅在连续循环前后同步的独立短测复核了无用梯度候选，均固定为 1 线程：

| 模型 | 原版连续更新 | 候选连续更新 |
| --- | ---: | ---: |
| Transformer | 27.89 ms | 21.10 ms |
| CNN matched | 11.73 ms | 9.52 ms |
| MLP matched | 8.30 ms | 7.26 ms |

两轮绝对时间略有波动，但优化方向一致。**这些是网络更新环节的收益，不能直接写成整段训练提速两倍。** 单次更新等价检查也不是所有可能训练轨迹的形式化证明；落地时仍需检查多步更新、保存/恢复和正式循环吞吐。

### 已测：动作采样中的重复检查

[take_action](/D:/paper/src/satellite_rl/agents/sac_transformer.py:280) 每步建立三个 `Categorical`。检查已安装的 PyTorch 2.3.1 源码确认，默认参数验证会对概率约束执行 `if not valid.all()`；概率在 CUDA 上时，这需要主机读取布尔结果。

保留相同概率归一化和采样调用，仅加 `no_grad()`、关闭每次 Categorical 构造时的参数验证，1 线程的 50 次动作采样均值如下：

| 模型 | 原版动作采样 | 候选动作采样 |
| --- | ---: | ---: |
| Transformer | 2.29 ms | 1.68 ms |
| CNN matched | 1.51 ms | 0.98 ms |
| MLP matched | 1.25 ms | 0.78 ms |

每个架构额外对照的 20 次动作序列及 CUDA RNG 状态相同。关闭验证只在有效概率路径上保持相同采样含义，正式实现仍须保留合适的有限值/概率健康检查，不能把检查全部取消后声称异常路径也完全等价。

把三个动作的 `.cpu().item()` 合并成一次传输，在本次采样中没有稳定的额外收益，暂不列为主要提速措施。

### 已测：CPU 环境重复几何计算

环境的主要耗时仍在 [_calc_group_slot_interference](/D:/paper/src/satellite_rl/envs/satellite.py:742)。本次通过内存子类原型逐步外提不变量和缓存，保持原公式、单位、判断和累加顺序：

| 环境原型 | 单步耗时中位数 | 相对原版 |
| --- | ---: | ---: |
| 原版 | 1.278 ms | 1.00× |
| 天线增益、距离、半径外提；当前波束每槽路径损耗缓存 | 1.049 ms | 1.22× |
| 再外提 interfere_power 的每波束不变量 | 1.033 ms | 1.24× |
| 再缓存坐标转换后的向量及范数 | 0.748 ms | 1.71× |

最后一项使环境单步时间减少约 **41.4%**。此处使用的固定动作场景与前面 CPU 环境采样不同，不能把不同场景的 3.041 ms 与此处 0.748 ms 拼成一次优化收益。

种子为 1024、2048、4096，各覆盖随机、拥挤、超功率三类场景；合计 **783 步**，含完整 100 步回合、有效零槽和第 61 步超功率终止。所有候选的每步状态、奖励、done、info、各波束速率、历史、占用及距离矩阵与原版均逐元素精确相等。时间采用三轮交替顺序测量，仍可能受到同机其他诊断影响。

正式落地要明确缓存的生命周期，随 reset 清理或绑定不可变场景，并补充跨 reset 验证。该原型没有修复此前发现的物理公式或单位问题；若同时修那些问题，就不再是纯粹的等价提速。

### 尚未量化或未证实有收益的候选

- [目标网络软更新](/D:/paper/src/satellite_rl/agents/sac_transformer.py:338)：逐参数做乘、加、复制，形成很多小算子。一对目标网络软更新单独均值约为 Transformer 2.81 ms、CNN matched 1.05 ms、MLP matched 0.68 ms，可再研究批量 foreach 操作。尚未实现候选；直接换成 lerp 或融合运算可能改变浮点舍入，不能提前宣称逐位等价。
- [回放采样](/D:/paper/src/satellite_rl/agents/sac_transformer.py:28)：deque/random.sample/zip/NumPy 拼批后，又在 update 创建张量。预分配连续数组、明确 dtype 可减少分配。此前 CPU 测量它不是首要开销； pinned memory、异步传输是否值得，需端到端验证，不能仅凭存在复制就判定 PCIe 带宽不足。
- [逐步 CSV](/D:/paper/src/satellite_rl/agents/sac_transformer.py:275)：每步开关文件并格式化。缓冲写入或按回合成批写可改善效率，需设计正常结束/异常时 flush；此前约 0.2–0.5 ms 的量级表明它排在网络更新之后。
- [城市 CSV](/D:/paper/src/satellite_rl/envs/satellite.py:342)：可缓存只读表，保留原 `DataFrame.sample()` 的随机数语义；未专项测量收益。
- alpha 虽仍位于 CPU，但把标量和其优化器搬到 GPU 的独立短测**没有一致改善**。该问题不能再仅凭代码设备位置被称为当前主要耗时瓶颈。

之后修正联合双 Q 时，应复用每个网络一次前向得到的三个 Q 头，广播或分块构造联合值，不要针对 10000 个动作逐个调用网络。batch=256 的一个 `[256,100,10,10]` float32 张量约 9.77 MiB，但两个 Q、概率、临时张量和梯度会继续占用显存，不能只按单个张量估算总峰值；这只是实现建议，数学修正尚未执行。

### 与待确认实验的关系

建议先将“GPU 宿主线程设置＋清理无用梯度＋环境计算缓存”作为共同的效率层进行落地验证，再用于原版公式与修正版公式两组实验。这样两阶段的差异仍集中于此前约定的两项学习公式。

AMP/TF32、关闭 Dropout、改变 batch、减少更新次数或并行环境采样会涉及数值或训练过程变化，应单独控制与验证，不能混入已证明等价的候选。本项目 PPO 模块导入时会设置全局 TF32 开关，正式实验也应显式固定和记录该设置，避免依赖导入顺序。

本次完成的是诊断和可审查原型。尚未改正式代码、核定完整训练加速倍数，也未获得前面 18 次训练方案的确认。

### 诊断产物

- [网络诊断汇总](/D:/paper/outputs/verification/cuda_perf_20260924/network_summary.md)
- [更新计时与单次参数等价结果](/D:/paper/outputs/verification/cuda_perf_20260924/network_results.json)
- [网络候选与复现脚本](/D:/paper/outputs/verification/cuda_perf_20260924/network_probe.py)
- [动作采样计时与 RNG 对照](/D:/paper/outputs/verification/cuda_perf_20260924/network_action_results.json)
- [温度参数设备及连续循环计时](/D:/paper/outputs/verification/cuda_perf_20260924/network_alpha_results.json)
- [环境原型](/D:/paper/outputs/verification/cuda_perf_20260924/env_candidates.py) 与 [计时/逐步等价结果](/D:/paper/outputs/verification/cuda_perf_20260924/env_results.json)

## 追问：环境计算能否放到 GPU，进一步减少 CPU/GPU 往返

回答日期：2026-09-24。

### 结论与范围

有空间，但要区分三个问题：减少重复计算、把多个环境合成批量计算、让状态与回放数据常驻 GPU。对当前单环境的小数组，单独将一个函数换成 CUDA 张量运算未必更快；数据常驻和批量化之后，GPU 才更可能发挥吞吐优势。

本次由子 agent 进行只读架构审查和局部 CUDA 可行性微基准。没有重写正式环境、修改训练协议或启动待确认实验。

### 当前往返发生在哪里

当前流程是：`CPU 状态 → GPU Actor → CPU 动作/环境 → CPU 回放 → GPU 更新`。

- [状态上传](/D:/paper/src/satellite_rl/agents/sac_transformer.py:282)：单个 NumPy 观测转为 CUDA float32 张量。
- [动作读取](/D:/paper/src/satellite_rl/agents/sac_transformer.py:297)：三个动作转成 Python 整数，CPU 环境等待 GPU 输出。
- [训练批次](/D:/paper/src/satellite_rl/agents/sac_transformer.py:365)：每次从 CPU 回放构造五类 GPU 张量。
- [串行循环](/D:/paper/src/satellite_rl/training/sac_runner.py:156)：单状态推理、环境推进、日志、回放采样、更新依次执行。

网络使用的一个 float32 观测仅 816 字节，三个 int64 动作仅 24 字节；环境原始 NumPy 观测可能为 float64，逻辑数组大小为 1632 字节。batch=256 的两份 float32 状态合计约 0.398 MiB。这些是数据结构大小，不是 profiler 实测的 PCIe 流量；不能仅凭往返存在就认定带宽已饱和。主要应检查依赖等待、分散的小操作、张量分配与同步。

只要 CPU 环境需要 GPU 的动作结果，就不能靠加一个 `non_blocking=True` 消除这条依赖。pinned memory、异步传输也需要预分配和可重叠的工作；在单次小数组传输前临时 pin 内存未必有收益。参考 [NVIDIA 主机与设备传输建议](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#data-transfer-between-host-and-device) 与 [PyTorch 异步传输说明](https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html)。此前合并三个动作回传的短测没有稳定收益，不能把它当作已证实的主要优化。

### 新的局部实测：单环境与批量环境的区别

从实际环境采样到第 8、32、64、96 步，保存历史快照，对 `_calc_group_slot_interference()` 的动态掩码、功率和干扰求和进行原型测试。采用 DQN、Python 3.11.11、PyTorch 2.3.1+cu121、NumPy 1.26.4、RTX 4070 SUPER，1 个 CPU intra-op 线程，种子 20260924。

下表使用 96 条历史的快照、float64，时间为每次调用均值：

| 实现 | 单个环境 | 16 份批量快照 | 64 份批量快照 |
| --- | ---: | ---: | ---: |
| 当前 Python 方法 | 2.814 ms | 未测 | 未测 |
| 缓存静态衰减因子后的 NumPy 动态计算 | 0.0305 ms | 0.766 ms/批 | 3.055 ms/批 |
| 同等 CUDA 动态计算，数据驻留 GPU | 0.185 ms | 0.169 ms/批 | 0.191 ms/批 |
| CUDA，动态输入上传且结果下载 | 0.250 ms | 0.235 ms/批 | 0.251 ms/批 |

据此，在这个已缓存的局部计算上，单环境 NumPy 比 CUDA 驻留实现更快；批量达到 16/64 后，CUDA 的批量吞吐优势明显。该结果支持优先研究批量化和驻留，不能泛化为“任何单环境 GPU 都慢”或“完整训练可以加速十几倍”。更融合的 CUDA 实现仍可能改变单环境成本，但不属于本次已测内容。

关键测量边界：

- 16、64 批量是复制同一个真实快照的合成吞吐测量，不是已经实现了 16、64 个完整独立环境。
- 快速动态计算假设静态几何、天线增益和路径损耗已缓存。当前脚本按原标量方式准备这个 96 历史快照本身约 **2.994 ms**；如果每步重新准备，收益会被抵消。正式方案要在场景初始化时向量化计算、有效复用或增量维护，并将初始化成本计入整个回合。
- 往返测试包含动态历史数组 H2D 与 100 槽输出 D2H，不含 Python 列表打包和完整 env.step。单环境对应 3072 字节输入与 800 字节输出，64 份对应 196608 与 51200 字节。
- float64 原型与原函数的最大相对误差约 **8.44e-15**，不再是逐位相同；float32 原公式版本最大相对误差约 **7.78e-6**。尚未完成这些改写对全部状态、奖励和边界分支的逐步验证。

因此，0.0305 ms 不能直接当作完整环境单步时间，更不能与前文完整 env.step 的 0.748 ms 混作同一指标。

### 比搬设备更先值得做的环境计算重构

1. **每场景预计算。** 波束位置固定时，波束对夹角/增益、接收波束与槽频率对应的路径损耗无需每步重算。可保存几何/增益 `[E,100,100]`、接收束×频槽路径损耗 `[E,100,100]`，分别广播使用，不必存完整 `[E,100,100,100]` 系数张量。
2. **干扰观测增量维护。** 每次新增一个分配，把该波束对后续接收波束各槽的贡献加入缓存；下次观测读取当前接收波束的一行。这样无需每步重新扫描全部历史。总的成对相互作用仍可能是二次规模，主要收益是减少重复几何/对数/指数及 Python 循环，不应未经推导声称变为线性复杂度。
3. **历史信息改为定长数组与掩码。** 将 Python 列表搜索改为按组、槽、所属波束索引访问；当前 SAC 动作最多 10 槽，可用 `[E,100,10]` 的历史槽记录，或在唯一占用约束下映射到 `[E,8,100]`。若支持其他基线的更宽槽动作，需相应扩展容量。

这些方法在 CPU NumPy 上也能实现，先做 CPU 版本可降低移植风险。原观察干扰与实际干扰的过滤规则、历史 CINR 更新顺序、5 dB 处理、奖励阈值等必须分别核对；不能把性能重构和物理修正混在一起。

### 真正减少往返的 GPU 方案

理想数据流是：

`GPU 批量观测 → GPU Actor → GPU 动作张量 → GPU 批量环境 → GPU 回放 → GPU 更新`

CPU 主要负责场景初始化/预取、低频统计、日志与保存。Python/CPU 仍会调度 GPU 工作，不表示整个程序不再使用 CPU。

可采用以下数据结构：

| 数据 | 批量张量形状示例 |
| --- | --- |
| 观测 | `[E,204]` |
| 频谱占用、历史槽状态 | `[E,8,100]` 或带有效掩码的槽记录 |
| 需求、位置、功率、吞吐量 | `[E,100,...]` |
| 几何、路径损耗与累计干扰 | 分别 `[E,100,100]` |
| 当前束编号、剩余功率、done/active | `[E]` |

当前 20,000 条回放若用 float32 连续数组保存两份 204 维状态，约占 **31.13 MiB**；加三个 int64 动作、float32 奖励和终止标志，逻辑数据总量约 **31.74 MiB**，另有索引和管理开销。显存容量上具备放入 GPU 的条件，但尚未实测其端到端收益。GPU 回放能减少训练批次反复上传；单独把回放搬过去，并不会消除 CPU 环境对动作结果的等待。

避免每步 `.item()`、`.cpu().numpy()` 或 Python `if cuda_tensor`；用张量掩码处理不同环境的终止与重置。必须先保存真实终态转移，再 reset，不能误把新回合首状态存成上一回合的 next_state。日志应低频批量取回。

### 批量化与实验公平性

应先测试例如 E=16、32、64 的小规模端到端吞吐，比较总环境交互数/秒、更新数/秒及完整回合时间，不能只看 GPU 利用率。

当前预热后约每产生一条 transition 更新一次。如果 E 个环境合并推进一步却仍只更新一次，就把更新/交互比例降低了 E 倍；这种“更快”混入了训练量减少。保持同一更新预算时，网络优化仍可能是总训练主耗时，环境和批量动作推理加速不能自动带来同倍率训练加速。

并行采集即使保持总更新数，也会改变数据混合、采样顺序和策略更新时机，不能宣称与单环境同 seed 的轨迹逐位相同。原公式与修正公式对照应采用相同采集方式，并明确记录变化。

建议顺序是：先完成场景预计算/增量计算的 CPU 版本和完整回合验证；试验 GPU 回放与批量动作推理；在端到端数据支持时再推进批量 GPU 环境。之前已验证的保守效率优化仍可先行。当前没有完整 GPU 环境，也没有启动或重新确认正式训练。

### 本轮产物

- [GPU 环境局部可行性报告](/D:/paper/outputs/verification/gpu_env_feasibility_20260924/kernel_summary.md)
- [干扰计算原型脚本](/D:/paper/outputs/verification/gpu_env_feasibility_20260924/kernel_probe.py)
- [计时、传输大小与数值误差](/D:/paper/outputs/verification/gpu_env_feasibility_20260924/kernel_results.json)

## 追问与实施：只做性能优化，原有框架不要动

实施日期：2026-09-24。

用户明确授权实施性能优化。本次已把保守原型落到生产源码，没有重构为多环境、GPU 仿真或新训练框架。完整说明见 [性能优化说明](../../docs/工程文档/04_性能优化说明.md)。

### 完成的修改

- 五种 SAC 的动作采样、TD 目标和 Actor 步 Critic 前向关闭无用梯度记录；Actor loss 的温度值 detach，正常 Actor/Critic/alpha 更新保留。
- 两套环境外提每波束不变量，增加局部路径损耗缓存、私有且有界的几何向量/范数缓存，正确处理 reset、标量类型和几何常量变化。
- 原 SAC 训练入口在 CUDA 调用期间使用 1 个 CPU intra-op 线程，并通过 finally 恢复调用方线程数；CPU 路径保持原设置。无需更换原调用方式。

网络结构、参数量、单环境流程、状态/动作、回放、更新比例、奖励、物理公式、终止、Dropout、概率校验和已有权重格式保持原样。此前讨论的目标熵、联合双 Q 公式尚未修正，正式两阶段训练仍待确认。

### 验证与实测

**35 项自动化测试全部通过。** 五种 SAC 各连续 3 次 CUDA 更新，对照原快照后，每步的五网络、alpha、四优化器状态及 RNG 逐位相等。两环境 18 案例、1,683 步的状态、奖励、done、info、动作裁剪和全部公开状态精确一致，包含换场景 reset。

三模型短循环包含采样、CPU 环境、回放、更新及 CSV，热身 5 步再测 30 步：

| 模型 | 原代码默认 20 线程 | 原代码 1 线程 | 新代码 1 线程 |
| --- | ---: | ---: | ---: |
| Transformer | 48.46 ms/步 | 32.20 ms/步 | 25.09 ms/步 |
| CNN matched | 28.73 ms/步 | 16.25 ms/步 | 12.62 ms/步 |
| MLP matched | 21.47 ms/步 | 12.42 ms/步 | 10.45 ms/步 |

固定 1 线程前后，该窗口的动作、观测、奖励、结束参数/优化器/RNG 及 CSV 均相同。包含线程配置的旧默认到新配置吞吐约提高为 1.93×、2.28×、2.05×；这是排除构造/reset 的短窗口，不是完整训练时间承诺，也不是学习效果提升结论。

环境独立整回合计时包含 reset：主环境 1.327→0.773 ms/步，103 变体 1.466→0.840 ms/步。Transformer 单独更新的 peak allocated 显存约 443→183 MiB。

五个网络类及原训练循环主体的 AST 保持一致，五份预训练权重、城市数据及依赖声明的 SHA256 未变。修改前源码、实际场景/动作、固定 batch、诊断初值/RNG、原始轨迹及计时均保存在 [本次验证目录](/D:/paper/outputs/verification/performance_only_20260924)。未启动 4000 回合或 18 次正式实验。

## 追问：一组训练多久，并行是否比串行快

记录日期：2026-09-24。用户在第三组完成后询问训练耗时与提速办法，并要求核对并行和串行哪种更快。以下更新使用本轮原始记录，不沿用前文尚未启动的历史状态。

### 已有同工作量对照

已复核启动前的 DQN/CUDA 校准：每轮都是相同的 Transformer、CNN matched、MLP matched 三个作业，均为修正公式、种子 51001；每个作业 2,600 次交互，其中 2,000 次为回放预热、600 次为实际参数更新，batch 为 256，关闭周期验证和终点考核，CPU 数值线程固定为 1。仅改变同时运行的作业数。

| 同时运行作业数 | 三个作业全部完成的总时间 | 更新活跃阶段的近似总吞吐 |
| ---: | ---: | ---: |
| 1（串行） | 61.484 秒 | 58.57 次更新/秒 |
| 2 | 43.172 秒 | 62.64 次更新/秒 |
| 3 | 37.078 秒 | 67.37 次更新/秒 |

**在这次同工作量短测中，三路并行更快**：总时间比串行减少 39.7%；按每两秒读取一次状态所得的更新活跃阶段估计，总更新吞吐提高约 15.0%。总时间包含启动、预热及保存等阶段，不能把 39.7% 全部归为参数更新提速。该校准每种并发条件只有一个短窗口，不是十八次四十万步完整训练的串行重跑，不能保证正式总时长同幅缩短，也不能据此证明任意模型组合下三路都是最优。

校准源码提交为 `10e974d7c0cbe78faf566ba1c9fff306e14694a2`；其算法和环境文件与正式训练 `36c14d74cbe6b9cf7410ee0595ee6d73083afdc8` 没有差异。训练入口后续新增了线程/精度的显式设置与记录、文件占用重试和评价回报记录。核算脚本、输入校验值及原始数值见 [本次核算](../../outputs/verification/concurrency_review_20260924/review_metrics.json)、[中文核算表](../../outputs/verification/concurrency_review_20260924/review_summary.md)。本次只读核算没有暂停训练或插入额外 GPU 测试。

### 正式作业为什么需要数小时

| 已完成作业（均为种子 51001） | 断点累计运行时间，含评价 | 从启动到完成的日历时间 |
| --- | ---: | ---: |
| CNN 原公式 | 3.92 小时 | 4.72 小时 |
| Transformer 原公式 | 6.46 小时 | 7.34 小时 |
| Transformer 修正公式 | 7.02 小时 | 7.84 小时 |

日历时间包含断电空档。断点累计时间来自恢复记录，可能缺少回滚前未保存的尾段，不能解释为完整计量的纯 CUDA 运算时间。三组都在共享 GPU 条件下运行；单组历时长，不等于所有作业串行后总时间更短。每组需要四十万次交互、三十九万八千次参数更新；十八组总预算是七百二十万次交互。

按事件日志近似划分，三组的周期验证分别约 18.4、20.7、27.0 分钟，预算终点后的验证与两套考核合计约 6.1、7.2、7.4 分钟。大部分时间仍在交互和参数更新阶段；减少评价不能直接消除数小时的训练开销。上述时间块包含相邻记录和保存开销，不是独立推理基准。

### 当前资源与执行决定

14:17–14:18 的 18.4 秒观察中，三路 CUDA 进程的 GPU 利用率平均为 93.3%（91%–97%），平均功率约 86.3 W；三个进程各消耗约 0.96–0.98 个 CPU 核的运行时间。原始采样见 [资源观察](../../outputs/verification/concurrency_review_20260924/live_resource_sample_20260924_141807.json)。这是当前 CNN 与两个 MLP 组合的短时观察，不代表整个训练平均值。

NVIDIA 将该 GPU 利用率定义为采样期间有内核运行的时间比例；高利用率不等于所有计算单元达到满吞吐，不能单凭这个数字选择并发数，见 [NVIDIA 说明](https://docs.nvidia.com/deploy/nvidia-smi/#utilization)。

当前继续采用已有实测支持的三路并行。进一步提速优先依据同工作量的端到端计时和具体耗时分析；网络结构、精度、batch、更新次数、评价频率和正在运行的固定源码保持本轮约定。六路短测曾出现 Windows 原生进程异常，未形成有效的六路成绩，不能宣称更多并发一定更快。其他代码提速若实施，应先在独立副本验证计算一致性及总吞吐，再决定下一轮对照如何采用。


## 追问：异步采集能否接近纯算法速度，并继续原实验

回答日期：2026-09-29。用户希望页面允许延迟显示，采用工程优化提速；有效则继续原实验，并要求设计报告和验证报告。

**已完成并接续。** 日志改为后台按产生顺序批量写入，前端增加同步/异步选项；同时把每个张量都等待GPU返回的有限值检查合并，指标集中传回CPU，页面增量读取日志。本机测量显示，主要收益来自减少GPU检查与等待。梯度裁剪、模型、奖励、种子、场景和更新预算未改变。

两类网络各3轮A/B/B/A、每臂500次真实更新的同工作量短测，吞吐为原来的2.66/2.46倍；保留异步记录比关闭观测I/O的参考多耗时约0.7%/2.1%。正式三任务共同训练窗口总吞吐观察为19.75→37.83次/秒。短测有三组正式任务竞争GPU，正式观察又处于不同训练位置，不能直接宣称整组40万步耗时同倍率下降。含周期验证的首窗也完整保留。

日志不会抽样丢弃；缓冲满时主线程等待，结束与保存前等待写完。完整验证、断点和场景保存仍保留一致的同步边界。2秒是目标刷新间隔，页面还受指标采样与轮询影响。351项CPU回归（1项跳过）、真实GPU更新、断点恢复及原始日志核对通过。原六ID、原进度、种子51001与各40万步保留；后续共同评价helper已恢复。模型效果仍待正式考核，工程提速不代表需求满足率已经提高。

证据与实现：[设计报告](../../docs/第二阶段/方案/08_异步采集与训练吞吐优化设计.md)、[验证报告与曲线](../../docs/第二阶段/实验报告/05_异步采集与训练提速验证.md)、[任务接续](../../outputs/verification/async_telemetry_20260929/migration/final_handoff.json)。代码主要在`workbench/telemetry.py`、`runner.py`、`monitoring.py`和`agents/workbench_algorithms.py`。

**2026-09-29 13:59运行补记：**A/B已完成40万步，C1/C3/C4继续。C2首次启动在0步发生CUDA驱动模块原生退出，原因尚未确定，不能据此断言是异步记录或显存不足。完整初始断点保留，现已通过工作台按同一任务ID恢复排队；实际更新须等其重新获得运行槽位后核验。共同评价helper已恢复，未修改算法或冻结源码。见[故障现场与恢复记录](../../outputs/experiments/europe_ablation_20260929/recovery_20260929_1350/recovery_summary.json)。

**2026-09-29 15:57恢复验证：**C1也已完成。C2已按原ID实际启动并从完整初始断点继续，现有32550次交互、30551次参数更新，当前没有新错误；C3/C4持续运行，共同评价helper正常。由此确认本次任务恢复已进入真实训练，但不代表已确定或修复驱动原生退出的底层原因。[后续实测记录](../../outputs/experiments/europe_ablation_20260929/recovery_20260929_1350/actual_restart_verified.json)。

### 追问：当前进度怎样，工程优化具体采用什么方案

回答日期：2026-09-29，14:11进度快照。A/B完成，C1为93.85%，C3为9.96%，C4为9.16%，C2恢复后排队；后台接续进程正常。六组共同独立考核尚未开始，当前只能汇报进度，不能给最终优劣排序。工作台原始快照见[本次进度记录](../../outputs/experiments/europe_ablation_20260929/preflight/user_progress_20260929.json)。

工程采用三层方案：训练记录进入有界缓冲，由后台线程按顺序写文件；把同类GPU有限值检查与诊断数值回传集中处理，减少反复等待；服务端缓存主要指标，只读取追加内容。主要速度收益来自第二项，单独异步日志的收益有限。模型学习、验证、场景归档和完整断点仍保留原有顺序与必要同步；缓存记录在异常强制退出时仍可能丢失未落盘尾部，恢复须按完整断点裁剪后续日志。已将同一主题的[方案08](../../docs/第二阶段/方案/08_异步采集与训练吞吐优化设计.md)完善为通俗报告，包含结构图、前后对照表、异步/同步范围、实测速度及限制，不重复新建同主题文档。

**23:32进度追问：**A/B/C1已完成；C2为290650步（72.66%），C3为357200步（89.30%），C4在350000步进行周期验证（87.50%）。六组累计完成约91.58%的训练交互预算，后续共同独立考核尚未开始，不能把这个比例理解为全部工作完成比例。活动进程、DQN/CUDA和helper均正常，无新错误，训练继续。[本次只读快照](../../outputs/experiments/europe_ablation_20260929/preflight/user_progress_20260929_2332.json)。

## 2026-09-30追问：将G09迁到远端4090D并停止本地

用户指定远端u2024110224@10.160.17.102:20418，密码读取BUPT_PASSWORD环境变量，要求远端跑通后停本地、尽量提高GPU利用率。已开始工程迁移准备，五组实验条件、原任务编号及40万步总预算保持；204/207维已学习约27万步，不能远端从头再算40万步。完整模型、优化器、经验池、所有随机状态、场景顺序与当前环境状态一起交接。

先保留本地继续训练，在远端核DQN/CUDA、实际冻结源及完整断点恢复；通过后用原工作台API停止本地，转交停止时的最终断点，再在远端同ID接续。并行数以同工作量总吞吐和显存实测选择，不用额外实验填GPU占用。第一阶段目录独立，前端通过SSH隧道查看远端真实任务，本地保留停止历史和归档。跨硬件后续随机轨迹不保证逐位相同，最终报告记录设备迁移段；工程迁移成功也不等于模型更好。本轮已实际完成本地停止和远端五路接续，详细门槛见[方案11](../../docs/第二阶段/方案/11_4090D迁移与并行运行方案.md)。

实际交接结果：两本地完整停点306200/300400与其优化器、20000经验、全部RNG和计数保真；只改迁移副本的运行路径，双SHA与非路径深比较通过。五个原ID在4090D都已真更新，服务仅改运维并行上限3→5，训练worker仍用ecb冻结源。五路正式60秒采样GPU均98.98%、峰99%，每组增650–700更新；这不是纯算法倍速或全程/模型收敛证明。远端页面改看8766（8765为本地停止历史），已实查204维13令牌及2946维32令牌、欧洲选项与源码/进度。新Jan19独立100场景已导入，共同评价待终点；数据、曲线和限制见[验证报告09](../../docs/第二阶段/实验报告/09_4090D迁移与并行验证.md)。

持续执行交接也已完成：远端v2监督/后处理和本地独立镜像已核身份，瞬时错误有限重试，非有限学习错误先排查，原手动停止任务不自动重训。浏览器关闭不影响远端训练；本机关闭会暂停隧道和本地镜像。正式训练与共同评价仍在后台按既定预算继续。[最终工程交接](../../outputs/verification/g09_remote_migration_20260930/final_handover.json)。

## 2026-10-02追问：剩余M2/M3分到两张GPU会不会更快？

从当前资源分配看，很可能提高总体训练速度。合理做法是M2、M3各用一张4090D；两组是独立实验，模型更新互不依赖，可以保持现有算法、种子和40万步预算。实际提速倍数尚未测量，不承诺翻倍，也不能只凭利用率推算剩余训练耗时。

本轮只读采样各25点，实际持续约32–34秒：

| 实例 | 北京时间采样窗口 | GPU平均利用率 | 范围 | 显存占用 | 观察 |
| --- | --- | ---: | ---: | ---: | --- |
| 20424 | 14:50:47–14:51:21 | 96.72% | 93–100% | 6315MiB | M2/M3均training，设备列出两个计算进程 |
| 20418 | 14:52:37–14:53:09 | 0% | 0% | 4MiB | 无GPU计算进程 |

两窗口并非同时进行，因此这是资源现状观察，尚不是“迁移前后提速对照”。20424窗口中M2/M3分别新增100/150次更新；工作台速率字段约3.36/3.75步每秒，该字段不是独占GPU速度。驱动返回的GPU进程号可能属于主机命名空间，不能仅按进程个数认定与容器PID一一对应。CPU处理、周期验证、断点和迁移本身的耗时也会影响最后完成时间。[原始数据](../../outputs/experiments/europe_tmg_20261001/resume_20261002T062332_824457Z/gpu_two_instance_readonly_20261002T065309_270211Z.json) · [数值复算](../../outputs/experiments/europe_tmg_20261001/resume_20261002T062332_824457Z/two_gpu_allocation_analysis_20261002.json)。

我建议优先考虑把M2交给空闲20418，M3保留在20424。但当前实现没有一个直接跨实例恢复的API：[任务存储](../../src/satellite_rl/workbench/storage.py)将run_id定位到本服务archive，[调度器resume及启动](../../src/satellite_rl/workbench/runtime.py)在本容器中启动worker。G11的[controller](../../outputs/verification/g11_remote_launch_20261001/g11_postprocess_controller.py)、[报告器](../../outputs/verification/g11_remote_launch_20261001/report_g11.py)与[审计器](../../outputs/verification/g11_remote_launch_20261001/audit_g11_raw.py)也按既定归属在20424查找M组。直接另起worker或让两个工作台共管同一archive可能重复调度，不能这样实施。

若实施，须先经原API保存完整断点并隔离原调度归属，再向另一实例独立归档交接同ID任务，重定位场景/来源路径并深核网络、优化器、经验池与历史、所有随机状态、场景顺序。学习设置保持，预算只补剩余步数；还须选择完成后回迁原归属再接Jan24评价，或单独审查并发布运维路由与报告读取的适配方案，保留原guard历史。前端需显示实际所在实例与同ID的迁移/恢复记录。随后比较相同训练阶段的有效更新吞吐和包含验证/保存成本的总耗时，才确认是否值得长期采用。

本次只完成资源观测、代码核对和建议，没有暂停或迁移任务，没有新训练组，现有M2/M3继续运行。此资源调整方案尚未实施，模型效果仍待既定共同评价。

### 后续执行：用户要求“迁移吧”

回答日期：2026-10-02。此要求已执行，覆盖上段“尚未实施”的历史状态：M2临时迁到20418，M3继续20424，两张4090D各运行一个独立实验。M2通过原工作台停止在262800交互/260801更新，完整保存后只转换运行路径，再由目标工作台同一ID续训；没有从头训练，也没有增加预算。15:49实时API核验M2为267050步、M3为273400步，均持续更新；M3进程出生时间保持不变。

断点保留了网络、优化器、20000条训练目标及15条序列历史、全部随机状态和场景顺序；路径转换前后深比较通过。训练仍导入原20424的不可变85f源码快照，共享/home使另一实例可以只读导入；输出与任务调度则在20418独立archive。原实例隐藏M2任务副本，避免两边同时调度。完整断点相等不承诺跨设备后续浮点轨迹逐位相同，最终实验报告须记录设备迁移。

迁移前后短时观察：

| 观察条件 | M2更新/秒 | M3更新/秒 | 合计更新/秒 |
| --- | ---: | ---: | ---: |
| 14:50–14:51，两组共用20424，约34秒 | 2.94 | 4.41 | 7.35 |
| 15:44–15:45，各用一张GPU，约60秒 | 6.56 | 5.82 | 12.38 |

该窗口合计约提高68%，支持此次分卡有吞吐收益。两卡平均利用率约76.1%/72.2%，虽低于原共用卡96.7%，总更新反而更多，说明不能只追求利用率数字。这里场景位置不同、状态计数有刷新间隔，而且未覆盖周期验证/保存和迁移停顿，不能视为严格同工作量倍速或整轮净节省68%；学习效果仍须原Jan24共同评价。[原始窗口与复算](../../outputs/experiments/europe_tmg_20261001/migration_m2_20261002T070813_436244Z/root_throughput_comparison.json)。

第一次目标恢复在262800步遇到已有同一步诊断快照的来源路径差异，零更新即退出。旧文件保留在迁移历史，目标重新生成该诊断快照，核16类科学字段和回放样本相等后再次同ID恢复。这个修复未改网络、训练配置、完整学习断点或科学源码；旧failure.json留存，不能据此再次重启正在运行的任务。

回迁与评价已配置持久后台接续：M2达到固定40万终点、权重/结果校验及worker退出后，把已完成记录回到原20424归属，再恢复原controller/collector、本地镜像与归档。这样原global_guard及Jan24十一模型、两基线的3500次分配计划不变；回迁和共同评价目前尚未实际完成。迁移期间原helper受控暂停，新returner/双端sidecar/本地watcher已核出生身份和脚本SHA。两端不共享loopback，使用各端自己的API和带唯一请求校验的共享文件回执；不上传密码。

前端无需改代码，已实际查看[20418的M2监控](http://127.0.0.1:8766/#monitor)与[20424的M3监控](http://127.0.0.1:8767/#monitor)。小时跟进已更新并精确回读，按迁移阶段读取正确API，继续每次简报。运维正确性、短窗吞吐和模型效果分别记录。[完整交接](../../outputs/experiments/europe_tmg_20261001/migration_m2_20261002T070813_436244Z/migration_live_handoff.json) · [根验收](../../outputs/experiments/europe_tmg_20261001/migration_m2_20261002T070813_436244Z/root_migration_verified_snapshot.json) · [自动化回读](../../outputs/experiments/europe_tmg_20261001/migration_m2_20261002T070813_436244Z/automation_migration_final_receipt.json)。

### 2026-10-02晚：分卡训练与自动回迁已经完成

M2、M3均已完成各400000步/398001更新，M2于21:42按原ID回迁20424归属；完整学习状态路径转换与终点权重SHA复核通过。原Jan24十一模型和两基线评价已完成，报告与逐束审计通过。此前短窗口约1.68倍仍只是工程观察，不等于整轮净耗时同倍率下降或模型效果提高。模型结论见[报告12](../../docs/第二阶段/实验报告/12_温度记忆与干扰图独立场景对照.md)；仅剩全部断点、历史日志等大资产的本地逐SHA同步，原训练和评价不会重复启动。
