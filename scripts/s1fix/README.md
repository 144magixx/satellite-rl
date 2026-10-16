# S1-FIX代码使用说明
来源：codex

代码只实现本次S1-FIX，不恢复旧AI17任务。正式实验由Claude在远端启动。本地已做数据抽取与小型冒烟，没有运行完整对照或完整PPO训练。

半径修订后本地37项小测试全部通过。静态与动态算子共用逐束半径上限，已删除间隔修复；拆分中心可沿连线外移，交替优化后继续按原预算增束。指派使用KD树邻接。本轮未执行正式远端对照或完整PPO训练。

## 文件和数据

`src/satellite_rl/coverage/s1fix/`中，`data.py`抽帧和选场景，`geom.py`处理几何，`skeleton.py`实现FLOW-AO与INIT，`bound.py`算下界，`downstream.py`连接贪心频率槽分配与SCA，`dynamic.py`实现动态算子和评价，`ppo_dyn.py`训练小型PPO，`io.py`保存CSV、布局和运行配置。小测试在`tests/test_s1fix_*.py`。

本地输入都来自`D:/paper/outputs/`；输出为`D:/paper/outputs/experiments/s1_fix_20261017/`。五个日包已各读一次，每天同时生成288个静态帧和961个动态帧，五份NPZ合计144,611,816字节（约137.9 MiB）。HIST12的12景、HIST20的20景飞机数全部一致，核对数值在`data/history_counts.csv`。NPZ使用`static_`、`dynamic_`前缀，分别保存`times`、`offsets`、`icao`、`lat`、`lon`、`gs_kt`和`track_deg`；时间为UTC Unix秒，经纬度保留float64，icao为uint32。

## 本地抽取与冒烟

在仓库根目录用PowerShell执行：

```powershell
$env:PYTHONPATH='src'
$env:PYTHONIOENCODING='utf-8'
$python='C:/Users/Admin/anaconda3/envs/DQN/python.exe'
& $python -m scripts.s1fix.extract_frames --workers 2
& $python -m scripts.s1fix.extract_frames --verify-only
& $python -m unittest discover -s tests -p 'test_s1fix_*.py' -v
```

## 远端运行

先把数据目录及历史对照的三份CSV传到远端。以下`DATA`、`OUT`、`HIST`换成远端真实绝对路径；`HIST`下面保持`round10/science_independent_statistics_01/endpoint_metrics.csv`、`round8/analysis/actual_v4_01/analysis20x3.csv`、`round8/analysis/mechanism_01/mechanism_rows.csv`的相对路径。仓库根目录运行，先完成A的开发/确认结果，再算下界；C的effort项读取A3原布局。

```bash
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
DATA=/absolute/path/s1_fix_20261017/data
OUT=/absolute/path/s1_fix_20261017
HIST=/absolute/path/first_stage_ai_search_20261013

python -m scripts.s1fix.run_A --data-dir "$DATA" --output-dir "$OUT" --history-dir "$HIST" --workers 16 --subexperiment all
python -m scripts.s1fix.run_C --data-dir "$DATA" --output-dir "$OUT" --workers 16 --subexperiment all

python -m scripts.s1fix.run_B cache --data-dir "$DATA" --output-dir "$OUT" --workers 16
python -m scripts.s1fix.run_B baselines --data-dir "$DATA" --output-dir "$OUT" --workers 10
python -m scripts.s1fix.run_B train --data-dir "$DATA" --output-dir "$OUT" --lam 0.2 --seed 51001
python -m scripts.s1fix.run_B eval --data-dir "$DATA" --output-dir "$OUT"
```

A可分开选`dev`（A0/A1）、`history`、`bounds`（A2）、`effort`（A3）、`confirm`。C可选`dev`、`confirm`、`effort`。用`--limit-scenes 1 --budgets 60 --max-iters 1`可缩小A/C冒烟；A下界另可设`--milp-seconds 1`，C可设`--sca-maxiter 1 --sca-seconds 1`。这些参数用于冒烟时不代表正式结果。

B共12个正式训练任务：预测版3个λ（0.05、0.2、0.8）乘3个种子（51001—51003），无预测版只用λ=0.2乘3个种子，在`train`后加`--no-forecast`。每个进程训练一个任务；full缓存必须预先完成。`--episodes 1 --episodes-per-update 1 --epochs 1 --sequence-limit 1`仅用于小型训练冒烟。评价默认读全部`*_final.pt`，不挑开发成绩最好的检查点。

## 输出与实现选择

- A保存静态结果、历史并列表、下界和布局；历史负载统一换算成架数，历史源文件只读。
- C三种常规布局共用INIT的实际束数，以同一合法INIT热启动；记录原`N_budget`和实际`comparison_N`。A3的effort=10布局直接读原件单列，其实际束数照实报告。另存逐束需求、容量、功率及槽位数值，方便复核。
- B保存逐帧/汇总评价、训练曲线与检查点。full按日期和动态帧缓存；读取时继承前帧束ID，求解耗时使用缓存原值。并行训练先写每run曲线，评价时汇总成`B/train_curves.csv`。
- 几何检查失败的结构动作撤销；动态patch失败按规格强制升级full。算法没有证明不可行时，结果中的`feasible=False`只表示本次没有找到合法布局。
- 下界仅算DEV24/CONF48的N=100、200，默认g=0.2°；g=0.1°敏感性只算DEV24前三个n<600的场景、N=200。LP每次限时300秒，超时保留已证下界；x上限为N，MILP也允许同格多束并保留整型性。MILP仍只算n<600，120秒由每个场景/预算共享。下界忽略间隔，主要用于衡量间隔关的算法差距；间隔代价另用同场景间隔开减间隔关衡量。
- SCA默认最多200次迭代，时间上限仍30秒；函数与块C脚本默认一致。
- `time_limit`是算法循环检查的软截止，正在进行的流或LP调用结束后才返回。1752架/200束预算的本地完整effort=1自测用时286.21秒，未达到30秒目标；远端正式速度仍待验证。
- 包络宽度的1e-9余量在1°上限处封顶，避免物理接口把数值余量当成真实超宽；真实超宽仍由合法性检查拒绝。
- `run_manifest.json`保存配置、git提交号、Python/依赖版本、主机名和起止时间。简单文件锁仅用于避免并发写覆盖；没有SHA计算、回执或预算账。既有共享物理函数内部的检查保持原样。

## 半径修订本地自测（2026-10-17）

DQN Python 3.11.11、NumPy 1.26.4、SciPy 1.11.4，`PYTHONPATH=src`，OMP/MKL/OpenBLAS线程均为1；同一DEV24最大帧（2024-01-12，静态142，UTC 11:52，1752架），间隔开、effort=1、默认30轮上限，无额外时间截止，四个布局全部通过最终检查。

| N预算 | N_used | 峰值架数 | 总耗时/秒 |
| --- | --- | --- | --- |
| 60 | 60 | 55 | 25.00 |
| 100 | 100 | 36 | 60.91 |
| 150 | 150 | 28 | 159.36 |
| 200 | 194 | 28 | 286.21 |

开发集第一段（2024-01-12 UTC 06:00—06:30，31帧）B2+、θ=1，repair按调用后目标严格改善计接受：0/15（0%），强制full为0，含初始full缓存构建用时36.96秒。该小样本没有证明repair有效，静态200预算的30秒速度目标也未达到。原始指标、四个布局、可复跑脚本和37项测试日志保存在`D:/paper/outputs/experiments/s1_fix_20261017/radius_revision_final/`。静态计算及动态full缓存使用不同截止设置，后者仍为30秒软截止。本轮不涉及工作台前端。

## 远端耗时估计

初版按16核CPU估计：A静态与A3约0.5—2小时，A2约1—4小时，C约1—3小时；B的3410个唯一full缓存约2—4小时，规则臂约0.5—3小时；12个PPO任务并行约8—36小时，再做评价约1—4小时。这些旧估计按每帧0.1—数秒推算，当前大场景本地耗时已超出该假设，不能直接用于修订版排期；远端首次运行后需据实更新。
