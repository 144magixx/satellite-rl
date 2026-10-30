# S2-FIX 代码与运行说明
来源：codex

实现依据是 `D:/paper/docs/Claude维护/S2-FIX第二阶段需求压力/06_S2-FIX给Codex的实现说明.md`。已完成第11节冒烟及第12节按需叠放修订；正式数据生成、九组训练和共同评价交给远端执行，效果待验证。

## 文件清单

新增算法代码位于 `src/satellite_rl/s2fix/`：

- `__init__.py`：包入口。
- `common.py`：随机种子、普通JSON、脚本运行信息。
- `data.py`：原飞机读取、三档请求、FLOW-AO v2布局与六个数据集合。
- `greedy.py`：容量感知贪心G1。
- `evaluate.py`：v2容量、逐飞机服务指标、每束50 W上限的SCA。
- `search_env.py`：六类候选、真实频率组、R/S1/S2共享环境。
- `i0.py`：L1训练和策略采样。
- `tsac.py`：L2/L3训练和策略采样。
- `experiments.py`：逐场景、逐方法、逐种子和重复评价。

新增脚本是本目录的 `build_data.py`、`train_i0.py`、`train_tsac.py`、`evaluate.py`、`summarize.py`、`smoke_check.py`、`check_stacking.py`。测试是 `tests/test_s2fix_{flow,physics,search,tsac,experiments,summary}.py`。

旧模块只增加可选路径：`coverage/s1fix/{skeleton,geom,io}.py`，`shared_interface/{spherical_gaussian,geometry,link,stages}.py`，`power_refinement/{physics,model_based}.py`，`sac_improvement/{config,agent}.py`，`power30/{agent,environment,menus,schema}.py`。原v1、21.6 Mbps和Transformer默认路径保留。根目录 `EXPERIMENT_STATUS.md` 增加本次实现状态。

## 远端运行

在源码根目录执行。以下变量都设成远端绝对路径；`INPUTS`含原W26三个published清单和objects，`FRAMES`含S1-FIX两天的NPZ。`PY`指DQN环境的Python。数据清单内的对象路径相对该清单保存，搬到远端后由读取器定位，不依赖当前目录。

```bash
export PYTHONPATH="$PWD/src"
PY=/home/u2025140002/second_stage_satellite_rl/envs/DQN/bin/python
O=/home/u2025140002/second_stage_satellite_rl/s2_fix_20261018
D="$O/data"
INPUTS=/absolute/path/to/inputs
FRAMES=/absolute/path/to/s1fix/data

"$PY" scripts/s2fix/build_data.py --output-dir "$D" --inputs-dir "$INPUTS" --frames-dir "$FRAMES" --workers 12
"$PY" scripts/s2fix/train_i0.py --data-dir "$D" --output-dir "$O/L1-s51001" --seed 51001 --steps 200000 --device cuda --threads 3
"$PY" scripts/s2fix/train_tsac.py --data-dir "$D" --output-dir "$O/L2-s51001" --seed 51001 --encoder transformer --steps 150000 --device cuda --threads 3
"$PY" scripts/s2fix/train_tsac.py --data-dir "$D" --output-dir "$O/L3-s51001" --seed 51001 --encoder mlp --steps 150000 --device cuda --threads 3
"$PY" scripts/s2fix/evaluate.py --data-dir "$D" --set dev_d864_n150 --methods G1 R S1 S2 L1 L2 L3 --models "L1=$O/L1-s51001/final.pt" "L2=$O/L2-s51001/model.pt" "L3=$O/L3-s51001/model.pt" --output-dir "$O/dev" --workers 6 --device cpu
"$PY" scripts/s2fix/summarize.py --input "$O/dev/results.csv" --output "$O/summary-dev"
```

训练命令分别使用51001、51002、51003；评价的 `--models` 允许重复传同一方法的三个模型。每个训练模型每景采样3次，R使用三个规定种子，G1/S1/S2各一次。`--set`可换成另外四个开发/鲁棒性集合或确认集合。全部集合合计744景布局任务：576训练、4×20开发、88确认。

L1每10000步保存 `checkpoint_<步数>.pt` 和 `curve.jsonl`，终点为 `final.pt`；保留6250回合上限，若先到回合上限会记录实际步数。L2/L3每10000步及终点保存 `policy_<步数>.pt`，通用终点为 `model.pt`，另有回合JSONL和曲线JSON。这里的检查点供保存与评价，命令没有续训选项。

评价CSV每行是一景、一方法、一个训练种子的一次采样；SCA前后指标分别以before/after开头。逐束NPZ含需求、成员数、频率组/起点/槽数、SCA前后功率、容量和有效承载。模型装载、G1初解、搜索/推理、SCA分别计时；共同G1实际计算一次，计入每个方法的完整成本。汇总先平均同种子的重复，再平均种子，再按同场景配对；区间按日期、日期内场景两层自举。

每个脚本追加配置、提交号、包版本、主机及起止时间到输出目录的 `run_manifest.json`。快照根目录有 `COMMIT` 就读该文件，否则读git提交号。无工作台、回执、预算账或远端启动程序。

## 本地第11节冒烟

本地Python为 `C:/Users/Admin/anaconda3/envs/DQN/python.exe`，`PYTHONPATH=src`。输出全部位于 `D:/paper/outputs/experiments/s2_fix_20261018/`。运行命令：

```powershell
$env:PYTHONPATH = 'src'
& 'C:/Users/Admin/anaconda3/envs/DQN/python.exe' scripts/s2fix/smoke_check.py --baseline-src D:/paper/outputs/experiments/s2_fix_20261018/before/src
```

`before/src`是改动前提交7d47b968的源码快照；v1输入使用W32已保存的开发场景：`D:/paper-w32/outputs/experiments/s2_w32_20261017/local/sample_new/dev_n100/objects/`。需要时用 `--v1-data` 替换该目录。该脚本只构建前两开发景、评价G1/R/S1、短训练L1/L2，再比较新旧v1环境。

| 开发时刻（UTC） | 飞机数 | 实际N | 叠放簇 | 最大束飞机数 | 超K束数 | 平均宽度 | 生成秒 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 00:37 | 161 | 109 | 0 | 3 | 0 | 0.300474° | 10.946 |
| 01:47 | 122 | 84 | 0 | 3 | 0 | 0.304029° | 10.937 |

两景的G1、R三个种子、S1结果均相同：

| 时刻 | W前→后 | 最差10%前→后 | SI前→后 | SGM前→后 | 功率W前→后 | 总槽数前→后 |
| --- | --- | --- | --- | --- | --- | --- |
| 00:37 | 1→1 | 1→1 | 1→1 | 0.951197→1 | 3000→502.186746 | 161→161 |
| 01:47 | 1→1 | 1→1 | 1→1 | 0.951612→1 | 3000→416.999869 | 122→122 |

| 时刻 | 方法 | 搜索秒 | SCA秒 | 总秒 |
| --- | --- | ---: | ---: | ---: |
| 00:37 | G1 | 0 | 0.296 | 0.319 |
| 00:37 | R（三种子均值） | 1.557 | 0.296 | 1.875 |
| 00:37 | S1 | 0.492 | 0.308 | 0.823 |
| 01:47 | G1 | 0 | 0.182 | 0.198 |
| 01:47 | R（三种子均值） | 1.107 | 0.189 | 1.311 |
| 01:47 | S1 | 0.355 | 0.180 | 0.551 |

R和S1每次均实际探测32次，10行评价全部满足约束。两景原飞机ID、位置逐位相同，每机86.4 Mbps、唯一归属和总请求守恒。夜间两景负载低，这些数字仅证明代码能运行，不证明完整开发集已饱和。

L1完成200步、L2完成300步；均仍在原2000步预热内，更新0次。小测试另用合成回放验证L1和MLP版T-SAC实际梯度更新。L2三回合平均回报85.757910仅为冒烟日志。L2与L3每个actor参数量分别419449与418759，差约0.16%。

v1对照的状态、奖励、终态容量、起点、槽数、功率全部逐位相同。测试合计75项通过，其中新增17项、受影响旧测试58项。完整数字、逐景CSV/NPZ和飞机核对分别见输出目录 `smoke/{numbers.json,evaluation/,aircraft_checks.json}`。

## 与规格的实现说明及耗时

按“不算SHA、不做身份校验”的明确要求，新数据采用普通JSON和plain输入路径，用物理版本号区分v2；旧v1的内容校验保持原样。保留了原始来源字段，没有生成新摘要或回执。

第12节修订后，整景默认300秒。先用完整N预算、原峰值目标求A，以K容量最大流计算未分配数U；U为0时直接收窄，否则取S=clip(ceil(U/K)+5, 0, 60)，以N−S重求骨架B。A/B各最多120秒，未用时间留给后续步骤，不再按S档位均分。按不同站点计算半径上限，每轮在合格未分配飞机最多的站点加一个副本，并对全部飞机重新做K容量流；达到N或时间不足仍未全分配时，用原峰值流保全全部飞机并记录超K束数。最后做K约束距离最小指派，同簇按全部成员的最小包络锥共同移动，仅接受满足站点间隔的移动。LP使用剩余时限；单次几何/最大流操作不能中途打断，可能略超时。

MLP隐藏层按原Transformer参数量选206维：展平14×128词元后，1792→206→128；词元投影等共同部分仍参与计算。

修订前远端粗估：单4090D和12—16核，完整数据约1—2小时，九组训练约8—12小时，共同评价约2—4小时，总计约11—18小时；两张卡训练约5—8小时。该估计来自稀疏两景与预热段，正式更新吞吐、完整开发集与密集场景SCA耗时需远端实测后修正。

## 第12节叠放回归

以下命令只生成最大开发景与夜间最小开发景的布局，使用每机86.4 Mbps、预算150、K=13，保存case、layout、逐景check与numbers.json：

```powershell
$env:PYTHONPATH = 'src'
& 'C:/Users/Admin/anaconda3/envs/DQN/python.exe' scripts/s2fix/check_stacking.py
& 'C:/Users/Admin/anaconda3/envs/DQN/python.exe' -m unittest discover -s tests -p 'test_s2fix_*.py' -v
```

| 开发时刻（2024-01-12 UTC） | 飞机数 | 实际N | 叠放簇数 | 簇内束数 | 最大束飞机数 | 超K束数 | 平均宽度 | 耗时（秒） |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 11:22 | 1815 | 148 | 16 | 36 | 13 | 0 | 0.323360° | 28.742 |
| 01:47 | 122 | 89 | 0 | 0 | 5 | 0 | 0.303496° | 7.389 |

结果在 `D:/paper/outputs/experiments/s2_fix_20261018/stacking_revision1/`。白天叠放共16簇，其中12簇各2束、4簇各3束，合计36束；原飞机ID、位置及每机请求核对通过，同簇中心完全相同，公共第二阶段输入构造通过。23项S2-FIX测试通过，包含旧默认行为、全局流重分、预算不足保全飞机、整簇收窄、A/B各120秒与总300秒、U=0跳过重求，以及case_ref路径同名时按sha256定位对象。Claude的source_cases改动一并保留。
