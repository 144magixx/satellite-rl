# 文献对照代码与本地检查
来源：codex

实现位于 `litcmp` 分支。未改 `s2fix/`、`coverage/s1fix/`、`shared_interface/`、`power_refinement/`，未做完整数据生成、完整评价或训练。

本次沙箱拒绝创建指定的 `D:/paper/outputs/experiments/litcmp_20261018/smoke/`，也拒绝写 `D:/paper/.git/worktrees/paper-litcmp/index.lock`。因此下列数值来自内存执行，尚无冒烟磁盘产物、CLI分片CSV验证或Git提交。不能将这些检查称为第12节全部完成。

## 文件与行数

`src/satellite_rl/litcmp/`：`__init__.py` 1、`geom_plane.py` 114、`hexgrid.py` 23、`kmeans_mec.py` 42、`voronoi_lb.py` 46、`layouts.py` 50、`moves.py` 90、`objective.py` 42、`anneal.py` 20、`genetic.py` 52、`convex_relax.py` 45、`power_lit.py` 71、`pipeline.py` 78、`data.py` 102。

`scripts/litcmp/`：`build_data.py` 24、`run_stage1.py` 45、`build_lit_layouts.py` 58、`evaluate.py` 44、`summarize_stage1.py` 42。

`tests/`：`test_litcmp_stage1.py` 80、`test_litcmp_moves.py` 61、`test_litcmp_search.py` 54、`test_litcmp_power.py` 75、`test_litcmp_pipeline.py` 52。共24个Python文件、1311行；另有本说明和实验状态摘要。

## 命令

以下是脚本调用示例，输出均限定在冒烟目录。正式运行由项目管理者安排。

```powershell
$py = 'C:/Users/Admin/anaconda3/envs/DQN/python.exe'
$env:PYTHONPATH = 'src'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$smoke = 'D:/paper/outputs/experiments/litcmp_20261018/smoke'
$frames = 'D:/paper/outputs/experiments/s1_fix_20261017/data'

& $py scripts/litcmp/build_data.py --sets dev_d864_n150 --limit 2 --workers 2 --output-dir "$smoke/data"
& $py scripts/litcmp/build_data.py --sets conf_d432_n150 --shard 0/44 --workers 2 --output-dir "$smoke/sharded_data"
& $py scripts/litcmp/run_stage1.py --data-dir $frames --sets dev24 --budgets 100 --methods A1 A2 A3 --seeds 51001 --workers 2 --shard 0/24 --output "$smoke/static.csv"
& $py scripts/litcmp/build_lit_layouts.py --data-dir $frames --methods A1 A2 A3 --requests 86400000 --N 150 --workers 2 --shard 0/44 --output-dir "$smoke/lit_layouts"
& $py scripts/litcmp/evaluate.py --set-dir "$smoke/data/dev_d864_n150" --methods B1_32 B2_32 B3 P0 C1 C2 --seeds 51001 --workers 2 --output-dir "$smoke/evaluation"
& $py scripts/litcmp/evaluate.py --set-dir "$smoke/data/dev_d864_n150" --methods P0 --workers 1 --shard 0/2 --output-dir "$smoke/shard0"
& $py scripts/litcmp/evaluate.py --set-dir "$smoke/data/dev_d864_n150" --methods P0 --workers 1 --shard 1/2 --output-dir "$smoke/shard1"
& $py scripts/litcmp/summarize_stage1.py --lit "$smoke/static_shard0of24.csv" --ref D:/paper/outputs/experiments/s1_fix_20261017/A/static_results.csv --output "$smoke/static_summary.csv"
& $py scripts/s2fix/summarize.py --input "$smoke/evaluation/results.csv" --output "$smoke/summary"
& $py -m unittest discover -s tests -p 'test_litcmp_*.py' -v
```

每个脚本复用 `s2fix.common.run_manifest` 记录配置、现有提交号、主机与起止时间，不计算资产SHA。Git因沙箱用户不同而报所有权错误时，本会话使用进程级 `GIT_CONFIG_COUNT=1`、`GIT_CONFIG_KEY_0=safe.directory`、`GIT_CONFIG_VALUE_0=D:/paper-litcmp`，未改全局Git配置。

分片对象保留原场景序号，清单和CSV带 `_shard{i}of{k}`。集合目录可含全部分片清单，读取时按对象序号排序；完整生成与分片生成使用不同输出目录，避免混入两套清单。各分片命令使用各自输出根目录，合并时归并对象和清单；同一目录并发追加运行清单不属于这里的调用方式。`--seeds`控制文献随机方法；G1/R/S1/S2保留S2-FIX原有运行规则，R固定三个原种子。

## 本地数值

第一阶段为1月12日分层24景最小/最大飞机数，预算100；A2种子51001。宽度为度，时间为秒。

| 场景/飞机数 | 方法 | N_used | 峰值 | 宽度均值 | spacing_violations | 时间 | feasible |
|---|---|---:|---:|---:|---:|---:|---|
| index19 / 123 | A1 | 68 | 5 | 0.300000 | 0 | 0.009088 | True |
| index19 / 123 | A2 | 100 | 3 | 0.300000 | 67 | 0.064566 | True |
| index19 / 123 | A3 | 58 | 7 | 0.402932 | 30 | 0.185219 | True |
| index142 / 1752 | A1 | 95 | 157 | 0.440000 | 0 | 0.234724 | True |
| index142 / 1752 | A2 | 100 | 77 | 0.330080 | 14 | 2.648274 | True |
| index142 / 1752 | A3 | 100 | 69 | 0.440806 | 64 | 7.933702 | True |

下游两景为开发集前两景，UTC 00:37/01:47，161/122架飞机，86.4 Mbps/机。内存调用原 `build_scene` 得FLOW-AO布局109/84束，K=13、峰值均3、超限均0；生成耗时10.944/10.937秒。因写权限失败未完成要求的 `build_set(limit=2)` 落盘。

文献布局预算150，经G1＋SCA；时间含布局、转换与评价。

| 景 | 布局 | N | W | 最差10% | 功率W | 时间s |
|---|---|---:|---:|---:|---:|---:|
| 0 | A1 | 85 | 1 | 1 | 639.195312 | 0.225260 |
| 0 | A2 | 150 | 1 | 1 | 3000 | 0.390720 |
| 0 | A3 | 82 | 1 | 1 | 926.793838 | 0.458871 |
| 1 | A1 | 73 | 1 | 1 | 478.445949 | 0.144017 |
| 1 | A2 | 121 | 1 | 1 | 3000 | 0.339008 |
| 1 | A3 | 65 | 0.997378 | 0.975682 | 777.957383 | 0.378798 |

六组布局均可行且 `UniformStage2Input(..., plain=True)` 构造通过。JSON对象在内存中用共享 `prepare_record` 恢复数组，等价于正常 `read_record` 的数组转换。

| 景 | 方法 | SCA前W | SCA后W | 总时间s |
|---|---|---:|---:|---:|
| 0 | B1_32 | 0.993207 | 0.993207 | 0.935395 |
| 0 | B2_32 | 0.997541 | 0.997541 | 0.916555 |
| 0 | B3 | 1 | 1 | 0.236901 |
| 1 | B1_32 | 0.996755 | 0.996755 | 0.738797 |
| 1 | B2_32 | 1 | 1 | 0.266657 |
| 1 | B3 | 1 | 1 | 0.174565 |

B1_320仅在景0运行一次，总时间2.686353秒，搜索1.933460秒。这里供需差目标允许W下降，不把W下降藏掉或改成W目标结果。

| 景 | 方法 | W | 功率W | 总时间s |
|---|---|---:|---:|---:|
| 0 | P0 | 1 | 3000 | 0.010634 |
| 0 | C1 | 1 | 501.907183 | 0.014893 |
| 0 | C2 | 1 | 501.907183 | 1.706961 |
| 1 | P0 | 1 | 3000 | 0.007786 |
| 1 | C1 | 1 | 416.838570 | 0.010782 |
| 1 | C2 | 1 | 416.838570 | 1.257690 |

A1布局上的E1_320：两景W为0.919871/0.961345，最差10%为0.553007/0.766278，功率232.752635/157.460324 W，时间1.743987/1.464008秒。

43.2 Mbps确认集分片0/44在88景中选索引0/44；内存生成分别为159/190架飞机、104/122束，K均26、峰值均3、超限均0、输入构造通过，耗时10.9436/10.9488秒。尚未输出该分片文件。分片互斥及并集覆盖通过小测试，真实dev分片CSV未生成。

新增12项测试、S2-FIX现有17项、从S1-FIX工作树只读加载的19项data/geometry/skeleton测试通过。S1-FIX其余bound/downstream/dynamic测试因本分支原本不含对应模块而无法导入；未修改或补入这些模块。五脚本的 `--help` 检查通过。

## 与实现说明的差异

- A1使用 `fixed_width=w` 校验，因为共享校验器的 `None` 要求宽度恰为逐束包络，与A1等宽定义冲突。
- A3的keep模式按实际保留中心的覆盖半角触发拆分。仅检查成员最小包围圆会让两指定景出现2.314°/1.266°波束；未拆分中心保持原值，拆分子簇按说明取包围圆。只有一个不同位置的超宽簇直接成为终止拆分产物，中心取该位置；123机景真实遇到一例。此修复遵循方案09的超宽区修复要求。
- A2中心数取预算与独立射线数的较小值；第二开发景存在重复位置，原K=150的D²初始化会除零。全部飞机、归属和请求均保留。
- B2按指定种群/代数为31/319/3199次计预算评价，起点另一次；B3没有Probe调用，记0。未人为补评价满足第10节中笼统的“B类等于预算”。
- 共享物理的逐槽signal已经除以槽数，计算gamma时乘回槽数；C2先逐束平均每槽干扰，再代入分母。C2第二阶段保证冻结近似目标不增，不能保证真实耦合干扰目标不增。
- B3的辅助变量以总需求归一化表示，数学问题不变，避免bps变量与槽数/功率尺度悬殊。
- 参考CSV存在同场景重复运行，先在同场景内取均值再配对，避免部分场景双重计权。A2也先在可行种子内按场景平均；可行率另报。
- 增加 `build_data.py --limit` 支持指定两景冒烟；没有增加完整运行或自动后继。
