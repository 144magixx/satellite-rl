# 文献对照实现说明（给Codex）

来源：cc

2026-10-18｜对应方案：[09 文献对照实验方案](09_文献对照实验方案.md)。Codex 负责实现、单元测试、本地冒烟；正式运行、核数与出图由项目管理者负责。

## 0. 工作方式（先读）

- **工作树**：`D:/paper-litcmp`，分支 `litcmp`，基于 S2-FIX 的 `s2fix` 分支（提交 5a54928）。只读复用 `s2fix/`、`coverage/s1fix/`、`shared_interface/`、`power_refinement/` 的代码，**不改它们的文件**。新代码放 `src/satellite_rl/litcmp/`，脚本放 `scripts/litcmp/`，测试放 `tests/test_litcmp_*.py`。提交到 `litcmp` 分支，前缀 `feat(litcmp):`，不合并 main，不碰 `D:/paper` 主工作区。
- **写法要求（用户明确）**：不做防御性编程，不为不会发生的情况加校验、守卫和回退；不算 SHA、不写回执和预算账；按正常路径写清楚，出错就报错。只写必要的小测试。新代码目标约 1500 行（含测试），脚本尽量薄，循环和汇总复用 s2fix 的。你写的说明文档标题下第一行写"来源：codex"。
- **本地环境**：`C:/Users/Admin/anaconda3/envs/DQN/python.exe`，`PYTHONPATH=src`。远端 Python 3.11、numpy 1.26.4、scipy 1.14.1；只用 scipy 1.11 已有接口（`scipy.spatial.Delaunay`、`scipy.optimize.minimize` 的 SLSQP）。
- **冒烟输出**一律写到 `D:/paper/outputs/experiments/litcmp_20261018/smoke/`，不要写进 `.../litcmp_20261018/data/`。
- **完成时**用中文简短回复：文件清单与行数、各脚本运行命令、冒烟数字（第 12 节）、与本文不同之处及原因。

## 1. 复用的接口（不改）

| 用途 | 接口 |
| --- | --- |
| 第一阶段场景 | `coverage.s1fix.data.load_day(data_dir, date)`、`stratified(day)`、`static_sets(data_dir)`（开发 24 景与确认 48 景的键与 S1-FIX 块 A 相同）；帧 `Frame(date, time, index, icao, positions(lat,lon), gs_kt, track_deg)`；数据 `outputs/experiments/s1_fix_20261017/data/frames_2024-01-{08,09,12,15,18}.npz` |
| 几何 | `coverage.s1fix.geom`：`rays(positions)`、`angles(a,b)`、`cone(member_rays)`→(中心射线, 半角)、`envelope(centers_rays, member_rays, owner)`、`validate_layout(layout, member_rays, N_budget, spacing, fixed_width)`、`metrics(layout)` |
| 布局 | `coverage.s1fix.skeleton.Layout(centers_latlon[N,2], widths[N], owner[n], beam_ids[N])`；`Result` 的字段 (layout, N0, feasible, t_init, t_opt, t_check, t_total) |
| 第二阶段数据 | `s2fix.data.SETS`、`source_cases(role)`、`build_scene(case, request_bps, N_budget)`、`build_set`；`shared_interface.stages.make_case`、`make_layout`、`UniformStage2Input(case, layout, None, PHYSICS_V2)` |
| 贪心与评价 | `s2fix.greedy.greedy_allocation(stage_input)`、`greedy.proportional_power(demand, budget=3000, cap=50)`；`s2fix.evaluate.metrics(...)`、`fixed_problem(...)`、`run_sca(...)` |
| 搜索与循环 | `s2fix.search_env.SearchEnv._legal`；`s2fix.experiments.evaluate_scene(job)`、`scripts/s2fix/evaluate.py`、`scripts/s2fix/summarize.py` |

常量同 S2-FIX 实现说明第 1 节：三档请求 86.4/43.2/21.6 Mbps；8 组 × 100 槽 × 25 MHz；每束 1–10 槽；3000 W、每束 50 W；频谱效率上限 4.5；实现损耗 1 dB；每束飞机上限 K＝floor(1.125e9÷请求)＝13/26/52。下文的"信噪比系数"γ_b 指每瓦每槽的信噪比：链路 v2 信号系数 ÷ 每槽噪声功率 × 10^(−0.1)，从 `fixed_problem` 的信号系数与噪声取；B＝25 MHz。

## 2. 文件

```
src/satellite_rl/litcmp/
  geom_plane.py     切平面投影、六边形格、拆分/合并修复、布局组装（3.0）
  hexgrid.py        A1（3.1）
  kmeans_mec.py     A2（3.2）
  voronoi_lb.py     A3（3.3）
  layouts.py        文献布局 → make_layout / UniformStage2Input（4）
  moves.py          分配表示、合法性、邻域、解码器、均匀起点（5.1）
  objective.py      真评价计数包装、供需差与 W（5.1）
  anneal.py         B1，C2 第一阶段也用（5.2）
  genetic.py        B2（5.3）
  convex_relax.py   B3（5.4）
  power_lit.py      C1、C2（6）
  pipeline.py       方法名注册与 evaluate_scene（7）
  data.py           两个新确认集的规格与按目录加载（8）
scripts/litcmp/
  build_data.py  run_stage1.py  build_lit_layouts.py  evaluate.py  summarize_stage1.py
tests/
  test_litcmp_stage1.py  test_litcmp_moves.py  test_litcmp_search.py
  test_litcmp_power.py   test_litcmp_pipeline.py
```

## 3. 第一阶段：A1、A2、A3

### 3.0 公共几何（`geom_plane.py`）

- `tangent(c0)`：以射线 c0 为法向建切平面基 (e1, e2)。`to_plane(rays, c0)` 做心射投影 x＝(r·e1)/(r·c0)，y＝(r·e2)/(r·c0)，乘 180/π 当作度；`from_plane(xy, c0)` 反投影为单位射线。场景跨度不到 5°，投影畸变不到 0.5%，只用于铺格、三角化和移动；张角、宽度仍用 `geom.angles`、`geom.cone` 在射线上算。
- `hex_lattice(R, d)`：平面上覆盖半径 R＋d 的六边形格点，行距 d×√3/2，奇数行错开 d/2。
- `fit_budget(member_rays, clusters, N_budget, center_mode)`：拆分/合并修复。`clusters` 是成员下标数组的列表。先循环：对 `geom.cone` 半角大于 0.5° 的簇做 2-means 拆分（初始中心取簇内最远两点，10 轮 Lloyd），直到没有簇超 1.0°。再在簇数大于 N_budget 时合并：候选为中心张角不超过 1.0° 的簇对，取合并后包围圆最小且宽度不超过 1.0° 的一对合并，重复到簇数不超过 N_budget；没有候选时返回 `feasible=False` 并保留当前簇。`center_mode="cone"` 时所有簇中心取包围圆中心；`"keep"` 时未拆分的簇保留传入的中心，拆分出的子簇取包围圆中心。返回 (clusters, centers_rays, feasible)。
- `assemble(frame, clusters, centers_rays, N_budget, t_init, t_opt, feasible)`：宽度＝max(0.3°, 2×成员到中心的最大张角)；组 `Layout`（中心转回经纬度，`beam_ids` 为 0..N−1，`owner` 由簇成员反推）；`geom.validate_layout(layout, member_rays, N_budget, spacing=False, fixed_width=None)` 计 `t_check`；另算 `spacing_violations`：中心间张角小于 0.15°，或半径大于到最近其他中心张角的波束数。返回 `Stage1Result(layout, feasible, t_init, t_opt, t_check, t_total, spacing_violations)`。

### 3.1 A1（`hexgrid.py: hex_layout(frame, N_budget)`）

```
c0, R = geom.cone(全部射线)
for w in 0.30, 0.32, …, 1.00:
    格点 = hex_lattice(R, 0.85 w) 反投影为射线
    owner = 平面距离最近的格点（格距留了余量，validate 负责核对张角 ≤ w/2）
    有飞机的格 → 簇；簇数 ≤ N_budget 则停
宽度全部为 w；中心＝格心；assemble
```
1.0° 仍超预算就用 1.0° 的结果，`feasible` 由 validate 的束数检查决定。

### 3.2 A2（`kmeans_mec.py: kmeans_layout(frame, N_budget, seed)`）

```
K = N_budget；rng = np.random.default_rng(seed)
中心 = K-means++ 初始化（D² 采样，距离用 geom.angles）
δ = 0（每簇）
重复最多 30 轮：
    Lloyd 最多 50 次：owner = argmin_b angles(a, c_b) × (1 + 0.5 × max(δ_b, 0))
                     c_b = 成员射线之和归一化；空簇中心重置为最大簇里离其中心最远的成员
    L_b = 成员数；L̄ = n/K；δ_b = (L_b − L̄)/L̄
    max|δ_b| < 0.2 时停
簇 = 各中心的成员；fit_budget(…, center_mode="cone")；assemble
```
`δ` 用上一轮的值参与本轮 Lloyd。种子 51001/51002/51003。

### 3.3 A3（`voronoi_lb.py: voronoi_layout(frame, N_budget)`）

```
c0, R = geom.cone(全部射线)
d0 = 2R/√N_budget；格点 = hex_lattice(R, d0) 中落在半径 R 内的点
    少于 N_budget 时 d0 乘 0.95 重试；多于时取离原点最近的 N_budget 个
生成点 g（平面坐标）；速度 v = 0
重复最多 200 轮：
    owner = 最近生成点（平面距离）；L_i = 成员数；L̄ = n/N_budget
    邻接 = scipy.spatial.Delaunay(g).simplices 的边
    F_i = Σ_{j∈邻(i)} (L_j − L_i)/L̄ × (g_j − g_i)/|g_j − g_i|
    v = 0.5 v + 0.1 d0 F；g = g + v
    max_i |L_i − L̄|/L̄ < 0.2 时停
删掉没有飞机的生成点；簇 = 各生成点成员
fit_budget(…, center_mode="keep", 传入生成点射线)；assemble
```

### 3.4 静态运行器（`scripts/litcmp/run_stage1.py`、`summarize_stage1.py`）

`run_stage1.py --data-dir --sets dev24 conf48 --budgets 60 100 150 200 --methods A1 A2 A3 --seeds 51001 51002 51003 --workers --shard i/k --output`。集合用 `static_sets(data_dir)` 里与 S1-FIX 块 A 相同的 24 景和 48 景。进程池按（场景, 预算, 方法, 种子）分任务，A1、A3 只跑一次。输出 CSV 列与 `outputs/experiments/s1_fix_20261017/A/static_results.csv` 相同（scene_set, date, frame_time, n, N_budget, arm, effort, spacing, feasible, t_init, t_opt, t_check, t_total, N_used, peak, top10, CV, Omega_sr, width_mean, width_min, width_max；列定义参照 `scripts/s1fix/run_A.py`，在 s1fix 工作树 `C:/Users/Admin/.codex/worktrees/s1fix/paper`，本分支没有就只照列名），`arm` 为 A1/A2/A3，`effort` 填种子（A1、A3 填 0），`spacing` 填 False，另加 `spacing_violations` 列。指标直接用 `geom.metrics(layout)`。

`summarize_stage1.py --lit <本轮CSV> --ref <S1-FIX static_results.csv> --output`：按集合 × 预算 × 方法给均值（A2 先按种子平均），再与 `ref` 里 FLOW-AO（加、不加间隔）和 INIT 按（场景, 预算）配对，给峰值比值的均值和"FLOW-AO 不加间隔峰值不高于文献峰值"的场景比例，只在双方可行的场景上配对；各方法另报可行率。

## 4. 文献布局接下游（`layouts.py`、`scripts/litcmp/build_lit_layouts.py`）

- `frame_from_case(case)`：用 case 的 ICAO 与 `lat_lon_deg` 造 `Frame`（`gs_kt`、`track_deg` 填 0；`date`、`time`、`index` 从 case 的场景 ID 取）。
- `to_stage2(result, case, request_bps)`：按 `s2fix.data.build_scene` 转格式那一段的做法，用 `make_layout` 写中心、宽度、归属、锚点飞机（离中心最近的成员）、每束需求（成员数 × 请求）、`stack_id` 全为 −1；再 `UniformStage2Input(case, layout, None, PHYSICS_V2)` 过一遍。返回 (case, layout) 两个可写 JSON 的对象。
- `build_lit_layouts.py --data-dir --methods A1 A2 A3 --requests 86400000 43200000 21600000 --N 150 --workers --shard i/k --output-dir`：对 `source_cases("conf")` 的 88 景，每景每方法算一次布局（A2 用种子 51001），再对每档请求写进集合目录 `conf_d{864|432|216}_n150_{A1|A2|A3}`，目录结构与 `build_set` 一致（`objects/NNNN_case.json`、`NNNN_layout.json`、清单 JSON、检查 CSV）。检查 CSV 列：n、N_used、最大束飞机数、超 K 的束数、宽度均值/最大、spacing_violations、耗时、feasible。

## 5. 第二阶段：B1、B2、B3

### 5.1 表示、合法性、邻域、目标（`moves.py`、`objective.py`）

- 状态 `Alloc(groups, starts, counts, powers)`，四个与 `beam_ids` 同序的数组。
- `legal(alloc)`：槽数 1–10、起点＋槽数 ≤ 100、同组槽区间不重叠。规则与 `search_env.SearchEnv._legal` 相同；测试里在随机状态上与 `_legal` 对拍（`_legal` 要环境实例就在测试里构造一个）。
- 四种邻域，`propose(alloc, rng, kinds)` 等概率选一种，只返回合法的新状态（不合法就重抽，20 次后返回原状态）：
  1. `count`：随机一束槽数 ±1；加槽与同组冲突时先试起点左移，再试右移。
  2. `power`：随机两束，从一束向另一束挪 U(0, 5) W，双方保持在 [0, 50]。
  3. `start`：随机一束，起点挪到同组一个随机的合法空位。
  4. `group`：随机一束，换到随机另一组的一个合法空位，起点随机。
- `pack(groups, counts, demand)`：解码器。每组内按需求从大到小连续铺放起点；组内槽数之和超 100 时按比例缩减（向下取整、每束至少 1，剩余槽按需求顺序补回到恰好 100）。B2、B3 用。
- `uniform_start(stage_input)`：E1 的均匀起点。N 束：组＝序号模 8，每束槽数＝min(10, floor(100 ÷ ceil(N/8)))，起点按序号在组内连续铺放，功率＝min(50, 3000/N)。
- `Probe(stage_input)`：包装 `evaluate.metrics`，记录调用次数、每次的 W 和供需差、最好的解与到达时的次数；`j_gap(result)＝Σ|C_b − D_b| ÷ ΣD`，`j_w(result)＝1 − W`；曲线在次数 32、64、128、320、640、1280、3200 处留点。B1、B2、C2 只通过它评价。

### 5.2 B1（`anneal.py: anneal(stage_input, start, probe, budget, seed, objective, kinds)`）

```
x = start；J = objective(probe(x))        起点这一次不计入预算
T0 = 0.05 × max(J0, 0.01)                 W 已到 1 时 J0 为 0，取下限
Tend = 0.02 T0；α = (Tend/T0)^(1/budget)；T = T0
for t in range(budget):
    y = propose(x, rng, kinds)；Jy = objective(probe(y))
    接受若 Jy ≤ J 或 rng.random() < exp(−(Jy − J)/T)
    T = T α
返回 probe 记录的最好解
```
`kinds` 默认四种；C2 只用 `power`。`objective` 为 `j_gap`（主）或 `j_w`（W 目标版）。

### 5.3 B2（`genetic.py: genetic(stage_input, start, probe, budget, seed, objective)`）

- 个体＝(counts[N] 整数 1–10, p_raw[N] 实数 0–50)；解码：组＝G1 的组，起点＝`pack`，功率＝`proportional_power(p_raw, 3000, 50)`。
- 种群与代数按预算：32→(8, 3)，320→(16, 19)，3200→(32, 99)；初始种群＝start 加 (种群−1) 个 start 的变异体，初始评价 (种群−1) 次，之后每代评价种群次，总次数不超过预算。
- 每代：锦标赛选择（2 选 1）成对产生子代；逐束均匀交叉（概率 0.9）；逐束变异（概率 1/N：槽数 ±1 或 p_raw 乘 U(0.8, 1.25)，二选一）；精英 1 个直接进入下一代。
- 返回 probe 记录的最好解。

### 5.4 B3（`convex_relax.py: convex_relax(stage_input, g1_alloc)`）

变量 k_b ∈ [1, 10]、p_b ∈ [0, 50]、s_b ≥ 0（辅助变量，代表 min(容量, 需求)）。最大化 Σ s_b，约束：

- s_b ≤ k_b B log2(1 ＋ γ_b p_b / k_b)
- s_b ≤ 4.5 B k_b
- s_b ≤ D_b
- Σ k_b ≤ 800，Σ p_b ≤ 3000

`scipy.optimize.minimize(method="SLSQP", jac=解析梯度, maxiter=500, ftol=1e-9)`，目标与容量约束除以 ΣD 归一；初值 k＝G1 槽数、p＝G1 功率、s＝相应容量与需求的较小者。γ_b 从 `fixed_problem` 取，干扰不进松弛。取整：counts＝round(k) 夹到 1–10；groups＝G1 的组；starts＝`pack`；powers＝p。返回 Alloc。

### 5.5 怎么接 G1、S1、S2

不重写。G1、R、S1、S2 走 `s2fix.experiments.evaluate_scene`；B1、B2、B3 走 `litcmp.pipeline.evaluate_scene`，内部先 `greedy_allocation` 得起点、再搜索、再 `run_sca`、再 `metrics`，分段计时和输出列与 s2fix 的一致。

## 6. 第三阶段：C1、C2（`power_lit.py`）

- `noise_limited(problem, alloc)`：从 `fixed_problem` 的信号系数与噪声得每束 γ_b；容量 C_b(p)＝k_b B min(log2(1 ＋ γ_b p / k_b), 4.5)；"刚好满足"功率 p^D_b＝k_b (2^{min(D_b, 4.5 B k_b)/(k_b B)} − 1)/γ_b。
- `waterfill(gamma, k, D, budget=3000, cap=50, omega=0.0)`：边际容量 dC/dp＝B γ_b/((1 ＋ γ_b p/k_b) ln 2)；给定水位 ν，p_b(ν)＝clip(k_b (B/((ν＋ω) ln 2) − 1/γ_b), 0, min(cap, p^D_b))。ν＝0 时 Σp ≤ budget 直接返回；否则对 ν 二分 40 次使 Σp＝budget。向量化，一次调用不到 1 毫秒。
- `C1(stage_input, g1_alloc)`：γ 用干扰置零的系数，`waterfill(omega=0)`；返回功率。
- `C2(stage_input, g1_alloc, seed, probe)`：第一阶段 `anneal(kinds=["power"], budget=320, objective=Σ(D−C)⁺ ＋ ω Σp)`，ω＝0.001 × ΣD ÷ 3000（bit/s 每瓦，即用满 3000 W 记 0.1% 总需求的代价）；第二阶段在粗解功率下用 `fixed_problem` 的干扰系数算每束受到的干扰 I_b，γ_b 改为 信号系数 ÷ (每槽噪声 ＋ I_b) × 10^(−0.1)，`waterfill(omega=ω)`；返回功率。
- 两者在 `pipeline` 里按方法名在"功率细调"段调用：原分配（G1 功率）、C1、C2、SCA 四条。

## 7. 方法注册与 E1（`pipeline.py`）

| 名称 | 布局 | 起点 | 搜索 | 功率细调 |
| --- | --- | --- | --- | --- |
| `B1_32` `B1_320` `B1_3200` | 集合自带 | G1 | 退火，供需差 | SCA |
| `B1W_320` | 同上 | G1 | 退火，W 目标 | SCA |
| `B2_32` `B2_320` `B2_3200` `B2W_320` | 同上 | G1 | 遗传 | SCA |
| `B3` | 同上 | G1 | 凸松弛取整 | SCA |
| `P0` `C1` `C2` | 同上 | G1 | 无 | 原分配 / C1 / C2 |
| `E1_320` `E1_3200` | 集合自带（跑在 A1 集合上） | 均匀起点 | 退火，供需差 | C1 |

`evaluate_scene(job)`：job 含集合目录、场景序号、方法名、种子。输出字典与 s2fix 的逐景行相同（功率细调前后全部指标、分段耗时 t_init/t_search/t_refine/t_total），加 `method`、`seed`、`probes`、`best_at`（最好解出现在第几次评价）和曲线留点列 `w_at_32`…`w_at_3200`（没到的留空）。

## 8. 数据（`data.py`、`scripts/litcmp/build_data.py`）

- **三档确认集已由项目管理者在本机建好**：`D:/paper/outputs/experiments/litcmp_20261018/data/{conf_d864_n150,conf_d432_n150,conf_d216_n150}/`，目录结构与 `build_set` 相同（`manifest.json`、`checks.csv`、`objects/NNNN_case.json`、`NNNN_layout.json`），用的是 s2fix 5a54928 的 `build_scene` 原样参数。不需要再建；冒烟和正式运行都直接读这个目录。
- **s2fix 读输入有一处错**：`s2fix.data.source_cases` 用 `Path(row["case_ref"]["path"]).name` 找对象，但 `collected/inputs/objects/` 下文件按 sha256 命名，应为 `root / "objects" / row["case_ref"]["sha256"]`。不改 s2fix 文件；在 `litcmp/data.py` 里放一份改正后的 `source_cases`（约 20 行，照抄原函数只改这一行），`build_data.py` 用它（把 `s2fix.data.source_cases` 替换为它后调 `build_set` 即可，`build_set` 引用的是模块全局名）。参照 `D:/paper/outputs/claude_scripts/litcmp_build_conf.py`。
- `build_data.py --sets ... --output-dir --workers` 保留，供 S2-FIX 改正 86.4 档数据后重建；新集合规格 `conf_d432_n150`＝("test", 43200000, 150)、`conf_d216_n150`＝("test", 21600000, 150)。
- `load_set(set_dir)`：按目录读清单，返回逐景 (case, layout, 场景 ID, 日期)。`evaluate.py` 一律用目录，不查 `SETS`。
- 已知情况：86.4 档 88 景中 37 景（飞机数 ≥1418）因 FLOW-AO v2 限时分配问题只用了 100 束且有超上限波束，已报给 S2-FIX 会话处理；对我们的代码没有影响，所有方法用同一份布局。

## 9. 脚本与命令行

| 脚本 | 作用 | 参数 |
| --- | --- | --- |
| `build_data.py` | 第 8 节 | `--sets --output-dir --workers --shard` |
| `run_stage1.py` | 第 3.4 节 | `--data-dir --sets --budgets --methods --seeds --workers --shard --output` |
| `build_lit_layouts.py` | 第 4 节 | `--data-dir --methods --requests --N --workers --shard --output-dir` |
| `evaluate.py` | 对集合目录跑方法列表；G1/R/S1/S2 走 s2fix 的 `evaluate_scene`，其余走 `pipeline.evaluate_scene`；逐景逐方法逐种子一行 CSV | `--set-dir --methods --seeds 51001 51002 51003 --workers --shard --output-dir` |
| `summarize_stage1.py` | 第 3.4 节 | `--lit --ref --output` |

- 切片：`--shard i/k` 按场景序号模 k 等于 i 取场景；输出文件名加 `_shard{i}of{k}`；汇总前把分片 CSV 直接拼接。
- 进程池用 `concurrent.futures.ProcessPoolExecutor`，`--workers` 默认取 CPU 数；每个任务只做一个（场景, 方法, 种子）；子进程里 `OMP_NUM_THREADS=1`。
- 第二、三阶段与端到端的汇总直接用 `scripts/s2fix/summarize.py`，输出列对齐即可。
- 每个脚本把配置、提交号、主机、起止时间追加到输出目录的 `run_manifest.json`，做法与 s2fix 脚本相同，能复用它的函数就复用。

## 10. 测试（`tests/test_litcmp_*.py`）

- `test_litcmp_stage1.py`：合成帧（三个高斯团共 300 架，另 3 架孤立在 2° 外），预算 20：A1、A2（一个种子）、A3 都返回 `validate_layout(spacing=False)` 通过的布局，束数不超过 20，宽度在 0.3–1.0°；A1 所有宽度相等；`fit_budget` 把一个直径 1.6° 的簇拆成两个且各不超过 1.0°，把超预算的簇列表合并到预算内。
- `test_litcmp_moves.py`：从一个小 `stage_input`（合成 case 经 `make_case`/`make_layout` 构造，10 束）的 G1 起点出发，连续 500 次 `propose` 全部 `legal`，且与 `SearchEnv._legal` 对拍一致；`pack` 在组内总槽 120 时缩到 100 且每束不少于 1；`uniform_start` 合法且总功率不超过 3000。
- `test_litcmp_search.py`：`anneal` 预算 32 恰好调用 `metrics` 32 次（起点不计）；`genetic` 预算 32 调用 28 到 32 次；两者返回的解合法且目标不差于起点；`convex_relax` 返回的 counts 在 1–10、总槽不超过 800、总功率不超过 3000.01，松弛目标值不低于起点的松弛目标值。
- `test_litcmp_power.py`：`waterfill` 预算宽裕时每束功率等于 min(p^D, 50)；预算紧张时总功率等于 3000 且功率为正、未封顶的束边际容量相等（相对误差 1e-3）；`C2` 第二阶段的（未满足需求 ＋ ω 总功率）不高于第一阶段。
- `test_litcmp_pipeline.py`：同一个小 `stage_input` 上每个方法名各跑一次，返回字典含全部指标列和四段耗时，`probes` 等于预算（B 类）、0（P0/C1）或 320（C2）。

## 11. 不需要做的事

不接工作台；不改 `s2fix/`、`coverage/s1fix/`、`shared_interface/`、`power_refinement/` 的任何文件；不重训 L1/L2；不写回执、预算账、SHA；不写正式报告；不做建议项与可选项。

## 12. 本地冒烟（必须做，并在回复里给出数字）

1. **第一阶段**：1 月 12 日分层 24 景里飞机最少和最多的各一景，预算 100：A1、A2（种子 51001）、A3 的 N_used、峰值、宽度均值、spacing_violations、耗时、feasible。
2. **接下游**：用改正后的 `source_cases` 加 `build_set(limit=2)` 在 `D:/paper/outputs/experiments/litcmp_20261018/smoke/data/` 建 `dev_d864_n150` 前 2 景（约 30 秒），对这 2 景的 case 做 A1、A2、A3 布局，86.4 档，经 G1＋SCA：W、最差 10%、总功率、耗时；`UniformStage2Input` 构造通过。
3. **第二阶段**：同 2 景（FLOW-AO v2 布局）：`B1_32`、`B2_32`、`B3` 的 SCA 前后 W 与耗时；`B1_320` 一次只看耗时。
4. **第三阶段**：同 2 景：`P0`、`C1`、`C2`（种子 51001）的 W、总功率、耗时。
5. **E1**：同 2 景用 A1 集合跑 `E1_320`：W、最差 10%、总功率。
6. `build_data.py --sets conf_d432_n150 --shard 0/44` 本地跑 2 景到冒烟目录，只看能产出、检查列里 K＝26 正确；不要跑全集，也不要写进正式数据目录。
7. `evaluate.py --shard 0/2` 与 `--shard 1/2` 在 dev 集上输出的场景集合互不重叠、并集完整。
8. 全部 `tests/test_litcmp_*.py` 通过；s2fix 与 s1fix 的旧测试不受影响（它们的文件没动）。
