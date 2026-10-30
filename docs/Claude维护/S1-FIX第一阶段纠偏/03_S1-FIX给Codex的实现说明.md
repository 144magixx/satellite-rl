# S1-FIX 实现说明（给Codex）

来源：cc

2026-10-17｜对应设计：[02 第一阶段纠偏实验设计](02_第一阶段纠偏实验设计.md)。本文是可直接照做的实现规格。Codex负责实现、冒烟测试和数据抽取；正式运行由Claude在远端发起。

## 0. 工作方式（先读）

- **分支与目录**：在git worktree `C:/Users/Admin/.codex/worktrees/s1fix/paper`（分支`s1fix`，基于`main`）里工作。新代码放`src/satellite_rl/coverage/s1fix/`，运行脚本放`scripts/s1fix/`。完成后在分支上提交（前缀`feat(s1fix):`），不合并main。
- **数据与输出用绝对路径**：原始数据在`D:/paper/outputs/...`（被git忽略，worktree里没有）；所有输出写到`D:/paper/outputs/experiments/s1_fix_20261017/`（下称OUT）。脚本用命令行参数接收路径。
- **写法要求（用户明确）**：
  - 不做防御性编程：不写层层校验、守卫、回退、断言；按正常路径写清楚，真出错就让它报错。只有`validate_layout`这一个检查函数是算法必需的。
  - 不算SHA/哈希、不写回执、不建"预算账本/ledger/identity"类对象。历史代码（`ai_search_r*`）里那一套一律不要搬。
  - 不要类层级，用函数加少量dataclass。numpy/scipy为主，需要时用torch。
  - 新代码总量目标约2500行（含测试），超过4000行说明设计过重。
  - 你写的任何说明文档，标题下第一行写"来源：codex"，用通俗中文。
- **测试**：只写必要的小测试（几何、合法性检查、流量指派、下界在玩具例上的正确性），放`tests/test_s1fix_*.py`。本地冒烟：每个块在1到2个小场景、少量迭代上跑通即可，**不要在本地跑完整实验或完整训练**。
- **本地环境**：`C:/Users/Admin/anaconda3/envs/DQN/python.exe`（numpy 1.26.4、scipy 1.11.4、torch 2.3.1）。远端是Python 3.11、numpy 1.26.4、scipy 1.14.1、torch 2.3.1+cu121，16核CPU、一张4090D；代码须两边都能跑，scipy只用1.11已有的接口。
- **完成时**：最后回复用中文简短列出：实现了哪些文件、各块怎么运行（命令）、冒烟测试结果、数据抽取结果（尤其第2节的历史场景核对）、与本文不同的地方及原因。

## 1. 问题定义（冻结）

- 卫星：GEO，13°E。飞机位置为地面投影（经纬度）。用`satellite_rl/coverage/geometry.py`里的`satellite_rays_numpy(positions,'europe_v1')`得到卫星指向各点的单位向量，角度用两向量夹角。
- 每架飞机请求21.6 Mbps；负载一律用"架数"计算和报告（1架＝21.6 Mbps）。
- 决策：波束数N、每束中心（地面经纬度）、每束全宽w、每架飞机唯一归属。
- 合法布局（`validate_layout`检查的全部内容）：
  1. 每架飞机恰好归属一个波束，且与所属波束中心夹角 ≤ w/2；
  2. 0.3° ≤ w ≤ 1.0°；w取"成员包络"：w = max(0.3°, 2×最远成员夹角)，数值上加1e-9余量；
  3. 非空波束数 N ≤ N预算（N预算 ≤ 200）；
  4. **最小中心间隔**（可开关，默认开）：任意两束中心夹角 ≥ max(wi, wj)/2，即任一束的中心都不落入另一束足迹内。
- Ω（各束立体角之和，2π(1−cos(w/2))）只报告，不作约束。
- 指标：峰值（最忙波束架数）、最忙10%波束（ceil(0.1N)个）平均架数、全束负载CV（总体标准差/均值）、实际N、Ω、宽度均值/最小/最大、分段耗时（初始解、优化、检查、总计）。

## 2. 数据

**原始日包**（readsb，每个tar内是逐机gzip JSON：`{"icao","timestamp","trace":[[dt,lat,lon,alt_ft或"ground",gs_kt,track_deg,flags,vrate,...],...]}`）：

| 日期 | 路径 | 角色 |
| --- | --- | --- |
| 2024-01-08 | `D:/paper/outputs/experiments/europe_adsb_20260928/raw/v2024.01.08-planes-readsb-prod-0.tar` | 训练（块B）；历史对照HIST12 |
| 2024-01-09 | 同目录 `v2024.01.09-...tar` | 训练（块B） |
| 2024-01-12 | 同目录 `v2024.01.12-...tar` | 开发 |
| 2024-01-15 | 同目录 `v2024.01.15-...tar` | 确认 |
| 2024-01-18 | `D:/paper/outputs/experiments/europe_ablation_20260929/data/raw/v2024.01.18-planes-readsb-prod-0.tar` | 确认 |

**取帧规则**（与历史场景一致，参考`src/satellite_rl/workbench/aviation.py`第86—300行）：对时刻t，取每架飞机时间戳≤t的最新一条记录；若它距t超过60秒、或该条无效、`flags&1`（stale）、`ground`、高度≤0或>25000 m，则该机不在场；再要求纬度25—80、经度−40—50、对13°E卫星仰角≥10°（`coverage/geometry.visible_numpy(positions,'europe_v1')`）。不插值。

**抽取两套时刻**，每个日包读一遍：
- 静态网格：UTC 00:02起每5分钟，共288帧；
- 动态网格：UTC 06:00:00至22:00:00每60秒，共961帧。

存为`OUT/data/frames_<日期>.npz`：两套网格分开存，每帧的icao（转uint32）、lat、lon、gs_kt、track_deg（缺失为NaN）用offsets拼接。

**历史场景核对（必须做）**：
- HIST12：1月8日静态第0、6、12、…、78帧（共12帧，即序号0,6,12,18,24,36,42,48,54,60,72,78）的飞机数应依次为293、216、204、178、175、185、235、288、318、423、946、1254。
- HIST20：1月12日静态第floor((i+0.5)·288/20)帧（i=0…19）的飞机数应等于`D:/paper/outputs/experiments/first_stage_ai_search_20261013/round8/analysis/mechanism_01/mechanism_rows.csv`中各case_index的`aircraft_count`（mother_id给出时刻）。
- 若不一致，查明原因并修正抽取规则，直到一致；在最终回复里报告核对结果。

**场景集**：
- DEV24（1月12日静态网格）：按飞机数分四层：<300、600—1000、1200—1600、>1600；每层内按时间排序，取第floor((k+0.5)·m/6)个（k=0…5，m为该层帧数）。
- CONF48：1月15日、1月18日各按同一规则取24帧。
- HIST12、HIST20：如上。
- 动态序列：每天06:00起每30分钟一段，每段31帧（含两端，即t0到t0+30 min），06:00—22:00共32段。训练用1月8、9日全部64段；开发用1月12日偶数段（16段）；确认用1月15、18日偶数段（各16段）。

## 3. 几何工具（`geom.py`）

- 卫星射线、夹角矩阵、KD树近邻（`scipy.spatial.cKDTree`在单位向量上用弦长 2sin(θ/2) 查询）。
- 最小包络锥：对一组成员射线，在其平均方向的切平面做gnomonic投影，用Welzl算法求最小外接圆，反投影回中心射线；返回中心与半角。中心射线转回地面经纬度（射线与地球球面求交）。
- 包络宽度、Ω、间隔检查。
- `forecast(lat,lon,gs_kt,track_deg,dt_s)`：按大圆航迹推算dt秒后位置；gs或track缺失时位置不变。

## 4. 骨架算法 FLOW-AO（`skeleton.py`）

核心思想：给定一组波束中心（每束覆盖半径上限0.5°，即宽度上限1.0°），"哪架飞机归哪个波束"可以用网络流精确求解；于是交替做"精确指派—重定中心—结构调整（拆最忙、并最闲）"。

**4.1 精确指派 `assign(centers, rays, mode)`**：二部图：飞机→其0.5°内的所有中心。
- `mode='peak'`：最小化峰值T：对T二分，用`scipy.sparse.csgraph.maximum_flow`判可行（源→飞机容量1，飞机→波束容量1，波束→汇容量T）。
- `mode='balance'`：在峰值≤T*下，最小化 Σ负载²，次要项为总夹角（权重小到不改变Σ负载²）。这是凸费用流，可写成LP（每束拆成T*条单位弧，第k条费用2k−1），用`scipy.optimize.linprog(method='highs')`求解，结果天然取整。
- 可选`prev_owner`与`kappa`：对"不在上一帧原波束"的飞机加费用kappa（块B的少换束指派用）。
- 可选`eligible_mask`：只允许指定飞机—波束对（块B的预测感知指派用）。

**4.2 初始覆盖 `initial_cover(rays)`**：贪心集合覆盖：候选中心＝各飞机位置，覆盖半径0.5°；每步选覆盖未覆盖飞机最多的候选，直到全覆盖；得到N0。N0>N预算时该场景在此N预算下不可行，记录N0并返回。

**4.3 主循环 `solve(frame, N_budget, spacing=True, effort=1, time_limit=None, fixed_width=None, init_layout=None)`**：
1. 初始中心：`init_layout`给定时用它（块B热启动），否则用`initial_cover`。
2. 增束：当 N < N预算：对当前最忙的波束，用其成员做2-means拆成两束（新中心取两组成员的最小包络锥中心），重新`assign('peak')`；直到N达到预算，或峰值已等于1。
3. 交替优化，至多`iters = 30×effort`轮：`assign('peak')` → `assign('balance')` → 各束重定中心为成员最小包络锥中心 → 结构动作：若存在可删除的最闲束（删后`assign('peak')`的峰值不变），删除它并把名额用于拆当前最忙束 → 间隔修复。目标按(峰值, 最忙10%均值, CV)字典序比较，连续3轮无改进则停。
4. effort>1时额外做`effort−1`次随机重启（打乱拆分顺序或初始覆盖候选顺序，种子固定为51001+重启序号），取字典序最好结果。
5. 间隔修复（spacing=True时）：对违反间隔的一对，先尝试合并（并集最小包络锥半角≤0.5°且负载≤当前峰值）；不行则删掉较轻的一束、其成员经`assign`分给其他束；仍不行则把较轻束中心沿两中心连线方向外移到满足间隔，再`assign`。修复后必须通过`validate_layout`；若某次结构动作导致无法修复，撤销该动作。
6. `fixed_width=0.5`：所有束宽度固定0.5°（覆盖半径0.25°），用于"宽度可变值多少"的消融。
7. 返回布局（中心经纬度、宽度、归属、束ID）和分段耗时。

**4.4 INIT基线**：`initial_cover`＋按4.3第2步增束，但指派用"就近中心"而不用流量；不做第3步。作为最简单的对照。

**性能目标**：n≈1800、N=200、effort=1时，单场景不超过30秒（远端单核）。性能不够时先换成增量max-flow或缩小LP，不要砍功能。

## 5. 下界（`bound.py`）

对每个(场景, N预算)：
- **平均下界**：ceil(n/N)。
- **LP下界**：候选中心取卫星角坐标上间距g=0.1°的网格点（只保留距某架飞机≤r的点），每个候选覆盖半径 r = 1.01×(0.5°+g/√2)（这样任何真实波束都被某个候选完全包含，下界才成立）。对整数T解LP：min Σx_c，s.t. 每架飞机 Σ_c y_ac = 1；y_ac ≤ x_c；Σ_a y_ac ≤ T·x_c；0≤x≤1，y≥0。最小的使LP最优值≤N的T即为下界LB_lp。T在[ceil(n/N), FLOW-AO峰值]内二分。最小中心间隔在下界中忽略（仍是合法下界）。
- 在3个场景上用g=0.05°复算，报告下界是否变化。
- **整数加强**（只对n<600的场景）：把x设为0/1，用`scipy.optimize.milp`限时120秒，若在某T上证明最小束数>N，则下界提高到T+1。
- 输出 `OUT/A/bounds.csv`：scene_set, frame_time, n, N_budget, LB_avg, LB_lp, LB_milp, g, lp_seconds, status。

## 6. 块A运行（`scripts/s1fix/run_A.py`）

用multiprocessing对(场景, 臂, N预算)并行。

| 实验 | 场景 | N预算 | 臂 |
| --- | --- | --- | --- |
| A0/A1 | DEV24 | 60、100、150、200 | FLOW-AO（间隔开）、FLOW-AO（间隔关）、FLOW-AO（固定0.5°，间隔开）、INIT |
| A1历史对照 | HIST12 | 每景取历史SH-C的实际N | FLOW-AO（间隔关，与历史口径一致） |
| A1历史对照 | HIST20 | 每景取`mechanism_rows.csv`的Ncap；另跑一次N=历史R8 G臂的实际N | FLOW-AO（间隔关） |
| A2 | DEV24、CONF48 | 60、100、150、200 | 下界 |
| A3 | DEV24中≥1200架的12景 | 200 | FLOW-AO（间隔开）effort=1、3、10 |
| 确认 | CONF48 | 60、100、150、200 | FLOW-AO（间隔开）、INIT |

历史结果文件（只读，用于对照表）：
- HIST12：`D:/paper/outputs/experiments/first_stage_ai_search_20261013/round10/science_independent_statistics_01/endpoint_metrics.csv`（method=SH-C、S2、B；maximum_count即峰值架数，N、Omega_sr、CV）。
- HIST20：`D:/paper/outputs/experiments/first_stage_ai_search_20261013/round8/analysis/actual_v4_01/analysis20x3.csv`（arm=A/G/R；maxload为bps，除以21600000得架数；N、CV、actual_Omega）。

输出：
- `OUT/A/static_results.csv`：scene_set, date, frame_time, n, N_budget, arm, effort, spacing, feasible, N0, N_used, peak, top10, CV, Omega_sr, width_mean, width_min, width_max, t_init, t_opt, t_check, t_total。
- `OUT/A/hist_compare.csv`：每个历史场景一行，历史方法与FLOW-AO的峰值、最忙10%、CV、N、Ω并列。
- `OUT/A/layouts/`：每个结果一个npz（centers_latlon, widths, owner, aircraft_index），供块C使用。

## 7. 块C：三阶段连通评价（`downstream.py`、`scripts/s1fix/run_C.py`）

固定下游，不用任何学习模型：

1. **布局转第二阶段**：用`src/satellite_rl/shared_interface/link.py`的`build_link_coefficients(layout, allocation, PHYSICS)`（`PHYSICS`来自`shared_interface/spherical_gaussian.py`，已支持逐束宽度）。layout与allocation直接构造所需字段的普通dict（`content_sha256`字段填任意字符串即可），不要走`stages.py`里的validate/seal流程。
2. **频率组（8组，组号奇偶表示极化）**：按需求从大到小依次给波束选组。代价＝该束与已分配的"同极化、不同组"波束之间的耦合增益之和（用同一物理模型的发射增益随夹角衰减）＋该组已占槽数超出100的惩罚。同组波束槽位互斥、互不干扰，异极化互不干扰。
3. **槽数与起点**：每束请求槽数 k = clip(ceil(D/100 Mbps), 1, 10)；若某组请求总和>100，按需求比例缩减（每束至少1槽）。组内槽位从小到大连续排；起点选择使与"同极化其他组"已占槽重叠最少（参考`src/satellite_rl/fair_comparison/environment.py`的`baseline_action`）。
4. **初始功率**：按需求比例分配，总和3000 W。
5. **第三阶段**：用`power_refinement/physics.py`的`FixedAllocationPhysics(demand_bps, groups, starts, slots, signal, interference, noise, slot_owner, 3000.)`直接以link系数构造（不走`from_scene`），调用`power_refinement/optimizer.py`的`optimize_power(problem, p0, 'sca', options)`。先在随机功率上核对`FixedAllocationPhysics`的容量与`link.capacity_bps`一致。
6. **服务指标**：每束容量C、需求D；每架飞机满足率 s = min(C/D, 1)（同束飞机请求相同，平分容量）。报告：飞机平均满足率（＝承载比例Σmin(C,D)/ΣD）、最差10%飞机平均满足率、满足率≥0.9的飞机比例、束均SI、SGM；分别给第二阶段贪心直出（SCA前）与SCA后两套；以及第二阶段、SCA耗时。

**布局变体**：DEV24与CONF48每个(场景, N预算)取三种同N布局：INIT、FLOW-AO（只做峰值指派，不做Σ负载²）、FLOW-AO（完整）。再加A3中effort=10的布局。输出`OUT/C/service.csv`：scene_set, frame_time, n, N_budget, variant, N_used, peak, top10, CV, width_mean, Omega_sr，以及上面的全部服务指标与耗时。

## 8. 块B：相邻时刻增量重配置（`dynamic.py`、`ppo_dyn.py`、`scripts/s1fix/run_B.py`）

N预算固定200，间隔规则开。

**8.1 波束身份与代价**
- 波束有持久ID。`patch`和`repair`保留ID；新开波束给新ID；`full`重解后用匈牙利算法（`scipy.optimize.linear_sum_assignment`，代价＝中心夹角，夹角>1.0°不匹配）对上一帧继承ID。
- 每帧代价：换束比例＝两帧都在场且所属波束ID改变的飞机数／两帧共有飞机数；波束事件＝新增数＋删除数＋中心移动>0.1°数＋宽度变化>0.05°数。

**8.2 三个算子**
- `patch(L, frame)`：中心不动。离场飞机删除；留在场且仍在原束0.5°内的保留并更新包络宽度；新进或越界的飞机分给0.5°内负载最小的束；无束可分时，若N<预算则在该机处开新束，否则分给最近束并重定该束中心（半角≤0.5°）。最后修复间隔。仍不合法时返回失败，由调用方升级为`full`（记为强制升级）。
- `repair(L, frame, region, iters, kappa, forecast)`：以`patch`结果为起点，用4.3的交替优化，`assign`带`prev_owner`和`kappa`。`region='heavy'`只动最忙的6个波束及中心在其1.0°内的波束（其余束和归属冻结）；`region='all'`动整景。`forecast=True`时：每架飞机只允许分给"当前位置和60秒后预测位置都在0.5°内"的束（若没有这样的束，退回只看当前位置），重定中心用当前与预测位置的中点。
- `full(frame)`：FLOW-AO从头解（effort=1，time_limit 30 s），再做身份匹配。`full`的结果只依赖该帧，**按帧预先计算并缓存**到`OUT/B/full_cache/`，训练和评价都读缓存，耗时按缓存中记录的求解时间计。

**8.3 对照臂**

| 臂 | 每帧做什么 | 参数 |
| --- | --- | --- |
| B0 | `full` | — |
| B1 | `patch` | — |
| B3(k) | 每k帧`full`，其余`patch` | k=5、10 |
| B2(θ) | `patch`；若峰值−参考峰值>θ则`repair(heavy, iters=10, kappa=1, forecast=False)`；参考峰值为上次repair/full后的峰值 | θ=1、2、4架 |
| B2+(θ) | 同B2，但触发看max(当前峰值, 预测60秒后峰值)，repair用forecast=True | θ=1、2、4架 |
| B4 | PPO策略每帧从下表5个动作中选一个 | λ=0.05、0.2、0.8；种子51001、51002、51003 |
| B4−预测 | 同B4，但不给预测特征，动作中的repair用forecast=False | λ=0.2；三个种子 |

预测峰值：把当前布局`patch`到预测位置（不改中心）后的峰值。

**8.4 PPO设定**
- 动作（5个）：0 `patch`；1 `repair(heavy, iters=5, kappa=3)`；2 `repair(heavy, iters=15, kappa=0.5)`；3 `repair(all, iters=10, kappa=0.5)`；4 `full`。B4的repair用forecast=True。
- 观测（约16维，做固定尺度归一化，不用数据统计量）：峰值/LB、峰值−LB、最忙10%/LB、CV、N/200、预测峰值/LB、参考峰值/LB、距上次repair帧数/30、距上次full帧数/30、新进飞机比例、patch前越界飞机比例、近5帧飞机数相对变化、小时的sin与cos、近5帧平均换束比例、剩余帧数/30。LB＝ceil(n/200)。B4−预测去掉预测峰值一项。
- 奖励：r_t = −(峰值_t − LB_t) − λ·100·换束比例_t。第0帧用`full`初始化且不计入奖励。
- 网络：MLP两层各64，ReLU；PPO：lr 3e-4，γ=0.99，GAE 0.95，clip 0.2，熵系数0.01，每次更新16个回合、4个epoch；共1200个回合（每回合30步），每100回合存一次检查点并记录曲线；正式用最后一个检查点，不按开发成绩挑。CPU训练即可（网络很小），每个训练任务一个进程。
- 训练时每回合从64段训练序列中均匀随机抽一段（由种子决定）。

**8.5 评价**：所有臂在开发16段与确认32段上各跑一遍（PPO取argmax）。输出：
- `OUT/B/eval_frames.csv`：arm, param, seed, day, seq, frame, n, LB, peak, top10, CV, N, churn, beam_events, action, forced_full, t_frame；
- `OUT/B/eval_summary.csv`：每臂每参数每种子的平均（峰值−LB）、平均峰值、平均换束比例、平均波束事件、平均每帧耗时，按开发/确认分开；
- `OUT/B/train_curves.csv`：run, episode, return, mean_gap, mean_churn, 各动作比例；
- `OUT/B/models/`：各PPO最后检查点。

## 9. 运行脚本与清单

- `scripts/s1fix/extract_frames.py`（本地运行，抽取五个日包；完成后把npz放到OUT/data）。
- `scripts/s1fix/run_A.py`、`run_C.py`、`run_B.py`：参数为数据目录、输出目录、worker数、要跑的子实验；能在远端用`python -m`或直接脚本运行（`PYTHONPATH=<repo>/src`）。`run_B.py`分子命令：`cache`（预计算full）、`baselines`、`train --lam --seed [--no-forecast]`、`eval`。
- 每个脚本把配置、git提交号、Python与包版本、主机名、起止时间写入`OUT/run_manifest.json`的对应条目（简单追加即可）。
- 远端Claude会用16核运行，请给出各子实验在远端的预计耗时。

## 10. 不需要做的事

不接工作台；不写独立审核、回执、预算账；不做场景身份SHA；不训练任何静态布局学习模型（设计里的A4条件性分支本次不做）；不改动`ai_search_r*`等历史模块；不写正式报告（Claude写）。

## 11. 修订一：半径上限法（2026-10-17 04:40，替换4.3第5步与全部间隔修复）

**远端实测问题**：1月12日最大场景（1752架）、N预算200、间隔开，FLOW-AO只用了33束、峰值104架、平均宽0.77°。原因是指派允许每束覆盖到0.5°，所有束都撑到接近1.0°宽，间隔规则于是要求中心相距约0.5°，拆出来的新束总被间隔修复并回去。

**新做法（数学上等价于间隔规则，且不再需要修复）**：
- 任意两中心夹角 ≥ s_min（可变宽度时s_min=0.15°，即0.3°最小宽度的一半）。
- 每束覆盖半径上限 ρ_j = min(ρ_max, 该束中心到最近其它中心的夹角)，ρ_max=0.5°。
- 飞机a只能分给满足 θ(a,c_j) ≤ ρ_j 的束。这样任何指派得到的包络宽度 w_j ≤ 2ρ_j，自动满足"中心不落入对方足迹"。
- 固定0.5°宽度：s_min=0.25°，ρ_j=0.25°，宽度恒为0.5°。
- 间隔关（历史对照口径）：不设s_min（只要求中心不重合），ρ_j恒为ρ_max。
- `assign`改为接受每束半径数组`radii`；删除`repair_spacing`及其所有调用。`validate_layout`保留不变，作为最终检查。

**初始覆盖改为0.5°网**：在未覆盖飞机中，选0.5°内未覆盖飞机最多的那架作为新中心（它必然距所有已有中心>0.5°，所以新中心半径上限就是0.5°），重复到全覆盖。得到的N0个中心两两相距>0.5°，指派一定可行。

**增束（拆分）**：对最忙束做2-means，得两个子中心；若某子中心距其它任一中心<s_min，沿两子中心连线向外微调或放弃该束改试下一忙束。加入后重算全部半径上限，`assign('peak')`；可行且峰值不升则接受，否则试下一忙束；所有束都拆不了或N达预算时停止。

**交替优化每轮**：`assign('peak')`→`assign('balance')`→各束重定中心为成员最小包络锥中心（仍须满足s_min）→重算半径上限→`assign('peak')`；可行且(峰值,最忙10%,CV)字典序变好才接受这一轮，否则回退。结构动作（删最闲可删束、拆最忙束）照旧，都在半径上限下判可行。

**INIT基线**：0.5°网＋同样增束，但每架飞机分给"半径上限内最近的中心"，不用流量、不做交替优化。

**块B同步修改**：
- `patch`：中心不动；为找不到合格束的飞机按0.5°网规则就地开新束（距所有中心≥s_min且预算未满）；重算半径上限；原归属仍合格的飞机保持不动，其余分给合格束中负载最小者；仍有飞机无合格束则失败（调用方升级为full）。
- `repair`：用半径上限代替间隔修复；冻结束的中心不动，移动束的新中心不得落入冻结束的当前半径上限内，冻结束成员必须仍在其半径上限内（不满足就拒绝该候选）。删掉第214—220行"修复后再复核预测一致性"那段：预测只用于指派时的合格性和重定中心。冻结束宽度比较改用`np.isclose`。

**修完后自测**：在1月12日最大场景上（可从`static_sets`的DEV24取飞机数最多的一帧），N=200、间隔开，要求实际用满或接近200束、峰值远低于33束时的104架；报告N=60/100/150/200四档的N_used、峰值、单景耗时；再跑一个动态序列的B2+，报告repair被接受的比例。只做这些冒烟，不跑完整实验。提交到s1fix分支。

## 12. 修订二：下界与SCA（2026-10-17 05:10）

- **下界实测太慢**：1752架场景上g=0.1°的LP一次求解超过20分钟。改为：网格g=0.2°（候选半径1.01×(0.5+0.2/√2)），x_c上限取N预算（不再依赖间隔规则，下界对"间隔开/关"都成立）；每次LP求解限时300秒（`linprog`的`time_limit`），超时则保留已证明的下界；只算N=100和N=200两档；场景为DEV24和CONF48。g=0.1°的敏感性复算只做n<600的3个场景。MILP加强保持只对n<600。
- 报告口径（Claude写报告用）：这个下界忽略间隔规则，用来衡量"间隔关"骨架的算法差距；间隔规则本身的代价由同场景"间隔开减间隔关"给出。
- **SCA**：`evaluate_layout`里`maxiter`由30改为200，`time_limit_seconds`保持30。

## 13. 修订三：随密度自适应的初始网与局部修复失效（2026-10-17 06:00）

**现象**：半径上限法下，1752架场景N预算150/200时峰值都停在28架（平均下界9架），200束预算只用到194束；动态B2+里repair触发15次、改善0次。逐束二分在密集区新中心总会压缩邻束半径上限，拆分被拒，峰值卡住。

**新初始构造 `adaptive_net(member_rays, k)`**：
1. 对每架飞机a，r_k(a)＝到其第k近邻飞机的夹角，裁剪到[s_min, 0.5°]（固定宽度变体裁剪到0.25°；间隔关变体同样用此半径，只是不再施加s_min）。
2. 按r_k从大到小（稀疏处先）、平手按飞机序号处理：若a不在任何已有中心c的r_k(c)以内，则在a处新开中心，记其半径r_k(a)。
3. 这样得到的中心两两相距大于先开中心的半径，因此每个中心的半径上限min(0.5°, 最近中心距离) ≥ 自身r_k，全部飞机一定可被覆盖，且自动满足s_min。
4. 中心数N(k)随k增大而减少。对k二分（k∈[1, n]），取满足N(k) ≤ N预算的最小k；再用4.3第2步把剩余名额拆给最忙束，然后进入交替优化。
5. 这一构造替换`initial_cover`在`solve`里的作用；`initial_cover`（0.5°网）只用于报告N0（0.5°网的最少束数，判断某N预算是否可能覆盖全部飞机）。INIT基线也改用`adaptive_net`的初始中心加"半径上限内最近中心"指派，便于看出流量指派与交替优化的贡献。

**局部修复失效**：先给`repair`加计数，统计`optimize`返回None的原因（指派不可行／冻结束检查失败／预测合格性限制等）和"可行但没变好"的次数，在一个序列上找出主因并修正。目标：B2+在峰值超阈值时，多数repair能降低峰值或最忙10%均值。修正后去掉计数代码。

**自测要求**：DEV24飞机最多的一帧，四档N预算报告N_used、峰值、单景耗时，N=200应用满或接近200束，峰值应明显低于28架；动态一段序列报告repair触发次数与改善次数；全部测试通过。提交到s1fix分支。

## 14. 修订四：修补算子在预算用满时失败（2026-10-17 07:50）

**现象**：远端两段训练序列（白天约1715架）上，B1只修补臂60帧里57帧被迫退回full，于是B1、B2、B2+、B3和各固定动作结果几乎相同。原因：full用满了100束，下一分钟只要有一架飞机落在所有半径上限之外，`patch`就无束可开而失败。

**新patch（中心不动）**：
1. 删除离场飞机和空束；用当前中心重算半径上限。
2. 原归属仍在半径上限内的飞机保持不动；其余飞机（新进或越界）分给半径上限内负载最小的束。
3. 仍覆盖不到的飞机逐架处理：预算未满时在该机处开新中心（它不在任何束的半径上限内，因而距所有中心都≥s_min），重算半径上限，因此掉出邻束上限的原成员回到第2步重新分配；预算已满时，先解散一个"成员都能改分给其他束"的最闲束（解散后邻束半径上限变大），再开新中心。
4. 找不到可解散的束才返回失败。
5. 不预留名额，已有full缓存不重算。

**自测**：在1月8日一段白天训练序列（如`sequence_sets(data,'train')[10]`）上本地算出31帧full布局（N=100），报告B1、B2+(θ=2)、B0三臂的平均峰值−LB、平均换束比例、强制full次数和单帧耗时。要求B1的强制full接近0、换束比例远低于B0。全部测试通过后提交到s1fix分支，最后用中文简短回复这些数字。不改skeleton.py和geom.py。
