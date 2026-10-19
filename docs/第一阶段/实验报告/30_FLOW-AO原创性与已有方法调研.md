# FLOW-AO 原创性与已有方法调研
来源：codex

2026-10-19。问题：FLOW-AO 是否属于全新自研算法，已有研究是否用过相近方法？

**结论：FLOW-AO 是本项目自行设计、组合和实现的方法，但主要数学机制已有明确先例。当前适合写“面向卫星覆盖约束的网络流与几何交替优化方法”，不宜写“从零提出的全新算法”或“首次使用网络流优化卫星波束”。**

名字里的 FLOW 指网络流，用来安排飞机归属；AO 指交替优化，即轮流调整飞机归属和波束位置。查到的先例既包括一般聚类中的相同求解框架，也包括卫星领域的容量约束覆盖、负载均衡和波束拆并。此次未找到与项目全部步骤完全一致的公开方法；名称检索没有命中，也不能据此认定组合首次出现。

## 1. 已有研究究竟重合到哪一步

下列五篇论文均核读了原文方法部分；专利核读了涉及网络流的技术说明。

| 原始研究 | 已有做法 | 与 FLOW-AO 的主要差别 |
| --- | --- | --- |
| Bradley、Bennett、Demiriz，2000，[Constrained K-Means Clustering](https://www.microsoft.com/en-us/research/publication/constrained-k-means-clustering/) | 固定中心，用最小费用流求约束归属，再更新中心，反复交替。原文 Algorithm 2.2、§3、Figure 1。 | 原文控制簇人数下限、优化距离平方和，并以均值更新中心；项目优先压最大束负载，加入覆盖允许边与几何包络。 |
| Harvey 等，2003／2006，[Semi-Matchings for Bipartite Graphs and Load Balancing](https://www.math.uwaterloo.ca/~harvey/Publications/SemiMatching/WADS-SemiMatching.pdf) | 每个任务分给一台允许相连的机器，通过网络流精确均衡负载；最优解同时最小化最大负载及平方和。 | 将“任务、机器、允许边”换成“飞机、波束、可覆盖关系”，就是项目固定覆盖图的分配核心；原文不优化几何布局。 |
| Lin、He、Xiao，2019，[Balanced Clustering: A Uniform Model and Fast Algorithm](https://www.ijcai.org/Proceedings/2019/414) | 网络流指派与中心更新交替；将凸人数费用拆成单位容量名额的边际费用。§4.1 式(14)给出费用差分。 | 原文将距离与均衡加权；项目优先控制最大负载，再做均衡，并使用受覆盖约束的几何调整。 |
| Bui 等，2022，[Joint Beam Placement and Load Balancing Optimization for Non-Geostationary Satellite Systems](https://arxiv.org/abs/2207.14633) | 第一阶段以图方法减少激活波束，第二阶段用 K-means 调整中心和用户归属，改善负载分配。见 Algorithm 1、§III-C。 | 原文主要追求少用波束，项目在给定预算内压峰值；原文第二阶段没有采用项目的精确流指派。 |
| Torkzaban 等，2023，[Capacitated Beam Placement for Multi-beam Non-Geostationary Satellite Systems](https://arxiv.org/abs/2301.07774) | 圆盘全覆盖、每用户唯一归属、每束容量上限；交替调整归属和中心，从繁忙或覆盖较差的波束产生新束，并合并相近中心；另有负载均衡扩展。§III、§IV-B/C。 | 原文采用确定性退火，即先允许概率归属、逐步收紧；项目用整数网络流处理固定几何下的指派，并采用不同的中心调整与接受规则。 |
| Telesat，2020 公开，[WO2020051685A1](https://patents.google.com/patent/WO2020051685A1/en) | 在卫星资源优化中，用网络流处理用户到卫星的链路选择；可用混合整数规划选择波束位置。见“Formulation as a Network Flow problem”段。 | 这是多星链路与资源分配，不是项目的单星波束峰值模型；但足以说明“卫星系统中使用网络流做用户关联”已有先例。 |

最直接的判断是：**分配核心最接近 2003 年的二分图负载均衡；“流分配＋挪中心”最接近约束聚类；卫星外层的容量约束和拆并最接近 2023 年的波束布局论文。**

## 2. 两个容易误当成原创的地方

**“名额费用 1、3、5……”是已有的凸费用流技巧。** 2019 年论文式(14)使用 `f(k)−f(k−1)`，并以人数平方作为一种费用。代入 `f(k)=k²` 就得到 `2k−1`，即 1、3、5……；装入 L 架飞机时，总费用为 L²。奇数序列是从原文公式直接推出的，不是该论文逐字列出的序列。[原文](https://www.ijcai.org/Proceedings/2019/0414.pdf)

**“固定中心后能精确压峰值”也是经典可解子问题。** 在固定覆盖图、所有飞机等权、只比较负载时，2003 年的最优半匹配已能同时最小化平方和与最大负载，见原文 §2、§3.3 的 Theorem 3—4。因此项目“二分峰值，再最小化平方和”的安排可以是清楚、实用的实现选择，单独拿出来不足以构成新的理论贡献。[作者原文](https://www.math.uwaterloo.ca/~harvey/Publications/SemiMatching/WADS-SemiMatching.pdf)

这里的“半匹配”只是说：每架飞机必须选一个波束，同一波束可以接多架飞机。若另加换束罚分、距离主目标或改变覆盖图，须按新的问题重新判断。

## 3. 本项目可以怎样定位贡献

| 可以保留的定位 | 还需要回答的问题 |
| --- | --- |
| 自行实现并命名的 FLOW-AO，属于网络流与几何局部搜索的组合方法 | 明确引用约束聚类与半匹配先例，说明新增步骤解决了哪些卫星约束。 |
| 具体的自适应初始布点、间隔规则处理、闲束迁移和忙束拆分组合 | 相比约束聚类与确定性退火，哪些步骤带来质量或耗时收益？需要同条件对照和去掉某一步的比较。 |
| 动态航空场景中的增量归属、局部修复和换束控制 | 在同样覆盖质量下，是否减少计算时间、换束和布局变化？这是应用方法层面的潜在贡献。 |
| 第一阶段与频率、功率阶段联动，考察布局对最终服务的影响 | 用最终需求满足率、较差飞机服务和耗时支撑贡献，不能只依靠第一阶段峰值更低。 |

新版 FLOW-AO v2 增加每束人数上限、热点同位置叠放和距离收窄。它的特点应按具体组合说明；2023 年论文已经包含容量上限及从已有中心产生新束，不能把“有容量限制、会加波束”单独称为首次。

项目已有[文献对照报告11](../../Claude维护/文献对照/11_文献对照实验报告.md)，覆盖蜂窝网格、He 2024 聚类和 Voronoi 均衡。**若要进一步证明算法贡献，应优先补“同一几何约束下的流式约束聚类”和“2023 年容量约束确定性退火”两个更接近的对照。** 这是文献比较建议，本次没有新增实验。

建议论文使用下面这段表述：

> 本文面向航空用户的多波束覆盖问题，设计 FLOW-AO 网络流与几何交替优化方法。在给定波束中心及覆盖关系时，利用网络流求解用户归属；通过自适应布点、中心调整与波束重构搜索几何布局，并在实际航迹与后续频率、功率分配中评价其效果。

若要使用“提出一种新方法”，应紧接着说明相对上述文献新增的具体机制。算法名称可以继续使用。

## 4. 源码核查与最优性说法

本次只读核查了项目原版与新版实现：

- [原版 skeleton.py](C:/Users/Admin/.codex/worktrees/s1fix/paper/src/satellite_rl/coverage/s1fix/skeleton.py:118)：固定中心最大流、峰值二分；第159行起为名额费用均衡；第301、350行起为重定中心和删拆波束。调用 SciPy 最大流和 HiGHS 线性规划求解器。
- [原版 geom.py](C:/Users/Admin/.codex/worktrees/s1fix/paper/src/satellite_rl/coverage/s1fix/geom.py:50)：先投影到平面，以 Welzl 方法求最小包围圆，再映回球面并按最远成员定宽度。
- [新版 skeleton.py](D:/paper-s2fix/src/satellite_rl/coverage/s1fix/skeleton.py:154)：加入每束人数上限与距离费用；第569行起处理热点叠放与束数预留。

**固定中心的归属可以精确，整个中心布局搜索仍是启发式。** 若某个可行结果的峰值恰好等于 `ceil(飞机数/波束预算)`，可以证明该景的峰值最优。现有[报告11](../../Claude维护/文献对照/11_文献对照实验报告.md)也有均值略高于下界的情况，因此早期“静态已经全部达到数学最优”的概括应收窄。

另两处写论文时应按实现表述：投影后的最小包围圆不能直接称为精确球面最小包络锥；新版“收窄”的流目标实际是飞机到所属中心的张角总和，不是直接最小化总束宽。这些是文字和数学保证的边界，本次未修改算法。

## 5. 检索范围与证据强度

这是截至 2026-10-19 的定向原创性核查，不是穷尽式系统综述。检索覆盖一般学术网页、Microsoft Research、IJCAI 官方论文、arXiv 作者全文、作者个人论文库及公开专利；检索后按原文核对方法。没有直接访问 IEEE Xplore 全库、Scopus 或 Web of Science 的完整导出。

主要查询式如下，均在 2026-10-19 使用：

| 目的 | 查询式 |
| --- | --- |
| 查同名公开方法 | `"FLOW-AO" algorithm`；`"FLOW-AO" satellite` |
| 查框架先例 | `"constrained k-means" "minimum cost flow" Bradley Bennett Demiriz 2000`；`"Balanced Clustering: A Uniform Model and Fast Algorithm"` |
| 查分配核心 | `"load balancing" "semi-matching" Harvey Ladner Lovasz Tamir 2003` |
| 查卫星交叉应用 | `"beam placement" "minimum cost flow"`；`"beam layout" "network flow" satellite`；`"satellite" "balanced clustering" "flow"` |
| 核同领域论文 | `"Capacitated Beam Placement" "10.1109"`；`"He" "Load-Balanced and Full-Coverage Beam Layout"` |

He 等 2024 的题录由[会议官方日程](https://iccspa.org/2024/program/index.html)核对，本次未取得可核的完整论文，因此没有把其详细步骤作为本次核心原创性判据。2023 年容量约束论文在[NSF 收录的正式版本](https://par.nsf.gov/servlets/purl/10568847)中给出 WCNC DOI。

此次证据足以确认上述基础机制已有先例；整套 FLOW-AO 是否具有可发表的新颖性，还取决于具体差异、最接近方法对照和论证质量。

## 6. 数据与图

本次文献检索、源码阅读、对照表和文字由 Codex 生成，没有新生成训练数据、实验结果或图。项目已有结果只用于核对实现和表述，其原来源保留在相应报告中。

## 参考文献与工具

1. Bradley P S, Bennett K P, Demiriz A. **Constrained K-Means Clustering**. Microsoft Research, MSR-TR-2000-65, 2000. [官方全文](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr-2000-65.pdf)。
2. Harvey N J A, Ladner R E, Lovász L, Tamir T. **Semi-Matchings for Bipartite Graphs and Load Balancing**. WADS, 2003. [作者全文](https://www.math.uwaterloo.ca/~harvey/Publications/SemiMatching/WADS-SemiMatching.pdf)。扩展期刊版题为 **Semi-matchings for bipartite graphs and load balancing**, Journal of Algorithms, 59(1):53–78, 2006，DOI：[10.1016/j.jalgor.2005.01.003](https://doi.org/10.1016/j.jalgor.2005.01.003)。
3. Lin W, He Z, Xiao M. **Balanced Clustering: A Uniform Model and Fast Algorithm**. IJCAI, 2019:2987–2993. DOI：[10.24963/ijcai.2019/414](https://doi.org/10.24963/ijcai.2019/414)。
4. Bui V P, Chien T V, Lagunas E, Grotz J, Chatzinotas S, Ottersten B. **Joint Beam Placement and Load Balancing Optimization for Non-Geostationary Satellite Systems**. IEEE MeditCom, 2022. [作者公开稿](https://arxiv.org/abs/2207.14633)。
5. Torkzaban N, Zoulkarni A, Gholami A, Baras J S. **Capacitated Beam Placement for Multi-beam Non-Geostationary Satellite Systems**. IEEE WCNC, 2023. DOI：[10.1109/WCNC55385.2023.10118814](https://doi.org/10.1109/WCNC55385.2023.10118814)。[作者公开稿](https://arxiv.org/abs/2301.07774)。
6. Choinière E, Minhas R; Telesat Canada. **Resource deployment optimizer for non-geostationary and/or geostationary communications satellites**. WO2020051685A1，2020-03-19 公开。[技术说明原文](https://patents.google.com/patent/WO2020051685A1/en)。此处只比较技术机制。

调研流程使用 literature-review 技能。工具参考：Kassis T, Agarwal V, He Y, Patel D, Brueckner A M. **Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents**. 2026. [arXiv:2609.00065](https://arxiv.org/abs/2609.00065)。该项仅说明调研工具来源，不属于 FLOW-AO 的算法先例。
