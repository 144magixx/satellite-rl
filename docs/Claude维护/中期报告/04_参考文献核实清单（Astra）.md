# HTS智能波束覆盖及频率—功率资源分配算法研究：已核实文献清单
来源：codex（GPT-6 Astra），当前接口无法设置 xhigh 思考档

核实日期：2026年10月29日。用途：北京邮电大学闵昱为硕士学位论文中期报告的“国内外研究现状”和“参考文献”。

## 统计与核实口径

- 去重后总条数：**53 条**，其中期刊／会议论文 **52 篇**，ITU-R 建议书 **1 项**。
- 2021—2026 年正式期刊／会议论文：**32 篇**；实际收录的近期论文发表于 2021—2024 年，未把预印本单独重复计数。
- 开题原有 20 条全部找到真实记录并逐条核实；末节重复列示核查结果，不计入总数。

| 主题 | 去重条数 | 2021—2026年条数 | 覆盖说明 |
| --- | ---: | ---: | --- |
| A | 10 | 10 | 10篇均直接涉及卫星布局／覆盖／波束放置 |
| B | 4 | 4 | 4篇均涉及功率与频率／带宽联合配置 |
| C | 11 | 9 | 8篇近期卫星相关研究，另1篇近期地面无线借鉴；2篇较早卫星工作 |
| D | 4 | 4 | 1篇星地融合网络Transformer论文，3篇注意力／神经组合决策会议论文 |
| E | 5 | 5 | 4篇近期卫星优化或混合方法，另1篇地面通感优化借鉴 |
| F | 19 | 0 | 基础算法、标准和指定对照的原始出处 |

总数按主归类去重。A、B、C、D、E 的近期覆盖分别为 10、4、8（卫星相关）、4、4（卫星相关）篇，均达到要求。跨主题关联通过文字说明，不重复扩充数量。

核实方式：逐条解析并打开 DOI，使用 DOI 官方内容协商接口取得出版机构提交的作者、题名、刊物、卷期页码和年份记录；有公开稿的再对照 arXiv，基础学习论文对照 PMLR／NeurIPS 正式论文集。IEEE 等网站的普通网页部分受访问限制，此时采用 DOI 返回的正式元数据；要点来自已取得的原摘要或公开正文，摘要索引入口另列，不能将其误读为所有论文均已阅读全文。

年份以正式卷期／会议出版年为准；文章编号按编号著录，不把它当起止页码。没有查到正式出版 DOI 的 PMLR／早期 NeurIPS 论文不虚配 DOI；arXiv 标识的 DOI 不冒充会议版 DOI。外文作者和题名保留原文，说明文字均为中文。

## A. 多波束布局、覆盖规划与负载均衡

- 【A｜2021】**A01** Honnaiah P J, Maturo N, Chatzinotas S, et al. Demand-Based Adaptive Multi-Beam Pattern and Footprint Planning for High Throughput GEO Satellite Systems[J]. IEEE Open Journal of the Communications Society, 2021, 2: 1526-1540. DOI: 10.1109/ojcoms.2021.3093106.
  - 核实：[DOI及正式出版信息](https://doi.org/10.1109/ojcoms.2021.3093106)；[摘要索引](https://api.openalex.org/works/https://doi.org/10.1109/ojcoms.2021.3093106)
  - 要点：针对航空、海事等空间分布与需求随时间变化的用户，利用用户空间聚类设计位置和尺寸均可调整的波束图样与足迹；仿真显示自适应布局优于固定图样，适合引出可变束宽和需求驱动覆盖。

- 【A｜2021】**A02** Angeletti P, Lizarraga Cubillos J. Traffic Balancing Multibeam Antennas for Communication Satellites[J]. IEEE Transactions on Antennas and Propagation, 2021, 69(12): 8291-8303. DOI: 10.1109/tap.2021.3096209.
  - 核实：[DOI及正式出版信息](https://doi.org/10.1109/tap.2021.3096209)；[摘要索引](https://api.openalex.org/works/https://doi.org/10.1109/tap.2021.3096209)
  - 要点：针对规则波束导致“部分束有剩余容量、部分束仍欠供”的问题，设计面向业务均衡的非规则多波束图样，使空间容量分布更贴近需求；适合支撑第一阶段从均匀铺设转向负载驱动布局的动机。

- 【A｜2022】**A03** Bui V P, Van Chien T, Lagunas E, et al. Joint Beam Placement and Load Balancing Optimization for Non-Geostationary Satellite Systems[C]//2022 IEEE International Mediterranean Conference on Communications and Networking (MeditCom). IEEE, 2022: 316-321. DOI: 10.1109/meditcom55741.2022.9928761.
  - 核实：[DOI及正式出版信息](https://doi.org/10.1109/meditcom55741.2022.9928761)；[摘要索引](https://api.openalex.org/works/https://doi.org/10.1109/meditcom55741.2022.9928761)；[公开稿／补充核实](https://arxiv.org/abs/2207.14633)
  - 要点：提出两阶段波束放置与负载均衡方法，先减少激活波束数，再在已有覆盖内重新分配用户以均衡负载；仿真改善载噪比和每束复用用户数，可用于讨论几何布局与用户归属的交替处理。

- 【A｜2023】**A04** Torkzaban N, Zoulkarni A, Gholami A, et al. Capacitated Beam Placement for Multi-beam Non-Geostationary Satellite Systems[C]//2023 IEEE Wireless Communications and Networking Conference (WCNC). IEEE, 2023: 1-6. DOI: 10.1109/wcnc55385.2023.10118814.
  - 核实：[DOI及正式出版信息](https://doi.org/10.1109/wcnc55385.2023.10118814)；[摘要索引](https://api.openalex.org/works/https://doi.org/10.1109/wcnc55385.2023.10118814)；[公开稿／补充核实](https://arxiv.org/abs/2301.07774)
  - 要点：将波束放置建模为具有容量限制的欧氏圆盘覆盖，利用确定性退火减少覆盖所需波束数，并扩展到束间负载均衡；原文同时考虑覆盖、需求和容量上限，与第一阶段容量约束指派关系紧密。

- 【A｜2024】**A05** He Y, Qi K, Feng X, et al. Load-Balanced and Full-Coverage Beam Layout for Multibeam Satellite Systems[C]//2024 6th International Conference on Communications, Signal Processing, and their Applications (ICCSPA). IEEE, 2024: 1-5. DOI: 10.1109/iccspa61559.2024.10794213.
  - 核实：[DOI及正式出版信息](https://doi.org/10.1109/iccspa61559.2024.10794213)；[摘要索引](https://api.openalex.org/works/https://doi.org/10.1109/iccspa61559.2024.10794213)；[公开稿／补充核实](https://api.semanticscholar.org/graph/v1/paper/DOI:10.1109/iccspa61559.2024.10794213?fields=title,abstract)
  - 要点：先按用户距离和簇流量采用加权 K-means++ 聚类，再依据中心距离及负载差迭代调整中心，最后用最小覆盖圆生成波束；仿真同时验证负载均衡与全部用户覆盖，是本课题第一阶段最直接的布局对照。

- 【A｜2022】**A06** Chaker H, Chougrani H, Martins W A, et al. Matching Traffic Demand in GEO Multibeam Satellites: The Joint Use of Dynamic Beamforming and Precoding Under Practical Constraints[J]. IEEE Transactions on Broadcasting, 2022, 68(4): 819-833. DOI: 10.1109/tbc.2022.3196173.
  - 核实：[DOI及正式出版信息](https://doi.org/10.1109/tbc.2022.3196173)；[摘要索引](https://api.openalex.org/works/https://doi.org/10.1109/tbc.2022.3196173)
  - 要点：在真实非均匀流量、载荷功率和天线约束下优化动态波束图样，以减小需求与容量失配，并结合星上动态波束成形和地面预编码处理束间干扰；其优化对象包含物理层容量，不能简化成纯几何的模式库切换。

- 【A｜2024】**A07** Yuan S, Sun Y, Peng M, et al. Joint Beam Direction Control and Radio Resource Allocation in Dynamic Multi-Beam LEO Satellite Networks[J]. IEEE Transactions on Vehicular Technology, 2024, 73(6): 8222-8237. DOI: 10.1109/tvt.2024.3353339.
  - 核实：[DOI及正式出版信息](https://doi.org/10.1109/tvt.2024.3353339)；[摘要索引](https://api.openalex.org/works/https://doi.org/10.1109/tvt.2024.3353339)；[公开稿／补充核实](https://arxiv.org/abs/2401.09711)
  - 要点：把动态 LEO 网络的波束指向、时隙、子信道和功率联合优化拆为三个子问题，结合有外部性的匹配与逐次凸近似迭代求解；仿真改善服务用户数和总速率，同时考虑用户公平性。

- 【A｜2021】**A08** Pachler de la Osa N, Guerster M, del Portillo Barrios I, et al. Static beam placement and frequency plan algorithms for LEO constellations[J]. International Journal of Satellite Communications and Networking, 2021, 39(1): 65-77. DOI: 10.1002/sat.1345.
  - 核实：[DOI及正式出版信息](https://doi.org/10.1002/sat.1345)；[摘要索引](https://api.openalex.org/works/https://doi.org/10.1002/sat.1345)
  - 要点：将静态波束放置转为边团覆盖问题，将频率规划转为约束满足问题并采用首次适配启发式；Starlink 类场景中，相较所设基准减少所需波束并增加服务用户，研究范围是静态布局和静态频率计划。

- 【A｜2021】**A09** Ortiz-Gomez F G, Salas-Natera M A, Martínez R, et al. Optimization in VHTS Satellite System Design with Irregular Beam Coverage for Non-Uniform Traffic Distribution[J]. Remote Sensing, 2021, 13(13): 2642. DOI: 10.3390/rs13132642.
  - 核实：[DOI及正式出版信息](https://doi.org/10.3390/rs13132642)；[摘要索引](https://api.openalex.org/works/https://doi.org/10.3390/rs13132642)
  - 要点：针对非均匀业务设计不规则覆盖和波束图样，同时考虑在轨每 Gbps 成本、覆盖误差及逐束容量误差；展示固定载荷也能通过非规则布局改善供需失配，和完全可重构载荷形成有用对照。

- 【A｜2023】**A10** Gaudry A, Li R, Mak‐Hau V. FlexBeamOpt: Hybrid solution methodologies for high‐throughput GEO satellite beam laydown and resource allocation[J]. International Journal of Satellite Communications and Networking, 2023, 41(6): 515-538. DOI: 10.1002/sat.1481.
  - 核实：[DOI及正式出版信息](https://doi.org/10.1002/sat.1481)；[摘要索引](https://api.openalex.org/works/https://doi.org/10.1002/sat.1481)
  - 要点：把定制启发式与整数线性规划、约束规划组合成三种 GEO HTS 布局和资源分配方案，比较不同用户数、空间分布和需求分布；结果表明覆盖质量、波束数和计算时间间存在明显权衡。

## B. 频率／带宽与功率联合分配：优化和元启发式

- 【B｜2021】**B01** Abdu T S, Kisseleff S, Lagunas E, et al. Flexible Resource Optimization for GEO Multibeam Satellite Communication System[J]. IEEE Transactions on Wireless Communications, 2021, 20(12): 7888-7902. DOI: 10.1109/twc.2021.3088609.
  - 核实：[DOI及正式出版信息](https://doi.org/10.1109/twc.2021.3088609)；[摘要索引](https://api.openalex.org/works/https://doi.org/10.1109/twc.2021.3088609)
  - 要点：针对非均匀 GEO 业务，在满足波束需求的同时联合降低发射功率和占用带宽，并考虑频谱复用引入的干扰；为非凸资源分配构造可处理的求解方案，说明仅追求总吞吐量不足以反映按需配置。

- 【B｜2021】**B02** Abdu T S, Kisseleff S, Lagunas E, et al. Power and Bandwidth Minimization for Demand-Aware GEO Satellite Systems[C]//2021 IEEE Global Communications Conference (GLOBECOM). IEEE, 2021: 1-6. DOI: 10.1109/globecom46510.2021.9685889.
  - 核实：[DOI及正式出版信息](https://doi.org/10.1109/globecom46510.2021.9685889)；[摘要索引](https://api.openalex.org/works/https://doi.org/10.1109/globecom46510.2021.9685889)
  - 要点：把满足用户需求下的功率、带宽最小化写成混合整数非线性问题，通过二次变换进行迭代求解；仿真在准确按需供给容量的同时提高带宽利用效率。

- 【B｜2022】**B03** Gao W, Wang L, Qu L. Research on Joint Resource Allocation for Multibeam Satellite Based on Metaheuristic Algorithms[J]. Entropy, 2022, 24(11): 1536. DOI: 10.3390/e24111536.
  - 核实：[DOI及正式出版信息](https://doi.org/10.3390/e24111536)；[摘要索引](https://api.openalex.org/works/https://doi.org/10.3390/e24111536)
  - 要点：提出非支配波束编码方法处理带宽重叠约束，并构造可用于多种元启发式的统一分配框架；实验显示编码和约束处理本身即可改善解质量、减少计算量，因此不能仅把该文概括为一种粒子群算法。

- 【B｜2024】**B04** Hika S F, Zhu L, Ji T, et al. Joint Allocation of Downlink Power, Bandwidth and Timeslots Based on Maximizing Energy Efficiency for GEO Satellite Communication Systems[C]//2024 IEEE 99th Vehicular Technology Conference (VTC2024-Spring). IEEE, 2024: 1-6. DOI: 10.1109/vtc2024-spring62846.2024.10683476.
  - 核实：[DOI及正式出版信息](https://doi.org/10.1109/vtc2024-spring62846.2024.10683476)；[摘要索引](https://api.openalex.org/works/https://doi.org/10.1109/vtc2024-spring62846.2024.10683476)
  - 要点：先用线性规划分配时隙，再结合 Dinkelbach 方法和逐次凸近似优化功率、带宽，以最大化 GEO 下行能效；其目标是能效，和本课题固定频率后优先保持供需匹配程度的目标不同。

## C. 深度强化学习与多智能体卫星资源分配

- 【C｜2022】**C01** Lin Z, Ni Z, Kuang L, et al. Dynamic Beam Pattern and Bandwidth Allocation Based on Multi-Agent Deep Reinforcement Learning for Beam Hopping Satellite Systems[J]. IEEE Transactions on Vehicular Technology, 2022, 71(4): 3917-3930. DOI: 10.1109/tvt.2022.3145848.
  - 核实：[DOI及正式出版信息](https://doi.org/10.1109/tvt.2022.3145848)；[摘要索引](https://api.openalex.org/works/https://doi.org/10.1109/tvt.2022.3145848)
  - 要点：用协作多智能体 DRL 联合决定跳波束图样和带宽，分别让智能体负责单束照射或带宽动作，以减小动作空间；共同奖励兼顾吞吐量与小区间时延公平，仿真验证在线实时分配及业务变化下的泛化。

- 【C｜2023】**C02** Li Y, Yang D, Yang H, et al. Joint Frequency Assignment and Power Allocation Based on Multi-Agent Deep Reinforcement Learning for Multi-Beam Satellite Systems[C]//2023 IEEE 97th Vehicular Technology Conference (VTC2023-Spring). IEEE, 2023: 1-6. DOI: 10.1109/vtc2023-spring57618.2023.10200414.
  - 核实：[DOI及正式出版信息](https://doi.org/10.1109/vtc2023-spring57618.2023.10200414)；[摘要索引](https://api.openalex.org/works/https://doi.org/10.1109/vtc2023-spring57618.2023.10200414)
  - 要点：把每个波束建模为智能体，采用集中训练、分布执行联合分配频率与功率，并以多离散动作转单离散动作的机制处理分层动作；目标是容量与业务需求匹配，和本课题第二阶段直接相关。

- 【C｜2024】**C03** Meng H, Xin N, Qin H, et al. A Recursive DRL-Based Resource Allocation Method for Multibeam Satellite Communication Systems[J]. Chinese Journal of Electronics, 2024, 33(5): 1286-1295. DOI: 10.23919/cje.2022.00.135.
  - 核实：[DOI及正式出版信息](https://doi.org/10.23919/cje.2022.00.135)；[摘要索引](https://api.openalex.org/works/https://doi.org/10.23919/cje.2022.00.135)
  - 要点：采用 LSTM 与 PPO 递归生成逐波束频率子动作，降低联合动作空间的探索代价，并利用已经生成的动作指导后续分配；仿真改善频谱效率和系统满意度，适合与本课题逐束离散决策对照。

- 【C｜2021】**C04** Luis J J G, Crawley E, Cameron B. Applicability and Challenges of Deep Reinforcement Learning for Satellite Frequency Plan Design[C]//2021 IEEE Aerospace Conference (50100). IEEE, 2021: 1-11. DOI: 10.1109/aero50100.2021.9438291.
  - 核实：[DOI及正式出版信息](https://doi.org/10.1109/aero50100.2021.9438291)；[摘要索引](https://api.openalex.org/works/https://doi.org/10.1109/aero50100.2021.9438291)；[公开稿／补充核实](https://arxiv.org/abs/2010.08015)
  - 要点：以多波束卫星频率规划为例，比较策略、优化器、状态、动作、奖励和训练环境六类设计选择；结果强调在线决策速度的优势，同时指出不存在适合所有维度和非平稳场景的单一 DRL 配置。

- 【C｜2024】**C05** Feng L, Zhang C, Zhang Q, et al. Double-Timescale Multi-Agent Deep Reinforcement Learning for Flexible Payload in VHTS Systems[J]. Electronics, 2024, 13(14): 2764. DOI: 10.3390/electronics13142764.
  - 核实：[DOI及正式出版信息](https://doi.org/10.3390/electronics13142764)；[摘要索引](https://api.openalex.org/works/https://doi.org/10.3390/electronics13142764)
  - 要点：针对带宽、功率调整速度不同以及业务时变的问题，提出双时间尺度的多智能体深度强化学习分配方案；仿真提高容量—需求匹配程度并降低功耗，采用离线训练、在线决策。

- 【C｜2023】**C06** Chang X, Wang C, Pang M, et al. Resource Allocation Using Deep Reinforcement Learning in GEO Multibeam Satellite System[C]//2023 IEEE International Symposium on Broadband Multimedia Systems and Broadcasting (BMSB). IEEE, 2023: 1-6. DOI: 10.1109/bmsb58369.2023.10211424.
  - 核实：[DOI及正式出版信息](https://doi.org/10.1109/bmsb58369.2023.10211424)；[摘要索引](https://api.openalex.org/works/https://doi.org/10.1109/bmsb58369.2023.10211424)
  - 要点：将 GEO 多波束联合功率—带宽分配建模为马尔可夫决策过程，用异步优势演员—评论家算法 A3C 学习分配策略；在匹配波束需求的同时兼顾吞吐量与用户公平性。

- 【C｜2021】**C07** Han Y, Zhang C, Zhang G. Dynamic Beam Hopping Resource Allocation Algorithm Based on Deep Reinforcement Learning in Multi-Beam Satellite Systems[C]//2021 3rd International Academic Exchange Conference on Science and Technology Innovation (IAECST). IEEE, 2021: 68-73. DOI: 10.1109/iaecst54258.2021.9695603.
  - 核实：[DOI及正式出版信息](https://doi.org/10.1109/iaecst54258.2021.9695603)；[摘要索引](https://api.openalex.org/works/https://doi.org/10.1109/iaecst54258.2021.9695603)
  - 要点：以卫星为智能体，将满足干扰规避规则的跳波束图样作为动作集合，利用 DRL 降低分组传输时延；仿真同时改善时延与吞吐量，适用于时域跳波束背景。

- 【C｜2024】**C08** Huang J, Yang Y, Lee J, et al. Deep Reinforcement Learning-Based Resource Allocation for RSMA in LEO Satellite-Terrestrial Networks[J]. IEEE Transactions on Communications, 2024, 72(3): 1341-1354. DOI: 10.1109/tcomm.2023.3331021.
  - 核实：[DOI及正式出版信息](https://doi.org/10.1109/tcomm.2023.3331021)；[摘要索引](https://api.openalex.org/works/https://doi.org/10.1109/tcomm.2023.3331021)
  - 要点：针对 LEO 星地融合网络的 RSMA 资源分配，分别用 DQN 处理离散资源选择、PPO 处理连续功率控制；可支持“离散—连续变量应区别建模”的论述，其星地共享与公共／私有流模型不同于单星 GEO 频率槽分配。

- 【C｜2021】**C09** Wang S, Lv T, Ni W, et al. Joint Resource Management for MC-NOMA: A Deep Reinforcement Learning Approach[J]. IEEE Transactions on Wireless Communications, 2021, 20(9): 5672-5688. DOI: 10.1109/twc.2021.3069240.
  - 核实：[DOI及正式出版信息](https://doi.org/10.1109/twc.2021.3069240)；[摘要索引](https://api.openalex.org/works/https://doi.org/10.1109/twc.2021.3069240)
  - 要点：面向含硬件灵敏度限制和不完美 SIC 的地面 MC-NOMA，采用 DDPG 将子载波分配与功率分配作为两个序贯子任务，并用多智能体结构降低功率分配复杂度；这是无线资源分配的方法借鉴，不能称为卫星资源分配论文。

- 【C｜2018】**C10** Hu X, Liu S, Chen R, et al. A Deep Reinforcement Learning-Based Framework for Dynamic Resource Allocation in Multibeam Satellite Systems[J]. IEEE Communications Letters, 2018, 22(8): 1612-1615. DOI: 10.1109/lcomm.2018.2844243.
  - 核实：[DOI及正式出版信息](https://doi.org/10.1109/lcomm.2018.2844243)；[摘要索引](https://api.openalex.org/works/https://doi.org/10.1109/lcomm.2018.2844243)
  - 要点：把多波束卫星动态资源分配表示为深度强化学习问题，并用类似图像的张量提取业务时空特征；动态信道分配实例验证了在未知时变环境中学习资源管理策略的可行性。

- 【C｜2020】**C11** Liao X, Hu X, Liu Z, et al. Distributed Intelligence: A Verification for Multi-Agent DRL-Based Multibeam Satellite Resource Allocation[J]. IEEE Communications Letters, 2020, 24(12): 2785-2789. DOI: 10.1109/lcomm.2020.3019437.
  - 核实：[DOI及正式出版信息](https://doi.org/10.1109/lcomm.2020.3019437)；[摘要索引](https://api.openalex.org/works/https://doi.org/10.1109/lcomm.2020.3019437)
  - 要点：提出协作多智能体 DRL 资源管理框架，以带宽分配为例分担集中式智能体的计算负担；仿真验证传输效率和低复杂度方面的收益。

## D. Transformer、注意力及组合决策

- 【D｜2024】**D01** Li J, Ying L, Ni C, et al. Dynamic Resource Allocation in Integrated Terrestrial-Satellite Networks: An Elastic Decision Transformer Approach[C]//2024 16th International Conference on Wireless Communications and Signal Processing (WCSP). IEEE, 2024: 931-936. DOI: 10.1109/wcsp62071.2024.10827284.
  - 核实：[DOI及正式出版信息](https://doi.org/10.1109/wcsp62071.2024.10827284)；[摘要索引](https://api.openalex.org/works/https://doi.org/10.1109/wcsp62071.2024.10827284)
  - 要点：用弹性决策 Transformer 联合优化星地融合网络的用户关联与功率控制，并通过轨迹拼接和历史长度自适应处理动态环境；仿真考察变长输入、泛化与可扩展性，但原文不等同于本课题的 SAC 策略网络。

- 【D｜2021】**D02** Kwon Y D, Choo J, Yoon I, et al. Matrix encoding networks for neural combinatorial optimization[C]//Advances in Neural Information Processing Systems. 2021, 34: 5138-5149.
  - 核实：[正式来源](https://proceedings.neurips.cc/paper/2021/hash/29539ed932d32f1c56324cded92c07c2-Abstract.html)
  - 要点：提出矩阵编码网络 MatNet，处理两类对象之间的关系矩阵，并用于非对称旅行商及柔性流水作业调度；可为“波束—频谱块”关系表示提供组合决策层面的方法借鉴，实验对象不是卫星系统。

- 【D｜2022】**D03** Kim M, Park J, Park J. Sym-NCO: Leveraging symmetricity for neural combinatorial optimization[C]//Advances in Neural Information Processing Systems. 2022, 35: 1936-1949. DOI: 10.52202/068431-0141.
  - 核实：[正式来源](https://proceedings.neurips.cc/paper_files/paper/2022/hash/0cddb777d3441326544e21b67f41bdc8-Abstract-Conference.html)；[公开正文](https://arxiv.org/html/2205.13209v2)
  - 要点：利用问题与解的对称性对神经组合优化训练进行正则化，在注意力式构造模型等框架上验证旅行商、容量车辆路径等任务；说明结构先验能改善学习型组合决策的泛化，不能把几何对称性直接移植到带实际干扰差异的卫星模型。

- 【D｜2023】**D04** Luo F, Lin X, Liu F, et al. Neural combinatorial optimization with heavy decoder: Toward large scale generalization[C]//Advances in Neural Information Processing Systems. 2023, 36: 8845-8864. DOI: 10.52202/075280-0387.
  - 核实：[正式来源](https://proceedings.neurips.cc/paper_files/paper/2023/hash/1c10d0c087c14689628124bbc8fa69f6-Abstract-Conference.html)；[公开正文](https://arxiv.org/html/2310.07985)
  - 要点：通过轻编码器、重注意力解码器，在每一步选择时重新刻画当前可用节点的关系，并结合部分解训练与随机重构；小规模训练可推广到较大旅行商和容量车辆路径实例，为变长候选集合的逐步打分提供依据。

## E. 连续功率、需求匹配及学习—优化混合方法

- 【E｜2023】**E01** Hu X, Wang Y, Liu Z, et al. Dynamic Power Allocation in High Throughput Satellite Communications: A Two-Stage Advanced Heuristic Learning Approach[J]. IEEE Transactions on Vehicular Technology, 2023, 72(3): 3502-3516. DOI: 10.1109/tvt.2022.3218565.
  - 核实：[DOI及正式出版信息](https://doi.org/10.1109/tvt.2022.3218565)；[摘要索引](https://api.openalex.org/works/https://doi.org/10.1109/tvt.2022.3218565)
  - 要点：用深度强化学习辅助启发式功率分配，以改善动态场景下纯学习方法的泛化并加快启发式收敛；摘要报告计算开销相对所比较方法降低约 6%—95%，是“学习给出引导、优化继续改进”的直接卫星依据。

- 【E｜2023】**E02** Cui H, Zhu L, Xiao Z, et al. Energy-Efficient RSMA for Multigroup Multicast and Multibeam Satellite Communications[J]. IEEE Wireless Communications Letters, 2023, 12(5): 838-842. DOI: 10.1109/lwc.2023.3245631.
  - 核实：[DOI及正式出版信息](https://doi.org/10.1109/lwc.2023.3245631)；[摘要索引](https://api.openalex.org/works/https://doi.org/10.1109/lwc.2023.3245631)
  - 要点：联合优化功率和速率，以降低未满足容量和通信能耗，并考虑逐馈源及总辐射功率约束；通过辅助变量和上图形式转化为差分凸结构，再用 SCA 得到次优解，直接支持需求匹配导向的连续优化综述。

- 【E｜2024】**E03** Chen L, Wu L, Lagunas E, et al. Joint Power Allocation and Beam Scheduling in Beam-Hopping Satellites: A Two-Stage Framework With a Probabilistic Perspective[J]. IEEE Transactions on Wireless Communications, 2024, 23(10): 14685-14701. DOI: 10.1109/twc.2024.3417707.
  - 核实：[DOI及正式出版信息](https://doi.org/10.1109/twc.2024.3417707)；[摘要索引](https://api.openalex.org/works/https://doi.org/10.1109/twc.2024.3417707)
  - 要点：从概率角度解耦波束激活状态与功率，先求统计意义上的能耗优化方案，再以二元二次规划和惩罚迭代生成具体照射时序；仿真验证按需匹配与能效收益，是不同于本课题固定频率功率细调的两阶段优化框架。

- 【E｜2021】**E04** Wang A, Lei L, Lagunas E, et al. Joint Beam-Hopping Scheduling and Power Allocation in NOMA-Assisted Satellite Systems[C]//2021 IEEE Wireless Communications and Networking Conference (WCNC). IEEE, 2021: 1-6. DOI: 10.1109/wcnc49053.2021.9417306.
  - 核实：[DOI及正式出版信息](https://doi.org/10.1109/wcnc49053.2021.9417306)；[摘要索引](https://api.openalex.org/works/https://doi.org/10.1109/wcnc49053.2021.9417306)
  - 要点：联合优化 NOMA 功率与跳波束调度，识别指数锥结构并改写成混合整数锥规划，同时提出逐时隙贪心近似降低复杂度；结果给出需求—容量失配与求解成本之间的权衡。

- 【E｜2022】**E05** Wang J, Ma L, Wei Z, et al. Multi-beam-based Downlink Modeling and Power Allocation Scheme for Integrated Sensing and Communication towards 6G[C]//2022 IEEE 95th Vehicular Technology Conference: (VTC2022-Spring). IEEE, 2022: 1-5. DOI: 10.1109/vtc2022-spring54318.2022.9860363.
  - 核实：[DOI及正式出版信息](https://doi.org/10.1109/vtc2022-spring54318.2022.9860363)；[摘要索引](https://api.openalex.org/works/https://doi.org/10.1109/vtc2022-spring54318.2022.9860363)
  - 要点：研究地面基站对车辆的多波束通感一体化下行模型，通过凸优化分配功率以兼顾通信速率和感知精度；原文使用 CVX，并与等功率、注水比较，但它不是多波束卫星功率分配论文。

## F. 方法原始文献、标准及经典对照

指定的近期原始文献已在 A 组收录：Angeletti 与 Lizarraga Cubillos 2021（A02）、Torkzaban 等 2023（A04）、He 等 2024（A05）；本节不再重复计数。

- 【F｜2018】**F01** Haarnoja T, Zhou A, Abbeel P, et al. Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor[C]//Proceedings of the 35th International Conference on Machine Learning. PMLR, 2018, 80: 1861-1870.
  - 核实：[正式来源](https://proceedings.mlr.press/v80/haarnoja18b.html)
  - 要点：提出离策略、最大熵的随机演员—评论家算法，在提高期望回报的同时鼓励策略熵，并在连续控制任务中验证样本效率和跨随机种子的稳定性；本课题若采用离散或多分支 SAC，需要另说明对原始连续动作算法的改动。

- 【F｜2017】**F02** Vaswani A, Shazeer N, Parmar N, et al. Attention is all you need[C]//Advances in Neural Information Processing Systems. 2017, 30: 5998-6008.
  - 核实：[正式来源](https://proceedings.neurips.cc/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html)；[论文集页码元数据](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Metadata.json)
  - 要点：提出以多头自注意力为核心的 Transformer 编码器—解码器，去除循环结构并提高并行训练能力；这是第二阶段全局波束关系编码的架构原始出处，原始实验是机器翻译。

- 【F｜2015】**F03** Vinyals O, Fortunato M, Jaitly N. Pointer networks[C]//Advances in Neural Information Processing Systems. 2015, 28: 2692-2700.
  - 核实：[正式来源](https://proceedings.neurips.cc/paper/2015/hash/29921001f2f04bd3baee84a12e98098f-Abstract.html)；[论文集页码元数据](https://proceedings.neurips.cc/paper_files/paper/2015/file/29921001f2f04bd3baee84a12e98098f-Metadata.json)
  - 要点：把注意力分布直接作为指向输入元素的输出，使输出候选数可随输入长度改变，并在几何与旅行商任务上验证；它支持“对频谱块候选逐项打分再选起点”的结构设计，但不自动保证资源约束可行。

- 【F｜1997】**F04** ITU-R. Satellite antenna radiation pattern for use as a design objective in the fixed-satellite service employing geostationary satellites: Recommendation ITU-R S.672-4[S]. Geneva: International Telecommunication Union, 1997.
  - 核实：[正式来源](https://www.itu.int/rec/R-REC-S.672-4-199709-I/en)
  - 要点：给出静止卫星固定卫星业务天线辐射图的设计目标，是建立波束方向图和离轴增益模型的标准依据；官方记录批准日期为 1997 年 9 月 18 日，网页上传或翻译版本日期不能替代建议书年份。

- 【F｜2018】**F05** Cocco G, de Cola T, Angelone M, et al. Radio Resource Management Optimization of Flexible Satellite Payloads for DVB-S2 Systems[J]. IEEE Transactions on Broadcasting, 2018, 64(2): 266-280. DOI: 10.1109/tbc.2017.2755263.
  - 核实：[DOI及正式出版信息](https://doi.org/10.1109/tbc.2017.2755263)；[摘要索引](https://api.openalex.org/works/https://doi.org/10.1109/tbc.2017.2755263)
  - 要点：在 DVB-S2 多波束前向链路中，用兼顾业务匹配和公平性的目标联合配置柔性带宽、功率，并用改进模拟退火求解非凸问题；提供复杂度分析和实际条件下的仿真，是模拟退火基线的直接出处。

- 【F｜2019】**F06** Paris A, Del Portillo I, Cameron B, et al. A Genetic Algorithm for Joint Power and Bandwidth Allocation in Multibeam Satellite Systems[C]//2019 IEEE Aerospace Conference. IEEE, 2019: 1-15. DOI: 10.1109/aero.2019.8742238.
  - 核实：[DOI及正式出版信息](https://doi.org/10.1109/aero.2019.8742238)；[摘要索引](https://api.openalex.org/works/https://doi.org/10.1109/aero.2019.8742238)
  - 要点：采用遗传算法联合搜索波束功率和带宽，并通过修复函数保证解的可行性、加快收敛；在含束间干扰和大气衰减的模型中，联合分配比只调功率更能降低未满足容量。

- 【F｜2005】**F07** Choi J P, Chan V W S. Optimum power and beam allocation based on traffic demands and channel conditions over satellite downlinks[J]. IEEE Transactions on Wireless Communications, 2005, 4(6): 2983-2993. DOI: 10.1109/twc.2005.858365.
  - 核实：[DOI及正式出版信息](https://doi.org/10.1109/twc.2005.858365)；[摘要索引](https://api.openalex.org/works/https://doi.org/10.1109/twc.2005.858365)
  - 要点：研究功率与时延约束下，按业务需求和信道条件分配多波束下行功率，并在激活波束数小于覆盖小区数时结合波束调度；结果表明较少并行激活波束可获得功率增益和合理比例公平性，不能只用“普通注水”概括整篇方法。

- 【F｜2015】**F08** Aravanis A I, Shankar B M R, Arapoglou P D, et al. Power Allocation in Multibeam Satellite Systems: A Two-Stage Multi-Objective Optimization[J]. IEEE Transactions on Wireless Communications, 2015, 14(6): 3171-3182. DOI: 10.1109/twc.2015.2402682.
  - 核实：[DOI及正式出版信息](https://doi.org/10.1109/twc.2015.2402682)；[摘要索引](https://api.openalex.org/works/https://doi.org/10.1109/twc.2015.2402682)
  - 要点：先获得业务匹配解，再用这些解增强同时考虑业务匹配和总功耗的多目标元启发式搜索，得到帕累托权衡；原文的“两阶段”不是“元启发式粗分后再做凸优化”，引用时应纠正这一常见误述。

- 【F｜1993】**F09** Foschini G J, Miljanic Z. A simple distributed autonomous power control algorithm and its convergence[J]. IEEE Transactions on Vehicular Technology, 1993, 42(4): 641-646. DOI: 10.1109/25.260747.
  - 核实：[DOI及正式出版信息](https://doi.org/10.1109/25.260747)；[摘要索引](https://api.openalex.org/works/https://doi.org/10.1109/25.260747)
  - 要点：在同信道蜂窝干扰系统中，用本地信号、干扰及噪声测量进行分布式功率更新，以达到给定信干比目标；证明在目标可行时的快速收敛，可作为目标容量转换为目标 SINR 后迭代控功率的原始依据。

- 【F｜2011】**F10** Shi Q, Razaviyayn M, Luo Z Q, et al. An Iteratively Weighted MMSE Approach to Distributed Sum-Utility Maximization for a MIMO Interfering Broadcast Channel[J]. IEEE Transactions on Signal Processing, 2011, 59(9): 4331-4340. DOI: 10.1109/tsp.2011.2147784.
  - 核实：[DOI及正式出版信息](https://doi.org/10.1109/tsp.2011.2147784)；[摘要索引](https://api.openalex.org/works/https://doi.org/10.1109/tsp.2011.2147784)
  - 要点：通过交替更新接收器、均方误差权重和发射端变量，将加权和速率优化与加权均方误差最小化联系起来，并证明收敛到驻点；原始 WMMSE 的加权和速率目标不等同于需求满足率或供需匹配程度。

- 【F｜2006】**F11** Harvey N J A, Ladner R E, Lovász L, et al. Semi-matchings for bipartite graphs and load balancing[J]. Journal of Algorithms, 2006, 59(1): 53-78. DOI: 10.1016/j.jalgor.2005.01.003.
  - 核实：[DOI及正式出版信息](https://doi.org/10.1016/j.jalgor.2005.01.003)；[作者公开摘要及稿件入口](https://people.csail.mit.edu/nickh/Publications/PublicationsCombAlg.html)
  - 要点：将二分图左侧顶点唯一分配给右侧顶点，研究使右侧负载均衡的最优半匹配，并提出推广匈牙利法和基于降成本路径的算法；其负载均衡性质包括最小化最大负载，可作为固定几何可达关系下精确用户指派的理论依据。

- 【F｜2014】**F12** Camino J T, Mourgues S, Artigues C, et al. A greedy approach combined with graph coloring for non-uniform beam layouts under antenna constraints in multibeam satellite systems[C]//2014 7th Advanced Satellite Multimedia Systems Conference and the 13th Signal Processing for Space Communications Workshop (ASMS/SPSC). IEEE, 2014: 374-381. DOI: 10.1109/asms-spsc.2014.6934570.
  - 核实：[DOI及正式出版信息](https://doi.org/10.1109/asms-spsc.2014.6934570)；[摘要索引](https://api.openalex.org/works/https://doi.org/10.1109/asms-spsc.2014.6934570)
  - 要点：采用随机多起点贪心构造非均匀波束布局，并用局部搜索、模拟退火式图重着色满足同反射面波束间最小角间距约束；原文研究的是受天线约束的非规则布局，不能把它本身称为固定六边形网格算法。

- 【F｜2016】**F13** Camino J T, Artigues C, Houssin L, et al. Mixed-integer linear programming for multibeam satellite systems design: Application to the beam layout optimization[C]//2016 Annual IEEE Systems Conference (SysCon). IEEE, 2016: 1-6. DOI: 10.1109/syscon.2016.7490613.
  - 核实：[DOI及正式出版信息](https://doi.org/10.1109/syscon.2016.7490613)；[摘要索引](https://api.openalex.org/works/https://doi.org/10.1109/syscon.2016.7490613)
  - 要点：以混合整数线性规划显式处理波束布局及天线子系统的技术可行性约束，强调布局与后续系统设计的耦合；适合作为候选布局集合上的数学规划参照。

- 【F｜2014】**F14** Wang H, Liu A, Pan X. Optimization of Joint Power and Bandwidth Allocation in Multi‐Spot‐Beam Satellite Communication Systems[J]. Mathematical Problems in Engineering, 2014, 2014(1): 683604. DOI: 10.1155/2014/683604.
  - 核实：[DOI及正式出版信息](https://doi.org/10.1155/2014/683604)；[摘要索引](https://api.openalex.org/works/https://doi.org/10.1155/2014/683604)
  - 要点：将连续功率与带宽联合分配建立为兼顾系统总容量和波束公平性的凸优化问题，利用对偶理论设计迭代求解；该原文不是对本课题离散频率槽及任意同频干扰模型直接成立的通用凸松弛。

- 【F｜2010】**F15** Lei J, Vazquez-Castro M A. Joint Power and Carrier Allocation for the Multibeam Satellite Downlink with Individual SINR Constraints[C]//2010 IEEE International Conference on Communications. IEEE, 2010: 1-5. DOI: 10.1109/icc.2010.5502506.
  - 核实：[DOI及正式出版信息](https://doi.org/10.1109/icc.2010.5502506)；[摘要索引](https://api.openalex.org/works/https://doi.org/10.1109/icc.2010.5502506)
  - 要点：在逐束 SINR 约束下联合配置功率和载波，以最大最小 SINR 均衡处理非对称业务及干扰，给出载波分配解析处理并迭代优化功率；原文比均匀带宽、功率方案获得更好的频谱利用和业务匹配。

- 【F｜2011】**F16** Lei J, Vázquez-Castro M Á. Multibeam satellite frequency/time duality study and capacity optimization[J]. Journal of Communications and Networks, 2011, 13(5): 472-480. DOI: 10.1109/jcn.2011.6112304.
  - 核实：[DOI及正式出版信息](https://doi.org/10.1109/jcn.2011.6112304)；[摘要索引](https://api.openalex.org/works/https://doi.org/10.1109/jcn.2011.6112304)；[公开稿／补充核实](https://arxiv.org/abs/1103.3866)
  - 要点：建立非正交频率复用与跳波束在频率域、时间域之间的对偶关系，并在逐束 SINR 和实际技术约束下研究容量优化；比较 Shannon 上界与实际调制编码后，说明理论对偶不意味着实际载荷性能完全等价。

- 【F｜2007】**F17** Chiang M, Tan C W, Palomar D P, et al. Power Control By Geometric Programming[J]. IEEE Transactions on Wireless Communications, 2007, 6(7): 2640-2651. DOI: 10.1109/twc.2007.05960.
  - 核实：[DOI及正式出版信息](https://doi.org/10.1109/twc.2007.05960)；[摘要索引](https://api.openalex.org/works/https://doi.org/10.1109/twc.2007.05960)
  - 要点：说明高信干比条件下若干功率控制问题可转化为几何规划，并在中低信干比条件下采用逐次凸近似求解一系列几何规划；适合作为凸化及局部迭代功率控制的基础出处，但原文不保证一般非凸情形的全局最优。

- 【F｜2020】**F18** Pachler N, Luis J J G, Guerster M, et al. Allocating Power and Bandwidth in Multibeam Satellite Systems using Particle Swarm Optimization[C]//2020 IEEE Aerospace Conference. IEEE, 2020: 1-11. DOI: 10.1109/aero47225.2020.9172694.
  - 核实：[DOI及正式出版信息](https://doi.org/10.1109/aero47225.2020.9172694)；[摘要索引](https://api.openalex.org/works/https://doi.org/10.1109/aero47225.2020.9172694)
  - 要点：在完整射频链路和实际业务、运行时限下，用粒子群联合优化功率、带宽，并与遗传算法混合以缓解局部最优；结果突出有限在线计算预算下的解质量与收敛速度权衡。

- 【F｜2020】**F19** Luis J J G, Pachler N, Guerster M, et al. Artificial Intelligence Algorithms for Power Allocation in High Throughput Satellites: A Comparison[C]//2020 IEEE Aerospace Conference. IEEE, 2020: 1-15. DOI: 10.1109/aero47225.2020.9172682.
  - 核实：[DOI及正式出版信息](https://doi.org/10.1109/aero47225.2020.9172682)；[摘要索引](https://api.openalex.org/works/https://doi.org/10.1109/aero47225.2020.9172682)
  - 要点：在运营商业务、完整链路和硬件／时间约束下，比较遗传、模拟退火、粒子群、DRL 及混合功率分配方法；评价维度明确包括收敛时间、持续运行能力、可扩展性与鲁棒性，可借鉴其公平比较框架。

## 指定对照方法清单的出处核查与必要更正

已读取用户指定的[三阶段文献对照方法清单](<D:/paper/docs/Claude维护/文献对照/08_三阶段文献对照方法清单.md>)，未读取用户禁止的目录和文件。以下核查针对原始文献与方法归属，不替代对项目实现代码的复现审查。

| 原清单中的方法说法 | 核实后的原始出处及结论 |
| --- | --- |
| Wang 2014“拉格朗日对偶或凸松弛的需求匹配” | **F14**：Wang H, Liu A, Pan X；文章编号683604，DOI 10.1155/2014/683604。原文直接建立连续联合功率—带宽的凸优化，权衡总容量和波束公平性，采用对偶迭代。不能据此声称本课题含离散槽起点及同频干扰的原问题可直接凸化。 |
| Lei 与 Vázquez-Castro ICC 2010／JCN 2011 | **F15、F16** 是两篇不同论文：前者为逐束SINR约束下的联合功率—载波分配和最大最小均衡，后者研究频率复用与跳波束的频率／时间对偶及容量优化；不宜合并成一篇“连续松弛再取整”。 |
| Aravanis 2015“两阶段：元启发式粗分，再凸优化细分” | **F08**：前阶段业务匹配解用于增强后阶段多目标元启发式优化，获得匹配—功耗帕累托权衡；“第二阶段为凸优化”与原摘要不符，必须改正。 |
| Choi–Chan 2005“凸需求差函数／注水” | **F07**：原文完整问题包含业务、信道、功率、时延和波束调度；可作为需求与信道自适应功率分配依据。若项目实现只是固定信道注水，应写“注水基线”并交代简化，不能直接宣称完整复现该文。 |
| Camino 2014/2016“固定六边形蜂窝网格出处” | **F12、F13** 的核心分别是受天线约束的非均匀布局启发式和布局MILP；它们讨论规则布局背景，但固定网格本身不是这两篇的所提算法。 |
| He 2024“流量加权K-means++＋最小覆盖圆” | **A05**：原摘要明确给出该步骤链，方法归属成立。 |
| Angeletti 与 Lizarraga Cubillos“Traffic Balancing Multibeam Antennas” | **A02**：完整题名含“for Communication Satellites”，IEEE TAP, 2021, 69(12): 8291-8303。正式出版作者为Piero Angeletti、Juan Lizarraga Cubillos，第二作者按期刊记录缩写为Lizarraga Cubillos J，不从专利记录推导额外首字母。 |
| 容量约束圆盘覆盖、确定性退火 | **A04**：Torkzaban N, Zoulkarni A, Gholami A, et al，WCNC 2023，1-6，DOI 10.1109/WCNC55385.2023.10118814；原摘要明确容量限制与均衡扩展。 |
| Bui 等先少束、再均衡 | **A03**：IEEE MeditCom 2022, 316-321，DOI 10.1109/MeditCom55741.2022.9928761；两阶段结构在原摘要中明确。 |
| Chaker 2022／Yuan 2024“模式库切换或局部搜索” | **A06、A07**：前者核心为动态波束成形与预编码的需求匹配，后者为指向与多维资源分解、匹配及SCA；不能以“模式库切换”概括两篇原始方法。 |
| Cocco 2018模拟退火、Paris 2019遗传、Pachler 2020粒子群 | 分别为 **F05、F06、F18**；原摘要均明确方法。Cocco是改进模拟退火，Paris包含可行性修复，Pachler还讨论PSO与GA的混合。 |
| Gao 2022“粒子群联合分配” | **B03**：核心创新是非支配波束编码及通用约束处理，在四种元启发式上应用；若复现，应明确选择的算法臂和编码方式。 |
| Chiang 2007几何规划 | **F17**：IEEE TWC, 6(7): 2640-2651；高信干比可转几何规划，一般中低信干比采用逐次凸近似，不能把所有情形都称全局最优。 |
| Hu 2023“DRL＋启发式” | **E01**：TVT 2023, 72(3): 3502-3516；该文支持学习加速启发式，但不能据此称为“DRL＋SCA”原始方法。 |
| Garau Luis 等2020比较框架 | **F19**：比较遗传、模拟退火、粒子群、DRL及混合方法，四维评价为收敛时间、持续运行、可扩展性、鲁棒性，原清单的范式归属成立。 |
| FlexBeamOpt | **A10**：Gaudry A, Li R, Mak-Hau V；IJSCN 2023, **41(6): 515-538**，补齐期号；原文是启发式、整数线性规划、约束规划的混合方案。 |

关于指标名称：Li 等2023（C02）已核实联合频率—功率分配及容量—需求匹配目标；仅凭摘要尚不能核定项目中 SGM 的具体公式、归一化和聚合方式与原文完全一致。正式报告若写“SGM出自该文”，应在已获得全文后补充公式级对照。按需求比例、等功率等规则基线无需强行指定一篇“发明论文”。

## 开题原有 20 条核实结果

以下“正确”指作者顺序、题名、刊物／会议、年份、卷期和页码与出版记录一致；仅统一大小写、刊名全称、连字符或补充 DOI 不算实质错误。超过三人的条目按 GB/T 7714 习惯列前三人及“et al”；第17条为消除争议，额外给出全部作者。

### [1] 正确（对应F12）

Camino J T, Mourgues S, Artigues C, et al. A greedy approach combined with graph coloring for non-uniform beam layouts under antenna constraints in multibeam satellite systems[C]//2014 7th Advanced Satellite Multimedia Systems Conference and the 13th Signal Processing for Space Communications Workshop (ASMS/SPSC). IEEE, 2014: 374-381. DOI: 10.1109/asms-spsc.2014.6934570.

- 核实：[DOI出版记录](https://doi.org/10.1109/asms-spsc.2014.6934570)；[摘要及作者记录](https://api.openalex.org/works/https://doi.org/10.1109/asms-spsc.2014.6934570)。
- 要点：采用随机多起点贪心构造非均匀波束布局，并用局部搜索、模拟退火式图重着色满足同反射面波束间最小角间距约束；原文研究的是受天线约束的非规则布局，不能把它本身称为固定六边形网格算法。

### [2] 正确（对应F13）

Camino J T, Artigues C, Houssin L, et al. Mixed-integer linear programming for multibeam satellite systems design: Application to the beam layout optimization[C]//2016 Annual IEEE Systems Conference (SysCon). IEEE, 2016: 1-6. DOI: 10.1109/syscon.2016.7490613.

- 核实：[DOI出版记录](https://doi.org/10.1109/syscon.2016.7490613)；[摘要及作者记录](https://api.openalex.org/works/https://doi.org/10.1109/syscon.2016.7490613)。
- 要点：以混合整数线性规划显式处理波束布局及天线子系统的技术可行性约束，强调布局与后续系统设计的耦合；适合作为候选布局集合上的数学规划参照。

### [3] 正确（对应A05）

He Y, Qi K, Feng X, et al. Load-Balanced and Full-Coverage Beam Layout for Multibeam Satellite Systems[C]//2024 6th International Conference on Communications, Signal Processing, and their Applications (ICCSPA). IEEE, 2024: 1-5. DOI: 10.1109/iccspa61559.2024.10794213.

- 核实：[DOI出版记录](https://doi.org/10.1109/iccspa61559.2024.10794213)；[摘要及作者记录](https://api.openalex.org/works/https://doi.org/10.1109/iccspa61559.2024.10794213)。
- 要点：先按用户距离和簇流量采用加权 K-means++ 聚类，再依据中心距离及负载差迭代调整中心，最后用最小覆盖圆生成波束；仿真同时验证负载均衡与全部用户覆盖，是本课题第一阶段最直接的布局对照。

### [4] 正确（对应A01）

Honnaiah P J, Maturo N, Chatzinotas S, et al. Demand-Based Adaptive Multi-Beam Pattern and Footprint Planning for High Throughput GEO Satellite Systems[J]. IEEE Open Journal of the Communications Society, 2021, 2: 1526-1540. DOI: 10.1109/ojcoms.2021.3093106.

- 核实：[DOI出版记录](https://doi.org/10.1109/ojcoms.2021.3093106)；[摘要及作者记录](https://api.openalex.org/works/https://doi.org/10.1109/ojcoms.2021.3093106)。
- 说明：2021年卷2、1526-1540正确。
- 要点：针对航空、海事等空间分布与需求随时间变化的用户，利用用户空间聚类设计位置和尺寸均可调整的波束图样与足迹；仿真显示自适应布局优于固定图样，适合引出可变束宽和需求驱动覆盖。

### [5] 正确（对应A06）

Chaker H, Chougrani H, Martins W A, et al. Matching Traffic Demand in GEO Multibeam Satellites: The Joint Use of Dynamic Beamforming and Precoding Under Practical Constraints[J]. IEEE Transactions on Broadcasting, 2022, 68(4): 819-833. DOI: 10.1109/tbc.2022.3196173.

- 核实：[DOI出版记录](https://doi.org/10.1109/tbc.2022.3196173)；[摘要及作者记录](https://api.openalex.org/works/https://doi.org/10.1109/tbc.2022.3196173)。
- 说明：2022年68(4)、819-833正确。
- 要点：在真实非均匀流量、载荷功率和天线约束下优化动态波束图样，以减小需求与容量失配，并结合星上动态波束成形和地面预编码处理束间干扰；其优化对象包含物理层容量，不能简化成纯几何的模式库切换。

### [6] 正确（对应A07）

Yuan S, Sun Y, Peng M, et al. Joint Beam Direction Control and Radio Resource Allocation in Dynamic Multi-Beam LEO Satellite Networks[J]. IEEE Transactions on Vehicular Technology, 2024, 73(6): 8222-8237. DOI: 10.1109/tvt.2024.3353339.

- 核实：[DOI出版记录](https://doi.org/10.1109/tvt.2024.3353339)；[摘要及作者记录](https://api.openalex.org/works/https://doi.org/10.1109/tvt.2024.3353339)。
- 要点：把动态 LEO 网络的波束指向、时隙、子信道和功率联合优化拆为三个子问题，结合有外部性的匹配与逐次凸近似迭代求解；仿真改善服务用户数和总速率，同时考虑用户公平性。

### [7] 正确（对应B04）

Hika S F, Zhu L, Ji T, et al. Joint Allocation of Downlink Power, Bandwidth and Timeslots Based on Maximizing Energy Efficiency for GEO Satellite Communication Systems[C]//2024 IEEE 99th Vehicular Technology Conference (VTC2024-Spring). IEEE, 2024: 1-6. DOI: 10.1109/vtc2024-spring62846.2024.10683476.

- 核实：[DOI出版记录](https://doi.org/10.1109/vtc2024-spring62846.2024.10683476)；[摘要及作者记录](https://api.openalex.org/works/https://doi.org/10.1109/vtc2024-spring62846.2024.10683476)。
- 要点：先用线性规划分配时隙，再结合 Dinkelbach 方法和逐次凸近似优化功率、带宽，以最大化 GEO 下行能效；其目标是能效，和本课题固定频率后优先保持供需匹配程度的目标不同。

### [8] 正确（对应C05）

Feng L, Zhang C, Zhang Q, et al. Double-Timescale Multi-Agent Deep Reinforcement Learning for Flexible Payload in VHTS Systems[J]. Electronics, 2024, 13(14): 2764. DOI: 10.3390/electronics13142764.

- 核实：[DOI出版记录](https://doi.org/10.3390/electronics13142764)；[摘要及作者记录](https://api.openalex.org/works/https://doi.org/10.3390/electronics13142764)。
- 说明：2764为文章编号。
- 要点：针对带宽、功率调整速度不同以及业务时变的问题，提出双时间尺度的多智能体深度强化学习分配方案；仿真提高容量—需求匹配程度并降低功耗，采用离线训练、在线决策。

### [9] 正确（对应F06）

Paris A, Del Portillo I, Cameron B, et al. A Genetic Algorithm for Joint Power and Bandwidth Allocation in Multibeam Satellite Systems[C]//2019 IEEE Aerospace Conference. IEEE, 2019: 1-15. DOI: 10.1109/aero.2019.8742238.

- 核实：[DOI出版记录](https://doi.org/10.1109/aero.2019.8742238)；[摘要及作者记录](https://api.openalex.org/works/https://doi.org/10.1109/aero.2019.8742238)。
- 要点：采用遗传算法联合搜索波束功率和带宽，并通过修复函数保证解的可行性、加快收敛；在含束间干扰和大气衰减的模型中，联合分配比只调功率更能降低未满足容量。

### [10] 正确（对应E01）

Hu X, Wang Y, Liu Z, et al. Dynamic Power Allocation in High Throughput Satellite Communications: A Two-Stage Advanced Heuristic Learning Approach[J]. IEEE Transactions on Vehicular Technology, 2023, 72(3): 3502-3516. DOI: 10.1109/tvt.2022.3218565.

- 核实：[DOI出版记录](https://doi.org/10.1109/tvt.2022.3218565)；[摘要及作者记录](https://api.openalex.org/works/https://doi.org/10.1109/tvt.2022.3218565)。
- 说明：正式卷期为2023年；DOI中的2022来自在线出版流程，不应把卷期年份改成2022。
- 要点：用深度强化学习辅助启发式功率分配，以改善动态场景下纯学习方法的泛化并加快启发式收敛；摘要报告计算开销相对所比较方法降低约 6%—95%，是“学习给出引导、优化继续改进”的直接卫星依据。

### [11] 正确（对应B03）

Gao W, Wang L, Qu L. Research on Joint Resource Allocation for Multibeam Satellite Based on Metaheuristic Algorithms[J]. Entropy, 2022, 24(11): 1536. DOI: 10.3390/e24111536.

- 核实：[DOI出版记录](https://doi.org/10.3390/e24111536)；[摘要及作者记录](https://api.openalex.org/works/https://doi.org/10.3390/e24111536)。
- 说明：1536为文章编号；正式归入2022年第11期，尽管在线日期为10月。
- 要点：提出非支配波束编码方法处理带宽重叠约束，并构造可用于多种元启发式的统一分配框架；实验显示编码和约束处理本身即可改善解质量、减少计算量，因此不能仅把该文概括为一种粒子群算法。

### [12] 正确（对应C10）

Hu X, Liu S, Chen R, et al. A Deep Reinforcement Learning-Based Framework for Dynamic Resource Allocation in Multibeam Satellite Systems[J]. IEEE Communications Letters, 2018, 22(8): 1612-1615. DOI: 10.1109/lcomm.2018.2844243.

- 核实：[DOI出版记录](https://doi.org/10.1109/lcomm.2018.2844243)；[摘要及作者记录](https://api.openalex.org/works/https://doi.org/10.1109/lcomm.2018.2844243)。
- 要点：把多波束卫星动态资源分配表示为深度强化学习问题，并用类似图像的张量提取业务时空特征；动态信道分配实例验证了在未知时变环境中学习资源管理策略的可行性。

### [13] 正确（对应C03）

Meng H, Xin N, Qin H, et al. A Recursive DRL-Based Resource Allocation Method for Multibeam Satellite Communication Systems[J]. Chinese Journal of Electronics, 2024, 33(5): 1286-1295. DOI: 10.23919/cje.2022.00.135.

- 核实：[DOI出版记录](https://doi.org/10.23919/cje.2022.00.135)；[摘要及作者记录](https://api.openalex.org/works/https://doi.org/10.23919/cje.2022.00.135)。
- 说明：正式卷期为2024年33(5)，DOI包含2022不代表正式出版年份为2022。
- 要点：采用 LSTM 与 PPO 递归生成逐波束频率子动作，降低联合动作空间的探索代价，并利用已经生成的动作指导后续分配；仿真改善频谱效率和系统满意度，适合与本课题逐束离散决策对照。

### [14] 正确（对应C04）

Luis J J G, Crawley E, Cameron B. Applicability and Challenges of Deep Reinforcement Learning for Satellite Frequency Plan Design[C]//2021 IEEE Aerospace Conference (50100). IEEE, 2021: 1-11. DOI: 10.1109/aero50100.2021.9438291.

- 核实：[DOI出版记录](https://doi.org/10.1109/aero50100.2021.9438291)；[摘要及作者记录](https://api.openalex.org/works/https://doi.org/10.1109/aero50100.2021.9438291)。
- 说明：作者在arXiv也写作Juan Jose Garau Luis，按DOI作者字段缩写为Luis J J G可接受；2020年预印本不应代替2021年会议年份。
- 要点：以多波束卫星频率规划为例，比较策略、优化器、状态、动作、奖励和训练环境六类设计选择；结果强调在线决策速度的优势，同时指出不存在适合所有维度和非平稳场景的单一 DRL 配置。

### [15] 正确（对应C08）

Huang J, Yang Y, Lee J, et al. Deep Reinforcement Learning-Based Resource Allocation for RSMA in LEO Satellite-Terrestrial Networks[J]. IEEE Transactions on Communications, 2024, 72(3): 1341-1354. DOI: 10.1109/tcomm.2023.3331021.

- 核实：[DOI出版记录](https://doi.org/10.1109/tcomm.2023.3331021)；[摘要及作者记录](https://api.openalex.org/works/https://doi.org/10.1109/tcomm.2023.3331021)。
- 说明：主要字段正确；是LEO星地融合RSMA研究。
- 要点：针对 LEO 星地融合网络的 RSMA 资源分配，分别用 DQN 处理离散资源选择、PPO 处理连续功率控制；可支持“离散—连续变量应区别建模”的论述，其星地共享与公共／私有流模型不同于单星 GEO 频率槽分配。

### [16] 正确（对应C09）

Wang S, Lv T, Ni W, et al. Joint Resource Management for MC-NOMA: A Deep Reinforcement Learning Approach[J]. IEEE Transactions on Wireless Communications, 2021, 20(9): 5672-5688. DOI: 10.1109/twc.2021.3069240.

- 核实：[DOI出版记录](https://doi.org/10.1109/twc.2021.3069240)；[摘要及作者记录](https://api.openalex.org/works/https://doi.org/10.1109/twc.2021.3069240)。
- 说明：书目信息正确；研究场景是地面MC-NOMA，不能归为卫星论文。
- 要点：面向含硬件灵敏度限制和不完美 SIC 的地面 MC-NOMA，采用 DDPG 将子载波分配与功率分配作为两个序贯子任务，并用多智能体结构降低功率分配复杂度；这是无线资源分配的方法借鉴，不能称为卫星资源分配论文。

### [17] 更正作者（对应C11）

Liao X, Hu X, Liu Z, Ma S, Xu L, Li X, Wang W, Ghannouchi F M. Distributed Intelligence: A Verification for Multi-Agent DRL-Based Multibeam Satellite Resource Allocation[J]. IEEE Communications Letters, 2020, 24(12): 2785-2789. DOI: 10.1109/LCOMM.2020.3019437.

- 核实：[DOI出版记录](https://doi.org/10.1109/lcomm.2020.3019437)；[摘要及作者记录](https://api.openalex.org/works/https://doi.org/10.1109/lcomm.2020.3019437)。
- 更正：原开题中的“Liao X, Lin M, Chen X, et al”错误；用户提示的“X. Liao, X. Hu, Z. Liu, S. Ma, L. Xu, X. Li, W. Wang, and F. M. Ghannouchi”与出版记录一致。题名、刊物、年份、卷期页码正确。
- 要点：提出协作多智能体 DRL 资源管理框架，以带宽分配为例分担集中式智能体的计算负担；仿真验证传输效率和低复杂度方面的收益。

### [18] 正确（对应C02）

Li Y, Yang D, Yang H, et al. Joint Frequency Assignment and Power Allocation Based on Multi-Agent Deep Reinforcement Learning for Multi-Beam Satellite Systems[C]//2023 IEEE 97th Vehicular Technology Conference (VTC2023-Spring). IEEE, 2023: 1-6. DOI: 10.1109/vtc2023-spring57618.2023.10200414.

- 核实：[DOI出版记录](https://doi.org/10.1109/vtc2023-spring57618.2023.10200414)；[摘要及作者记录](https://api.openalex.org/works/https://doi.org/10.1109/vtc2023-spring57618.2023.10200414)。
- 要点：把每个波束建模为智能体，采用集中训练、分布执行联合分配频率与功率，并以多离散动作转单离散动作的机制处理分层动作；目标是容量与业务需求匹配，和本课题第二阶段直接相关。

### [19] 正确（对应F08）

Aravanis A I, Shankar B M R, Arapoglou P D, et al. Power Allocation in Multibeam Satellite Systems: A Two-Stage Multi-Objective Optimization[J]. IEEE Transactions on Wireless Communications, 2015, 14(6): 3171-3182. DOI: 10.1109/twc.2015.2402682.

- 核实：[DOI出版记录](https://doi.org/10.1109/twc.2015.2402682)；[摘要及作者记录](https://api.openalex.org/works/https://doi.org/10.1109/twc.2015.2402682)。
- 说明：原书目信息正确；需要纠正的是方法描述，第二阶段并非凸优化。
- 要点：先获得业务匹配解，再用这些解增强同时考虑业务匹配和总功耗的多目标元启发式搜索，得到帕累托权衡；原文的“两阶段”不是“元启发式粗分后再做凸优化”，引用时应纠正这一常见误述。

### [20] 正确（对应E05）

Wang J, Ma L, Wei Z, et al. Multi-beam-based Downlink Modeling and Power Allocation Scheme for Integrated Sensing and Communication towards 6G[C]//2022 IEEE 95th Vehicular Technology Conference: (VTC2022-Spring). IEEE, 2022: 1-5. DOI: 10.1109/vtc2022-spring54318.2022.9860363.

- 核实：[DOI出版记录](https://doi.org/10.1109/vtc2022-spring54318.2022.9860363)；[摘要及作者记录](https://api.openalex.org/works/https://doi.org/10.1109/vtc2022-spring54318.2022.9860363)。
- 说明：书目信息正确；研究对象为地面基站—车辆通感下行，不能归为卫星论文。
- 要点：研究地面基站对车辆的多波束通感一体化下行模型，通过凸优化分配功率以兼顾通信速率和感知精度；原文使用 CVX，并与等功率、注水比较，但它不是多波束卫星功率分配论文。

## 未纳入统计的待定候选及写作提醒

检索到 Ivari 等的《On Optimal Power Allocation in Multibeam Multicast NOMA for Satellite Communication Systems》，DOI为[10.1109/TAES.2023.3287475](https://doi.org/10.1109/TAES.2023.3287475)，但本次取得的出版注册记录仍仅给出2023年、1-20的提前发表信息，未完成正式卷期页码核定，因此未列入A—F清单、未计入总数。

第一阶段可按“规则布局局限—需求驱动变束宽／变波位—容量约束与负载均衡—动态重配置”组织研究现状；第二阶段可按“数学优化和元启发式—单智能体与多智能体DRL—结构化注意力及候选指针”组织；第三阶段可按“需求匹配功率控制—干扰耦合下SCA／WMMSE—学习与优化混合”组织。比较时分别说明是否允许改变布局、频率、时隙和功率，避免把目标不同的能效、总速率和需求满足率直接当作同一指标。

## 补充：2026-10-29 按用户批注新增

以下 6 条由 Claude 按批注新增，元数据经 Crossref 逐条核对；DRL 覆盖类文献的摘要与局限见 `outputs/midterm_report/review_20261029/drl_coverage_refs.md`，DVB-S2X 数值取自 ETSI 官方 PDF 的 Table 20a。

- 【A｜2018】**A11** Liu J, Shi Y, Fadlullah Z M, et al. Space-Air-Ground Integrated Network: A Survey[J]. IEEE Communications Surveys & Tutorials, 2018, 20(4): 2714-2741. DOI: 10.1109/comst.2018.2841996.
- 【A｜2023】**A12** Xu G, Tan F, Ran Y, et al. Joint Beam-Hopping Scheduling and Coverage Control in Multibeam Satellite Systems[J]. IEEE Wireless Communications Letters, 2023, 12(2): 267-271. DOI: 10.1109/lwc.2022.3223507.
- 【A｜2025】**A13** Yuan Y, Ji Z, Wu S, et al. User-Centric Beam Coverage and Resource Optimization for Multi-Beam Satellite Systems via Multi-Agent Deep Reinforcement Learning[C]//2025 11th International Conference on Computer and Communications (ICCC). IEEE, 2025: 1845-1852. DOI: 10.1109/iccc68654.2025.11438171.
- 【A｜2025】**A14** Ngo Q T, He Y, Jayawickrama B, et al. A Fast Fuzzy DRL-Based Joint Beam Design and Power Allocation for Multi-Beam GEO-LEO Coexisting Satellite Networks[J]. IEEE Transactions on Wireless Communications, 2025, 24(10): 8741-8755. DOI: 10.1109/twc.2025.3568780.
- 【A｜2020】**A15** Hu X, Zhang Y, Liao X, et al. Dynamic Beam Hopping Method Based on Multi-Objective Deep Reinforcement Learning for Next Generation Satellite Broadband Systems[J]. IEEE Transactions on Broadcasting, 2020, 66(3): 630-646. DOI: 10.1109/tbc.2019.2960940.
- 【F｜2024】**F20** ETSI. Digital Video Broadcasting (DVB); Second generation framing structure, channel coding and modulation systems for Broadcasting, Interactive Services, News Gathering and other broadband satellite applications; Part 2: DVB-S2 Extensions (DVB-S2X): ETSI EN 302 307-2 V1.4.1[S]. Sophia Antipolis: ETSI, 2024.
