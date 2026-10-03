# 实验状态与新窗口接手说明

**S2-P17核心已实现并核验，正式训练尚未启动（2026-10-04）：**核心提交0da4ca6已合入main（909a98d）；20424全量526项/4跳过、真实CUDA小更新/恢复、T0严格加载及真实保存初解→SCA A/B不重新推理检查通过；实际页面V0/V1、210维/14令牌、SCA来源与预算操作验收通过。596原CSV逐SHA上传；Jan12合格12母/144输入与Jan29 20母/440输入已核验。训练原576母仅520合格，56不满足200正需求用户，已请用户选择520/7280或补56母；答复前不改范围、不提交训练。两组0/2、正式交互0/800000，效果待验证。每小时s2-p17保持ACTIVE，已更新待选约束并精确回读。正式报告保留第二阶段直接与SCA最终两列、逐数量配对和分段耗时；固定/贪心新增合法动作适配明示，不称原规则未变。所有正式任务只允许20424。[方案实施记录](docs/第二阶段/方案/17_可变波束数量泛化训练方案.md#11-2026-10-04实施与启动前核验记录) · [运行清单](outputs/experiments/variable_beams_20261004/run_manifest.json) · [工程证据](outputs/experiments/variable_beams_20261004/preflight/engineering_preflight_summary.json)。

**S2-P17训练母场景资格待用户选择（2026-10-04）：**20424核验原576训练母快照仅520个满足至少200正需求用户，56个不满足（最少140）；原20验证中18合格，可按既定规则选12，首次不合格选取导致的0/144数据准备失败现场已保留并修正。已询问采用520母/7280训练景且维持两组40万步及考核预算，或另取原两天56个合格时刻补576；答复前不生成或提交依赖该选择的正式训练。工程实现与20424验证继续：DQN/CUDA及冻结包路径通过，核心/采样/案例16项、旧环境45项、SCA与保存初解来源39项专项通过，真实V0/V1更新和完整恢复、旧T0严格加载及100/200束兼容检查通过；均为工程验证，不是新训练效果。596源CSV逐SHA上传，Jan29官方下载持续，12合格验证母景的144输入在后台生成。正式任务0；每小时`s2-p17`已ACTIVE并精确回读。[运行清单与资格证据](outputs/experiments/variable_beams_20261004/run_manifest.json) · [自动化回读](outputs/experiments/variable_beams_20261004/followup/verified_readback.json)。

**S1-02已获执行授权，正在实现和数据准备（2026-10-04）：**用户批准“先A阶段，再按方案05做B阶段”，本轮仅20418。范围落实为A开发80/正式400方法结果，B验证集五菜单筛选100结果，X0加两个候选各409600交互/seed51001（共1228800），训练验证1200及新100景三模型加三搜索600结果；原两奖励目标矩阵不额外执行。开发worktree为`C:/Users/Admin/.codex/worktrees/stage1-balance-actions/paper`，分支`codex/stage1-balance-actions`；核心、工作台与运行管线并行实现，尚未冻结新科学提交或提交正式任务，效果待验证。

实际远端核验北京时间02:53:24：20418 DQN/CUDA可用、4090D无训练占用，默认包仍指旧repo，正式将显式绑定新snapshot。A Jan28与B Jan30各100时刻已在看场景/成绩前冻结，避开另一个第二阶段任务的Jan29；原train576/val20来源已核，原包下载由两持久后台进程执行，下载不算训练。每小时`s1-02-a-b`已ACTIVE并精确回读。入口：[执行说明06](docs/第一阶段/方案/06_A阶段与动作空间B阶段远端执行说明.md) · [运行清单](outputs/experiments/first_stage_s1_02_20261004/run_manifest.json) · [真实远端预检](outputs/experiments/first_stage_s1_02_20261004/preflight/remote_readonly_20418.json) · [定时配置回读](outputs/experiments/first_stage_s1_02_20261004/preflight/automation_readback.json)。旧S1-01及其他阶段保持原记录，本条优先于以下未授权候选历史。

**S2-P17已获执行授权，正在实现与预检（2026-10-04）：**用户批准方案17开始，并明确本轮只在`10.160.17.102:20424`执行，覆盖原双实例建议。两新组各400000交互/seed51001、总800000；6048验证分配、4840独立初始分配及同数SCA完整细调预算保持。最终报告须并列第二阶段直接结果与第三阶段SCA最终结果及分段/总耗时。已从70572f7建立独立开发worktree与`codex/variable-beams-20261004`分支，正在做可变N、210维状态、混合采样、工作台和数据适配；源码尚未冻结，正式任务尚未提交，效果待验证。真实远端观测与数据资格见[运行清单](outputs/experiments/variable_beams_20261004/run_manifest.json)及preflight，不用旧完成记录代替本轮检查。[方案17](docs/第二阶段/方案/17_可变波束数量泛化训练方案.md)。

**S1-02动作空间候选扩展（2026-10-04，仅方案）：**按用户希望增加思路、后续共同对比的要求，新增[第一阶段方案05](docs/第一阶段/方案/05_波束调整动作空间候选与对照方案.md)。优先X0四向5动作/X1八方向9动作/X2四向多步13动作/X3业务引导位置/X4相邻双束；主动选束、用户归属、连续位移和可变宽度另列后续。共同A2起点、覆盖/主体口径及移动范围；单束最多10轮/1000次束处理，双束占两次额度并同时比较候选核验和完整时间。拟20开发场景×5菜单=100个非学习结果，之后再选基准加两候选训练；3×409600=1228800交互或全部5×409600=2048000仅成本备选，seed51001，不与方案04原两目标矩阵自动交叉/叠加。若选三菜单，独立考核3模型加3同菜单搜索共600方法结果为候选规模，日期/预算/任务ID/冻结科学版本仍待确认；未实现、未启动训练/正式评价/定时跟进，效果待验证。仅内容/链接/算术检查，无需本轮改前端，实施须同步动作配置和结果。S1-01已完成及其他组状态保持原证据，本轮未刷新API/SSH。

**S1-02动作与最大轮数补充（2026-10-04，仅方案）：**用户提出四向移动若干格/固定格数或不动，直到全局理想状态，并要求未达时有合理轮询上限。[方案04第5.1节](docs/第一阶段/方案/04_覆盖优先与主体波束负载均衡方案.md#51-四向格点移动与全局停止条件)建议先固定1格5动作，多步长1/2/4格13动作备选；每轮100束依次决策，最多10轮/1000次调位，达标提前结束，未达保留最好可行布局和实际终态/停止原因。10轮是待验证的工程起点，替代此前200次调位建议；覆盖修补另最多200次，原候选核验/时间限额继续累计，不因未达而自动延长。成对动作保留扩展，A3/B0/B1采用相同选定动作与停止规则，不新增组数；两PPO仍各409600交互，建议128环境×64采集以便观察1000步完整回合，每批8192/50批/原学习设置1600更新，须正式前核验冻结。前端动作/格宽/轮数/停止展示及逐场景早停尚未实现，未创建任务/定时跟进，未训练或新评价；无需本轮改前端。旧S1-01结果与完成状态不变。

**S2-P17可变波束数量正式方案已归档（2026-10-04，待确认执行）：**用户指出前次仅问答未单独写方案，现已补[第二阶段方案17](docs/第二阶段/方案/17_可变波束数量泛化训练方案.md)。建议接口1–200、首轮科学范围50–200（1–49仅工程检查）；两组同210维数量感知T-SAC，V0固定100束、V1保留25%100束并混合其余数量，各400000交互/seed51001、合计800000。留出60/90/140/180束，拟训练8064场景、144验证场景×21节点×2模型=6048次验证分配；候选Jan29的20母快照×11束数×2负载=440衍生场景，V0/V1/H0各3动作加固定/贪心各1，4840初始分配＋同数统一S3-P03 B路线细调，共9680细调前后终态。固定3000W、原频谱，B每次含A及追加两段共最多4000调用/60秒；20母快照不能冒充440独立来源。Jan29为避开第一阶段候选Jan28的建议，仍须核真实可用性及数据隔离；上述全部预算为待确认建议。T0原207维中的两段100是频率槽，旧权重跨N只作兼容后历史参照，不保证200束效果。代码/前端同步、正确性与真实更新、计时/逐束审计及报告跟进门槛已写明，但尚未实现、冻结新科学版本、创建任务或定时跟进；效果待验证。本轮未刷新API/SSH，已完成旧组保持原交付核验。[问答](Q&A/算法与训练/可变波束数量的泛化训练.md)。

**S1-02候选方案归档（2026-10-04，尚未授权执行）：**根据用户“100%或接近100%覆盖下尽量主体均衡，允许少数边缘专用束轻载”的要求，结合开题第2.1节和S1-01原始结果形成[第一阶段方案04](docs/第一阶段/方案/04_覆盖优先与主体波束负载均衡方案.md)。建议双100%优先、双99.9%备用目标，按共同候选几何认证少量例外，主体CV≤0.20且全部100束中至少90束处于主体均值±20%为待确认目标，全体CV与覆盖明细继续保留。A阶段拟旧C1/加覆盖修补/再加负载感知归属/再加协调调位四路线，候选新日期100场景各1次、400结果且训练0；日期和有限计算上限正式执行前核验冻结。B阶段两新PPO各409600步、seed51001/合计819200仅为后续建议，须根据A报告另行确认；宽度扩展不在两阶段候选预算内。尚无任务ID、冻结新科学提交或权重，未实现或启动正式计算/定时跟进，效果待验证。

本轮只读复核旧C1的100个2024-01-22终态：双≥99.9%为50/100，双100%为36/100，64场景漏用户、最多7架；原100/100可行使用双99%门槛，历史结果不改写。[复核原始来源与SHA](outputs/verification/stage1_balance_plan_20261004/c1_threshold_review.json) · [问答](Q&A/环境仿真/第一阶段波束覆盖规划与两阶段衔接.md)。这是归档结果重算，不是新评价/训练；本轮没有刷新实时API/SSH，S1-01完成状态仍依据2026-09-30最终交付核验。文档无需前端改动，未来实施须同步配置、校验、任务/断点元数据和结果展示。

**SI上限表述澄清（2026-10-04，只读问答）：**S3-P03支持当前固定槽、布局、预算与基本服务条件下，从已有SCA继续功率微调的收益明显递减；C对A仅+0.00229/+0.00248个百分点、完整耗时约11秒对3.9秒，可作为降低继续微调投入优先级的依据。尚无能够约束所有可行方案最好SI的参照值，B/C也共享对应A起点，未证明全局或物理上限；约94.2%仅为本批Jan27场景平均成绩。不把“步长小/预算用尽停止”当最优证明，不把未调整的频率、布局或跨日期效果写成已验证。仅更新[通俗问答](Q&A/算法与训练/第三阶段功率细调方案.md)，无新计算任务，S3-P03完整交付和跟进暂停保持不变。

**S3-P03完整交付已完成（2026-10-04；任务实际API/SSH核验为10月3日23:35，10月4日完成审计/报告复核）：**两原任务各900/900，完成2/2、1800/1800（100%），新增训练0。Jan27共同100场景×3动作×A/B/C，T0固定150000步/seed51001/α0.01，科学e85a942未改变。3696文件/1,403,813,436字节逐SHA归档、18万终态束与12万最高SI候选束物理复算、基本服务/SI门槛及完整计时审计均通过。中文[报告03](docs/第三阶段/实验报告/03_SI优先功率细调与容量近似对照.md)、15图、配对/阶段/耗时表及生成脚本齐全。

本轮追加调整使SI微增，未出现明显提升：3000W下A/B/C SI为94.224274%/94.225854%/94.226564%，SGM为0.943430/0.943229/0.943162，单次完整时间3.916/5.529/11.331秒；初始总功率上限下SI为94.128265%/94.129973%/94.130745%，SGM为0.942520/0.942313/0.942244，时间3.934/5.603/10.945秒。B对A平均SI+0.00158/+0.00171个百分点，C+0.00229/+0.00248，两预算均100景略升；SGM大多下降，C并非每景优于B。B/C无相对A新增严格基本服务跨下；C恢复36/28条临界束，原A距门槛仅数bps，不代表承载大幅改善。C有1次时间上限停止，30.004320秒如实记录；单种子、单Jan27、固定频率与有限求解范围不构成全局最优证据。

后处理曾因Windows MKL与远端OpenBLAS小数尾差误报4个B案例5束的最高SI候选违规（最大约3.71e-5bps）；原远端冻结DQN复算与保存值一致且严格达标。只修本地审计的严格记录判定与数值一致性检查分工（提交ec37863），不改科学源码、原结果或容差；失败现场和复核证据保留。原watcher重试已核身份后停止，后处理由根幂等接续完成，没有重推理/优化/评价。报告53项资产已备份授权远端共享/home，并两端逐成员复核；这是一个共享副本。每小时`s3-p03-si`已在完整交付核对后暂停为PAUSED，并[精确回读](outputs/experiments/stage3_si_priority_20261003/delivery/automation_closed_verified.json)；不关机、不归档对话、不新开实验。

[运行清单](outputs/experiments/stage3_si_priority_20261003/run_manifest.json) · [原始归档终核](outputs/experiments/stage3_si_priority_20261003/completion/root_archive_verification_final.json) · [全量物理与计时审计](outputs/experiments/stage3_si_priority_20261003/completion/beam_audit.json) · [独立指标/阶段复算](outputs/experiments/stage3_si_priority_20261003/completion/root_summary_check.json) · [审计误报诊断](outputs/experiments/stage3_si_priority_20261003/diagnostics/conclusion.md) · [报告复核](outputs/experiments/stage3_si_priority_20261003/completion/root_report_verification.json) · [报告备份](outputs/experiments/stage3_si_priority_20261003/delivery/operation_receipt.json)。以下运行中、待确认和审计待诊断内容是之前的过程记录。

**S3-P03 SI优先三组已在双远端正式运行（2026-10-03，北京时间22:30:25–26主任务API/SSH核验）：**20418总3000W任务`refine_sac_20261003_222353_313e994d`为119/900，20424初始总功率上限任务`refine_sac_20261003_222356_db93000c`为144/900；均evaluating、完成0/2，合计263/1800（14.61%），新增训练0。冻结科学源码`e85a9429461a358ba2e11614a3cce4671d12b68e`；T0固定150000步终点/seed51001/α0.01，Jan27共同100景×3动作×A/B/C不变。A为旧SCA，B优先SI，C改进功率估计且优先SI；B/C分别从同A终值追加，允许SGM下降，之后锁住最高已找到可行SI。每段300/2000/30，B/C完整路线计入A，最多4000容量调用、60秒两段求解；连续功率非负且仅总预算，空槽0。

两端首景各9条结果通过：跨预算T0初始SHA一致，本预算B/C引用同一A分配及案例SHA，SI锁定/基本服务与引用时间核验通过。部分首景SGM低于A是本轮目标允许的取舍，完整SI/SGM收益仍待全部评价。497项远端全量（3项Windows专用跳过）、最后冻结版双端各44项专项、Jan12两景双预算12条开发检查及真实worker入口通过；原A核心源码与b3等价。22:33另核两正式worker的CVXPY1.6.5/Clarabel0.11.1及隔离路径正确。页面实际三组选择/冻结参数只读/0动作被拒/3动作通过；同端口隧道8776/8777保留原同源检查。输入下载、依赖传递及dev静态资源问题均在正式任务前解决，无当前运行异常。

持久watcher38592负责完成后逐SHA归档及1800case/18万束、最高SI候选与计时审计；本对话每小时`s3-p03-si`已创建ACTIVE并精确回读，每次发中文简报，完整报告03与曲线交付后暂停。今后未指定本地的正式实验默认这两远端，已写入AGENTS。S3-P01/P02完成/暂停及冻结资产不变；下文同日“待确认/未启动”为之前讨论历史。

[方案04](docs/第三阶段/方案/04_SCA需求满足率提升与剩余空间诊断.md) · [运行清单](outputs/experiments/stage3_si_priority_20261003/run_manifest.json) · [根实时核验](outputs/experiments/stage3_si_priority_20261003/preflight/root_launch_acceptance.json) · [正式worker依赖核验](outputs/experiments/stage3_si_priority_20261003/preflight/root_worker_dependency_acceptance.json) · [首景来源](outputs/experiments/stage3_si_priority_20261003/preflight/first_scene_verified.json) · [首景SI与逐束守护](outputs/experiments/stage3_si_priority_20261003/preflight/first_scene_guard_verified.json) · [定时跟进精确回读](outputs/experiments/stage3_si_priority_20261003/followup/verified_readback.json)。

**SI优先目标更正（2026-10-03，仅方案）：**用户明确先确保高SI、再尽可能提高SGM。方案04已改为分级目标：SI阶段移除旧SGM下限，保留功率与逐束基本服务；SGM阶段锁住已找到的最佳可行SI，默认仅1e-8数值容差且门槛不累计下滑。旧实现的SCA线性子问题与真实接受检查两处都有SGM保护，尚未修改。上一版“SGM不得低于当前SCA”的建议被覆盖，最终SGM可能因优先SI而低于旧SCA。B/C候选共同采用SI优先、等追加预算，C再换容量凹下界；B对A是目标与追加搜索整体效果。目标顺序已明确，正式实施及候选1800结果预算仍待确认，未新建任务/训练/监控；S3-P02完成/暂停不变。[方案04](docs/第三阶段/方案/04_SCA需求满足率提升与剩余空间诊断.md) · [问答](Q&A/算法与训练/第三阶段功率细调方案.md)。

**SCA进一步提高SI的前次方案讨论（2026-10-03，SGM保护建议已由上段更正）：**已只读复核S3-P02的600条SCA轨迹。98%/99%的调用封顶均发生在SGM阶段；两端各299/300条SI阶段按65%预算切换，其末尾约100调用（窗口中位105次）SI增益中位仅0.000139/0.000156个百分点。每场景最差10束贡献约87%累计SI缺口，不能据此认定缺口全可恢复。建议从相同SCA终值比较原解/原SCA追加求解/容量凹下界SCA，守住SGM与基本服务，额外预算相同并计入总耗时；候选Jan27的100景×3动作×3方法×两预算共1800结果，新增训练0。数学、预算及条件见[方案04](docs/第三阶段/方案/04_SCA需求满足率提升与剩余空间诊断.md)；[阶段证据](outputs/verification/sca_si_followup_20261003/review/sca_history_review.json)与[缺口统计](outputs/verification/sca_si_followup_20261003/beam_deficit.json)来自已用Jan26，非新算法效果。当前仅方案，须用户确认后实施；未新建任务或自动化，S3-P02完整交付与暂停不变。

**S3-P02七路线评价、耗时审计与正式报告已完成（2026-10-03；16:35实际API/SSH核验）：**两端原任务均completed，各2100/2100，完成2/2、4200/4200（100%），新增训练0。共同Jan26的100场景×3动作×7方法、T0固定150000步终点/seed51001、冻结科学源码`b3c94c993457919bd17e30c71f22f87e5349f012`不变；相同初始SHA、42万束物理/指标与1400场景×方法计时行通过审计，6094原始文件/2,272,275,339字节逐SHA复核通过。watcher已正常完成归档并退出，无需恢复评价。

同场景先平均3动作：原T0为SI89.7898%、SGM0.707955、2276.483W。3000W预算下SCA为93.7783%/0.940246/2999.682W、单次分段总时间3.867秒，SLSQP为93.7473%/0.937759/2928.120W、12.013秒；初始总功率上限下SCA为93.7003%/0.939547/2276.383W、3.890秒，SLSQP为93.6711%/0.939406/2268.334W、13.139秒。SCA在本轮平均指标略高且更快，但功率略高、部分场景不如SLSQP；DE/PSO和WMMSE效果与计算开销见报告，不称全局最优。严格基本服务计数保留下降；SCA最大服务下限短差约3.418bps（最大需求比例8.56e-9）在既定容差内，不能全部解释成机器舍入。SLSQP局部非收敛103/131次，SCA多数触及2000容量调用上限，WMMSE每端70/300回退；零超时、无超预算或容差外服务守护违规。

中文[报告02](docs/第三阶段/实验报告/02_七路线功率细调与逐场景耗时对照.md)及14图已完成并复核，终态指标与单次/每景3动作的时间同表，均值、中位/P95和逐景原始表齐全；公共T0初始解只实际生成一次，总时间为分段实测合计，包含初始解而非只计算细调。单训练seed、单Jan26和设备实现限制保留。报告资产已备份到授权远端共享/home的一处目录，两SSH端逐成员核验通过，不称两份独立备份。本对话`s3-p02`每小时跟进已在完整交付复核后暂停为PAUSED，并保存[操作及精确回读](outputs/experiments/stage3_all_routes_20261003/delivery/automation_closed_verified.json)；不新增实验、不关机，旧S3-P01保持完成和暂停。

[运行清单](outputs/experiments/stage3_all_routes_20261003/run_manifest.json) · [物理与计时审计](outputs/experiments/stage3_all_routes_20261003/completion/beam_audit.json) · [全量SHA复核](outputs/experiments/stage3_all_routes_20261003/completion/root_archive_verification.json) · [指标与阈值独立复算](outputs/experiments/stage3_all_routes_20261003/completion/root_supplement_audit.json) · [报告和图表复核](outputs/experiments/stage3_all_routes_20261003/completion/root_report_verification.json) · [报告备份](outputs/experiments/stage3_all_routes_20261003/delivery/operation_receipt.json)。原两任务ID仍为20418 `refine_sac_20261003_142302_c8c68fcb`、20424 `refine_sac_20261003_142305_a2665f99`，不重复评价。

**S3-P01完整评价、审计与正式报告已完成（2026-10-03；03:57:55–56远端API核验，04:18原始归档全量复核，04:25报告复核）：**两原任务均900/900，2/2完成，1800/1800方法结果（100%），新增训练0；冻结T0的15万终点/seed51001和科学2b3ef54不变。100个Jan25共同场景内先平均3次动作：原分配SI91.5300%、SGM0.705797、功率2184.800W；连续细调在3000W预算下为95.0546%/0.951469/2907.237W，在初始总功率上限下为94.9681%/0.953006/2172.408W，两预算各100/100场景SI与SGM均提高。严格基本服务束数和全束达标比例有下降，逐束诊断确认全部跨越差在预设数值容差内（最大约0.000153bps、相对7.375e-13），原值保留，不宣称严格计数不降；连续求解分别131/156次非收敛，收益包含初始化与放宽单束功率范围，单seed/单日期且无全局最优保证。3740文件/1,206,945,833字节逐SHA最终核验、1800case/18万束物理与指标独立审计、中文报告与8图均已完成。报告ZIP已备份到两SSH端共享/home的一处目录并逐成员核验，不称两份独立远端备份。此前本地辅助进程退出原因未证，结果API门槛与审计写缓存两项运维问题已保留现场后修复；不重评、不改冻结科学源码。

[正式报告01](docs/第三阶段/实验报告/01_T0初始解的连续功率细调对照.md) · [报告及图表SHA](outputs/experiments/stage3_power_20261003/report_20261003/artifact_manifest.json) · [原始归档最终复核](outputs/experiments/stage3_power_20261003/completion/root_archive_verification.json) · [报告根复核](outputs/experiments/stage3_power_20261003/completion/root_report_verification.json) · [原始审计/配对完成回执](outputs/experiments/stage3_power_20261003/completion/receipt.json) · [报告备份回执](outputs/experiments/stage3_power_20261003/delivery/operation_receipt.json)。本轮交付复核已完成，04:30已通过应用工具将`s3-p01`暂停为PAUSED，目标对话和完整prompt保持不变并精确回读；[暂停回执](outputs/experiments/stage3_power_20261003/delivery/automation_closed_verified.json)。接续watcher32124已完成归档后正常退出，工作台与查看隧道保留；以下“运行中/归档中/报告待完成”均为注明时点的历史记录，不代表当前阶段。

**S3-P01两项评价均已完成，正在恢复本地归档并编写报告（2026-10-03，北京时间03:57:55–56 API/SSH核验）：**20418与20424原任务均completed、各900/900，共1800/1800（100%），新增训练0；冻结2b3ef54与T0终点不变，worker均已正常退出。本地watcher39804及原隧道进程已退出，旧watch_state停03:25，确切退出原因未证实；已保存故障前文件/日志SHA和新现场，恢复本地只读辅助链，不重复评价。接续归档首轮发现结果API正文不含stage，旧运维脚本错误将其当完成门槛；正在按真实job/status已completed加结果research_stage/summary/900scenes修复，原现场保留。正式总报告、18万束独立审计与本地归档尚未完成，定时跟进保持ACTIVE。[本次实时核验](outputs/experiments/stage3_power_20261003/preflight/heartbeat_live_20261002T195755_962475Z.json) · [本地辅助异常现场](outputs/experiments/stage3_power_20261003/preflight/helper_recovery_preflight_20261002T195925_390847Z.json)。

**S3-P01结束后正式实验报告已安排（2026-10-03 02:56配置精确回读）：**用户明确任务结束后要有实验报告。已新建本对话每小时heartbeat `s3-p01`（S3-P01实验报告与完成验收），状态ACTIVE，目标对话、完整prompt与配置均精确回读相等；每次跟进中文简报，完成后核双任务/逐SHA归档/1800case与18万束独立审计，交付两预算口径的中文正式总报告及曲线，更新目录、问答与实验状态，交付后暂停本跟进。已有60秒后台watcher继续负责原观察和后处理，不新增训练/评价或改冻结源码。正式报告计划为`docs/第三阶段/实验报告/01_T0初始解的连续功率细调对照.md`，目前未生成；须保留失败/非收敛/回退与单种子单日期限制。[创建及精确回读](outputs/verification/stage3_report_followup_20261003/verified_readback.json)。下方02:43“尚未创建聊天任务”是此前状态，已由本段更新。

**S3-P01第三阶段T0功率细调已在双远端正式运行（2026-10-03，北京时间02:32主任务API复核）：**20418任务`refine_sac_20261003_022328_6deff6bb`按3000W总预算执行，200/900方法结果；20424任务`refine_sac_20261003_022330_571c2455`按初始总功率上限执行，137/900；均evaluating、0/2任务完成，合计337/1800（18.72%），新增训练交互0。两端均用T0固定15万步终点（seed51001、固定α0.01、权重SHA ae095e8246…），冻结科学源码`2b3ef546391198004b82d86631258846bb3587f6`，新Jan25共同100场景×3动作×none/discrete/SLSQP含需求匹配初始化。首景两端各9结果通过，同初始分配SHA一致；CUDA、实际冻结包路径、server/worker进程出生核验通过。连续功率非负且仅总预算约束，无5W下限或50W单束上限。当前无运行异常；完整效果、1800case/18万束审计与归档待完成，不将部分场景当正式结论。持久watcher39804与8787/8788隧道运行，完成后自动逐SHA归档、逐束物理/指标复算及场景配对汇总。本地全量462项（2跳过）、最终T0专项29项、远端两端各29项通过；真实T0两Jan12场景两预算口径短检查通过，局部非收敛如实保留。原M1仅开发/部署预核，零正式任务。 [运行清单](outputs/experiments/stage3_power_20261003/run_manifest.json) · [主任务02:32核验](outputs/experiments/stage3_power_20261003/preflight/root_launch_acceptance.json) · [首景一致性](outputs/experiments/stage3_power_20261003/preflight/first_scene_verified.json) · [实施方案02](docs/第三阶段/方案/02_连续功率优化与双实例运行.md) · [当前持久跟进状态](outputs/experiments/stage3_power_20261003/preflight/watch_state.json)。以下同日“准备/未提交”是已被本段覆盖的历史里程碑。

**S3监控层次澄清（2026-10-03 02:43只读核对）：**本轮是每轮约60秒的本地持久watcher，PID39804/创建02:23:51仍运行；尚未创建S3的Codex聊天定时唤醒任务，因此不会自动在本对话发送简报。任务异常只标needs_attention，未实现自动恢复或聊天通知；完成后的归档/逐束审计/配对汇总已接续。应用现有五项旧实验自动化均PAUSED，没有S3条目。本次未创建或更改定时任务。[核验及问答](Q&A/算法与训练/第三阶段功率细调方案.md#2026-10-03追问是否设置了定时监控任务)。

**初始模型更正（2026-10-03，用户最新指令）：**今后用T0组固定终点模型生成初始解，覆盖此前M1工程起点建议。第三阶段原M1版本仅完成双端源码部署与检查，尚未启动工作台、导入或提交正式任务；正同步修改为T0，Jan25共同100场景、三方法与两预算口径不变，不新增训练。以下M1实现记录保留为更正前里程碑，不能按其启动正式评价。

**S3-P01第三阶段已获实现与双实例运行授权（2026-10-03，开发/验证中）：**用户明确连续功率可取任意非负值，只受总功率约束，覆盖P09最初的[5,50]W建议；今后“远端”默认10.160.17.102:20418与20424，认证只读BUPT_PASSWORD。首轮已按用户后续指令改为冻结G11 T0的15万步终点（固定α=0.01），固定实际频率槽，比较none、原档位discrete、连续SLSQP含需求匹配初始化。20418使用3000W总预算，20424使用同初始总功率上限；共同Jan25预选100场景×3动作×3方法，每端900条方法结果，共1800条，新增训练交互0、原训练seed51001，动作seed53001–53003。数据已下载及逐景核验，两GPU/DQN/CUDA已预核；源码、API/前端与恢复流程正完成验收，原M1版本仅部署验证，未启动工作台或正式任务；T0版本正式任务尚未提交、效果待验证。[实施方案02](docs/第三阶段/方案/02_连续功率优化与双实例运行.md) · [远端预核](outputs/experiments/stage3_power_20261003/preflight/remote_preflight_20261002T174108_775029Z.json) · [Jan25数据回执](outputs/experiments/stage3_power_20261003/preflight/remote_jan25_data_receipt.json)。下段P09为本日早先讨论，授权状态已由本段覆盖。

**当前最佳候选及T0高SGM补充诊断（2026-10-03，只读统计）：**从十一模型已归档Jan24逐束终态和原始results复算，T0的SGM0.697289、原奖励平均回报117.151、有效承载估计18.329Gbit/s均为本轮十一模型最高，功率2129.18W最低；M0的平均SI92.967%最高，T0为91.898%。按旧奖励/供需匹配优先保留T0，按平均SI选M0；此前M1是高SI/功率折中建议，不是唯一综合冠军。T0相对同15万步J0将超过2倍需求的记录比例38.63%降至24.49%，超额容量10.56降至5.78Gbit/s；训练末α0.01/策略熵1.02，对照α0.244/熵6.39，支持动作更集中但不证明全部收益由温度引起。有效承载是模型估计，按需求量加权，与逐束等权SI有取舍。单seed、单Jan24及不同训练预算限制保留，不拼历史榜或宣称上限；未实施第三阶段或改动原方案运行范围。[最佳候选问答](Q&A/算法与训练/模型是否学到了有效策略.md#2026-10-03追问当前历史最佳的模型是哪个) · [SGM原因](Q&A/算法与训练/当前奖励函数.md#2026-10-03追问t0的sgm为什么这么高) · [原数据复算与SHA](outputs/verification/g11_t0_sgm_20261003/beam_diagnosis.json)。G11完整交付与自动跟进已结束状态不变。

**历史最佳与上限口径澄清（2026-10-03，只读问答）：**M0/M1/G2是当前较强候选，不能称M/G两个系列全面历史最佳或接近所有强化学习算法的上限。Jan19/23/24结果不可直接拼榜；同Jan24的J0_150k已达SI92.861%，距M0仅0.106个百分点，而T0的SGM0.6973高于M/G。曲线平台只说明当前设置下改善趋缓，尚无同约束最优性差距证据。可按工程投入产出推进第三阶段讨论，但不以上限已证实为理由；本次未新增任何运行。[详细问答](Q&A/算法与训练/模型是否学到了有效策略.md)。

**第三阶段功率细调进入方案讨论（2026-10-03）：**用户提出第二阶段暂告一段落，改为研究“冻结较好模型给出频率功率初始分配，再固定频率槽优化功率”。本轮只读核对本地G11报告/原始汇总与代码，并形成[P09/第三阶段方案01](docs/第三阶段/方案/01_冻结模型初始解与功率细调方案.md)及[问答](Q&A/算法与训练/第三阶段功率细调方案.md)；未重新API/SSH观测、未实现优化器、未提交训练/评价或定时任务。M0/M1/G2的Jan24 SI分别92.967%/92.898%/92.714%，SGM为0.572476/0.576259/0.564436，没有唯一双指标赢家；训练平台不等于物理上限。建议M1作首版起点、M0参照，原5W档位调整作基线、连续[5,50]W的SQP作主候选，SCA后续研究；低于5W另列资源规则变更。效果待验证，正式运行范围与预算未确定。G11已完成的状态以以下00:40最终验收为依据，不恢复历史任务。

**G11完整交付已完成（2026-10-03，北京时间00:10归档完成、00:40最终复核）：**九训练3100000交互、十一模型Jan24共同评价和两基线3500次分配、350000终态束独立审计、正式报告12及9张曲线、分阶段620000条供需日志全部完成。原始训练/代码/历史断点等61,144文件、20,917,500,639字节已全量本地逐SHA归档；根再次核全部文件存在与大小、15个报告输出、9终点权重及66个补充日志输出SHA。归档器36948在成功写入最终回执后正常退出，不是新故障。保留所有原始结果、失败现场和修复来源，不重训、不重评、不关机、不归档对话。本次g11定时跟进已在验收后删除，原配置与删除回执另存保留。下方“同步中”各段是带日期的历史里程碑。[最终验收](outputs/experiments/europe_tmg_20261001/delivery/final_root_acceptance_20261002T164041_279791Z.json) · [全量归档回执](outputs/verification/g11_remote_launch_20261001/archive_receipt_g11.json) · [正式报告12](docs/第二阶段/实验报告/12_温度记忆与干扰图独立场景对照.md)。

**G11训练、共同评价及正式报告已完成，完整资产本地归档中（2026-10-02晚）：**21:59 API与22:04独立终点复核确认9/9训练完成，总3100000交互；T各150000/148001更新，M/G各400000/398001更新，原worker均退出，固定终点policy/metadata/results SHA与controller回执一致。M2已于21:42自动回原20424归属，保留原ID/科学85f/guard及学习状态；11模型Jan24共同100场景×3动作、固定/贪心各100次均完成，共3500分配。报告前后两次独立审计通过350000终态束，9张中文曲线和15个输出已本地逐SHA核验。[正式报告12](docs/第二阶段/实验报告/12_温度记忆与干扰图独立场景对照.md) · [训练和评价原始核验](outputs/experiments/europe_tmg_20261001/preflight/terminal_and_common_raw_20261002T140424_906058Z.json) · [报告本地核验](outputs/experiments/europe_tmg_20261001/verified_report_20261002/download_receipt.json)。

**分模块结论：**固定α的T0相对同15万步J0，SGM+0.1315且100场景均提高、功率−362.48W，但SI−0.963个百分点；T1相对J1的SI+2.678个百分点，仍未优于原加性Q参照。M1相对M0的SI几乎相当（−0.069个百分点），功率−153.03W；GRU/因果Transformer未提升本轮SI。G2相对G1的SI+0.615个百分点、SGM+0.0041，但功率+107.75W。仅限单seed51001、单Jan24日期及既定终点，不能称多种子稳定性或跨日期泛化已证实；不启动额外实验。

报告首attempt在共享数据根解析及CSV/NPZ来源层次比对失败，已保存现场后修复运维审计；从guard双端archive定位数据根，核原来源/导入清单及100×2场景七个NPY成员字节一致，不更改公式、容差、场景或权重、不重评。当前collector43290已正常完成退出，旧失败39184仅为历史。原资产归档器36008在进度JSON原子替换遇Windows读锁退出，已保留11175文件及错误现场；随后补有界重试、非关键进度写失败继续以及远端POSIX路径兼容修复，4项短测通过。当前唯一v3归档器36948（UTC14:49:36.200895，脚本SHA6eff51f8…）已重核并超过旧停点，继续逐SHA同步61,144文件、20,917,500,639字节；进度以[总归档状态](outputs/verification/g11_remote_launch_20261001/archive_mirror_status.json)为准，**尚未宣布全量资产本地归档完成**。补充逐束CSV/JSONL的22数据集、620000终态束已本地归档，并独立逐行复算/CSV对照通过；本步历史缺项仍为null。[补充日志验收](outputs/experiments/europe_tmg_20261001/beam_service_logs/root_independent_export_audit.json) · [修复后报告完成凭据](outputs/verification/g11_remote_launch_20261001/verify_repaired_report_completion_20261002T141103_918965Z.json)。g11继续每小时简报归档进度，直到全量回执、正式文档及远端备份复核完毕才结束跟进。下一次不要再按已退出worker/controller/returner的旧PID恢复训练或重复评价。

**G11训练结束后的逐束供需日志已接入等待流程（2026-10-02）：**用户要求记录D、C、min(C,D)、未封顶C/D、SI和SGM，并区分本步与终值。当前85f冻结worker保持原样；已有评价结果的终态beams含D/C/SI/SGM，可保真导出。历史抽样训练与旧评价trace缺少本步容量，明确null和missing_reason，不用终值或封顶SI补造。独立导出程序17:22在20418启动PID16827/出生ticks752694431，本地逐SHA补充归档器17:23启动PID2392；均只读已有结果，不进行模型推理或新增分配。先等九训练完成及M2回迁原归属，导出Jan15九终点测试2700分配/270000束；再等原Jan24报告和独立审计完成，导出原3500分配/350000束。两阶段分开，共22数据集、620000终态束，不混成新的训练或考核预算。

输出为远端`/home/u2024110224/second_stage_satellite_rl/g11_beam_logs_20261002`、本地`outputs/experiments/europe_tmg_20261001/beam_service_logs`；与原总归档目录隔离，避免扫描正在写的补充文件。当前状态为等待，**尚未生成正式全量日志**；原G11交付门槛外还须核补充日志completion与本地archive_receipt的全SHA。6项数据导出检查在本地DQN和远端均通过，使用已完成T0的真实100束样本，另核过供、step/final不同、未服务、输入变更拒绝和幂等发布。后续新任务的完整逐步/全束终态记录已由独立分支合并main（0a6c39a、9f87a49），原数值/RNG对照与页面CSV/JSONL下载核验通过；全量432项中2跳过、最终补丁23项聚焦通过，未测全程吞吐，也未替换G11运行源码。[远端启动与测试](outputs/verification/g11_beam_service_logging_20261002/remote_launch_receipt.json) · [本地归档启动](outputs/verification/g11_beam_service_logging_20261002/local_mirror_launch_receipt.json) · [自动跟进精确回读](outputs/verification/g11_beam_service_logging_20261002/automation_beam_log_receipt.json) · [记录口径问答](Q&A/工程维护/训练日志与进度监控改进建议.md)。本段为记录方案里程碑，训练实时进度仍以latest_verified_snapshot为准。

**G11 M2分卡迁移已完成并续训（2026-10-02，北京时间15:49 API、15:50进程复核）：**M2原ID从262800交互/260801更新的完整断点迁至20418独立archive，经目标工作台同ID恢复；M3原worker在20424持续运行。当前M2为267050/265051、M3为273400/271401，均training；7/9完成，累计2840450/3100000交互（91.63%，不含验证/测试）。M2的网络、优化器、20000目标＋15历史回放、全部RNG及场景顺序在路径白名单转换后深比较相等，原种子51001、40万总预算、科学85f与global_guard不变；source_root仍为共享/home中的原20424冻结snapshot。源20424不再调度M2，404是有意隔离，不得恢复历史副本。

迁移中首个目标启动因同一步诊断快照已有旧来源路径退出，停点仍262800且零新增更新；原快照已完整保留，目标重建后16类科学字段与回放样本相等，再经同ID恢复。历史failure保留，不代表当前故障。新M2 worker6400/出生ticks752067759、原M3 worker573/751678235、returner6000、双端sidecar5137/5831与本地watcher30664均复核身份；两GPU UUID不同。15:44–15:45约60秒观察合计12.38更新/秒，迁移前短窗7.35，约1.68倍；不同训练位置、短窗计数，不能外推全程净节省或模型效果。[根验收及新原始入口](outputs/experiments/europe_tmg_20261001/migration_m2_20261002T070813_436244Z/root_migration_verified_snapshot.json) · [交接与来源链](outputs/experiments/europe_tmg_20261001/migration_m2_20261002T070813_436244Z/migration_live_handoff.json) · [短窗吞吐复算](outputs/experiments/europe_tmg_20261001/migration_m2_20261002T070813_436244Z/root_throughput_comparison.json)。

**当前接手方式：**M2看8766、M3看8767；两前端已实际验证任务列表、211维/K16与更新增长，无需改前端代码。g11仍ACTIVE每小时简报，已精确回读新路由。优先用本地base解释器执行`outputs/verification/g11_remote_launch_20261001/m2_migration_local_watcher.py --read-only --once --snapshot`，脚本SHA为37d65b27a6faeeeaa4084e3c9e9f22d97bb20c697be0446bdcb83d8856e6810a；不用旧固定20424路由读M2。原controller/collector及本地mirror/archive在此期间受控暂停，持久returner、两sidecar和本地watcher负责M2完成/终点SHA/worker退出核验后的自动回迁及原流程重启。Jan24共同评价须等九训练完整完成且M2回原20424归属；自动回迁、共同评价和终点归档尚待执行。两实例共享/home但不共享loopback，跨端动作经各自sidecar与唯一请求ACK完成。实际阶段以运行清单migration_context和本轮远端migration_state为准；旧helper暂停不当新故障。[冻结合同](outputs/experiments/europe_tmg_20261001/migration_m2_20261002T070813_436244Z/migration_contract.json) · [自动跟进精确回读](outputs/experiments/europe_tmg_20261001/migration_m2_20261002T070813_436244Z/automation_migration_final_receipt.json)。

**G11双GPU分配询问（2026-10-02，仅只读分析与建议，未迁移）：**14:50–14:53分别短时采样各25点，20424上M2/M3均在训练，共用GPU平均利用率96.72%（93–100%）、显存6315MiB；20418样本为0%、无GPU计算进程。支持尝试两独立实验各占一张GPU以提高总体吞吐，但两窗口非同时采样，未测迁移后速度，不能承诺两倍或给净节省时间。当前工作台resume仅管理本实例archive，G11 guard/controller/report也绑定M组到20424；规范迁移需保原ID和完整断点、隔离调度，并处理评价归属与校验，不能直接另开worker。建议优先M2→20418、M3留20424，实施方案与实际提速仍待验证。本次没有停止、迁移、改配置或新增训练，现有续训和小时跟进继续。[原始资源样本](outputs/experiments/europe_tmg_20261001/resume_20261002T062332_824457Z/gpu_two_instance_readonly_20261002T065309_270211Z.json) · [复算与限制](outputs/experiments/europe_tmg_20261001/resume_20261002T062332_824457Z/two_gpu_allocation_analysis_20261002.json)。

**G11重启后已恢复续训（2026-10-02，北京时间14:26恢复、14:35根运行核验、14:36镜像核验）：**用户明确“开始吧”，本段解除下方历史暂停状态。M2/M3经原工作台同ID从249600/247601、252400/250401完整断点恢复，training_resumed事件的配置哈希和场景顺序与原训练一致；新worker为574/751678237、573/751678235，实际DQN/CUDA与85f冻结快照路径通过核验。14:35已到251150/249151、254550/252551并持续更新，仍各40万总预算；七已完成组保持不变，Jan24考核继续等待两组完成。两实例重启后仍为独立4090D，实际GPU UUID重分配已记入恢复回执，科学源码/global_guard未改。两工作台、controller、collector、隧道、镜像和归档等待器已按新PID出生登记接续；14:36本地镜像九任务SHA检查恢复，旧job字节已保留且仅接受工作台新增resume:true字段。g11自动跟进已ACTIVE并精确回读，继续每小时简报。后续从运行清单latest_verified_snapshot及current_helper_registry.json读取新身份，helper采集改用read_helpers_from_registry.py（本地base解释器），勿再使用旧固定PID脚本。[恢复交接](outputs/experiments/europe_tmg_20261001/resume_20261002T062332_824457Z/resume_final_receipt.json) · [根验收与当前快照](outputs/experiments/europe_tmg_20261001/resume_20261002T062332_824457Z/root_resume_verified_snapshot.json) · [自动跟进回读](outputs/experiments/europe_tmg_20261001/resume_20261002T062332_824457Z/automation_resume_final_receipt.json)。

**G11按用户要求暂停，准备远端重启（2026-10-02，北京时间14:03停止请求、14:08退出复核）：**因断网后需要重启服务器，已通过原20424工作台8771的停止API优雅停止M2/M3，原worker均退出。M2停249600交互/247601更新，M3停252400交互/250401更新；完整checkpoint含五网络、四优化器、序列经验池及K16历史前缀、全部RNG、下一回合与场景顺序，配置/来源核验通过；两checkpoint及job/config/status/source_manifest均已本地备份并逐SHA复核。其余七组已完成，未重训或改动第一阶段。g11自动跟进已PAUSED并回读，Jan24共同评价未提交。**保持暂停；用户告知重启完成后按本次授权从两原任务ID续训到各40万总预算。** 恢复前核新主机/进程身份、DQN/CUDA、冻结85f与断点SHA；controller遇stopped会进入needs_diagnosis，不自动resume，重启后应按原guard及请求ID接续，不能据此另建训练。断点审计曾将20015行误当超出20000容量，已按SequenceReplay允许20000目标＋15历史行更正检查，原诊断留存，断点未改。[暂停状态入口](outputs/experiments/europe_tmg_20261001/pause_20261002T060300_404442Z/pause_state.json) · [恢复交接](outputs/experiments/europe_tmg_20261001/pause_20261002T060300_404442Z/pause_final_receipt.json) · [根SHA与完整性复核](outputs/experiments/europe_tmg_20261001/pause_20261002T060300_404442Z/root_pause_verification.json)。代理没有重启服务器或发出恢复请求。

**G11 M0/M1完整完成（2026-10-02，北京时间11:48训练观测、11:51终点复核）：**两组均已`completed`，400000交互/398001更新，原worker正常退出；各自policy/model/results/evaluations SHA与两controller回执一致，seed51001/冻结85f/固定终点、验证2580条及100场景×3次测试齐全。M0为207维/14令牌，M1为211维/15令牌并包含上一实际动作。当前7/9完成，仅M2/M3训练，分别219800/222550步；累计2742350/3100000步（88.46%，不含验证和测试）。两端后台与镜像身份核对正常，Jan24共同评价尚未提交，预算与冻结源码未变。[有效快照](outputs/experiments/europe_tmg_20261001/preflight/root_verified_snapshot_20261002T035325_408454Z.json) · [M0/M1独立终点核验](outputs/verification/g11_remote_launch_20261001/m01_terminal_compact_20261002T035127_320177Z.json)。

**G11 M0/M1达到40万步（2026-10-02，北京时间10:48核验）：**两组均400000交互/398001更新，当前为`evaluating`，原worker仍存活，正在进行既定Jan15内部终点测试，尚未计为completed。T0/T1/G0/G1/G2五组保持完成，M2/M3分别206150/209650步并持续更新；累计2715800/3100000步（87.61%，不含验证与测试）。两端controller仍等待四个M组完整结束，Jan24共同独立考核未提交。实际PID/出生、DQN/CUDA冻结源码及helper身份/guard均核对通过，本轮无训练或配置变更。[有效快照](outputs/experiments/europe_tmg_20261001/preflight/root_verified_snapshot_20261002T025020_260874Z.json)。

**G11 G1完整完成（2026-10-02，北京时间09:47训练观测、09:48终点复核）：**G1已完成400000交互/398001更新并正常退出；固定终点policy/model/results/evaluations SHA及两controller回执一致，seed51001/冻结85f、2946维/122令牌/逐束编码、2580条验证与100场景×3次终点测试均核验通过。当前T0/T1/G0/G1/G2共5/9完成，M0–M3四组持续训练；观测累计2654550/3100000步（85.63%，不含验证与测试），Jan24共同评价未提交。两端后台身份、源码SHA与本地镜像核对正常。只读helper脚本在本地DQN中因缺psutil提前退出，已用原有运维解释器完成补核；正式训练仍为DQN/CUDA，未安装包或改动训练。后续该helper脚本使用`C:/Users/Admin/anaconda3/python.exe`，训练快照脚本继续用DQN。[有效快照](outputs/experiments/europe_tmg_20261001/preflight/root_verified_snapshot_20261002T015022_338815Z.json) · [G1独立终点核验](outputs/verification/g11_remote_launch_20261001/g1_terminal_compact_20261002T014854_564684Z.json)。

**G11 G2完整完成与监控取证更正（2026-10-02，北京时间08:45训练观测、08:54独立复核）：**G2已`completed`，400000交互/398001更新，原worker退出；独立核对final policy/model、results/evaluations SHA与两controller回执一致，seed51001/冻结85f/固定终点、Jan12验证2580条及Jan15测试300条齐全。T0/T1/G0/G2共4/9完成，其余五组继续训练或周期验证；08:45累计2556450/3100000步（82.47%），Jan24评价仍未提交。08:44轮本地Luna汇总混入旧来源路径、helper时间和计数，已拒用并保留；本轮采用实际API/SSH及新独立helper/终点原始证据从头重建，未操作训练进程或冻结源码。g11自动跟进已更新为原始来源与时间校验、每次简报不变并精确回读。后续增长基线使用运行清单`latest_verified_snapshot`，不能用被拒摘要。[有效快照](outputs/experiments/europe_tmg_20261001/preflight/root_verified_snapshot_20261002T005730_267586Z.json) · [独立原始核验](outputs/verification/g11_remote_launch_20261001/helper_live_raw_20261002T005446_074844Z.json) · [汇总拒用记录](outputs/experiments/europe_tmg_20261001/preflight/rejected_summary_20261002T005505_613096Z.json) · [自动跟进修正回执](outputs/verification/g11_remote_launch_20261001/monitoring_provenance_repair_20261002.json)。

**G11 G0完整完成（2026-10-02，北京时间06:45观测、06:46终点复核）：**G0已`completed`，400000交互/398001更新，原worker正常退出。final的policy.pt/model.json、results.json和evaluations.jsonl SHA已核，元数据为seed51001/冻结85f/固定40万步终点，Jan12验证2580条、Jan15内部测试300条齐全，两controller完成回执匹配。现在T0/T1/G0共3/9完成，其余M0–M3、G1/G2六组持续训练；观测累计2311650/3100000步（74.57%，不含验证和测试）。Jan24共同独立考核尚未提交，科学配置和预算未变。证据：[本轮更正后的完整核验摘要](outputs/experiments/europe_tmg_20261001/preflight/g11_hourly_readonly_20261001T224905_248625Z.json) · [根复核](outputs/experiments/europe_tmg_20261001/preflight/root_hourly_review_20261001T224953_401020Z.json)。原首份摘要继承的G0旧阶段与controller旧别名字段已保留并明确更正，实际API/SSH及终点证据未改写。

**G11首个40万步节点（2026-10-02，北京时间05:44核验）：**G0已达到400000交互/398001更新，当前为`evaluating`，正在执行既定Jan15内部终点测试；真实worker仍为PID6448/出生ticks743734948，尚未计为completed。T0/T1保持完成，其余M0–M3、G1/G2六组继续训练，九组累计2185000/3100000步（70.48%，不含验证与测试）。DQN/CUDA、冻结85f源码与原任务身份核对一致。Jan24共同独立考核仍须等待全部九组完整终点；当前2/9任务完成，不依据达到训练预算提前宣告评价或模型效果完成。证据：[实际API](outputs/experiments/europe_tmg_20261001/preflight/hourly-20261002-0543_20261001T214437_357289Z.json) · [SSH及进程核验](outputs/verification/g11_remote_launch_20261001/live_snapshot_20261001T214449_771059Z.json)。

**2026-10-02用户进度简报偏好更改：要求改为每小时必报，自动化实际保存结果以本次操作凭据为准。** G11 自动化更新回执确认任务仍为 `ACTIVE`、原频率与九组范围不变，并已改为每次运行后在对话中发送简短中文进度简报（包括正常无变化）；见[更新与精确回读回执](outputs/verification/g11_reporting_preference_20261002/change_receipt.json)。本条记录的是汇报偏好和自动化提示，不更改下方实验训练事实。

**G11 T两组完整结束、G1已接入（2026-10-01，北京时间21:29观测、21:33终点复核）：**T0/T1均为`completed`，各150000交互/148001更新，原worker正常退出；final目录的policy.pt/model.json与results/evaluations SHA、元数据seed51001/冻结85f/固定终点、Jan15内部测试各300条及Jan12验证各1080条均已核验。G1已按原任务ID从队列启动，PID27640/出生ticks745261874，DQN/CUDA及冻结包路径一致，21:29为24100步/22101更新。当前2/9完成、其余7组运行，无排队或新故障；九组共967950/3100000步（31.22%，不含验证/测试）。两controller均已登记T0/T1完成并等待剩余七组，Jan24共同考核尚未提交。未修改科学源、任务、种子或预算，尚未形成正式模型效果排名。入口：[本轮实时与终点SHA证据](outputs/experiments/europe_tmg_20261001/preflight/g11_hourly_readonly_20261001T133632_381538Z.json) · [真实API快照](outputs/experiments/europe_tmg_20261001/preflight/hourly-20261001-2128_20261001T132906_063676Z.json)。

**G11首个训练预算节点（2026-10-01，北京时间20:31根核验）：**T1已达150000交互/148001更新，当前仍为`evaluating`，正在进行预先安排的Jan15内部终点测试；尚未核齐最终metadata/results，不计作completed。T0为145600/150000步，其余六个已运行长组持续推进，G1仍排队。九组本次共821950/3100000步，completed仍为0/9。这不是Jan24共同独立考核开始；两个controller仍等待九组完整终点，未新增独立评价任务。worker/辅助进程身份、冻结85f源码和guard均保持，未停止、恢复、改源或增加预算，尚无正式模型效果结论。入口：[根实时API](outputs/experiments/europe_tmg_20261001/preflight/root-t1-budget-milestone_20261001T123118_343660Z.json) · [本轮SSH/进程核验](outputs/experiments/europe_tmg_20261001/preflight/g11_hourly_readonly_20261001T123012_238990Z.json)。

**G11九组已在双4090D启动（2026-10-01，北京时间16:31根核验）：**T0/T1各15万步，M0–M3/G0–G2各40万步，共9组310万交互，seed51001。八个真实worker均已有CUDA参数更新，G1在20418按四路上限排队；合计43050步，0/9完成，不能提前判断模型效果。科学冻结`85f667cd5731e284c19b2590b5dcf5308a58f234`；本地及两远端各自425项全量检查通过（平台跳过见日志），默认J0/J1数值等价、M完整恢复、G物理关系及前端贯通通过。20418承担T/G，20424承担M，两GPU独立但共享/home，repo/archive/锁已隔离，前端8766/8767。Jan24预选100场景及796场景50/40dBi合同通过；九终点＋旧J0/J1的15万步参考＋两基线共3500分配尚待执行。guard SHA`139197aced9821b861d7faaf624cd983505dfea11ef4af895e2a4fc176e946c4`，两controller/报告collector/本地SHA镜像与完整归档等待进程均已启动，九训练全部完成才触发独立评价。g11每小时跟进ACTIVE，正常无重要变化静默，旧g10保持PAUSED；不关机、不开额外组/种子/预算。入口：[执行说明16](docs/第二阶段/方案/16_温度记忆与干扰图双实例实验说明.md) · [运行清单](outputs/experiments/europe_tmg_20261001/run_manifest.json) · [根PID与更新核验](outputs/verification/g11_remote_launch_20261001/live_snapshot_20261001T083145_176520Z.json) · [持久启动交接](outputs/verification/g11_remote_launch_20261001/final_launch_handoff.json) · [辅助进程补核](outputs/verification/g11_remote_launch_20261001/final_launch_handoff_helper_identity_addendum.json)。

**P08后续方案（2026-10-01，仅建议）：**G10完成后，建议先读取既有1/3/5/8/10/15万步学习快照，拆分联合Q目标的价值项与熵项；再新增原分头策略下“加性Q＋固定α0.01”和“完整联合Q＋固定α0.01”两组，各15万步，候选新增30万交互。满足默认分支等价、初始化和场景/验证条件后，复用历史J0/J1同期节点，不把15万与40万混排。固定原初值仅隔离温度自适应，不代表最佳温度；后续候选为加性Q＋槽数—功率交互，共220输出，不在30万预算中。Jan23已参与开发，正式确认需新留出日期。本次只核文件清单、源码格式、证据与方案，未运行模型探针、训练/评价、改前端或恢复g10。入口：[方案15](docs/第二阶段/方案/15_联合Q失稳诊断与结构化价值改进方案.md) · [后续问答](Q&A/算法与训练/Transformer记忆与算法创新.md#2026-10-01追问后续怎么做)。

**G10组间差距解释补充（2026-10-01）：**用户追问J0/J2为何优于J1/J3。已核四组共同分界是加性/完整联合Critic，动作空间仍最多10000，改变的是每Critic120分量输出→10000组合输出及直接监督共享方式。原始日志新增窗口统计显示J1/J3前段批熵长期低于目标，后段α/Q/TD同时大幅漂移；加性结构更易学稳是本轮证据支持的解释，稀疏直接监督与温度/自举反馈是机制假说，不把α定为唯一初因、也不推导83倍样本需求。日志α×当前熵仅作数量级代理，非下一状态TD目标精确分解。仅读取既有代码/日志和维护问答，未改算法/前端、未新增训练/评价或恢复g10跟进。入口：[解释问答](Q&A/算法与训练/Transformer记忆与算法创新.md#2026-10-01追问为什么j0j2比j1j3好这么多) · [窗口证据](outputs/verification/g10_completion_20261001/why_joint_q/learning_windows.json)。

**G10全部完成并通过最终核验（2026-10-01 14:43）：**J0–J3各40万步/398001更新，四固定终点同Jan23的100场景×3动作及两基线各100，共1400分配/140000逐束轨迹，全部completed。报告v2、独立审计和根标准库逐束重算一致；20618文件/7.183GB双端SHA归档、四规范模型、6个中文补图/统计文件以及Jan23原包1.504GB双端SHA均完成。科学冻结cc27不变；仅修报告把final test行误当validation的分类错误，原失败和运维史保留，无重训/重评。J0/J1/J2/J3独立SI=93.11/83.54/93.41/87.41%，SGM=.5518/.4237/.5643/.3530；J2平均SI比J0高0.295个百分点、SGM高.01251，但全100束基本服务88%→85.33%，仅小幅取舍。完整联合Q的J1/J3后段价值误差与温度持续增大、回报退步，不支持替代J0；单seed/单日期/参数量差异限制保留。正式报告11含训练原始及100回合均值、43节点原生验证回报、对数学习曲线、共同考核和资源账。g10已通过应用工具暂停并回读PAUSED，中文提示无乱码；最后额度9%、1张重置卡未用，不关机或归档本对话。入口：[正式报告11](docs/第二阶段/实验报告/11_联合动作四组独立场景对照.md) · [最终资产交接](outputs/verification/g10_completion_20261001/final_archive_handover.json) · [根最终核验](outputs/verification/g10_completion_20261001/root_final_delivery_verification.json) · [规范模型](models/runs/europe_joint_action_20261001/) · [运行清单](outputs/experiments/europe_joint_action_20261001/run_manifest.json)。

**G10训练与共同评价已完成，报告统计修复中（2026-10-01 14:16）：**根实时API及远端逐文件核对，J0–J3各400000步/398001更新、4000回合完成，四worker正常退出；四个已登记Jan23评价也全部completed，J0为500分配（含两基线）、其余各300，共1400/140000逐束轨迹，终点policy与metadata SHA一致。controller因报告器把训练`evaluations.jsonl`中的正常final test300行也要求为validation而退出，错误为`J0: unexpected validation row`；实际验证为43×20×3=2580行，另final test300，共2880行。原失败现场、脚本、不可变输入与SHA已保留；仅修报告/运维接续，禁止重训或重复评价，科学冻结cc27不变。逐项报告审计与本地归档尚待完成，暂不发布模型优劣结论。入口：[根八任务完成核验](outputs/verification/g10_completion_20261001/root_api_completion_milestone.json) · [原报告失败](outputs/verification/g10_completion_20261001/failure_preserved/report_failure.json) · [终点及原始数量核验](outputs/verification/g10_completion_20261001/completion_probe_initial.json)。

**G10跟进提示编码已修复（2026-10-01 10:15）：**上次更新自动化时，经Windows子进程读取非ASCII JSON的中文发生有损转换，导致本次提示主体及名称乱码；原任务约束在已保存的Unicode提示中完整保留。本轮恢复原中文并保留09:12监控修复条款，ASCII JSON/显式UTF8回读与期望逐字符相同，替换字符为0，g10仍ACTIVE、每小时/正常静默及严格剩余<3%用卡规则不变。新只读脚本按已核SHA真实执行，唯一快照确认J0/J1/J2/J3为230900/232900/228100/232750步，总924650/1600000，四组持续更新，controller/mirror正常，无训练操作。本轮额度16%，未用卡。入口：[有效快照](outputs/experiments/europe_joint_action_20261001/preflight/heartbeat_live_hourly-20261001-1012_20261001T021344_832598Z.json) · [根回读核验](outputs/experiments/europe_joint_action_20261001/preflight/root_review_20261001_1012.json)；乱码原配置已留在运维证据目录，未掩盖历史。

**G10监控取证误报已排除（2026-10-01 09:22）：**09:12心跳记录混入08:13旧观测并误覆盖原08:11本地快照，不能据此认定训练停滞。09:14根独立no-cache API与09:17新API/SFTP核对四组实际为189500/190000/186000/190000步，更新187501/188001/184001/188001，原PID/出生tick、cc27源码、controller15798及mirror25220身份/SHA均正常；没有停止、恢复或修改训练。误写现场保留，原08:11完整字节未恢复，仅已核总558550步及四组数字可重建；旧响应进入新文件的具体路径因缺独立原始请求记录仍未确定。新增唯一UTC微秒文件名、拒绝覆盖、保留raw响应和新鲜度核对的只读入口`g10_live_heartbeat.py`，SHA89d8424b52a1c9501786da090485e0d85b757500fe467c96cb488348e2bf47d9，本地无网络安全自检通过；长验证/评价/正常退出交状态审阅，不据旧心跳直接判失败。g10跟进已更新为此入口，仍ACTIVE，训练/评价预算不变。本轮额度剩余17%，未用卡。入口：[诊断与限制](outputs/verification/g10_remote_launch_20261001/heartbeat_0912_diagnosis.json) · [新的有效快照](outputs/experiments/europe_joint_action_20261001/preflight/heartbeat_live_post0912-audit_20261001T011744_516361Z.json)。

**G10四组已在远端4090D真实训练（2026-10-01 05:13根API核对）：**J0/J1/J2/J3已到4500/4250/3850/3600步，更新2501/2251/1851/1601，真实PID15471/15590/15704/15831及出生tick核对稳定；四组207维14令牌、旧奖励/dropout0/shuffle/joint_feasible、seed51001各40万步共160万，初始温度沿G09实际0.01。科学冻结`cc27cd7f7a4725d217c2fce1feadfb2c3cbc8358`，source清单SHA`7d31929e8fd658f08dee82ab5d2fdccc8c15bd04cebf807540bc777b38464174`；本地及远端Git保留，新worker显式导入第二阶段冻结snapshot，第一阶段/旧G09不动。Windows403通过/skip2、Linux403通过/skip3，旧J0同场景动作/两次更新及optimizer/RNG全等、J四组本地/远端CUDA恢复与实际页面通过；这些不等于学习效果提升。05:13–05:14短窗口约54.3更新/秒，含5000步验证，GPU98–99%/约3164MiB，非长期吞吐结论。Jan23 staging-0独立100场景已导入`scenes_20261001_050411_a2df67d5`，首错误sidecar ID导入留预检历史，100数组未变；四固定终点后自动同100×3加两基线各100共1400分配，报告/独立审计/本地SHA归档已接持久后处理。g10每小时跟进ACTIVE，正常静默、里程碑与异常通知。用户额度规则严格剩余<3%才可使用已有卡，最近剩余20%、1张未用。学习效果及最终归档仍待完成。入口：[训练说明14](docs/第二阶段/方案/14_联合动作四组远端训练说明.md) · [G10清单与四任务ID](outputs/experiments/europe_joint_action_20261001/run_manifest.json) · [根实时核验](outputs/experiments/europe_joint_action_20261001/preflight/root_live_launch_check.json) · [运维证据](outputs/verification/g10_remote_launch_20261001/)。

**P07算法创新调研（2026-10-01，仅方案）：**用户要求在字段消融外研究Transformer记忆及根本性算法改进。当前代码与原论文核对：Transformer仅处理同一步状态令牌，无跨步骤记忆；t是静态场景内逐束分配，不是ADS-B真实时间。联合掩码已实现，但策略合法频率/槽数打分仍加性、功率独立，双Q各自三头相加。推荐优先J模块原/条件策略×加性/联合Q四组，207维旧奖励/dropout0/shuffle、候选seed51001各40万步共160万；历史记忆M四组与逐束干扰关系G三组分别列候选，不自动累计或替代P06。时序方案包含真实上一动作/有效位、同窗口GRU与因果注意力、连续回放与无未来泄漏；图方案要求同信息对照。定向原论文调研不证明首次创新或实际提升。尚未实现、未建任务或训练/评价，G09完成/跟进暂停保持。入口：[方案13](docs/第二阶段/方案/13_分配记忆与联合动作建模算法调研方案.md) · [算法创新问答](Q&A/算法与训练/Transformer记忆与算法创新.md) · [调研记录](outputs/verification/algorithm_innovation_research_20261001/)。

**P06下一轮建议（2026-10-01，仅方案）：**基于G09结果，推荐同4090D从零比较完整207维及分别去掉平均需求/处理进度/剩余需求的三种206维，共四组、seed51001各40万步，候选总160万步；旧奖励/dropout0/shuffle及真实维数不补0。四组只检验另外两字段存在时的单项作用；若要检验三项整体相对原状态的作用，204维第五组须单列，候选总预算才变200万。平均需求仍用于环境旧奖励，消融只删除观测；字段相关、必要投影参数和初始化差异不隐去。先诊断基本服务不足，再做固定验证及新日期共同考核；01-19已用于开发判断、01-22已用于第一阶段，候选01-23尚未核验可用性。新字段选择与步0/早期验证需前后端接入，目前未实现、未提交任务、未训练/评价或启用自动跟进，G09保持完成。入口：[方案12](docs/第二阶段/方案/12_207维全局信息消融与稳定性验证方案.md) · [问答](Q&A/算法与训练/模型是否学到了有效策略.md#2026-10-01追问根据g09结论下一轮实验怎么做)。

**G09已全部完成并通过独立复核（2026-10-01 03:57）：**五原任务各400000步/398001更新、4000回合、40验证节点，八个固定终点模型同01-19独立100场景×3动作，另固定/贪心各100，共2600分配均completed。全100束最终SI/SGM/基本服务、功率/槽数、2600评价记录与260000轨迹、权重与源SHA复算一致，科学训练ecb、评价/服务6042不变。N0/N1/N2/N3/N4的SI=86.67/91.56/91.67/89.40/90.40%，SGM=.5586/.5980/.5973/.5932/.5907；N2平均SI微高，N1基本服务99.65束、300次分配全100束达标73.33%（N2为98.64/44%），综合优先参照N1但非全指标最优。N0回报/服务在迁移前已退步，不能称五组均收敛。验证NPZ SHA差异经5×20数组及ZIP层证实只是Windows/Linux封装create_system标记，实际场景相同。单seed/单日期、自然编码参数量和N0/N1跨设备限制保留。规范模型/完整断点和日志、2.798GB全量运行资产已remote+local SHA归档；原始1.66GB ADS-B上游包本地已核，额外异地副本未建。最终报告、分面回报/验证及学习诊断曲线已完成；本轮没有新训/重评。g09每小时跟进已通过应用工具核验为PAUSED，页面与原始记录保留。入口：[报告10](docs/第二阶段/实验报告/10_欧洲自然维状态与独立场景对照.md) · [独立审计](outputs/verification/g09_completion_20261001/audit/audit_summary.json) · [全量归档](outputs/verification/g09_completion_20261001/archive_receipt.json) · [规范模型](models/runs/europe_natural_state_20260930/) · [实时完成核验](outputs/experiments/europe_natural_state_20260930/preflight/completion_check_20261001.json)。

**G09每小时进度跟进已按用户新授权启用（2026-09-30 22:03）：**自动化g09（本对话heartbeat，ACTIVE，每小时一次）每次输出五组维数/当前40万步/百分比/阶段及总体交互预算进度；此前的G08/G01/SAC跟进保持暂停。每轮刷新真实工作台/API/PID出生与更新，不以旧镜像或502误判训练。五固定终点及文件SHA齐后确保既定Jan19八模型同100×3、固定/贪心各100共2600评价执行，优先沿用远端现有v2后处理与已登记请求ID，避免重复；不加种子/组数/预算。评价阶段继续每小时报告，最终原始核验、中文报告/曲线/归档完成后暂停g09。训练和后处理依旧为远端持久进程；定时本地跟进需要本机和应用运行。入口：[自动化创建核验](outputs/experiments/europe_natural_state_20260930/preflight/hourly_followup_created.json)。

**G09本地通道已恢复（2026-09-30 21:26核对）：**新tunnel34164（21:22:11）、mirror3420（21:22:30）身份/脚本SHA已核，8766实际API恢复、镜像沿原账本14→15轮error空，三项隔离重连检查通过。原SSH banner/通道超时现场保留，辅助进程退出根因未定；本轮仅恢复本地运营通道，未停/重训/恢复远端任何训练。新API N0/N1各400000 completed，N2/N3/N4为129050/124650/120700步，PID仍49148/49926/49927且训练中；总交互预算58.72%，独立Jan19共同评价仍待三组终点。入口：[恢复凭据](outputs/verification/g09_remote_migration_20260930/local_helper_recovery_20260930_211837/recovery_receipt.json) · [恢复后API进度](outputs/experiments/europe_natural_state_20260930/preflight/user_progress_20260930_212646_root.json)。

**G09进度及本地查看通道异常（2026-09-30 21:14–21:17核对）：**远端工作台/SSH实际确认N0(204维)、N1(207维)均completed400000步/398001更新，worker退出，final policy/model.json/results.json齐且SHA已记录；N2/N3/N4为119550/115200/111500步（29.89%/28.80%/27.88%），真实PID49148/49926/49927及/proc出生tick匹配，科学源ecb/清单50f124未变。合计1146250/2000000交互步，57.3125%，不含验证/考核工作量。远端服务49918和后处理55124仍活着、脚本SHA匹配，独立Jan19评价任务尚未创建，phase waiting_five_fixed_endpoints，不能先排名两完成模型。本地隧道34444和mirror27268实际已死、8766本地拒绝连接/代理502；镜像旧步数落后且末次有SSH banner错误，不能当训练停了。已按既定授权派Sol只恢复本地辅助通道，远端训练/服务/源码/评价未动；恢复结果待实核。入口：[本次真实进度快照](outputs/experiments/europe_natural_state_20260930/preflight/user_progress_20260930_211738.json)。

**G09已迁往4090D并五路运行（2026-09-30 19:00核对）：**用户指定u2024110224@10.160.17.102:20418，BUPT_PASSWORD仅环境认证。五原ID、seed51001、每组40万总预算、旧奖励/dropout0/shuffle和自然维输入均保持。8765两本地worker已退出，N0/N1完整停点306200/300400、更新304201/298401；三零步队列经API取消，原日志/断点保留。远端独立archive以同ID接续，本次五组步数N0/N1/N2/N3/N4为335000/329500/30000/26500/25800，训/验证中且均真更新；科学worker仍ecb74e89e637dced320f2f8ea9892a55d28565d7、原源清单SHA50f124...，服务只以6042e715调度cap3→5（main7d2f133，16相关检查通过）。60.005秒生产窗口GPU均98.98%、峰99%、显存峰3761MiB，每组增650–700更新，不能当全程提速或学习结论。新01-19独立100场景已核并导入scenes_20260930_182924_9417de52，旧G07/A/B固定终点也已复制；共同2600评价仍待五训练结束。查看远端8766，不把8765停止历史误当远端停训。原本地后处理/监督已退出或身份核停，数据监督独立；远端v2后处理55124已核身份/脚本SHA/完整断点与非有限恢复门槛，local镜像v2 PID27268（独立remote_mirror）至少3轮且error空；旧helper退出已核，CJK报告小流水线通过。工程交接已完成，科学学习与评价待完成。入口：[方案11](docs/第二阶段/方案/11_4090D迁移与并行运行方案.md) · [验证报告09及曲线](docs/第二阶段/实验报告/09_4090D迁移与并行验证.md) · [任务来源与现态](outputs/experiments/europe_natural_state_20260930/run_manifest.json) · [最终交接](outputs/verification/g09_remote_migration_20260930/final_handover.json) · [根任务核验](outputs/verification/g09_remote_migration_20260930/root_final_audit.json)。

**G09自然维状态实验已授权并启动（2026-09-30）：**用户明确旧奖励＋dropout0＋每轮重排场景，按各档实际维数输入，不为网络对齐填0并开始训练。五组N0/N1/N2/N3/N4为204/207/246/746/2946维，令牌13/14/22/32/32，seed51001各400000步（共200万）。源码冻结`ecb74e89e637dced320f2f8ea9892a55d28565d7`，API8765服务源固定`C:/Users/Admin/.codex/worktrees/natural-state-ablation/paper`、归档D:/paper，worker用同提交source_snapshots；运行期间勿修改该固定工作区src或切分支。最终全量356测试、五档真实CUDA更新、原13基线初始逐位一致及实际前端选择通过，效果仍待训练。N0/N1已真实更新、N2–N4排队，初始max-workers=2；服务PID30640、持久后处理33292及恢复监督31632的创建时间与脚本SHA已核；监督只管五原ID，完整断点同ID恢复最多两次，手动停止/取消及非有限学习错误不自动恢复。01-19独立包尚在断点下载，日期/100预选时刻/业务规则已先冻结；完整校验并API导入后才评价5新模型＋旧G07/A/B，同100×3及基线共2600次，不冒充旧01-18。P05门控211维/新奖励只是旧建议，本轮未执行；P04第一阶段不受此授权影响。入口：[方案10](docs/第二阶段/方案/10_欧洲自然维状态对照训练方案.md) · [正式任务清单](outputs/experiments/europe_natural_state_20260930/run_manifest.json) · [启动现场](outputs/experiments/europe_natural_state_20260930/preflight/live_launch_snapshot.json) · [根API复核](outputs/experiments/europe_natural_state_20260930/preflight/root_api_snapshot.json) · [交付核验](outputs/experiments/europe_natural_state_20260930/preflight/root_final_review.json) · [恢复监督](outputs/experiments/europe_natural_state_20260930/preflight/supervisor_launch.json)。后续先刷新真实API/PID创建时间和更新增长，不恢复已结束G08。

**S1-01覆盖图补充（2026-09-30）：**按用户要求，七组各同10时刻（idx5/15/.../95）生成70波束覆盖单图＋7组总览＋10跨组对照及本地可筛选图库；取自已审计的01-22共同考核原始终态，没有重训/新评价或按成绩挑图。真实0.25°GEO半角锥与球面近交点、欧洲/10°仰角截断，7000束几何QA和92文件SHA通过；七组同场景同坐标范围、全70同线性色标。P0展示十图中035/055/075不达覆盖门槛（图子集3/10不替代完整18/100），055图未覆盖26架、C1为3架。新的静态图库实际筛选通过，原工作台和第二阶段G09实验不变。入口：[报告03与全部图](docs/第一阶段/实验报告/03_七组十场景波束覆盖图.md) · [图库文件](outputs/experiments/first_stage_s1_01_20260930/coverage_maps_10_scenes/index.html) · [图及几何清单](outputs/experiments/first_stage_s1_01_20260930/coverage_maps_10_scenes/manifest.json) · [SHA/数值核验](outputs/verification/coverage_maps_20260930/download_check_receipt.json)。

**S1-01七组已完成并通过独立复核（2026-09-30）：**三PPO各409600交互/1600优化更新、四基线＋三模型各同01-22留出100场景，10工作台任务全部completed且worker退出；冻结源`b3a11c51e9da6612a43cb005a79c9e89a37e448d`未改。独立700场景复算、原始业务/NPZ/结果/权重SHA通过；初次956条告警仅float32用户覆盖舍入（最大2.98e-8，均在原数值契约），修正审计侧精度后0问题，原失败现场保留。R0/R1/R2/C0/C1/C2/P0需求覆盖99.8495/99.9299/99.8658/99.8849/99.9391/99.9077/99.4725%，CV0.907098/0.854229/0.912277/0.840619/0.805150/0.844620/0.794456；R/C六组100/100可行，P0为82/100且无空束，18不达标来自覆盖。P0负载较均匀但覆盖较低；本轮C1兼顾覆盖与负载，PPO微调未优于同初始化局部搜索。仅single seed51001/该留出日期，不泛化为稳定结论。三模型checkpoint/policy/model.json已本地保存9文件SHA一致；138项suite资产已归档，新修正审计另存且SHA通过，原始ADS-B/任务NPZ/源码快照本地完整备份仍有缺口。训练/评价/controller/审计/归档工作均结束，页面服务与SSH隧道保留供查看，不创建定时任务或新训练；第二阶段G09独立继续。入口：[正式报告02与曲线](docs/第一阶段/实验报告/02_七组覆盖规划独立场景对照.md) · [最终交付与10任务核对](outputs/experiments/first_stage_s1_01_20260930/final_delivery.json) · [通过审计](outputs/experiments/first_stage_s1_01_20260930/audit_precision_review/audit.json) · [原始结果](outputs/experiments/first_stage_s1_01_20260930/results/) · [本地权重](models/runs/first_stage_s1_01_20260930/)。以下启动/进行中条目保留历史时点，不作为当前运行事实。

**S1-01三训练完成、七组考核进行中（2026-09-30）：**R2/C2/P0三原任务均completed，各409600交互/1600优化更新/50rollout，原始完整回合2048/2048/4096；固定终点模型已导出，源`b3a11c51e9da6612a43cb005a79c9e89a37e448d`不变。原始训练/验证/负载/学习曲线已SHA核对并在本地归档，[报告01](docs/第一阶段/实验报告/01_七组覆盖规划训练记录与考核进度.md)附全部回合散点，避免将512环境同时终止的回合连成连续趋势。终点20场景开发验证：R2/C2/P0需求覆盖99.8820%/99.8846%/99.5569%，负载CV0.894887/0.828985/0.793118，可行20/20、20/20、19/20；这不是最终考核、不能说P0整体更好。七考核已同API建档，R0/C0各100完成，R1/C1局部搜索进行中，三模型评价排队；最新真实进度看[根API快照](outputs/experiments/first_stage_s1_01_20260930/root_current_snapshot.json)。controller22500与独立审计等待器29154持续后台，后者每30秒/最多2小时且结果齐才重算；异常不算通过。三模型checkpoint/final policy/model.json已同步本地并逐SHA核实；其余资产由隐藏归档器38144等待考核与审计后归档，最终ledger仍待生成，原始ADS-B/任务NPZ/源码快照本地备份缺口保留。完整效果待七考核、逐场景审计与正式结果更新，第二阶段G09独立继续。

**S1-01第一阶段七组正式任务已启动（2026-09-30 14:43）：**用户授权远端4090D与新增无基础纯PPO，范围R0/R1/R2/C0/C1/C2/P0；4基线评价＋3PPO各409600步/seed51001，共1228800。冻结源码`b3a11c51e9da6612a43cb005a79c9e89a37e448d`，本地D:/paper main与远端Git均保留；386项集成回归/3平台跳过，三PPO真实更新、P0空布局构建和工作台停止/恢复通过。API核R2/C2真实更新已到98304步/384次，P0排队；max-workers=2，两路总吞吐35499高于三路20866步/秒，实测GPU均79.5%/83.4%，不以占用率牺牲总吞吐。远端DQN3.11.11/torch2.3.1+cu121/CUDA12.1，实际包从冻结source_snapshots导入；server22188、controller22500与进程创建时间已核。01-08/09训练576、01-12验证20、01-22独立考核100，各初始化登记，共9套；P0每场景初始active0、2048候选上限。后台controller完成三训练后经同API提交七项共同考核，效果仍待完整评价及独立审计。以下P04是历史建议，不改变第二阶段G09。入口：[方案03](docs/第一阶段/方案/03_远端4090D部署与七组并行实验技术方案.md) · [正式计划与清单](outputs/experiments/first_stage_s1_01_20260930/) · [主任务API复核](outputs/experiments/first_stage_s1_01_20260930/root_api_launch_snapshot.json) · [386测试与源等价](outputs/verification/remote_stage1_20260930/integrated_source_equivalence.json) · [GPU与恢复证据](outputs/verification/remote_stage1_20260930/)。

**P04第一阶段首轮分组建议（2026-09-30）：**拟6个比较组：现有规则不微调/局部搜索/PPO（R0/R1/R2），有约束聚类不微调/同搜索/PPO（C0/C1/C2）。只有R2/C2为2次PPO训练候选，其他4组是基线评价；共用seed51001、100束/0.5°和业务/动作/评价条件。先规则侧工程跑通再接聚类侧，两PPO正式预算在几何及真实更新短核验后统一冻结；本次未授权预算、实现布局功能、创建任务或训练/评价。第二阶段P05是另一计划，不合并运行，G08既有完成状态不变。入口：[第一阶段方案02](docs/第一阶段/方案/02_基础布局与PPO微调首轮实验分组.md)。

**P05历史奖励失败后的优先级更正（2026-09-30）：**用户提醒融合奖励曾以448.25W得到SI35.73%，低功率伴随服务损失，不能将新奖励默认视为改进。现建议R0/R1旧奖励主线（A+B、204/211维，各40万步，共80万），新奖励R2须当前欧洲/联合掩码功率反馈及服务风险诊断后才考虑，R3仅为交互归因条件；四格160万为备选总框架，不是默认预算。历史固定频率槽的30W重放提高SI及原融合累计奖励，不能唯一归因罚项过强，也不证明重新训练能学好。新J归一化功率系数与旧融合相同，9项代数检查不等于功率学习验证。主线用固定验证旧奖励回报和终态服务验收，未实施、未创建任务；G08完成/跟进暂停与P04计划不变。入口：[方案09第11节](docs/第二阶段/方案/09_关闭dropout与随机场景后的稳定训练方案.md#11-历史融合低功率失败后的优先级更正) · [奖励问答](Q&A/算法与训练/当前奖励函数.md)。

**第一阶段分层PPO方案补充（2026-09-30）：**用户提出聚类或固定规则生成基础布局，再用PPO微调；已细化P04/B为第一阶段内部两层，基础规则冻结、PPO独立学习中心修正，第二阶段T-SAC仍独立分配频率槽与功率。首版100束/0.5°、固定轮询与相对初始布局的累计位移范围；终态覆盖门槛、全量重算用户归属、原始终态/回退和初始化/学习收益分开。现有规则训练需求覆盖约99.79%为576快照汇总，不是PPO效果。仅方案和问答更新，未实现布局环境、PPO掩码或前端任务，未授权新增预算或启动训练/评价。入口：[方案01的B方案](docs/第一阶段/方案/01_第一阶段波束覆盖规划与两阶段解耦方案.md#b-基础布局加ppo微调) · [问答](Q&A/环境仿真/第一阶段波束覆盖规划与两阶段衔接.md)。

**第一阶段编号更正（2026-09-30）：**按用户追问，第一阶段首个方案由原全项目序号09更正为本阶段01，文件名、正文标题、目录与相关链接已同步，后续本系列从02递增。第二阶段既有编号不变；仅文档编号更正，无算法、前端或实验任务变化。入口：[第一阶段方案01](docs/第一阶段/方案/01_第一阶段波束覆盖规划与两阶段解耦方案.md)。

**文档按阶段归档（2026-09-30）：**用户要求docs下分开第一阶段与第二阶段。覆盖规划方案归入 `docs/第一阶段/`，资源分配方案、报告与规范归入 `docs/第二阶段/`，第二阶段和共用资料原编号保留，第一阶段首个方案按用户更正为01；跨阶段方案04、工作台、工程文档与资料保持共用原路径。全仓库维护中的文档引用同步修正，冻结实验产物不改写。仅文档移动与引用检查，无算法、前端或任务变化。入口：[文档总目录](docs/00_目录.md) · [第一阶段](docs/第一阶段/00_目录.md) · [第二阶段](docs/第二阶段/00_目录.md)。

**G08状态空间表现解释补充（2026-09-30）：**C组不如204维的直接证据是后段价值误差、Q/target和温度明显增大；C组还保留dropout0.1与固定顺序，且13→32令牌编码与信息一起改变，不能纯归因于状态更全面。冻结代码确认未开放字段置零但对应令牌仍参加注意力、每10束压为一个令牌；这些表达是否造成退步仍是推测，未作新的因果对照。相同32令牌内部C4比C1在99/100场景SI提高、平均+2.51个百分点，但SGM下降0.0412。A/B分别有效仍限定单seed51001、40万步与当前欧洲考核，A+B组合尚未验证。仅读取既有结果和冻结源码、更新问答，无新训练/评价，G08与跟进状态不变。入口：[通俗问答与曲线](Q&A/算法与训练/模型是否学到了有效策略.md#2026-09-30追问为什么c组状态更全却不如204维版本) · [数值与源码指纹](outputs/experiments/europe_ablation_20260929/analysis/c_state_interpretation_20260930/evidence.json)。

**第一阶段方案审查（2026-09-30）：**已阅读开题报告第2.1节，结合现行欧洲环境和G08已完成结果形成P04四个覆盖规划候选方案；推荐有约束初始布局＋单智能体集中轮询修正，第一阶段以覆盖/负载独立训练，第二阶段按保存的覆盖方案分配频率槽和功率。首版100束固定0.5°仅验证中心，逐束宽度须另立物理/接口版本。仅方案与问答归档，算法未实施、效果待验证，未批准新预算或创建训练/评价任务；没有恢复已完成G08或自动跟进。核对正式清单完成状态及源码接口，未刷新工作台实时服务/进程。入口：[方案01](docs/第一阶段/方案/01_第一阶段波束覆盖规划与两阶段解耦方案.md) · [问答](Q&A/环境仿真/第一阶段波束覆盖规划与两阶段衔接.md)。

**频率槽粒度追问核验（2026-09-30）：**正式欧洲环境最小1槽25MHz＋整束5W；A的3641/30000条记录已为最小动作且最终容量仍超过需求（占全部12.14%、过供评分缺口18.65%），确认小需求存在粒度限制。A/B约64%的过供缺口仍来自多槽记录，不能仅按实际动作分类认定全部过供不可避免或策略错误。仅对实际单槽记录固定频点和其他束入射干扰，将本束功率解析降至5W后，A/B单槽记录的69.40%/64.63%仍过供；该检查不代表全体资源调整效果，未重放模型或新增评价。冻结评价源码a0556de与结果SHA核对；G08仍已完成、自动跟进暂停。入口：[报告08第5节](docs/第二阶段/实验报告/08_SGM低分的过供欠供诊断.md#5-追问是不是一个槽也太大小需求只能过供) · [统计和代码指纹](outputs/experiments/europe_ablation_20260929/analysis/slot_granularity_20260930/summary.json)。

**SGM低分事后诊断（2026-09-30）：**对G08已保存的01-18共同考核逐束容量、需求与SGM复算，七模型的评分缺口主要来自容量过供，占73.74%–85.20%；G07/A/B分别为85.20%/83.75%/84.18%。A/B虽改善，仍有38.45%/43.31%的记录容量超过需求2倍；也同时有24.54%/24.13%的束欠供，不能以统一减功率或槽数代替逐束调整。这里分析的是全部干扰后的最终容量，并未确定功率、槽数或干扰各自的因果贡献。仅CPU统计既有结果，没有新增训练、评价或前端操作，G08仍全部完成、自动跟进保持暂停。入口：[报告08与曲线](docs/第二阶段/实验报告/08_SGM低分的过供欠供诊断.md) · [原始统计和输入SHA](outputs/experiments/europe_ablation_20260929/analysis/sgm_supply_diagnosis_20260930/summary.json)。

**交付与运行结束（2026-09-30 02:20）：**正式报告07及独立审计已通过；本轮训练、评价和报告任务全部结束，`g08`自动跟进已通过应用工具暂停并核实为PAUSED，原两个历史跟进未恢复。用户要求任务结束后关机，将在保存最终交付清单与本地提交后安排正常关机；实际请求记录见本组`final_delivery.json`和`shutdown/`目录（只有记录存在且确认成功时才视为已安排，不把本文等同于已关机）。本轮没有追加下一阶段训练，关机不应触发新一轮自动恢复。

**G08全部完成并通过复核（2026-09-30）：**六个原训练任务均完成400000步/398001更新及各300条原历史考核；G07＋六模型的七项01-18共同独立考核均completed，模型各100场景×3动作，固定/贪心各100场景×1，共2300次分配。所有场景/模型SHA、逐束终态指标及逐步资源账已独立复算通过，报告与曲线已归档到[报告07](docs/第二阶段/实验报告/07_欧洲六组独立场景对照.md)。独立考核G07/A/B SI为89.92%/91.40%/90.99%，SGM0.4033/0.5741/0.5198；A平均功率1680.17W、B2164.73W，A平均服务/匹配/功率更好，B全100束基本达标70.00%（A59.67%）和实际槽数更好。C1/C2/C3/C4 SI为85.90%/85.12%/82.87%/88.41%，当前设置下均未超过G07；C4相对同结构C1提高2.51个百分点但SGM下降，C组学习仍明显不稳定。七模型无越界/冲突/裁剪/零容量；固定原规则越界16槽/裁剪4次，贪心冲突10槽/零容量2束，比较时已并列限制。单seed51001、合成通信需求与独立日期范围限制保留。报告收尾编码异常已修复并严格复核既有产物后发布完成状态，没有重跑训练或七评价；C2早期原生退出及恢复历史保留。后续新组合仅建议，未启动。入口：[最终任务清单](outputs/experiments/europe_ablation_20260929/run_manifest.json) · [独立审计](outputs/experiments/europe_ablation_20260929/analysis/final_audit/summary.json) · [收尾修复](outputs/experiments/europe_ablation_20260929/finalization_recovery_20260930/finalization_receipt.json)。以下带时间的运行中记录为历史过程，不应触发重复提交或恢复。

**G08新完成核验（2026-09-30 01:00）：**C3完整布局与需求组已完成400000步/398001次更新，300条原历史考核结果、实际100场景元数据、完成事件以及最终权重SHA均核验一致。目前A/B/C1/C3共4组完成；C4已达到400000步/398001更新，正在终点评价，不能计作已完成；C2训练到342300步/340301更新。C2/C4进程正常，helper8844存活、阶段`six_real_updates_verified`，共同独立考核任务尚未提交。源e25ae56、seed51001、预算与原任务ID未变，没有新故障。[本次快照](outputs/experiments/europe_ablation_20260929/preflight/heartbeat_20260930_0055.json) · [主任务完成核验](outputs/experiments/europe_ablation_20260929/preflight/root_C3_completion_20260930.json) · [C3固定终点权重](models/runs/workbench/train_sac_20260929_022430_295fe5ed/final/policy.pt)。四档状态的效果结论仍待共同评价，不以完成先后排名。

**G08完成与恢复里程碑（2026-09-29 15:55–15:57核对）：**A/B/C1均已完成400000步、398001次更新及各300条原历史场景终点评价；新增核验C1最终权重SHA与模型元数据相符。C2已于15:02:52以原任务ID和完整初始断点实际恢复，worker PID19732创建时间/`--resume`命令匹配，15:57已到32550步/30551更新，当前状态及异步写入无新错误；13:59“只恢复排队、尚未实际训练”的限制在本时点已解除，原生驱动退出根因仍未确定。C3/C4在15:55分别到99200/95300步并持续更新。当前活动worker为19732/5652/11756，helper8844存活、脚本SHA一致、阶段`six_real_updates_verified`，仍等待全部六组结束后执行01-18共同独立考核。六组源e25ae56、seed51001和各40万步预算未变。入口：[现场核验与C1产物](outputs/experiments/europe_ablation_20260929/preflight/heartbeat_20260929_1551.json) · [主任务确认实际恢复](outputs/experiments/europe_ablation_20260929/recovery_20260929_1350/actual_restart_verified.json) · [C1固定终点权重](models/runs/workbench/train_sac_20260929_022420_7df81644/final/policy.pt)。C1状态档位效果尚未作共同对照，不提前判定六组胜负。

**G08 A/B原考核阶段结果（2026-09-29）：**对已完成G07/A/B的同100场景×3随机动作原始结果逐束复算，场景/动作种子规则和约束口径已对齐。SI为88.93%/90.55%/89.50%，SGM为0.4149/0.5799/0.5343，平均功率2942.97/1726.25/2209.17W；A/B相对G07的SI改善为1.62/0.57个百分点，100场景配对SI分别81升19降、65升35降。A主要指标和拟合改善更明显，B使用槽数更少；两者总容量下降但按需求封顶的有效承载上升。尾5万步Critic损失中位G07约757/712、A约0.99/0.99、B约10.44/10.64，不能仅凭loss断言严格收敛。此为单seed51001、已有开发考核集上的阶段结论，**不是01-18新日期共同独立考核**，不证明泛化提升或A+B叠加收益。原始数据、学习/验证曲线、配置与来源限制见[报告06](docs/第二阶段/实验报告/06_欧洲AB阶段结果对照.md)。此次仅读取统计与绘图，C组继续运行，预算未变。

**最近用户进度查询（2026-09-29 23:32）：**A/B/C1已完成各400000步/398001更新；C2训练290650步（72.66%），C3训练357200步（89.30%），C4在350000步进行周期验证（87.50%，尚未完成）。六组累计2197850/2400000交互步，约91.58%的训练步数预算；不包含后续共同考核工作量。活动worker身份核对通过，C2/C3有实际更新增长、C4验证进程正常，当前无新错误。helper8844存活、脚本SHA匹配、阶段`six_real_updates_verified`，仍等待六组结束后执行01-18共同独立考核。[本次工作台快照](outputs/experiments/europe_ablation_20260929/preflight/user_progress_20260929_2332.json)。本次仅读取与记录进度，没有修改训练配置或新增任务。

**G08进度与恢复（2026-09-29 13:59核对）：**A关闭dropout、B随机场景均已完成400000步/398001次更新，各完成原100个历史考核场景×3次动作的终点评价，结果/最终权重SHA已核对；尚未进行01-18共同独立考核，不作优劣结论。C1为368950步，C3为32350步，C4为30000步（验证中），三者持续运行。C2曾于13:05启动后在0步以`0xC0000409`退出；Windows事件指向`nvcuda64.dll`，根因未定，不能归因为显存或异步采集。故障现场已保留，完整0步断点/科学配置哈希确认后，经工作台原ID恢复为queued，等空闲槽位后再核真实更新，尚未实际重启训练。源仍为e25ae56、seed51001、各40万步；未改变算法、环境或运行中源码。原helper24192因C2中断退出，已保存历史并以PID8844（13:55:33.186325）重启，继续等待原六组与后续共同评价。API10108仍在，活动worker为18676/5652/11756，均核对创建时间。入口：[完成与现场核验](outputs/experiments/europe_ablation_20260929/preflight/heartbeat_20260929_1350.json) · [恢复记录](outputs/experiments/europe_ablation_20260929/recovery_20260929_1350/recovery_summary.json) · [主任务复核](outputs/experiments/europe_ablation_20260929/recovery_20260929_1350/root_recovery_audit.json)。后续先查C2实际启动及helper状态，勿重复提交新训练。

**工程提速已完成并接续G08（2026-09-29 06:38）：**六个原ID已全部绑定`e25ae56cb55ec402e0ba96013e383ab3da324bf6`；A/B/C1从96700/96600/82300步完整断点继续，C2–C4仍排队，种子51001与每组40万步预算不变。24次稳态试验/12对数值、两布局真实短恢复/验证/保存及197项凭据检查通过；13/32令牌稳态吞吐2.66/2.46倍，纯算法参考额外观测耗时约0.7%/2.1%。正式共同训练窗口三任务总更新19.75→37.83次/秒（观察值，非全程同工作量）。新API PID10108（06:27:12）、worker PID13936/18744/18676、共同考核helper PID24192（06:32:06）已核验，派发保留释放，原helper19440/API16668已退出。更新/回合日志连续、六份来源链与科学配置通过，前端已验证异步选项与新源进度；[接续与原始证据](outputs/verification/async_telemetry_20260929/migration/final_handoff.json) · [主任务审计](outputs/verification/async_telemetry_20260929/migration/root_continuation_audit.json) · [设计报告](docs/第二阶段/方案/08_异步采集与训练吞吐优化设计.md) · [验证报告与曲线](docs/第二阶段/实验报告/05_异步采集与训练提速验证.md)。工程效果已验证，六组模型效果待共同评价；本对话`g08`每小时静默跟进至实质变化，两个历史跟进保持暂停。接手须刷新实际任务/进程创建时间，不重复迁移或提交。

**G08最初恢复记录（2026-09-29，北京时间02:26，升级前历史）：**用户重新说“开始”后，工程核验完成，6组均已通过工作台提交，冻结源码`48022f4499823def40d573ea9436e3f7bdbf87fa`。A/B/C1正在训练并完成真实GPU更新，C2/C3/C4排队，max-workers=3。此前01:29的暂停已解除，旧PID不再有效。[实际任务清单](outputs/experiments/europe_ablation_20260929/run_manifest.json) · [主任务启动复核](outputs/experiments/europe_ablation_20260929/preflight/root_launch_verification.json)。

**G08工程与数据核验（2026-09-29）：**四档同2946维/32令牌实现、前端选项与任务参数贯通完成；320项全量测试通过，原G07配置哈希/权重兼容、真实场景状态/奖励/资源一致性及代表GPU短更新通过。新01-18独立100场景数据与SHA核验完成，已导入为`scenes_20260929_022130_6c00e5a6`。1/2/3路同工作量短测选定3路并行。完整范围见[方案07](docs/第二阶段/方案/07_欧洲六组训练机制与状态对照.md)，效果仍待正式训练与共同评价。

最后核对：**2026-09-28，北京时间00:18–00:27**。维护方式见 [AGENTS.md](AGENTS.md)。这是已核对事实的摘要；涉及正在运行的任务时，接手者必须刷新实时状态。表中百分数经过四舍五入，精确值以各组原始汇总为准。

方案补充：**2026-09-28，北京时间00:45**。本次只核对训练场景代码、开题报告接口并新增P02数据泛化建议；未重新核验实时任务、未启动实验，以上运行状态核对时间保持原记录。

数据调研补充：**2026-09-28，北京时间01:08**。P02新增公开数据与合成业务方法调研；两份小型公开包完成格式检查，未导入正式数据、未改源码或训练，未刷新实时任务。

工程交付补充：**2026-09-28，295项最终回归于北京时间01:32通过**。P02欧洲独立环境、航空导入与前端实现完成，DQN/CUDA短真实更新通过。用户明确本晚不做完整训练，跑通后关机、明天再训；正式训练未启动，未创建定时任务。

最新G07完成核对：**2026-09-28，北京时间22:48。** 工作台API、逐任务状态与完成事件一致，任务于20:45完成400000步、398001次参数更新，原工作进程已退出。最终100场景×3次随机考核需求满足率88.93%；最终权重及元数据存在且SHA一致，冻结提交01457f245bf0ab0e23873616e44975693014e3d0。仅核对已有记录与汇总，未重新评价；泛化改善仍需受控对照，详见G07。

G07专项诊断更新：**2026-09-28晚核查，2026-09-29凌晨归档。** API确认任务仍为completed；新增全量日志分析、最终断点CPU只读探针与状态/奖励小例，确认价值拟合问题并形成P03。未恢复训练或改写权重，诊断产物及哈希见[清单](outputs/experiments/europe_adsb_20260928/diagnosis/diagnosis_manifest.json)。

G07基线曲线补充：**2026-09-29。** 用户后续授权的欧洲模型/固定/贪心同场景评价已完成，500次分配及两张曲线见[报告04](docs/第二阶段/实验报告/04_欧洲模型与基线逐场景对比.md)。模型SI88.94%、SGM0.4213；未启动P03新训练。

## 1. 先读这里

- **已执行实验范围：第一阶段覆盖规划、第二阶段频率功率分配与第三阶段功率细调，分别管理。** 第一阶段S1-01七组已完成训练/考核；第二阶段最新G11已完成九训练、共同评价、报告与资产归档。2026-10-03第三阶段S3-P01已完整交付；S3-P02七路线及耗时对照也已完成评价、审计、报告和归档，详情见页首。优先沿用“波束、业务需求、需求满足率、供需匹配程度”等用语。
- **历史参照：G01旧奖励＋有/无联合掩码。** 两组各40万步、终点评价、同场景基线和正式核验均完成；2026-09-27已补齐奖励及服务曲线报告。它是历史掩码证据，不是最新一轮结果。
- **G07训练及终点评价已完成。** 任务 train_sac_20260928_142100_38559f37，40万步；100考核场景×3次随机动作的需求满足率88.93%，平均99.24/100束达到10%基本服务，全部100束达标比例54%。最终模型已保存，未证明相较原数据的泛化改善。
- **G07异常已做专项诊断，后续范围已扩为G08六组。** 固定验证SI有小幅改善，但Critic终止Q拟合明显失准，训练/评估态dropout模式差已测得。P03原分阶段建议保留；用户已授权G08的dropout、顺序与四档状态实验，旧奖励不变，详见G08及方案07。
- 跟进自动化 `automation`（核验前端T-SAC掩码对照）实际配置为 **PAUSED**，已收尾。不要因旧心跳文本再次恢复已完成训练。
- **第二阶段G09、G10及G11已全部完成。** 报告10/11/12保留各自口径与限制，最新G11交付见页首最终验收。P05新奖励/211维门控等仍为历史候选，M组的211维是另一套实际动作字段，不恢复已完成任务或增加种子。
- **2026-09-28 P02→G07：**欧洲航空环境及前端已完成；用户授权的ADS-B场景已导入，一组训练及考核已完成。P01奖励对照及P02原/新数据两组对照仍未获本轮运行授权。
- 正式训练与配套长评价必须从**前端工作台或其同一任务API**发起/恢复；使用DQN/CUDA、冻结源码和后台子agent。Luna负责简单清点，Sol负责实现与分析；主任务汇总结果。已限定训练种子51001，不能擅自增加51002/51003。

**阅读顺序：**页首最新状态 → 本节 → 第2节共同口径 → 所关心实验组及报告/清单。第二阶段最新完成结果见[G11报告12](docs/第二阶段/实验报告/12_温度记忆与干扰图独立场景对照.md)，第三阶段已完成结果见[报告01](docs/第三阶段/实验报告/01_T0初始解的连续功率细调对照.md)，S3-P02见[报告02](docs/第三阶段/实验报告/02_七路线功率细调与逐场景耗时对照.md)及页首运行清单。P08方案15的固定温度部分已执行，其220输出结构仍为候选，P05方案09保留历史。第一阶段另见S1-01。历史奖励讨论见[当前奖励问答](Q&A/算法与训练/当前奖励函数.md)，训练数据见[训练场景说明](Q&A/环境仿真/训练场景与随机需求生成.md)。

## 2. 共同上下文与容易混淆的概念

| 项目 | 接手时要记住的含义 |
| --- | --- |
| SI | 以全部100束最终容量计算的平均截断需求满足率，含后续干扰与未服务波束；以各组报告的实际口径为准 |
| SGM | 当前代码的供需匹配评分，反映容量与需求接近程度；用户关注容量/需求比接近1，不能把代码SGM数值直接当作这个比值 |
| 基本服务 | 每束最终容量达到本束需求的10%；平均达标束数、每次100束全部达标比例必须分开 |
| 零槽与零容量 | 分配到正槽数、产生正容量、达到10%基本服务，是三件不同的事 |
| `legacy` / `corrected` | G03/G04里的SAC学习公式差异：目标熵与双Q聚合；不是两种奖励函数 |
| `legacy_repaired` / `fused` | G02/G01中的奖励预设：旧奖励修正功率记账 / 融合奖励；G01两组都用旧奖励 |
| 原奖励 | 阈值满意度项＋本束SGM－供需偏差罚－占用冲突罚，没有直接的发射功率成本项 |
| 原融合奖励 | `2Δ基本服务进度＋ΔSI总分＋0.2ΔSGM总分－本步实际功率/3000 W`；G02效果下降，尚未做有掩码重训 |
| 联合掩码 | 筛选频点和槽数的组合，同时保留后续最低资源、限制功率并配套动态目标熵；本轮结果是整套方案的效果 |
| 训练数据 | 本轮工作台固定100城市坐标；每回合重新抽100–1000 Mbit/s需求档位，再加标准差20 Mbit/s扰动，按需求降序；序号%8分组 |
| 业务可见与网络输入 | 用户确认本轮分配前已知全部需求；G01保留的204维仍仅直接包含当前束位置/需求、剩余功率、100槽干扰及占用，没有自动加入全部未来需求 |
| 随机范围 | 城市集合不变，无坐标扰动；需求和排序变化。最终考核也是同一城市库的新需求组合，尚未验证任意新地理布局或真实时序流量 |
| 预算与模型选择 | 正式模型一般各40万交互步，采用固定预算终点；历史组398000更新、工作台组398001更新，不要混写 |
| 单种子 | 仅训练种子51001；100场景×3动作采样不是300个独立训练种子 |

**各组的场景、环境、功率记账和学习实现不同，以下分组结果不能直接拼成排行榜。** 不同奖励的回报不能直接排名；基线是同场景评价，不是模型训练。诊断或短检查不等于正式学习效果。

## 3. 实验组总览

| 组号 | 实验与目录 | 实际范围 | 当前状态 |
| --- | --- | --- | --- |
| G09 | `europe_natural_state_20260930`：自然维状态输入 | 204/207/246/746/2946维；旧奖励/dropout0/shuffle，seed51001各40万步 | 五训练、八项01-19共同考核及两基线共2600分配、逐束复核与报告10/曲线/归档均完成；N1综合服务较好、N2 SI微高、N0后段退步 |
| G10 | `europe_joint_action_20261001`：联合动作表示 | J0–J3：原/条件策略×加性/联合Q；207维旧奖励/dropout0/shuffle，seed51001各40万步 | 四训练、1400共同分配、审计/报告11/曲线/双端归档全部完成，g10跟进暂停；J2小幅指标取舍，完整联合Q两组后期失稳 |
| G11 | `europe_tmg_20261001`：温度、记忆与干扰图 | T2×15万＋M4×40万＋G3×40万，单seed51001；双独立4090D、共享存储隔离 | 已完成9/9训练、310万交互、11模型与两基线3500次Jan24共同分配、审计、报告及全资产归档；2026-10-03 00:40最终验收，见页首凭据 |
| G08 | `europe_ablation_20260929`：欧洲训练机制与状态空间 | 用户由3组扩为6组：dropout=0、每轮场景打乱、compact/summary/layout/full四档状态；均保留旧奖励，seed51001各40万步 | 六训练及七项01-18共同考核全部完成，逐束/资源账复算与带曲线报告07已通过；本轮无待运行任务 |
| G01 | `workbench_mask_20260925`：前端T-SAC掩码对照 | 旧奖励，掩码关闭/开启2次训练＋原规则贪心/固定基线 | 已完成并核验；最新参照 |
| G02 | `coverage_reward_20260924`：奖励对照 | 无掩码的旧奖励/融合奖励2次正式v2训练＋重评/基线 | 已完成；融合奖励未达到目标 |
| G03 | `sac_comparison_20260924_r2`：网络×原/修学习公式 | Transformer/CNN/MLP × 2公式，共6次正式训练 | 已完成；原18组范围已取消 |
| G04 | `sac_formula_factorial_20260925`：四公式组合 | 复用G03六组＋新增交叉组合六组，共12个结果单元 | 已完成；不是额外12次训练 |
| G05 | 原预训练模型与原指标资料 | 五模型统一重评、T-SAC早期学习诊断、旧扫描数组 | 已有历史评价；口径及来源有限制 |
| G06 | 环境/学习诊断、工程验收与短检查 | 204维诊断、功率重放、零槽诊断、DQN/CUDA与恢复检查等 | 证据见各入口；不记作新训练成果 |
| G07 | 欧洲ADS-B首轮训练，目录 europe_adsb_20260928 | 真实位置＋合成需求，旧奖励/联合掩码T-SAC；单种子51001、40万步 | 训练与100场景×3终点评价完成，SI88.93%；泛化改善尚无同场景对照证据 |
| S1-01 | 第一阶段覆盖规划七路线 | 规则/聚类不调整、局部搜索、PPO，以及空布局纯PPO；3×409600训练＋7×100共同考核 | 全部完成并独立审计通过；P0覆盖较低/82%可行，C1本轮较好，单种子限制 |
| P01 | 固定掩码下旧/融合奖励比较 | 拟复用G01参照，候选新增1组40万步 | 仅建议，待用户决定，未启动 |
| P02 | 训练数据泛化与第一阶段衔接 | 数据调研、欧洲航空独立环境、前端与数据冻结流程；295项及GPU短验证 | 工程实现完成；后续已授权并启动的单组训练见G07，原/新数据对照仍为建议 |
| P03 | 欧洲训练不稳定原因与下一阶段对照 | 全量日志、最终断点只读探针、状态/奖励小例；建议先dropout单因素 | 诊断与方案已完成；后续已授权并启动的六组见G08 |
| P04 | 第一阶段覆盖规划与两阶段解耦 | 四候选方案已收窄并扩为远端7组 | 用户已授权进入S1-01工程实施，效果待验证；初始建议与来历保留 |
| S1-01 | 第一阶段4090D规则/聚类微调与纯PPO构建 | 用户授权7组：4基线、3PPO单seed51001各409600步；当前GPU工程与P0新输入网络验证 | 远端与本地Git、数据、工作台已接入；工程恢复问题修复中，正式训练尚未启动、效果待验证 |
| P05 | A+B基础上的第二阶段稳定训练候选历史 | 曾拟211维门控/新奖励条件矩阵；保留奖励代数与风险解释 | 本轮执行已被G09自然维五组指令替换，未启动P05新奖励或其它专项 |
| P06 | G09后的207维全局信息消融与稳定性验证 | 推荐207参照＋三种206维删除单字段；候选四组共160万步，旧奖励/dropout0/shuffle/seed51001 | 仅方案12与问答，未实现或启动；204第五组是条件扩展，不在默认预算内 |
| P07 | 分配记忆、干扰关系与联合动作算法调研 | J模块G10已完成；M四组、G三组已获后续明确授权进入G11 | 具体执行见G10/方案14及G11/方案16；其它未选择方向仍仅建议，不累加P06预算 |
| P08 | G10联合Q失稳诊断与结构化价值改进 | 两组固定α0.01各15万步已获授权进入G11；220输出交互Q单列 | 24快照离线分解完成，T执行进度见G11；220输出结构仍未授权，不恢复已完成G10 |
| P09 → S3-P01 | 第三阶段T0固定频率槽功率细调 | T0/Jan25共同100场景×3动作×3方法，两预算口径各900结果，新增训练0 | 两任务、1800结果、18万束独立审计、3740文件归档、报告01及8图均完成；严格基本服务阈值边界与非收敛限制见报告，旧任务不重复运行 |
| S3-P02 | 七路线功率细调与逐场景耗时 | T0/Jan26共同100场景×3动作×7方法，两预算各2100，共4200，新增训练0 | 两端4200结果、42万束与计时审计、6094文件归档、报告02和14图已完成；源码b3c94c9，单seed/日期与严格阈值限制见报告 |
| S3-P03 | SI优先及容量估计对照 | T0/Jan27 100景×3动作×A/B/C，两预算各900、总1800，0训练 | 两端1800结果、18万终态束与12万候选束、完整时间审计、3696文件归档、报告03及15图均完成；SI微增与SGM/耗时代价并列 |

## 4. G01——前端T-SAC联合掩码对照（当前参照）

**条件：**原T-SAC13令牌Transformer、原204维顺序、旧奖励 `legacy_repaired`、seed51001；各400000交互/398001更新。训练场景151001–155000，共4000回合，两组逐回合场景哈希相同。验证274000–274019；考核275000–275099，模型每场景3次随机动作，基线每场景1次。主成绩取最终权重。网络结构已对齐原T-SAC，但工作台学习实现与历史入口不完全相同；不能将跨入口差值只归因于掩码。

| 方法 | SI | SGM | 基本达标束/100 | 零容量束 | 实际功率 W | 实际槽数 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 关闭掩码 | 65.82% | 70.03% | 88.74 | 10.37 | 2020.42 | 573.47 |
| 联合掩码＋最低预留 | 75.55% | 80.55% | 98.94 | 0.00 | 2528.02 | 627.26 |
| 贪心原规则 | 68.27% | 63.49% | 95.43 | 2.00 | 2409.75 | 490.00 |
| 固定原规则 | 72.16% | 59.08% | 98.24 | 0.00 | 3000.00 | 784.00 |

**结论：**掩码组相对关闭组SI＋9.73个百分点、SGM＋10.52个百分点、达标＋10.20束、功率＋507.60 W；按场景平均后100/100场景三个服务指标均提高。掩码组300次考核零槽、裁剪、越界、冲突及超功率均为0；两组均处理全部100束。增加功率伴随服务改善，不能称相对无掩码节能。

**仍有缺口：**掩码组只有106/300次（35.3%）全部100束达10%需求，最低95束；固定策略40/100次（40%），最低93束。两者平均达标束数与全体达标概率不能混用。只证明联合筛选＋最低预留＋动态熵整体方案在本次单种子实验有效，尚未拆开规则作用和学习贡献。

**运行与版本：**两个训练任务分别为 `train_sac_20260925_152308_c6a4bc88`（关）、`train_sac_20260925_152308_dcba4f43`（开）；基线评价任务 `evaluate_sac_20260926_030645_95e399db`。初始冻结 `847f7a7cb99a3bcab25f1c8bb4eaa9c821df6feb`；Windows文件写入故障后，从80000/40000步经工作台恢复到等价I/O修复提交 **`5da891fdbcf5fa0255d912677840620d977367c9`**。原日志/快照保留。比较器修复提交451b239仅使掩码两组可同场景配对，不改变训练或评价。

**入口：**[实验记录](docs/第二阶段/实验报告/02_原T-SAC联合掩码对照实验.md) · [实验清单/哈希](outputs/experiments/workbench_mask_20260925/experiment_manifest.json) · [正式结果汇总](outputs/experiments/workbench_mask_20260925/comparison/summary.json) · [100场景配对表](outputs/experiments/workbench_mask_20260925/comparison/paired_scenes.csv) · [带曲线HTML报告](outputs/experiments/workbench_mask_20260925/comparison/training_curves/带曲线实验报告.html) · [曲线日志核验](outputs/experiments/workbench_mask_20260925/comparison/training_curves/log_audit.json)。

原始任务数据在 `outputs/workbench/jobs/<任务编号>/`（配置、状态、日志、实际场景、results）；最终推理权重分别为[关闭掩码](models/runs/workbench/train_sac_20260925_152308_c6a4bc88/final/policy.pt)、[开启掩码](models/runs/workbench/train_sac_20260925_152308_dcba4f43/final/policy.pt)，元数据为各自同目录 `model.json`。两权重与40万步断点Actor一致，80000束终态已两轮复算。

## 5. 其他正式实验与新实验准备

### G08——六组欧洲训练机制与状态空间对照

**最终状态：**六训练、七共同独立考核与结果复核均已完成。独立日期核心结果如下；基线请求/裁剪差异、完整资源及容量表、旧日期单列、学习不稳定与单种子限制详见[正式报告07](docs/第二阶段/实验报告/07_欧洲六组独立场景对照.md)。不得把此前A/B原历史考核结论直接当作本表的新日期结果。

| 模型 | 01-18 SI | 01-18 SGM | 全100束基本达标 | 平均功率W |
| --- | ---: | ---: | ---: | ---: |
| G07 | 89.92% | 0.4033 | 48.67% | 2895.00 |
| A 关闭dropout | 91.40% | 0.5741 | 59.67% | 1680.17 |
| B 打乱场景 | 90.99% | 0.5198 | 70.00% | 2164.73 |
| C1 少量全局 | 85.90% | 0.4049 | 29.67% | 3000.00 |
| C2 资源服务概况 | 85.12% | 0.3797 | 26.00% | 2860.95 |
| C3 布局与需求 | 82.87% | 0.3924 | 23.67% | 2544.78 |
| C4 布局、分配与干扰 | 88.41% | 0.3638 | 45.00% | 2300.17 |

训练有效冻结e25ae56，共同评价冻结a0556de，两者src/data/pyproject的Git对象一致；G07参照权重原源01457f。七评价ID、来源与SHA见[最终运行清单](outputs/experiments/europe_ablation_20260929/run_manifest.json)；模型位于`models/runs/workbench/<原训练ID>/final/`，原始结果在`outputs/workbench/jobs/<评价ID>/results.json`，曲线与核验脚本在`outputs/experiments/europe_ablation_20260929/analysis/`。本轮无增加种子或追加训练预算。下列启动/接续记录保留为历史。

**09-30 01:00最新执行状态：**新增C3完成并核验300条结果和最终权重；C4达到训练预算但终点评价仍在进行，C2继续训练。共同01-18考核尚未开始，helper8844正常等待。[核验入口](outputs/experiments/europe_ablation_20260929/preflight/root_C3_completion_20260930.json)。

**15:57最新执行状态：**A/B/C1完成，C2/C3/C4训练中。C1完整预算/终点评价/权重SHA已核对，C2原任务断点恢复后已产生30551次真实更新；helper8844已确认全部六组均产生过实际更新。此状态取代下面13:59和14:11时点的排队描述，历史经过保留。[完成与恢复证据](outputs/experiments/europe_ablation_20260929/recovery_20260929_1350/actual_restart_verified.json)。

**A/B效果补充：**原历史100场景×3动作对照中A/B分别SI90.55%/89.50%、SGM0.5799/0.5343，均高于G07的88.93%/0.4149；A平均功率1726.25W，B2209.17W。全部分配完整、无超功率/冲突/裁剪/零容量。A改善更明显，B更省频率槽；仍有部分场景SI下降，独立日期和六组最终结论待后续共同评价。详见[阶段报告与曲线](docs/第二阶段/实验报告/06_欧洲AB阶段结果对照.md)及[逐场景/逐束核验](outputs/experiments/europe_ablation_20260929/analysis_ab_legacy_test_20260929/metrics/summary.json)。

**13:59当前状态：**A/B已完成，最终权重和各300条终点评价原始记录已核验；C1/C3/C4运行，C2异常后从原ID/完整初始断点恢复排队。此处的终点评价使用原历史100场景，不是后续01-18共同独立考核。当前helper为8844；恢复经过、创建时间及下一步见页首与[恢复汇总](outputs/experiments/europe_ablation_20260929/recovery_20260929_1350/recovery_summary.json)。A/B权重入口：[A固定终点](models/runs/workbench/train_sac_20260929_022411_a4654185/final/policy.pt) · [B固定终点](models/runs/workbench/train_sac_20260929_022415_f5552810/final/policy.pt)；结果及完整SHA见[完成证据](outputs/experiments/europe_ablation_20260929/preflight/heartbeat_20260929_1350.json)。以下06:38接续和02:26启动记录保留为历史时点。

**06:38工程接续记录（历史时点）：**六个原ID已全部绑定`e25ae56cb55ec402e0ba96013e383ab3da324bf6`；A/B/C1从96700/96600/82300步完整断点继续，C2–C4仍排队，种子51001与每组40万步预算不变。24次稳态试验/12对数值、两布局真实短恢复/验证/保存及197项凭据检查通过；13/32令牌稳态吞吐2.66/2.46倍，纯算法参考额外观测耗时约0.7%/2.1%。正式共同训练窗口三任务总更新19.75→37.83次/秒（观察值，非全程同工作量）。新API PID10108（06:27:12）、worker PID13936/18744/18676、共同考核helper PID24192（06:32:06）已核验，派发保留释放，原helper19440/API16668已退出。更新/回合日志连续、六份来源链与科学配置通过，前端已验证异步选项与新源进度；[接续与原始证据](outputs/verification/async_telemetry_20260929/migration/final_handoff.json) · [主任务审计](outputs/verification/async_telemetry_20260929/migration/root_continuation_audit.json) · [设计报告](docs/第二阶段/方案/08_异步采集与训练吞吐优化设计.md) · [验证报告与曲线](docs/第二阶段/实验报告/05_异步采集与训练提速验证.md)。工程效果已验证，六组模型效果待共同评价；本对话`g08`每小时静默跟进至实质变化，两个历史跟进保持暂停。接手须刷新实际任务/进程创建时间，不重复迁移或提交。

**正式运行记录：**2026-09-29 02:24逐组提交，同一冻结提交`48022f4499823def40d573ea9436e3f7bdbf87fa`，DQN/PyTorch2.3.1+cu121、CUDA12.1、RTX4070SUPER，主模型固定40万步终点。服务PID16668创建于02:20:55（恢复时需重新核对身份），8765工作台并发上限3；首波3个worker均已核验独立冻结导入路径、启动时间、非空updates及真实更新。02:26主任务API复核A为2850/851、B为2800/801、C1为2550/551（交互/更新）；其余0步排队。没有正式效果结论。

| 变体 | 工作台任务 | 02:26状态 |
| --- | --- | --- |
| A 关闭dropout | `train_sac_20260929_022411_a4654185` | training |
| B 每轮随机场景 | `train_sac_20260929_022415_f5552810` | training |
| C1 少量全局状态 | `train_sac_20260929_022420_7df81644` | training |
| C2 资源服务概况 | `train_sac_20260929_022424_6af7c280` | queued |
| C3 完整布局需求 | `train_sac_20260929_022430_295fe5ed` | queued |
| C4 完整布局分配干扰 | `train_sac_20260929_022435_171eb319` | queued |

共同训练576/验证20/历史比较100场景复用G07，新增01-18独立100场景已按SHA核验导入，留给G07和6个固定终点模型共同评价。后续跟进以工作台逐任务状态和本组`preflight/run_status.json`为准，不能因本文快照中的queued就重复提交。一次性提交helper已于02:25确认首波真实更新后正常退出，训练由工作台持久worker继续。

**最初后台接续（02:32，升级前历史）：**一次性进程PID19440、创建时间02:32:38，读取已提交六ID等待完成；不会再提交训练。完成后经同一工作台创建G07＋6新模型在01-18的7个共同评价任务，固定/贪心只在G07评价中各执行一次，再核对原始结果并生成中文报告与曲线。当前实际阶段为`waiting_six_training_completions`、stderr空；当时未创建定时自动化，现时接续/跟进见页首。入口：[接续进程身份与脚本哈希](outputs/experiments/europe_ablation_20260929/preflight/continuation_launch.json) · [阶段状态](outputs/experiments/europe_ablation_20260929/preflight/run_status.json)。PID只对应本次核验，接手须重查。

**恢复补记：**2026-09-29用户明确“开始”，下列暂停接手列表保留历史经过；恢复后的实际启动状态见上面的正式运行记录，不再以暂停时的未完成事项阻止已授权任务。

**工程准备记录：**四档实现已完成，同结构参数2307801（策略461560）；320项回归通过、旧权重严格兼容、四档同动作的原奖励及资源结果一致。前端已实际选择四档、随机场景顺序并保存工程配置；隔离8877仅核验页面，结束后已停止，正式服务为8765。新01-18包SHA`bf7b473755f4af70cada5aacde1980fb46734bc25105eaa2dbd6bf011b724241`，100预选场景无拒绝，来源、数组及明细SHA核验通过。3路短吞吐18.09更新/秒，比串行高53.8%，GPU峰值2475/12282MB。配置/数据/证据入口见[六组方案](docs/第二阶段/方案/07_欧洲六组训练机制与状态对照.md)及[实验计划](outputs/experiments/europe_ablation_20260929/experiment_plan.json)。

- **最新用户范围：**原G07作已有参照；新增A仅dropout=0、B仅场景每轮随机排列；用户明确第三方向只补状态、保留旧奖励，后又要求多档状态，已告知扩为4档，合计6个新模型。共同种子51001、各400000步，合计240万新交互步；这是获准范围，不因已有G07完成而自动增减；01:29暂停已经被后续“开始”指令解除。
- **暂停时工作区：**复用`C:/Users/Admin/.codex/worktrees/europe-environment/paper`，分支`codex/europe-three-ablation`（沿用名称，无须改名），基于主线`c328c83`。源码修改仅在该worktree，主线尚未合并；暂停时已保存WIP提交，提交消息明确集成未完成。原main及G07冻结源码未修改。
- **暂停时已有检查：**原207维/14令牌扩展、旧13令牌兼容、scene_order及轻量学习快照已有相关测试；初始三个候选各3次真实GPU更新通过，证据在`outputs/verification/g08_engineering_20260929/`。这些检查早于用户扩为4档状态，**不能当作新32令牌四档均通过**。并行吞吐测试尚未开始。
- **四档最新设计：**同一2946维封装、32令牌骨干（原13＋全局1＋每10束一段共10＋8组资源），features为原四项后加`global_context`[10]、`beam_context`[10,10,11]、`resource_context`[8,204]；context_level为compact/summary/layout/full，未开放字段归零。逐档开放少量全局→资源服务概况→全100束几何需求→已执行动作/当前服务/逐槽占用和历史干扰。保留旧奖励，不读取未来动作或未来容量；网络编码不等于数学上已证明完整MDP。
- **暂停时未完成、现已补齐的集成：**32令牌网络/环境字段实现刚开始，config已部分加入；需核对网络/字段/归一化、config_summary布局描述、默认context_level='none'的旧权重hash兼容、四档前端选项、导出/恢复/比较元数据及真实页面操作。UI目前只有先前13/14令牌与场景顺序改动，32档位还未接完。保留已完成14令牌功能，但本次4档正式对照要共用32令牌。勿在当前WIP启动训练。
- **暂停时数据与设备记录：**G07训练576/验证20/旧考核100场景可复用，ID见G07；旧100用于历史共同比较。新独立日预先固定为2024-01-18、同ADSB.lol prod-0、100均匀预选时刻和原需求参数，官方包1,663,856,640字节；下载/准备已按用户要求停止，分片1,648,361,472字节保留。状态及来源在本组`data/`。DQN/CUDA曾核对可用；恢复后须重新验证实际解释器、GPU、工作台队列、源码及进程，不能复用旧PID。
- **暂停时列出的恢复顺序（已执行至正式启动）：**仅在用户明确说开始后，核对WIP/暂停记录并续传新数据；完成4档实现、相关正确性/GPU检查和前端同步；测代表网络并发吞吐后确定工作台max-workers；提交并冻结源码，再由后台子agent通过工作台提交6任务。统一比较G07＋6新模型，最终新日期共同评价另记工作台任务。尚无G08模型效果。

### G07——欧洲ADS-B首轮训练（2026-09-28新增）

**状态：训练与最终考核已完成。** 工作台任务`train_sac_20260928_142100_38559f37`，冻结源码`01457f245bf0ab0e23873616e44975693014e3d0`；DQN/PyTorch2.3.1+cu121与RTX4070SUPER，训练种子51001。于北京时间14:21启动、20:45完成，交互400000步、4000回合、398001次更新，任务状态`completed`，原进程PID7460已退出。主模型为固定预算终点权重；100考核场景各随机动作3次全部完成，300条分配记录/30000条逐束数据，不能称300个独立场景。

**原训练终点评价结果与限制：**最终需求满足率88.9307%，SGM均值0.414919；平均99.24/100束达到本束10%需求，100束全部达标162/300次（54%）；平均功率2942.97 W、最高3000 W，零容量束、超功率、越界与裁剪记录均为0。22:48核对API/完成事件/结果元数据、逐场景汇总和最终权重SHA，当时未重新加载模型或重算物理链路。单训练种子、四个冬季工作日与合成需求限制结论，不能直接比中国旧成绩或8个训练场景的规则检查；原/新训练数据对照仍未完成。同场景规则基线对照后续已授权完成，单列于下文，不混用两次评价的动作种子与均值。

**后续诊断补记（2026-09-28）：**本次已在DQN环境加载冻结最终断点作少量CPU只读前向，区别于上段22:48的文件核验；没有独立重评或更新权重。64个固定随机终止样本Q均值−46.13、奖励+0.065，Critic训练/评估态Q差17.10。全部398001次更新的Critic裁剪前梯度超过10约99.08%；早末窗口TD中位1.18→13.77。固定验证1万→40万步SI86.92%→88.69%、回报57.11→28.47，故不能称完全未学习或稳定收敛。场景顺序/隐藏β/目标取舍的证据和因果边界见[方案06](docs/第二阶段/方案/06_欧洲训练异常诊断与下一阶段方案.md)。

场景ID：训练`scenes_20260928_142049_09674a59`（576），验证`scenes_20260928_142053_e9083bfc`（20），最终考核`scenes_20260928_142054_9bc86790`（100）。四日原包、逐帧CSV/NPZ/外置明细SHA、日期隔离及8个训练场景短规则检查已核验；需求仍是工程假设。

**欧洲同场景基线评价（2026-09-29完成）：**用户要求的两张逐场景SI/SGM曲线已生成。工作台`evaluate_sac_20260929_005523_382b5e60`于北京时间00:58:50完成，冻结提交`61d888ccc60567979acab8a7d75e95d6a166dee3`，与G07训练版源码/数据内容相同。复用最终权重与100个欧洲考核场景，模型3次、固定/贪心各1次，共500次分配；DQN/CUDA、冻结导入路径、权重及场景SHA均核对，50000条逐束SI/SGM/功率复算通过。模型/固定/贪心SI分别88.94%/88.04%/89.27%，SGM0.4213/0.2436/0.3240，功率2885.23/3000/2066.60 W。模型SGM更高，SI略低于贪心；模型无裁剪/零容量，固定原规则有越界裁剪，贪心有冲突裁剪及平均2个零容量束。独立评价动作种子为每场景275000/275001/275002，与原训练终点评价派生规则不同，不能混写历史均值；不是新训练。详见[报告与两张曲线](docs/第二阶段/实验报告/04_欧洲模型与基线逐场景对比.md)，产物在`outputs/experiments/europe_adsb_20260928/baseline_comparison_20260929/`。

入口：[含训练/验证/考核曲线的结果报告](docs/第二阶段/实验报告/03_欧洲ADS-B首轮训练结果.md) · [通俗说明](docs/第二阶段/方案/05_欧洲ADS-B首轮训练说明.md) · [工作台任务](outputs/workbench/jobs/train_sac_20260928_142100_38559f37) · [考核原始数据](outputs/workbench/jobs/train_sac_20260928_142100_38559f37/results.json) · [最终权重](models/runs/workbench/train_sac_20260928_142100_38559f37/final/policy.pt) · [来源清单](outputs/experiments/europe_adsb_20260928/source_manifest.json) · [场景统计](outputs/experiments/europe_adsb_20260928/preflight/actual_data_summary.json)。2026-09-28补图：原始指标、验证评价与最终考核各有曲线，绘图脚本及哈希见报告；训练曲线周期起伏且损失波动，不能只凭最终均值断言稳定收敛。下一步可分析低基本服务波束；任何新对照训练按后续授权执行。

### G02——旧奖励与融合奖励（无掩码）

**目的/条件：**在零槽不计实际功率的新环境中，从头训练两组Transformer SAC，比较旧奖励 `legacy_repaired` 与融合奖励 `fused`；不是G01工作台训练。seed51001，各40万步/398000更新，固定终点；验证82000–82019，考核83000–83099，模型每场景动作种子91001–91003。源码 **`5a73516cd69c632258d37daba51a0f39d5fccc1c`**。正式任务均已completed。

| 方法 | SI | SGM | 基本达标束/100 | 零容量束 | 实际功率 W |
| --- | ---: | ---: | ---: | ---: | ---: |
| 正式旧奖励重训v2 | 66.64% | 70.11% | 89.53 | 9.54 | 2066.55 |
| 正式融合奖励重训v2 | 35.73% | 33.15% | 73.93 | 24.27 | 448.25 |
| G03修正式Transformer权重新环境重评（非重训） | 63.37% | 68.96% | 91.93 | 7.07 | 1637.88 |
| 贪心 | 68.28% | 62.98% | 95.09 | 2.00 | 2408.30 |
| 固定策略 | 71.55% | 58.71% | 97.93 | 0.00 | 3000.00 |

**结论：**融合奖励低功率伴随服务明显下降和零槽增多，本次实验未达到目标。正式两组均处理100束且未超3000 W。后续诊断：融合7280次零槽全部发生在仍有空位时，直接原因是选已占起点；固定槽位改为30 W重放可提高其SI与自身融合奖励，不能把失败单归因于功率罚项过重。G01尚未重训融合奖励，不能宣称这次失败已被修复。

**入口：**[清单](outputs/experiments/coverage_reward_20260924/experiment_manifest.json) · [正式报告](outputs/experiments/coverage_reward_20260924/comparison/中文前后对照报告.md) · [汇总](outputs/experiments/coverage_reward_20260924/comparison/summary.json) · [配对表](outputs/experiments/coverage_reward_20260924/comparison/paired_scenes.csv) · [功率诊断](outputs/experiments/coverage_reward_20260924/comparison/power_counterfactual/中文原因诊断.md) · [零槽诊断](outputs/experiments/coverage_reward_20260924/comparison/zero_slot_diagnostic_20260925/中文零槽原因核对.md) · [20场景资源图说明](outputs/experiments/coverage_reward_20260924/comparison/visualization_random20_20260925/图表说明.md)。20场景是既有考核子集，不能视为新增独立考核。

正式目录名：`transformer_old_reward_repaired_s51001_v2`、`transformer_fused_s51001_v2`，配置/状态/逐束评价位于 `outputs/experiments/coverage_reward_20260924/<组名>/`，权重位于 `models/runs/coverage_reward_20260924/<组名>/final.pt`。重评旧权重来源与SHA见[旧权重重评摘要](outputs/experiments/coverage_reward_20260924/old_weight_repaired_eval/summary.json)，不是从头训练第三组。首次无v2预热作业已作废，不计入结果。

### G03——三网络×原/修学习公式（六组）

**条件：**Transformer、参数量接近的CNN/MLP，各原式/修正式；seed51001，场景种子61001，各40万步/398000更新，主结果固定终点。验证71000–71019、考核81000–81099，100场景×3动作种子91001–91003。六组均completed，训练冻结 **`36c14d74cbe6b9cf7410ee0595ee6d73083afdc8`**；分析提交508bc4d。其环境仍包含零槽请求功率记账等旧约定，不能与G02/G01直接排名。

`legacy` 同时使用目标熵−6.9和逐头取双Q较小值；`corrected` 同时使用＋6.9和联合动作Q取较小值。两个因素同时变化，奖励公式在该组内保持一致。

| run_id | SI | SGM | 功率W（该组旧记账） | 超功率比例 | 完整处理100束比例 |
| --- | ---: | ---: | ---: | ---: | ---: |
| transformer_legacy_s51001 | 67.18% | 69.49% | 3026.47 | 100% | 0% |
| transformer_corrected_s51001 | 62.97% | 67.86% | 1748.80 | 0% | 100% |
| cnn_matched_legacy_s51001 | 52.53% | 54.22% | 3029.00 | 100% | 0% |
| cnn_matched_corrected_s51001 | 41.45% | 39.75% | 3028.85 | 100% | 0% |
| mlp_matched_legacy_s51001 | 51.92% | 53.99% | 3030.78 | 100% | 0% |
| mlp_matched_corrected_s51001 | 47.39% | 46.45% | 2022.78 | 0% | 100% |

**结论/限制：**修正式没有带来三网SI整体提高；原式三组均超功率、提前结束，不能只按SI宣称更优。CNN修正式出现数值失稳，训练结束不等于算法稳定。只有一个训练种子。204维信息缺口及功率/槽数动作的联合表达问题另有诊断，不能由这些关联直接判定主要原因。

该组同场景基线如下（均未超功率、均完整处理100束；仍为G03口径）：

| 基线 | SI | SGM | 功率 W |
| --- | ---: | ---: | ---: |
| 贪心 | 68.48% | 63.56% | 2452.1 |
| 固定 | 70.90% | 58.21% | 3000.0 |
| 四色现有实现 | 48.90% | 47.04% | 2452.1 |
| 随机动作 | 45.02% | 43.27% | 2726.7 |

前三种每场景一次，随机每场景三次；[基线来源](outputs/experiments/sac_comparison_20260924_r2/single_seed_51001/analysis/baseline_reference.json)和报告保存具体规则。四色实现存在可换组、空位估计和连续槽段裁剪等限制，不能代表理想四色策略。

**入口：**[六组范围清单](outputs/experiments/sac_comparison_20260924_r2/single_seed_51001/scope_manifest.json) · [中文报告](outputs/experiments/sac_comparison_20260924_r2/single_seed_51001/analysis/中文单种子六组报告.md) · [结果汇总](outputs/experiments/sac_comparison_20260924_r2/single_seed_51001/analysis/run_summary.json) · [分析来源](outputs/experiments/sac_comparison_20260924_r2/single_seed_51001/analysis/analysis_manifest.json) · [204维诊断](outputs/experiments/sac_comparison_20260924_r2/single_seed_51001/state_diagnosis/中文204维状态诊断.md)。

逐任务原始目录 `outputs/experiments/sac_comparison_20260924_r2/<run_id>/`，权重 `models/runs/sac_comparison_20260924_r2/<run_id>/final.pt`；各任务 `manifest.json` 保存版本、配置与校验值。原始冻结目录登记为 `D:/paper_worktrees/frozen-sac-20260924-r2`，使用前核查存在及源码一致性。

### G04——拆开目标熵与双Q聚合的四组合

**目的：**在G03基础上补交叉组合，区分两项学习公式因素。共12个网络×公式结果，其中A/D六组复用G03，**只新增B/C六次训练**，均completed。新增seed51001、场景种子61001、各40万步/398000更新，固定终点；使用与G03相同实际验证/考核场景和3个动作种子。新增冻结 **`09505290aae1a754a0e4b9d4b9a9ae3c39bf1425`**，旧组仍为36c14d；不是同一个提交重新训练12组。

| 组合 | 目标熵 | 双Q取小方式 | 来历 |
| --- | --- | --- | --- |
| A | −6.9 | 三动作头分别取小后相加 | G03 `legacy` |
| B | −6.9 | 联合动作Q先聚合再取小 | 新增 `negative_joint` |
| C | ＋6.9 | 三动作头分别取小后相加 | 新增 `positive_per_head` |
| D | ＋6.9 | 联合动作Q先聚合再取小 | G03 `corrected` |

下表列新增六组；A/D结果见G03，完整12格见原报告。

| run_id | SI | SGM | 功率W（该组旧记账） | 超功率比例 | 完整处理100束比例 |
| --- | ---: | ---: | ---: | ---: | ---: |
| transformer_negative_joint_s51001 | 71.19% | 74.39% | 3028.25 | 100% | 0% |
| transformer_positive_per_head_s51001 | 65.35% | 69.63% | 2036.52 | 0% | 100% |
| cnn_matched_negative_joint_s51001 | 47.51% | 47.83% | 3026.48 | 100% | 0% |
| cnn_matched_positive_per_head_s51001 | 40.10% | 40.36% | 3021.98 | 100% | 0% |
| mlp_matched_negative_joint_s51001 | 55.54% | 57.29% | 3029.80 | 100% | 0% |
| mlp_matched_positive_per_head_s51001 | 39.06% | 40.11% | 1765.68 | 0% | 100% |

**结论：**目标熵与双Q方式会共同影响学习、功耗和是否完成分配；不能笼统断言某一方式必然提高性能。负目标熵组合均超功率，CNN正目标熵也有超限/失稳风险；成绩必须连约束一起看。不是奖励函数实验，也不是掩码实验。

**入口：**[四组合报告](outputs/experiments/sac_formula_factorial_20260925/analysis/中文四组合公式对照报告.md) · [12格汇总](outputs/experiments/sac_formula_factorial_20260925/analysis/run_summary.json) · [配对效应](outputs/experiments/sac_formula_factorial_20260925/analysis/paired_effects.json) · [分析来源](outputs/experiments/sac_formula_factorial_20260925/analysis/analysis_manifest.json) · [相关问答](Q&A/算法与训练/匹配规模SAC训练与公式修复对照.md)。新增组原始数据 `outputs/experiments/sac_formula_factorial_20260925/<run_id>/`，权重 `models/runs/sac_formula_factorial_20260925/<run_id>/final.pt`，冻结目录登记 `D:/paper_worktrees/frozen-sac-factorial-20260925`。

## 6. 历史评价、诊断及不纳入正式结果的记录

### G05——旧权重和历史指标

- **五套预训练权重统一重评：**Transformer/CNN/MLP SAC、PPO direct、DQN，原环境默认100场景及额外压力条件、未训练初始化参照。默认旧口径满意度总分依次95.36、67.94、80.75、67.11、73.23（满分100）；贪心77.23。CNN和MLP默认100/100回合超功率。结果说明这些权重对当时程序学到了有效选择，不能当作当前修正版环境的排名。原训练集和过程不齐全，不能确认或排除过拟合。入口：[五模型问答](Q&A/评估与指标/五套预训练模型的策略偏好与泛化评估.md)、[动作偏好](outputs/verification/model_policy_audit_DQN_20260924/preference_summary.md)、[压力分析](outputs/verification/model_policy_audit_DQN_20260924/generalization_summary.md)；原始数据 `outputs/verification/model_policy_audit_DQN_20260924/outputs/full/`，原权重 `models/pretrained/`。
- **更早T-SAC 20场景学习诊断：**旧权重95.6167/100，未训练初始化均值52.2095、随机53.2787、贪心76.7507；为旧环境下的另一批诊断，不能与五模型评价拼接统计。入口：[是否学到策略](Q&A/算法与训练/模型是否学到了有效策略.md)。
- **历史需求扫描/八算法资料：**保存数组在 `outputs/metrics/demand/`，部分缺种子、权重校验、版本及完整场景；部分绘图脚本曾改内存数据，应以原数组与说明为准。不能据这些旧曲线建立当前模型排行榜。入口：[原始结果汇总](Q&A/评估与指标/各算法现存原始结果汇总.md)。拉格朗日等无可靠完整结果的条目继续标缺失，不补造数据。

### G06——诊断与工程检查（非正式新训练）

| 类别 | 已有证据与边界 | 入口 |
| --- | --- | --- |
| 204维状态信息 | 有严格同观测、不同未来需求导致奖励不同的小例；未证明它是成绩差的主因 | [诊断报告](outputs/experiments/sac_comparison_20260924_r2/single_seed_51001/state_diagnosis/中文204维状态诊断.md) |
| 融合奖励低功率/零槽 | 固定槽位功率重放及30000步占用核对；为原轨迹诊断，不是新策略重训 | G02中的功率、零槽诊断链接 |
| DQN/CUDA | 2026-09-24验证Python3.11.11、Torch2.3.1+cu121、RTX4070SUPER及真实GPU更新；启动前仍需现场复核 | [环境/耗时记录](Q&A/算法与训练/训练耗时瓶颈与GPU利用率.md) |
| 工作台与掩码 | 9种短GPU组合、网络权重映射及掩码概率/真实更新等检查；实现检查不替代正式性能评价 | [工作台验收](docs/工作台/02_实验工作台验收记录.md)、[G01记录](docs/第二阶段/实验报告/02_原T-SAC联合掩码对照实验.md) |
| 状态写入与恢复 | G01等价I/O修复、断点前后一次更新精确核对，原失败现场保留 | [恢复摘要](outputs/experiments/workbench_mask_20260925/recovery_20260925_1730/recovery_summary.json) |
| 校准/探针 | `sac_comparison_20260924_calibration`、`sac_comparison_20260924_checks`、各`outputs/verification/`短运行用于吞吐/恢复/正确性 | 不将校准种子51002等视作已授权正式多种子结果 |

### 停止、作废与旧状态陷阱

1. **原18组计划已收窄为G03六组。** `transformer_legacy_s51002`已停止并保留断点；本次读到旧状态仍写training/71257步，不代表仍在运行。禁止恢复额外种子。依据：[范围变更清单](outputs/experiments/sac_comparison_20260924_r2/scope_change_single_seed_20260924/scope_change_manifest.json)。
2. **G02无v2旧奖励预热作业作废。** `transformer_old_reward_repaired_s51001`旧状态仍写training/2531步；正式清单记 `aborted_with_record`，不能计入正式结果。旧权重重评目录的initializing也是残留，应看已产出的完整评价与汇总。
3. **更早`sac_comparison_20260924`启动与G03恢复历史保留。** 以r2六组已完成结果为准，旧suite计划或backup中的running不能用于判断现状；`analysis_before_*`、`analysis_superseded_*`保留历史，优先读各组当前`analysis/`。
4. 工作台工程验收任务 `train_sac_20260925_052426_ed3f1ca1` 为cancelled/0步，不是正式算法训练成果。G01根层旧`failure.json`为已恢复的Windows故障，不能只凭存在判失败。
5. 项目根目录现有零散 `status.json`、`job.json`、`checkpoint.pt`、`*.jsonl`等未跟踪任务文件归属尚未整理；**正式来源是实验目录与工作台任务目录**。这些文件不作为实时状态依据，不擅自覆盖、提交或删除。另有四策略20场景问答/报告的既存未提交改动，本次未接管。

## 7. 下一步与授权边界

**S3-P01已完整交付（2026-10-03）：**两预算口径原任务各900结果、独立审计、本地归档、中文[报告01](docs/第三阶段/实验报告/01_T0初始解的连续功率细调对照.md)与8图完成。用户随后已授权其它算法路线与逐场景耗时对照，S3-P02七路线、完整耗时、审计、14图与报告02现已交付，下一步回到讨论。严格基本服务安全裕量、频率槽调整或更多预算仍未授权。原范围和历史来历见[方案02](docs/第三阶段/方案/02_连续功率优化与双实例运行.md)。

**G11已执行范围的历史记录（2026-10-01授权，2026-10-03验收）：**P08的T0/T1两组各15万步，P07的M四组/G三组各40万步，共310万交互，单seed51001；Jan24固定终点共同评价已完成，不重复新开。15万筛查不等于40万稳定。完整条件与授权边界见[方案16](docs/第二阶段/方案/16_温度记忆与干扰图双实例实验说明.md)，初始原因与候选论证保留于[方案15](docs/第二阶段/方案/15_联合Q失稳诊断与结构化价值改进方案.md)。下文保留早期方案历史。

**P01仅为2026-09-27讨论建议，未新建训练任务。**

1. 先查G01未达10%需求的波束：刚分配就不足，还是后续干扰使其下降；结合槽数、功率、顺序及剩余预算判断。该专项诊断尚未执行，不能预先断言功率不足是主因。
2. 保留当前原13令牌Transformer/204维输入、相同联合掩码/预留/动态熵，拟比较旧奖励与原融合奖励。条件核对一致时复用G01参照，候选从头训练1组seed51001、40万步；若改动破坏可比条件，需重新明确参照与预算。
3. 已用来分析的83000/81000/275000系列场景是既有诊断资料。后续方案选定后保存新的共同验证/最终考核场景，同场景重评旧参照和基线，不拿历史分数直接排名。验收优先基本服务，再分别看SI、SGM、功率和约束，明确取舍。
4. 若合法动作下功率学习仍异常，再根据诊断单独讨论让功率选择参考所选频点/槽数。未批准扩展第一/三阶段、增加训练种子、盲目加训练步数或同时更改多个因素。

详细方案：[奖励问答最新追问](Q&A/算法与训练/当前奖励函数.md)。旧[第二阶段候选方案](docs/第二阶段/方案/03_融合奖励失败后的第二阶段改进方案.md)与四公式问答含历史优先顺序，实际已执行范围以G01为准，当前建议以本节及最新用户指令为准。

### P02：训练场景泛化与覆盖接口（2026-09-28新增建议）

- **目的：**从固定100城市、固定需求生成分布扩展到用户/航空器业务图驱动的波束布局与聚合需求；第一版保持100束、固定宽度和预算，以静态时刻快照训练，后续才接入可变宽度和连续重配置。
- **事实与限制（2026-09-28晚间按G07更正）：**204维含本束位置和当前干扰，未含完整未来位置/需求；新增欧洲版已支持显式三用途场景及地区对应的默认参考场景，原“内置考核只用默认城市”是实现前缺口。G07四日ADS-B位置驱动的576/20/100场景训练与考核已完成；人数/接入/活跃/速率仍是工程假设，未通过受控对照验证相较原数据的新地理泛化改善。
- **候选对照：**原数据与多场景数据共用输入/网络、旧奖励、联合掩码及物理口径；建议seed51001、各40万步（候选80万步，未授权）。改变输入时两组均重训。新地理、新业务与联合变化用共同留出场景评价，不能直接沿用历史分数。按需另行讨论两组单因素补充，不增加训练种子。
- **后续调研（2026-09-28）：**找到航空轨迹、人口栅格、通信活动与卫星仿真需求等公开原料，建议真实位置/轨迹＋可解释业务模型。已读取71束24小时需求样例、6G-NTN一周船舶栅格两份小包（合计约2.54 MB），存于`outputs/verification/dataset_research_20260928/`；属于格式/含义核验。船舶数、CDR活动及已实现吞吐均不能直接当请求Mbps。欧洲数据与当前122.2°E卫星需处理物理范围匹配。
- **实现与核验：**源码`e7e38dd08b7ed1ea997289655ebb5c0ebd763472`增加`europe_v1`（13°E、10°最小仰角工程约定）、航空CSV/readsb快照、人数/接入/活跃/速率需求模型、100束过渡覆盖器、地区/日期/角色隔离与冻结数据/权重。前端选项、导入及三用途场景选择同步。295项测试通过；与旧源码3场景×100步中国数值逐位一致，中欧各6转移/1次GPU真实更新、G01两套权重严格加载通过；均为代码检查，不是新训练成果。
- **状态与授权（2026-09-28中午更新）：**工程实现完成；用户后续明确导入ADS-B开始训练，授权的新单组欧洲实验另记G07。此前夜间延期及一次关机是历史操作，不适用于G07。本轮不自动扩展原/新数据两组对照，P01仍未获运行授权。
- **处理流程说明（2026-09-28追问）：**原始轨迹按UTC时刻选非未来最新位置→每机合成需求→100个需求加权代表位置→按0.25°视角半径唯一归属→汇总每束需求并保留未覆盖量。当前为过渡覆盖规则和中心代表用户近似，不是第一阶段DRL；需求D与资源分配得到的容量C分开。详细算例及时间/身份限制已追加至训练数据问答，未改代码或启动训练。
- **入口：**[完整方案](docs/方案/04_训练场景泛化与第一阶段衔接方案.md) · [数据调研](docs/资料/03_公开数据集与合成业务场景调研.md) · [欧洲环境使用与核验](docs/工作台/04_欧洲航空环境使用与核验.md) · [原始验证](outputs/verification/europe_environment_20260928) · [训练数据问答](Q&A/环境仿真/训练场景与随机需求生成.md)。

### P03：欧洲训练异常诊断与下一阶段方案（2026-09-28晚至09-29凌晨）

- **本轮已完成：**原始日志全量统计、576场景跨轮配对、固定20场景验证分析、冻结SAC路径审计、最终断点分层随机128转移CPU只读前向、相同204维观测/动作却不同奖励的确定性小例。图片来自保存日志，证据位于`outputs/experiments/europe_adsb_20260928/diagnosis/`；API仍为completed，未恢复任务。
- **已证实与未证实：**终止价值拟合失准、Critic训练/评估态模式差、固定顺序及状态缺项存在；哪项造成多少学习下降尚无受控重训证据。SAC终止掩码、双Q合成顺序、概率与温度符号未发现明确公式错误。Actor loss上升不能单独当成策略失败。
- **建议顺序：**首批复用G07参照，仅新增1组dropout=0/seed51001/40万步，从头训练，其他条件不变；12万步诊断检查，固定40万步主结果。其后视证据单独对照每轮随机场景顺序、状态信息、奖励或其他学习设置，不打包修改。补规则/未训练同结构验证参照，最终保留新日期独立考核；当前01-15结果已参与开发诊断。
- **授权与前端：**用户本次要求找原因与方案，未授权新训练；没有改生产代码或发起正式长评价。本轮文档无需前端变化；实施时已有dropout配置须贯通验证，新场景顺序/诊断统计/状态版本须同步前后端及权重元数据。
- **入口：**[诊断与方案](docs/第二阶段/方案/06_欧洲训练异常诊断与下一阶段方案.md) · [问答](Q&A/算法与训练/模型是否学到了有效策略.md) · [原结果报告](docs/第二阶段/实验报告/03_欧洲ADS-B首轮训练结果.md)。

### S1-01：远端七组波束覆盖规划（2026-09-30，已完成）

- **展示与解释补充（2026-10-04）：**报告01/02各实验表与配对差值表已补全“编号＋实验组合”，原始任务与成绩不变。P0（无基础布局，纯PPO）平均CV最低，但两项覆盖最低且82/100可行，不能据此称整体最好；C1（聚类＋局部搜索）仍是本轮覆盖与负载质量参照。该补充仅解释既有共同100场景结果，未新增训练、评价或实时任务核验。

- **当前核对：**2026-09-30最终API确认10任务均completed、worker退出，[最终交付](outputs/experiments/first_stage_s1_01_20260930/final_delivery.json)含核对时间；以下14:43启动与增长证据保留历史。正式训练源`b3a11c51e9da6612a43cb005a79c9e89a37e448d`；386项测试的bfa0bed与该提交在src/data/pyproject逐文件Git差异为零。运行中不修改远端repo或source_snapshots。
- **组与任务：**R2 `train_ppo_20260930_144259_874e5464`、C2 `train_ppo_20260930_144259_78db8328`、P0 `train_ppo_20260930_144300_e05c586b`；三组已completed，各409600交互、50rollout/1600优化更新，原始日志已核；固定预算终点模型，验证不改变训练随机源。
- **共同条件：**train/validation/test分别576/20/100，UTC日期01-08/09、01-12、01-22；逐机原始位置与合成通信需求相同。100束全宽0.5°；964维状态与4维候选特征；646914参数，512并行环境×16采集，4更新轮、1024小批量，gamma1、学习差分奖励×0.01。R/C9局部动作/200步一回合，P0空布局全局候选最多2048/100步，移动不适用，不把路线差值纯归因初始化。
- **评价安排：**三训练完成后，工作台已提交R0/R1/C0/C1四基线及R2/C2/P0三模型，在01-22同100场景确定性各1次评价；覆盖与可行性按全部场景、失败保留，负载均值注明有效布局分母。本组覆盖比例不是第二阶段SI/SGM。
- **工程失败与恢复：**旧35ee短验证恢复ByteTensor设备错，保留原快照；修复后新P0 `train_ppo_20260930_143508_b45eda5d`停于384步/6更新，断点SHA与文件一致，同API恢复到12800/200更新、用途verification。另一次6400短验证自然先完成，未发生恢复；两者均不计正式训练成果，PowerShell中文名编码问题留存，正式请求由UTF-8 Python生成。
- **证据与待办：**[方案03](docs/第一阶段/方案/03_远端4090D部署与七组并行实验技术方案.md)、[本地计划与原始数据入口](outputs/experiments/first_stage_s1_01_20260930/)、[GPU核验](outputs/verification/remote_stage1_20260930/gpu_validation_v4/report.md)、[恢复记录](outputs/verification/remote_stage1_20260930/workbench_v2_resume/workbench_v2_resume_summary.json)。模型远端在`/home/u2024110224/first_stage_satellite_rl/models/runs/workbench/`；三组模型已同步本地models/runs并核SHA，原始数据/任务NPZ/源码快照本地完整备份仍有缺口。三训练与七评价完成，逐场景独立重算通过，原始曲线与[正式对照报告02](docs/第一阶段/实验报告/02_七组覆盖规划独立场景对照.md)已保存；初次审计精度问题及备份缺口保留，未追加实验；单种子与冬季合成业务限制保留。未增加种子、第二阶段适配训练或定时任务。

### P04：第一阶段波束覆盖规划与两阶段解耦（历史建议，已进入S1-01）

**本节下列条目保留初始方案时点。** 用户后续已授权远端实施并新增P0，当前状态以S1-01和页首为准，不再把“未授权/未实现”历史条目作为本轮事实。

- **目的与范围：**在开题报告第一阶段独立优化指向与宽度、覆盖和负载均衡，并延续已有第二阶段。推荐有约束初始布局＋单智能体轮询修正；首版固定100束/0.5°先研究中心，随后适配逐束宽度及连续时序，首版不能称完整第一阶段已完成。
- **分层细化（2026-09-30）：**基础层采用冻结的聚类或固定覆盖规则，PPO层按稳定ID轮询微调中心，限定相对初始位置的累计移动；主结果是终态覆盖/负载/调整代价，不依赖第二阶段奖励。与同初始化不调整及非学习局部搜索比较，分别报告候选检查次数/耗时。现行PPO与SAC掩码不能直接当作布局PPO，需未来接入独立环境和工作台配置；本次仅方案。
- **首轮矩阵（2026-09-30）：**R0/R1/R2为规则初始化的不调整、局部搜索、PPO；C0/C1/C2为聚类初始化对应三组。2个PPO独立训练、4个基线评价候选，单seed51001；共同预算、轮询、覆盖门槛和留出考核须工程核验后冻结，未授权或提交任务。R2-C2只比较完整路线，不能纯归因初始化。详见[第一阶段方案02](docs/第一阶段/方案/02_基础布局与PPO微调首轮实验分组.md)，第二阶段P05范围不变。
- **接口与限制：**共享一致几何/物理定义，独立奖励、权重和训练，保存用户归属、未覆盖需求、稳定波束ID、中心、宽度、需求及顺序/组号映射。现行场景要求100束正需求，方向图宽度固定，组号由处理序号%8决定；零需求/可变宽度/任意分组需新版本。第一阶段不调用T-SAC奖励；旧权重能加载不证明新布局有效。
- **建议对照：**同一初始化无修正、同一初始化＋非学习局部搜索、同一初始化＋DRL修正；固定业务图/候选/计算预算，保存覆盖、负载、失败及原始曲线。冻结二阶段A/B仅作新布局适应性诊断，衔接评价计入未覆盖业务，必要的二阶段适配训练另行确定。
- **ADS-B复用追问核对（2026-09-30）：**现存576/20/100份训练/验证/原考核场景及对应逐机明细，可复用位置与合成需求供第一阶段静态覆盖训练；输入应取逐机明细、重新计算布局归属，100束NPZ只作资源分配接口。四日原始包存在、长度相符，本次未重算大包SHA。01-15/18已开发使用，新最终考核另留；连续重配置需另建时间序列。来源清单旧not_imported字段不代表G07/G08当前未导入。仅可行性建议，无新训练授权或任务。
- **实际状态与授权：**本轮已核对报告、源码和G08完成清单，仅文档方案；没有实现第一阶段、开展训练/长评价、增加种子或恢复自动跟进。未刷新实时服务/进程，不以旧PID判断现状。新预算、工作台任务和物理版本尚待后续确定；本轮无需前端改动，实施时须接入前端配置/校验/任务及模型结果展示。
- **入口：**[完整方案01](docs/第一阶段/方案/01_第一阶段波束覆盖规划与两阶段解耦方案.md) · [中文问答](Q&A/环境仿真/第一阶段波束覆盖规划与两阶段衔接.md) · [读取证据与来源SHA](outputs/verification/stage1_plan_review_20260930/review_evidence.json)。无新增正式任务编号或训练权重。

## 8. 以后怎样维护本文件

- **更新时点：**正式任务创建、范围/配置改变、异常/恢复、训练和评价完成、结果更正及下一步决定；填北京时间和实际核对来源。普通心跳不需要逐次写入。
- **每个实验组至少包含：**编号/目的、实际变体和基线、状态、种子与步数/更新数、奖励/环境/网络/掩码差异、源码提交和任务ID、实际场景/动作采样/模型选择、SI/SGM/基本服务/功率/约束、结论与限制、报告/数据/权重/哈希入口、待办及授权状态。未记录的指标写“未记录”，未核验写“待核验”，不能填0。
- **核对层次：**清单规定范围与版本；逐任务状态和真实进程确认进度；完整结果、实际场景和校验文件确认完成。历史结果引用现有核验报告，本文件初始化没有重新跑全部训练或评价。
- **简明交接：**本文件放概要和必要结果，详细诊断留报告；同组追加或更正，不制造多个互相矛盾的“当前状态”文件。重要更正带日期与原因，保留历史入口。
- **保存与备份：**本文件及维护规则纳入本地Git；模型/日志等仍保存在被忽略的产物目录。尚未确认异地备份位置，不声称这些实验资产已完成外部备份。

### 维护记录

| 日期 | 变更 | 核对范围 |
| --- | --- | --- |
| 2026-10-01 | G10 09:12监控旧观测/快照覆盖误报修复，更新只读跟进入口 | 独立实时API/SFTP确认训练健康，原现场与不完整恢复限制保留；唯一输出/拒覆盖/时间和raw响应检查通过，不动训练、前端或冻结源 |
| 2026-10-01 | 用户授权G10联合动作四组，前后端实现/双平台验证并启动远端四路，自动后处理与g10跟进 | 单seed51001/总160万，cc27冻结；真实API/PID出生/CUDA/更新增长，Jan23新100场景与数据SHA；未启动P06/M/G，额度20%未用卡 |
| 2026-10-01 | 新增P07算法创新调研：联合动作、分配记忆、干扰关系及分模块对照 | 当前代码/原T-SAC PDF/G09报告、原论文和两项Sol审查；仅方案文档，未修改算法/前端或启动训练评价 |
| 2026-10-01 | 新增P06：207维三个全局字段逐项删除的下一轮方案 | G09报告10/独立审计、自然状态与旧奖励代码、Sol独立方案复核；仅文档与引用检查，无源码修改、训练或评价 |
| 2026-09-30 | docs按第一阶段和第二阶段分开，编号保留、共用资料保持原路径 | 维护中的Markdown引用、README/AGENTS/问答/状态入口与图片代码链接；仅文档归档，无训练/评价/前端或冻结资产变更 |
| 2026-09-30 | 新增P04第一阶段多方案与两阶段解耦建议，归档第一阶段方案（原09，现01）和问答 | 开题报告第2.1节、欧洲场景/方向图/分组接口、G08完成清单及报告07；仅文档审查，未实现/训练/长评价/改前端，效果待验证 |
| 2026-09-28 | 首次建立统一状态入口，AGENTS规定新窗口先读与关键节点维护，README/Q&A加入入口 | G01–G04清单、汇总及逐任务状态；G05历史报告；真实Python进程、工作台访问、自动化暂停状态；仅文档整理，未训练/评价 |
| 2026-09-28 | 新增P02训练场景泛化与第一阶段衔接建议，同步方案和问答 | 核对G01生成/状态/导入/考核代码、报告航空业务与覆盖接口、公开数据来源；仅文档方案，未刷新任务或启动实验 |
| 2026-09-28 | P02补充公开数据、访问条件、合成业务方法与两份小包核验 | 区分真实位置、活动量、请求与吞吐；样例未接入正式训练，未改源码/训练，效果待验证 |
| 2026-09-28 | P02欧洲航空环境及前端实现完成，记录用户本晚停止于跑通并关机的要求 | 295项测试、旧中国数值对照、DQN/CUDA短真实更新、权重兼容和页面导入；正式训练未启动，效果待验证 |
| 2026-09-28 | 用户后续授权导入ADS-B并开始G07，先交付通俗说明；数据准备中 | 单组欧洲T-SAC、seed51001/40万步；DQN/CUDA、空队列与服务核查；候选日期与场景已预定，未提交训练 |
| 2026-09-28 | G07四日数据及三组场景核验完成，待提交训练 | 576/20/100场景SHA、日期隔离、8个训练场景规则与资源账；非训练结果 |
| 2026-09-28 | G07正式任务已创建并核验真实GPU更新 | 工作台 train_sac_20260928_142100_38559f37，冻结 01457f245bf0ab0e23873616e44975693014e3d0，核验时2200步/201次更新；效果待验证 |
| 2026-09-28 | 22:48核对G07于20:45完成40万步及终点评价，归档报告 | API/完成事件/100场景×3结果与模型SHA一致；SI88.93%、基本服务99.24束、全束达标54%；未重训或独立复测，泛化改善待对照 |
| 2026-09-28 | G07结果报告补齐训练、验证、学习过程和考核分布曲线；以后正式实验报告均附相关曲线 | 原始工作台日志绘图，保存脚本、输入/图像哈希及平滑口径；未改训练/前端或追加评价，损失波动不等于已证明不收敛 |
| 2026-09-29 | G07训练异常完成分项诊断与P03方案归档 | 全量日志、终止Q探针、同权重模式差、状态/奖励确定性小例；优先建议dropout单因素，未改生产代码或启动新训练 |
