# 中期报告数学模型与推导
来源：codex（GPT-6 Astra）。当前接口无法设置 xhigh 思考档。

本研究面向北京邮电大学硕士学位论文《HTS智能波束覆盖及频率-功率资源分配算法研究》，构建“波束覆盖规划—频率与离散功率联合分配—连续功率精细优化”的三阶段模型。第一阶段采用网络流与几何交替优化方法 FLOW-AO v2，确定飞机归属、波束中心与宽度；第二阶段采用频谱块指针 T-SAC，逐束分配连续频率槽和离散功率；第三阶段采用匹配约束信赖域逐次凸逼近方法 MCTR-SCA，在保持第二阶段供需匹配程度的约束下提高需求满足率。

## 记号表

| 记号 | 含义与单位 |
| --- | --- |
| $n,m,N$ | 当前场景飞机数、实际非空波束数、波束数预算，单位为架、束、束；主链路 $N=150$ |
| $a,i,j,q,g,t$ | 飞机、接收波束、发射波束、频率槽、频率组、逐束决策步的索引；频率槽 $q=0,\ldots,99$，组 $g=0,\ldots,7$ |
| $R_{\mathrm E},R_{\mathrm O},\lambda_{\mathrm s}$ | 地球半径、地心至卫星距离、星下点经度，分别为 $6371000$ m、$42164000$ m、$13^\circ$ |
| $\boldsymbol u_a,\boldsymbol v_i,d_i$ | 卫星至飞机地面投影的单位方向、卫星至波束中心的单位方向、卫星至波束中心距离；前两者无量纲，$d_i$ 为 m |
| $\theta_{ai},\theta_{ij},b_i,r_i$ | 飞机相对束中心的离轴角、束中心间夹角、半功率全束宽、指派允许半径，单位为度 |
| $x_{ai},\mathcal U_i,L_i$ | 飞机归属二元变量、波束成员集合、成员数；$L_i=|\mathcal U_i|$ |
| $d_{\mathrm a},D_i,C_i$ | 单机需求、波束需求、波束容量，单位 bit/s；$D_i=L_i d_{\mathrm a}$ |
| $g_i,s_i,c_i,p_i,\mathcal S_i$ | 波束的组号、起始槽号、实际槽数、整束功率和占用槽集合；$p_i$ 为 W |
| $B,f_q,N_0,\eta$ | 槽宽 $25\times10^6$ Hz、槽中心频率 Hz、单槽噪声 W、频谱效率上限 $5.5$ bit/s/Hz |
| $S_i,H_{ij},\ell_{iq},\Gamma_{iq}$ | 有用信号系数、接收束 $i$ 对发射束 $j$ 的耦合系数、自由空间路径增益、逐槽 SINR，均为线性量 |
| $\beta,\Lambda_i,F,W,\sigma_i$ | 平均波束需求 bit/s、单束 SGM、场景平均 SGM、需求满足率、需求封顶的单束满足率；后四者无量纲 |
| $P,K,\kappa$ | 第一阶段峰值人数、每束人数目标上限、换束费用权重；$P,K$ 为架 |
| $o_t,a_t,\pi,Q$ | 智能体观测、完整动作、策略分布、软动作价值函数 |
| $\alpha,\gamma,\tau$ | SAC 温度 $0.01$、折扣 $0.99$、目标网络软更新率 $0.005$ |
| $\mathcal I,M,\boldsymbol p^0,F_0,\delta$ | 有槽束集合及其大小、第二阶段整束功率、其平均 SGM、允许的 SGM 降幅 |
| $J_{ij},\boldsymbol\Delta,R,\mu$ | 容量雅可比 bit/s/W、功率增量 W、信赖域半径 W、匹配约束曲率余量系数 W$^{-2}$ |

下文 $\mathbb{1}\{\cdot\}$ 表示条件成立取 1、否则取 0；$\|\cdot\|_2$ 为欧氏范数，$\|\cdot\|_\infty$ 为各分量绝对值的最大值。三角函数的输入用弧度，几何角度输出统一换为度。场景平均均只包含真实波束，不包含批量计算的填充位置。

对应代码：[最终配置:7–50](<D:/paper/outputs/experiments/s2lab_round4_20261028/account_B/runs4/r4-ptr-s1/config.json#L7-L50>)；[场景配置:21–34](<D:/paper-s3lab/src/satellite_rl/s2lab/scenes.py#L21-L34>)；[物理参数:21–56](<D:/paper-s3lab/src/satellite_rl/s2lab/physics.py#L21-L56>)。

## 一、系统场景与卫星视角几何

本研究考虑单颗地球静止轨道卫星服务欧洲航空用户的下行场景。飞机经纬度来自场景输入，覆盖与链路几何采用其地面投影，未把飞行高度加入距离。设纬度、经度为 $\phi,\lambda$，地面投影位置 $\boldsymbol z$ 与卫星位置 $\boldsymbol z_{\mathrm s}$ 均以 m 为单位：

$$ \boldsymbol z(\phi,\lambda)=R_{\mathrm E}(\mathrm{cos}\phi\,\mathrm{cos}\lambda,\mathrm{cos}\phi\,\mathrm{sin}\lambda,\mathrm{sin}\phi),\qquad \boldsymbol z_{\mathrm s}=R_{\mathrm O}(\mathrm{cos}\lambda_{\mathrm s},\mathrm{sin}\lambda_{\mathrm s},0). $$（1）

任一位置对应的卫星视线方向与距离为：

$$ \boldsymbol u(\phi,\lambda)=\frac{\boldsymbol z(\phi,\lambda)-\boldsymbol z_{\mathrm s}}{\|\boldsymbol z(\phi,\lambda)-\boldsymbol z_{\mathrm s}\|_2},\qquad d(\phi,\lambda)=\|\boldsymbol z(\phi,\lambda)-\boldsymbol z_{\mathrm s}\|_2. $$（2）

波束中心经纬度同样通过式（2）转换为 $\boldsymbol v_i,d_i$。两个单位视线的张角采用弦长计算；链路代码也使用截断内积后的反余弦，二者数学等价：

$$ \theta(\boldsymbol u,\boldsymbol v)=\frac{180}{\pi}\,2\mathrm{arcsin}\left(\min\left(\frac{\|\boldsymbol u-\boldsymbol v\|_2}{2},1\right)\right). $$（3）

记 $\theta_{ai}=\theta(\boldsymbol u_a,\boldsymbol v_i)$、$\theta_{ij}=\theta(\boldsymbol v_i,\boldsymbol v_j)$。飞机归属满足唯一性和覆盖约束，发布宽度按成员包络确定：

$$ x_{ai}\in\{0,1\},\qquad \sum_{i=1}^{m}x_{ai}=1,\qquad x_{ai}=1\ \Longrightarrow\ \theta_{ai}\le\frac{b_i}{2},\qquad b_i=\max\left(0.3^\circ,2\max_{a\in\mathcal U_i}\theta_{ai}\right). $$（4）

实现对包络增加 $10^{-9}$ 度数值余量，在 $1^\circ$ 边界处作容差回收。共享几何检查要求纬度 $25^\circ$—$80^\circ$、经度 $-40^\circ$—$50^\circ$、仰角至少 $10^\circ$。布局生成使用 $43.2$ Mbps/机；后续冻结几何，在每个资源分配场景内给全部飞机使用同一单机需求，训练时该值在 $30$—$80$ Mbps 抽取，评价时由清单固定。三个阶段承接同一归属，第二、三阶段的逐束需求均按 $D_i=L_i d_{\mathrm a}$ 计算。

对应代码：[共享几何:5–36](<D:/paper-s3lab/src/satellite_rl/shared_interface/geometry.py#L5-L36>)；[第一阶段几何:21–29、100–161](<D:/paper-s3lab/src/satellite_rl/coverage/s1fix/geom.py#L100-L161>)；[场景构造:79–110](<D:/paper-s3lab/src/satellite_rl/s2fix/data.py#L79-L110>)；[环境需求:157–173](<D:/paper-s3lab/src/satellite_rl/s2lab/env.py#L157-L173>)。

## 二、波束方向图、干扰与容量

### 2.1 参考方向图与传播系数

为刻画变宽波束的主瓣和旁瓣干扰，本研究采用代码实现的 ITU-R S.672-4 单馈源圆波束参考方向图。设半功率半宽 $\psi_i=b_i/2$，相对离轴角 $v=\theta/\psi_i$，近旁瓣相对电平 $L_{\mathrm N}=-20$ dB。峰值增益 $G_i^{\max}$ 与方向图增益 $G_i(\theta)$ 均以 dBi 表示：

$$ G_i^{\max}=50+20\log_{10}\left(\frac{0.5^\circ}{b_i}\right). $$（5）

$$ G_i(\theta)=\begin{cases}G_i^{\max}-3v^2,&v\le(-L_{\mathrm N}/3)^{1/2},\\G_i^{\max}+L_{\mathrm N},&(-L_{\mathrm N}/3)^{1/2}<v\le6.32,\\\max\left(G_i^{\max}+L_{\mathrm N}+20-25\log_{10}v,0\right),&v>6.32.\end{cases} $$（6）

远旁瓣的地板为绝对增益 $0$ dBi，近旁瓣平台则为相对峰值的 $-20$ dB。接收增益为 $40$ dBi，其它损耗为 $5$ dB，二者形成公共线性因子 $A_{\mathrm r}=10^{40/10}10^{-5/10}$。有用信号采用成员飞机发射增益的线性平均；干扰采用发射束在接收束中心方向上的增益：

$$ S_i=\frac{A_{\mathrm r}}{L_i}\sum_{a\in\mathcal U_i}10^{G_i(\theta_{ai})/10},\qquad H_{ij}=\begin{cases}A_{\mathrm r}10^{G_j(\theta_{ij})/10},&j\ne i,\\0,&j=i.\end{cases} $$（7）

因此，$H_{ij}$ 一般不对称：两束夹角相同，但方向图使用发射束 $j$ 的宽度。自由空间路径增益以接收束中心距离计算。记光速 $c_0=299792458$ m/s、频带起点 $f_0=17.7\times10^9$ Hz：

$$ f_q=f_0+(q+0.5)B,\qquad \ell_{iq}=\left(\frac{c_0}{4\pi d_i f_q}\right)^2,\qquad N_0=k_{\mathrm B}T_{\mathrm n}B. $$（8）

其中玻尔兹曼常数 $k_{\mathrm B}=1.380649\times10^{-23}$ J/K，噪声温度 $T_{\mathrm n}=631$ K。实现损耗 $1$ dB 通过 $\zeta=10^{-1/10}$ 乘入 SINR；异极化隔离度 $20$ dB 对应泄漏系数 $\chi=10^{-20/10}=0.01$。当前实现的传播项为自由空间路径增益和固定其它损耗，没有另行计算逐场景雨衰、云衰。

对应代码：[方向图与参数:21–84](<D:/paper-s3lab/src/satellite_rl/s2lab/physics.py#L21-L84>)；[场景物理张量:112–144](<D:/paper-s3lab/src/satellite_rl/s2lab/physics.py#L112-L144>)。

### 2.2 逐槽 SINR 与需求封顶前容量

频率资源池包含八组，每组一百个槽。组号奇偶表示极化，同一组同一槽最多分配给一束。不同组复用同一组槽中心频率，组号本身不增加频率偏移。令 $o_{iq}=\mathbb{1}\{q\in\mathcal S_i\}$，有槽时 $\mathcal S_i=\{s_i,\ldots,s_i+c_i-1\}$，否则为空集。整束功率按实际槽数均摊：

$$ \sum_{i:g_i=g}o_{iq}\le1,\qquad p_{iq}=\begin{cases}o_{iq}p_i/c_i,&c_i>0,\\0,&c_i=0.\end{cases} $$（9）

在合法频率方案中，同组波束没有重叠槽；不同组只有槽号重叠时才产生干扰。同极化异组的系数为 1，异极化为 $0.01$。将这一有效干扰关系记为 $\rho_{ij}$：

$$ \rho_{ij}=\begin{cases}0,&i=j\text{ 或 }g_i=g_j,\\1,&g_i\ne g_j\text{ 且 }g_i\bmod2=g_j\bmod2,\\\chi,&g_i\bmod2\ne g_j\bmod2.\end{cases} $$（10）

每个占用槽上的干扰、SINR 和波束容量为：

$$ I_{iq}=\ell_{iq}\sum_{j\ne i}\rho_{ij}H_{ij}p_{jq},\qquad \Gamma_{iq}=\frac{\zeta S_i\ell_{iq}p_i/c_i}{N_0+I_{iq}},\qquad q\in\mathcal S_i. $$（11）

$$ C_i=B\sum_{q\in\mathcal S_i}\min\left(\log_2(1+\Gamma_{iq}),\eta\right),\qquad \eta=5.5. $$（12）

无槽束容量为零；有槽束即使继续增大发射功率，单槽速率也不超过 $137.5$ Mbps，十槽容量不超过 $1.375$ Gbps。式（12）先逐槽封顶再求和，不能把平均 SINR 直接代入容量。环境和第三阶段按极化累加全部已分配发射束，未单独排除同组；同组不重叠的分配约束使其在接收束的占用槽上贡献为零，故与式（10）一致。

对应代码：[参考容量:163–192](<D:/paper-s3lab/src/satellite_rl/s2lab/physics.py#L163-L192>)；[环境容量:199–224、405–411](<D:/paper-s3lab/src/satellite_rl/s2lab/env.py#L199-L224>)；[固定频率系数:55–80](<D:/paper-s3lab/src/satellite_rl/s3lab/problem.py#L55-L80>)。

## 三、供需匹配与服务指标

本研究分别评价“容量是否贴近需求”和“实际满足了多少业务”。前者使用 SGM，后者使用需求满足率 $W$。令平均波束需求为 $\beta$、容量需求比为 $z_i$，SGM 的实部 $u_i$ 与虚部 $v_i$ 为：

$$ \beta=\frac1m\sum_{i=1}^{m}D_i,\qquad z_i=\frac{C_i}{D_i},\qquad u_i=\begin{cases}z_i-1,&z_i\ge1,\\1-1/z_i,&0<z_i<1,\end{cases}\qquad v_i=\frac{C_i-D_i}{\beta}. $$（13）

这里“实部、虚部”用于构造复合偏差的模，不代表链路复信号。定义 $h_i=(u_i^2+v_i^2)^{1/2}$，得到单束与场景匹配程度：

$$ \Lambda_i=\begin{cases}1-\left(1-\exp(-h_i)\right)^3,&C_i>0,\\0,&C_i=0,\end{cases}\qquad F=\mathrm{SGM}=\frac1m\sum_{i=1}^{m}\Lambda_i. $$（14）

当 $C_i=D_i$ 时 $\Lambda_i=1$；欠供、过供都会降低该值。服务量按需求封顶，单束满足率 $\sigma_i$、业务量加权满足率 $W$ 和束平均满足率 $\mathrm{SI}$ 为：

$$ \sigma_i=\min\left(\frac{C_i}{D_i},1\right),\qquad W=\frac{\sum_i\min(C_i,D_i)}{\sum_iD_i}=\frac{\sum_iD_i\sigma_i}{\sum_iD_i},\qquad \mathrm{SI}=\frac1m\sum_i\sigma_i. $$（15）

由于同景单机需求相同，$W$ 也等于按飞机人数加权的满足率。最差 10% 飞机指标把每束 $\sigma_i$ 重复 $L_i$ 次，形成 $n$ 个值；从小到大排序为 $\sigma_{(1)},\ldots,\sigma_{(n)}$，取向上取整后的最低 10% 平均：

$$ k_{10}=\max\left(1,\lceil0.1n\rceil\right),\qquad W_{10}=\frac1{k_{10}}\sum_{a=1}^{k_{10}}\sigma_{(a)}. $$（16）

这一指标衡量波束容量按成员共同满足比例展开后的飞机服务水平；当前代码没有逐机独立调度容量。未服务束与其成员均保留在指标分母中。

对应代码：[SGM:147–160](<D:/paper-s3lab/src/satellite_rl/s2lab/physics.py#L147-L160>)；[环境指标:226–244、443–503](<D:/paper-s3lab/src/satellite_rl/s2lab/env.py#L443-L503>)；[第三阶段指标:90–133](<D:/paper-s3lab/src/satellite_rl/s3lab/problem.py#L90-L133>)。

## 四、第一阶段：FLOW-AO 覆盖规划

### 4.1 覆盖约束、峰值目标与下界

热点飞机集中时，就近归属容易使少数波束过载。本研究先在覆盖、唯一归属、可变束宽和束数限制下压低最大成员数，再通过几何调整改善分配条件。基本峰值模型为：

$$ \min_{m,\{\boldsymbol v_i,b_i\},\{x_{ai}\}}P,\qquad L_i=\sum_{a=1}^{n}x_{ai}\le P,\qquad 1\le m\le\min(N,200),\qquad 0.3^\circ\le b_i\le1.0^\circ,\qquad L_i\ge1. $$（17）

式（17）同时满足式（4）。非叠放束之间的间隔规则为中心不进入另一束半功率足迹内部，边界相切允许：

$$ \theta_{ij}\ge\frac{\max(b_i,b_j)}2,\qquad i\ne j\text{ 且不属于同一热点叠放组}. $$（18）

因 $n=\sum_iL_i\le mP\le NP$ 且 $P$ 为整数，任何可行布局均有：

$$ P\ge\left\lceil\frac nN\right\rceil. $$（19）

因此，只要某景找到的可行布局峰值等于式（19），即可证明该景峰值全局最优。几何外层是否找到这种布局，由实际搜索结果决定。

固定中心时，令 $\mathcal J_i$ 为除自身及同叠放组副本外的中心集合。允许指派半径与中心最小间距为：

$$ r_i=\min\left(0.5^\circ,\min_{j\in\mathcal J_i}\theta_{ij}\right),\qquad \theta_{ij}\ge0.15^\circ\quad(j\in\mathcal J_i). $$（20）

没有其它中心时取 $r_i=0.5^\circ$。若所有成员满足 $\theta_{ai}\le r_i$，则包络半宽不超过 $r_i$，从而满足式（18）；反过来，满足式（18）与包络条件的归属也在这些允许边内。这给出了间隔规则的等价指派实现。关闭间隔规则时半径固定为 $0.5^\circ$。

对应代码：[允许半径与间隔:47–89](<D:/paper-s3lab/src/satellite_rl/coverage/s1fix/skeleton.py#L47-L89>)；[几何合法性:113–161](<D:/paper-s3lab/src/satellite_rl/coverage/s1fix/geom.py#L113-L161>)。

### 4.2 最大流判定与凸费用均衡

固定中心后，允许边集合为 $\mathcal E=\{(a,i):\theta_{ai}\le r_i\}$。给定整数峰值 $P$，建立“源点—飞机—波束—汇点”网络，三类边容量依次为 $1,1,P$。等价指派约束为：

$$ \sum_{i:(a,i)\in\mathcal E}x_{ai}=1,\qquad \sum_{a:(a,i)\in\mathcal E}x_{ai}\le P,\qquad x_{ai}\in\{0,1\}. $$（21）

最大流为 $n$ 当且仅当该峰值可行。可行性对 $P$ 单调，故代码以当前中心数 $m$ 对应的 $\lceil n/m\rceil$ 为二分下界，以允许边内最近中心归属的峰值为上界，找到固定中心下最小可行峰值 $P^*$。设置人数上限时，流的束端容量取 $\min(P,K)$。空束通过交替路径从人数超过一的束接收飞机，保持峰值不升；仍为空的束在布局构造时删除。

在固定 $P^*$ 下，均衡模式为每束设置 $P^*$ 个单位名额。第 $k$ 个名额变量为 $y_{ik}\in[0,1]$，边际费用为 $2k-1$。记允许边的弦长为 $d_{ai}^{\mathrm{ch}}=\|\boldsymbol u_a-\boldsymbol v_i\|_2$，小距离费用系数为 $\epsilon=0.1/[n\,2\mathrm{sin}(0.5\pi/360)]$，线性规划为：

$$ \min_{x,y}\ \sum_i\sum_{k=1}^{P^*}(2k-1)y_{ik}+\epsilon\sum_{(a,i)\in\mathcal E}d_{ai}^{\mathrm{ch}}x_{ai},\qquad \sum_i x_{ai}=1,\quad \sum_a x_{ai}=\sum_k y_{ik},\quad 0\le x_{ai},y_{ik}\le1. $$（22）

名额费用递增，人数为 $L_i$ 时总使用费用为：

$$ \sum_{k=1}^{L_i}(2k-1)=L_i^2. $$（23）

因此主费用为负载平方和。小距离项在全部飞机上的总贡献至多约 $0.1$，只用于整数主费用相同后的取舍。将飞机节点约束乘以 $-1$ 后，约束矩阵为网络节点—弧关联矩阵；其全幺模性与整数右端、整数容量保证存在整数最优基本解，可直接用线性规划获得归属。该结论来自网络结构，费用加入距离或换束罚分不会破坏整数性。

对应代码：[最大流:30–44](<D:/paper-s3lab/src/satellite_rl/coverage/s1fix/skeleton.py#L30-L44>)；[峰值二分、名额费用与线性规划:127–218](<D:/paper-s3lab/src/satellite_rl/coverage/s1fix/skeleton.py#L127-L218>)；[空束交替路径:92–124](<D:/paper-s3lab/src/satellite_rl/coverage/s1fix/skeleton.py#L92-L124>)。

### 4.3 几何交替、人数上限与热点叠放

初始中心采用自适应覆盖网：以第 $k$ 近邻角距形成局部半径，夹在 $0.15^\circ$—$0.5^\circ$，按半径由大到小先覆盖稀疏区。代码二分寻找使中心数不超过预算的最小 $k$，然后在预算允许时拆分繁忙波束。

基础外层交替执行重新指派、逐束重定中心、删闲束后拆忙束。重新指派及删拆组合按“峰值、最忙 10% 束的平均人数、人数变异系数”的字典序接受改进。重定中心在保持成员归属时只检查几何合法性；增加新束以峰值不升和非空束数增加为条件，因而不是每个几何动作都要求三元组严格下降。连续三轮没有进展时停止，默认轮数上限为 $30$ 倍工作量参数。

重定中心先把成员方向投影到局部切平面，用最小包围圆确定平面中心，再映回单位球面并重新计算成员包络。若归一化均值轴为 $\boldsymbol e$，其正交切向基为 $\boldsymbol e_1,\boldsymbol e_2$，平面坐标与映回方向为：

$$ \boldsymbol z_a^{\mathrm{tan}}=\frac{(\boldsymbol u_a\cdot\boldsymbol e_1,\boldsymbol u_a\cdot\boldsymbol e_2)}{\boldsymbol u_a\cdot\boldsymbol e},\qquad \hat{\boldsymbol v}=\frac{\boldsymbol e+z_1^*\boldsymbol e_1+z_2^*\boldsymbol e_2}{\|\boldsymbol e+z_1^*\boldsymbol e_1+z_2^*\boldsymbol e_2\|_2}. $$（24）

其中 $(z_1^*,z_2^*)$ 为平面最小包围圆圆心。这是投影几何更新，不等同于直接精确求解球面最小包络锥。

v2 把后续链路能够承担的人数引入第一阶段。布局生成入口使用固定参考容量 $1.125$ Gbps，因此：

$$ K=\left\lfloor\frac{1.125\times10^9}{43.2\times10^6}\right\rfloor=26. $$（25）

该数值继承布局生成参考，未随物理 v3 的 $1.375$ Gbps 十槽上限更新。v2 先生成基础布局，再用容量为 $K$ 的部分最大流检查热点是否还有未归属飞机。若有，按未归属人数预留额外波束名额，缩减基础网预算后重新布点；随后优先复制能够覆盖最多未归属飞机的中心。副本共享叠放组标识，同组副本豁免式（18），但与其它站点仍执行间隔约束。

“尽量窄”模式在固定当前峰值上限 $\bar P$ 下求解张角费用流：

$$ \min_x\sum_{(a,i)\in\mathcal E}\theta_{ai}x_{ai},\qquad \sum_i x_{ai}=1,\quad \sum_a x_{ai}\le\min(\bar P,K),\quad x_{ai}\in\{0,1\}. $$（26）

这里 $\bar P$ 为当前布局峰值；初始化未给出峰值时使用最近中心归属的峰值。此分支不再二分最小峰值，也不使用 $2k-1$ 均衡费用。指派后填补可用空束，同一叠放站点的副本一起重定中心，只有最终张角总和严格下降才接受收窄迭代。因此“尽量窄”是张角总和目标，经成员包络间接形成较窄波束。

若预算或时间用尽仍无法让每束不超过 $K$，叠放过程退回保留全部飞机的峰值指派，并记录超限束数。最终几何合法性检查未传入人数上限，故本研究把 $K$ 写成 v2 的容量控制目标与可行时的约束，并同时保留超限记录。

对应代码：[初始化与拆分:238–382](<D:/paper-s3lab/src/satellite_rl/coverage/s1fix/skeleton.py#L238-L382>)；[基础外层:385–500](<D:/paper-s3lab/src/satellite_rl/coverage/s1fix/skeleton.py#L385-L500>)；[v2 收窄、叠放与返回:531–637](<D:/paper-s3lab/src/satellite_rl/coverage/s1fix/skeleton.py#L531-L637>)；[平面包围圆:51–84](<D:/paper-s3lab/src/satellite_rl/coverage/s1fix/geom.py#L51-L84>)；[K 的实际入口:79–110](<D:/paper-s3lab/src/satellite_rl/s2fix/data.py#L79-L110>)。

### 4.4 动态修补与轻量局部修复

相邻航迹帧之间，本研究先保持已有中心，按稳定飞机编号继承归属。仍落在原束允许半径内的飞机留在原束；移动出界或新进入的飞机归给可覆盖的最闲束；无束覆盖时在该机位置新开中心。预算已满时先尝试解散一个成员能被其它束接收的轻载束，随后继续修补。离场飞机及空束删除，宽度按当前成员更新；修补失败时使用整景布局。

轻量局部修复选取最忙六束及其 $1^\circ$ 内邻束，其余束的中心、宽度、成员和编号冻结。局部执行峰值指派、均衡、重定中心及删闲拆忙，最多五轮，换束费用 $\kappa=3$。若飞机上一帧所连波束仍存在，记其当前列号为 $i_a^-$，则允许边费用为：

$$ c_{ai}^{\mathrm{dyn}}=\epsilon d_{ai}^{\mathrm{ch}}+\kappa\mathbb{1}\{i_a^-\ge0\text{ 且 }i\ne i_a^-\}. $$（27）

峰值先通过最大流确定，随后在该峰值内最小化边费用；均衡模式还叠加式（22）的名额费用。因此 $\kappa$ 不允许用更高峰值换取少换束，而是在同一峰值内与均衡费用共同影响归属。新飞机或原束已删除的飞机不计该项。

启用预测时，按地速与航向外推 60 s，优先保留同时覆盖当前及预测位置的允许边；只有某机存在这种边时才收紧其选择。重定中心使用当前与预测单位方向的归一化和作为中间方向。速度或航向缺失时保持原位置。当前第三阶段工作树不含动态模块，以上动态机制对应原第一阶段工作树，不参与当前静态 v2 场景生成入口。

对应代码：[动态修补:62–121](<C:/Users/Admin/.codex/worktrees/s1fix/paper/src/satellite_rl/coverage/s1fix/dynamic.py#L62-L121>)；[局部修复与预测约束:136–265](<C:/Users/Admin/.codex/worktrees/s1fix/paper/src/satellite_rl/coverage/s1fix/dynamic.py#L136-L265>)；[轻量配置与失败回退:353–377](<C:/Users/Admin/.codex/worktrees/s1fix/paper/src/satellite_rl/coverage/s1fix/dynamic.py#L353-L377>)；[原动态指派费用:118–187](<C:/Users/Admin/.codex/worktrees/s1fix/paper/src/satellite_rl/coverage/s1fix/skeleton.py#L118-L187>)；[运动外推:166–176](<C:/Users/Admin/.codex/worktrees/s1fix/paper/src/satellite_rl/coverage/s1fix/geom.py#L166-L176>)。

## 五、第二阶段：频谱块指针 T-SAC

### 5.1 逐束决策、规则定组与动作掩码

本研究将一景资源分配建模为卫星中心马尔可夫决策过程。场景内几何、需求和处理顺序固定，环境状态保存全部已执行分配、频率占用和剩余功率，按需求从大到小处理波束。每步分配一束，完成 $m$ 束后回合结束；神经网络接收由环境状态提取的十四令牌观测 $o_t$。

频率组由环境规则确定。设 $O_{gq}$ 表示组 $g$ 的槽 $q$ 已被占用，$F_g=\sum_q(1-O_{gq})$ 为该组空闲槽数，$I_{t,h,q}$ 为当前束在极化 $h$ 上受到的槽干扰，$h\in\{0,1\}$。定组规则为：

$$ g_t=\arg\min_{g\in\{0,\ldots,7\}}\left[\frac{1}{100N_0}\sum_{q=0}^{99}I_{t,g\bmod2,q}+10^9\mathbb{1}\{F_g=0\}-10^{-6}F_g\right]. $$（28）

即优先选平均干扰较小且有空闲槽的组，空闲槽数作为微小费用项。该规则没有代入当前需求估算容量。容量感知贪心规则基线另外使用式（12）的无干扰近似估槽数、按需求份额和所需功率取小，并选择低干扰窗口；学习策略仅沿用式（28）的定组。

完整动作为申请槽数 $\hat c_t$、起点 $s_t$、功率档 $\ell_t$，最终配置采用三个独立分布。十三档功率集合为：

$$ a_t=(\hat c_t,s_t,\ell_t),\qquad \hat c_t\in\{1,\ldots,10\},\quad s_t\in\{0,\ldots,99\},\quad p_t\in\mathcal P=\{0.5,1,2,5,10,15,20,25,30,35,40,45,50\}\ \mathrm W. $$（29）

设 $R_{gs}$ 为组 $g$ 从槽 $s$ 开始的连续空闲长度，$P_{\mathrm{left}}$ 为未用功率，$h_t=m-t-1$ 为当前束之后的束数。扣除后续每束 $0.5$ W 预留后的功率为 $P_{\mathrm{av}}=\max(P_{\mathrm{left}}-0.5h_t,0)$。令 $U=\sum_gF_g$，槽数上限为：

$$ c_{\max}=\max\left(1,\min\left(10,\max_sR_{g_ts},U-h_t\right)\right). $$（30）

三个头分别允许 $R_{g_ts}\ge1$ 的空闲起点、$1\le\hat c_t\le c_{\max}$ 的槽数、$p_t\le P_{\mathrm{av}}+10^{-6}$ W 的功率档。独立策略没有要求所选起点能完整容纳申请槽数，环境执行时取：

$$ c_t=\min\left(\hat c_t,100-s_t,R_{g_ts_t},c_{\max}\right),\qquad p_t^{\mathrm{exec}}=\begin{cases}\min(p_t,P_{\mathrm{left}}),&c_t>0,\\0,&c_t=0.\end{cases} $$（31）

当某类掩码全空时，代码保留首项作为可执行占位，随后由环境截短和功率裁剪处理，可能得到零槽、零功率。完整窗口条件化掩码要求 $R_{g_ts}\ge\hat c_t$，只在其它策略分支使用；最终 `policy=indep` 未启用该分支。动作入回放池保存的是申请动作，环境状态与第三阶段保存的是实际执行槽数、功率。

对应代码：[定组、掩码与观测:247–332](<D:/paper-s3lab/src/satellite_rl/s2lab/env.py#L247-L332>)；[执行截短:372–415](<D:/paper-s3lab/src/satellite_rl/s2lab/env.py#L372-L415>)；[容量感知规则:28–81](<D:/paper-s3lab/src/satellite_rl/s2lab/rules.py#L28-L81>)；[条件化掩码:36–53](<D:/paper-s3lab/src/satellite_rl/s2lab/masks.py#L36-L53>)；[最终独立动作分支:188–210、225–252](<D:/paper-s3lab/src/satellite_rl/s2lab/sac3.py#L188-L210>)。

### 5.2 十四令牌状态与七项补充

本研究把当前束几何、需求、功率、频谱结构及后续压力映射为固定长度令牌。令 $N_{\mathrm{pad}}$ 为批量场景的最大束数，$D_{\mathrm{rem}}$ 为包含当前束的剩余总需求。基础槽干扰输入为 $I/N_0$ 的 dB 映射：

$$ \tilde I_q=\min\left(1,\max\left(0,\frac{10\log_{10}(I_{t,g_t\bmod2,q}/N_0+10^{-12})+20}{40}\right)\right). $$（32）

十四令牌的实际顺序如下。经纬度归一化前以度计，宽度直接使用度值；需求除以 $10^9$ 相当于换为 Gbps。

| 位置 | 令牌及输入内容 | 投影前维数 |
| --- | --- | ---: |
| 1 | 位置：当前束纬度/90、经度/180、宽度 $b_i$ | 3 |
| 2 | 需求：$D_i/10^9$，加所需功率特征 `pneed` | 31 |
| 3 | 剩余功率：$P_{\mathrm{left}}/2000$，加需求份额特征 `share` | 2 |
| 4 | 频谱块 0：本组槽 0—9 的六列特征 | 60 |
| 5 | 频谱块 1：本组槽 10—19 的六列特征 | 60 |
| 6 | 频谱块 2：本组槽 20—29 的六列特征 | 60 |
| 7 | 频谱块 3：本组槽 30—39 的六列特征 | 60 |
| 8 | 频谱块 4：本组槽 40—49 的六列特征 | 60 |
| 9 | 频谱块 5：本组槽 50—59 的六列特征 | 60 |
| 10 | 频谱块 6：本组槽 60—69 的六列特征 | 60 |
| 11 | 频谱块 7：本组槽 70—79 的六列特征 | 60 |
| 12 | 频谱块 8：本组槽 80—89 的六列特征 | 60 |
| 13 | 频谱块 9：本组槽 90—99 的六列特征 | 60 |
| 14 | 全局：$\beta/10^9$、$t/m$、$D_{\mathrm{rem}}/(N_{\mathrm{pad}}10^9)$，加 `future` 四维和 `gspec` 十七维 | 24 |

六列槽特征依次为归一化干扰、占用、外向干扰、一瓦信噪比、一瓦信干噪比、匹配敏感度。一个块内部按“同一特征的十个槽值”依次拼接。环境还生成成员数、总束数等标量，十四令牌编码器只读取上表所列字段。

为完整给出七项补充的缩放，定义无量纲对数映射 $\mathcal D_{l,h}$，其中 $l,h$ 为 dB 上下界；以下 $\log$ 无下标时为自然对数。一瓦特征使用参考每槽功率 $p_{\mathrm{ref}}=1$ W；代码以 W 为功率数值单位，故该乘数数值为 1：

$$ \mathcal D_{l,h}(z)=\min\left(1,\max\left(0,\frac{10\log_{10}(z+10^{-30})-l}{h-l}\right)\right). $$（33）

**所需功率 `pneed`。** 使用当前束平均路径增益 $\bar\ell_i=100^{-1}\sum_q\ell_{iq}$，估计无干扰下每种槽数的功率需求。令 $A_i=\zeta S_i\bar\ell_i/N_0$，单位 W$^{-1}$，对 $c=1,\ldots,10$：

$$ e_c=\frac{D_i}{cB},\qquad \hat p_c=\frac c{A_i}\left(2^{\min(e_c,\eta)}-1\right),\qquad u_c=\mathbb{1}\{e_c>\eta\},\qquad v_c=\mathbb{1}\{\hat p_c>50\text{ 且 }e_c\le\eta\}. $$（34）

$$ z_c=\begin{cases}1.2,&u_c=1,\\\min\left(1.2,\max\left(0,\frac{\log_2(\max(\hat p_c,10^{-3})/0.5)}{\log_2 200}\right)\right),&u_c=0.\end{cases} $$（35）

三组十维向量 $z,u,v$ 按此顺序进入需求令牌，区分缺槽不可达、功率超过单束上限及可达时的功率量级。

**需求份额 `share`。** 按当前需求占剩余需求的比例分配可用功率，并封顶到 50 W，再归一化进入功率令牌：

$$ f_{\mathrm{share}}=\frac1{50}\min\left(\frac{P_{\mathrm{av}}D_i}{\max(D_{\mathrm{rem}},1)},50\right). $$（36）

**外向干扰 `outint`。** 令 $\mathcal A_t$ 为已分配束集合，$\rho^{\mathrm{pol}}_{ji}$ 只按极化取 1 或 $\chi$。估计当前束在某槽发射一瓦时，给全部已占该槽的接收束新增的干噪比总和：

$$ f_{\mathrm{out},q}=\mathcal D_{-40,20}\left(\sum_{j\in\mathcal A_t}\frac{p_{\mathrm{ref}}H_{ji}\ell_{jq}\rho^{\mathrm{pol}}_{ji}o_{jq}}{N_0}\right). $$（37）

该特征包含已分配的同组束；同组已占槽随后被起点掩码排除。它不等同于实际选定窗口后的总干扰代价。

**信噪条件 `snr`。** 两列分别为一瓦每槽的无干扰信噪比与当前干扰下的信干噪比，包含实现损耗：

$$ f_{\mathrm{snr},q}=\mathcal D_{-10,20}\left(\frac{p_{\mathrm{ref}}\zeta S_i\ell_{iq}}{N_0}\right),\qquad f_{\mathrm{sinr},q}=\mathcal D_{-10,20}\left(\frac{p_{\mathrm{ref}}\zeta S_i\ell_{iq}}{N_0+I_{iq}}\right). $$（38）

**匹配敏感度 `sens`。** 以当前容量靠近需求的程度作为权重，只纳入已分配、同极化、不同组的接收束：

$$ f_{\mathrm{sens},q}=\mathcal D_{-30,5}\left(\sum_{j\in\mathcal A_t:\ g_j\ne g_t,\ g_j\bmod2=g_t\bmod2}\exp\left(-\left|\frac{C_j}{D_j}-1\right|\right)\frac{p_{\mathrm{ref}}H_{ji}\ell_{jq}o_{jq}}{N_0+I_{jq}}\right). $$（39）

式（37）—（39）进入频谱块令牌。敏感度是加权干扰代理量，不是 SGM 对功率的解析梯度。

**后续邻域压力 `future`。** 令 $\mathcal V_t$ 为处理顺序在当前束之后、且中心夹角严格小于 $1^\circ$ 的波束集合。定义入向耦合和 $U_{\mathrm{in}}$、出向耦合和 $U_{\mathrm{out}}$：

$$ U_{\mathrm{in}}=\frac{p_{\mathrm{ref}}\bar\ell_i}{N_0}\sum_{j\in\mathcal V_t}H_{ij},\qquad U_{\mathrm{out}}=\frac{p_{\mathrm{ref}}}{N_0}\sum_{j\in\mathcal V_t}H_{ji}\bar\ell_j,\qquad f_{\mathrm{future}}=\left(\frac{\log(1+|\mathcal V_t|)}{\log151},\frac{\log(1+\sum_{j\in\mathcal V_t}D_j/10^9)}{\log151},\frac{\log_{10}(1+U_{\mathrm{in}})}3,\frac{\log_{10}(1+U_{\mathrm{out}})}3\right). $$（40）

未来束尚未定组，这里不乘极化系数，也不预设占用槽。

**全组频谱余量 `gspec`。** 令 $R_g^{\max}=\max_sR_{gs}$，十七维摘要为八组空闲比例、八组最长空闲段比例及剩余每束可用槽数的对数映射：

$$ f_{\mathrm{gspec}}=\left(\{F_g/100\}_{g=0}^{7},\{R_g^{\max}/100\}_{g=0}^{7},\frac{\log(1+U/\max(m-t,1))}{\log801}\right). $$（41）

式（40）、（41）进入全局令牌，使只观察当前组十个频谱块的网络仍能读取其它组余量和后续空间压力。

对应代码：[七项特征及维数:10–88](<D:/paper-s3lab/src/satellite_rl/s2lab/features.py#L10-L88>)；[基础归一化:279–332](<D:/paper-s3lab/src/satellite_rl/s2lab/env.py#L279-L332>)；[十四令牌实际拼接:74–128](<D:/paper-s3lab/src/satellite_rl/s2lab/nets.py#L74-L128>)；[配置到环境的传递:45–58](<D:/paper-s3lab/src/satellite_rl/s2lab/train.py#L45-L58>)。

### 5.3 场景目标增量奖励

新增一束会改善本束服务，也会改变已分配束的干扰。因此本研究在每步前后重新评价全部波束，以场景目标的真实增量奖励动作。记 $C_i^{(t)}$ 为分配前容量，场景目标 $\mathcal J_t$ 为：

$$ \mathcal J_t=\sum_{i=1}^{m}\left[\Lambda_i(C_i^{(t)},D_i)+\min\left(\frac{C_i^{(t)}}{D_i},1\right)\right]=m(F_t+\mathrm{SI}_t),\qquad r_t=\mathcal J_{t+1}-\mathcal J_t. $$（42）

未分配束容量为零，故初始目标为零，未折扣的整回合奖励和等于终态目标。训练使用 $\gamma=0.99$，仍按折扣回报更新。奖励中的满足率是束等权封顶项，不能直接替换为业务量加权的 $W$；最终配置也没有额外加入干扰损失罚项。

为区分历史口径，环境中 `reward_mode=paper` 实际计算当前束 $i$ 的满足正负奖励、SGM 与超出 50 Mbps 的偏差罚项：

$$ r_t^{\mathrm{paper\ code}}=\begin{cases}1,&C_i\ge D_i,\\-1,&C_i<D_i\end{cases}+\Lambda_i-\frac{\max(|C_i-D_i|-50\times10^6,0)}{10^9}. $$（43）

该历史代码模式只奖励当前束，没有扣回当前束给先前束带来的服务损失。所给已发表论文文字的式（10）则写为 SGM、未封顶容量需求比和非法动作罚项之和：

$$ r_t^{\mathrm{paper\ text}}=\Lambda_i+\frac{C_i}{D_i}+\Psi_i. $$（44）

其中 $\Psi_i$ 是论文针对同组冲突、总功率超限设置的罚项。因此论文原版文字、当前环境的历史命名分支、最终增量奖励应分别表述。

对应代码：[目标定义:226–244](<D:/paper-s3lab/src/satellite_rl/s2lab/env.py#L226-L244>)；[奖励计算:399–434](<D:/paper-s3lab/src/satellite_rl/s2lab/env.py#L399-L434>)。文字对照：[已发表论文奖励式:427–435](<D:/paper/outputs/midterm_report/build/paper_text.txt#L427-L435>)。

### 5.4 Transformer 编码与频谱块指针

本研究将不同语义、不同长度的输入先投影到统一维数 $d=128$。设第 $k$ 个原始令牌为 $\boldsymbol z_k$，其语义投影为 $W_k,b_k$；十个频谱块共享同一个投影。加上可学习位置参数 $\boldsymbol e_k$ 后：

$$ \boldsymbol h_k^{(0)}=W_k\boldsymbol z_k+\boldsymbol b_k+\boldsymbol e_k,\qquad k=1,\ldots,14. $$（45）

位置参数初始化为零。编码器包含两层、每层八个注意力头，每头维数 $d_h=16$，前馈中间宽度 512，激活为 ReLU，随机失活概率为零。把十四个令牌按行组成矩阵 $H^{\mathrm{tok}}$。某注意力头的查询、键、值分别记为 $Q_{\mathrm{att}},K_{\mathrm{att}},V_{\mathrm{att}}$，由该矩阵线性投影形成：

$$ Q_{\mathrm{att}}=H^{\mathrm{tok}}W_Q+\boldsymbol b_Q,\qquad K_{\mathrm{att}}=H^{\mathrm{tok}}W_K+\boldsymbol b_K,\qquad V_{\mathrm{att}}=H^{\mathrm{tok}}W_V+\boldsymbol b_V,\qquad A_{\mathrm{att}}=\mathrm{softmax}\left(\frac{Q_{\mathrm{att}}K_{\mathrm{att}}^{\mathrm T}}{d_h^{1/2}}\right)V_{\mathrm{att}}. $$（46）

各头输出拼接后经输出投影形成多头注意力 $\mathrm{MHA}$。每层采用后置层归一化，偏置按行广播：

$$ U^{\mathrm{tok}}=\mathrm{LN}\left(H^{\mathrm{tok}}+\mathrm{MHA}(H^{\mathrm{tok}})\right),\qquad H^{\mathrm{tok},+}=\mathrm{LN}\left(U^{\mathrm{tok}}+\max(0,U^{\mathrm{tok}}W_1+\boldsymbol b_1)W_2+\boldsymbol b_2\right). $$（47）

式（47）的线性映射逐令牌作用，$W_1$ 为 $128\times512$ 矩阵，$W_2$ 为 $512\times128$ 矩阵。最终取位置令牌输出 $\boldsymbol h_{\mathrm{geo}}$ 作为全局读出，保留十个块令牌输出 $\boldsymbol h_b$，$b=0,\ldots,9$。槽数头和功率头对位置读出作线性映射；起点由频谱块指针头产生。令查询向量 $\boldsymbol q=W_{\mathrm q}\boldsymbol h_{\mathrm{geo}}+\boldsymbol b_{\mathrm q}$，块内偏移向量为 $W_{\mathrm o}\boldsymbol h_b+\boldsymbol b_{\mathrm o}$，则起点 $s=10b+j$ 的分数为：

$$ z_{10b+j}=\frac{\boldsymbol h_b^{\mathrm T}\boldsymbol q}{d^{1/2}}+\left(W_{\mathrm o}\boldsymbol h_b+\boldsymbol b_{\mathrm o}\right)_j,\qquad j=0,\ldots,9. $$（48）

演员将式（48）作为起点分布的未归一化分数，评论家将其作为起点价值项。演员、两个评论家及目标评论家使用各自网络参数；最终配置的查询不以槽数为条件。此处指针是“块相关性分数＋该块内十个偏移分数”，不是另设一百个独立槽令牌。

对应代码：[十四令牌编码层:79–128](<D:/paper-s3lab/src/satellite_rl/s2lab/nets.py#L79-L128>)；[实际指针头:37–100](<D:/paper-s3lab/src/satellite_rl/s2lab/sac3.py#L37-L100>)；[演员与评论家创建:154–174](<D:/paper-s3lab/src/satellite_rl/s2lab/sac3.py#L154-L174>)。

### 5.5 完整动作分布上的离散 SAC 更新

本研究用完整动作联合分布计算软价值与熵。本小节 $c$ 表示申请槽数 $\hat c_t$，$\ell=1,\ldots,13$ 按式（29）的顺序表示功率档。最终 `policy=indep`、离散功率、`q_power=plain` 对应：

$$ \pi(a\mid o)=\pi_c(c\mid o)\pi_s(s\mid o)\pi_p(\ell\mid o),\qquad Q_j(o,c,s,\ell)=q_{j,c}(o,c)+q_{j,s}(o,s)+q_{j,p}(o,\ell),\quad j=1,2. $$（49）

每个头先对不允许选项赋分 $-10^4$，再作对数柔性最大归一化；浮点实现中这些选项的概率下溢为零。配置中的 `pi_power=count` 用于连续功率分支的条件化选择，在当前独立离散策略下不把功率分布改成槽数条件分布。

令 $w_a=\pi(a\mid o)$。对十种槽数、一百个起点、十三档功率形成的完整动作数组精确求和，不通过单次动作采样估计期望：

$$ V_j(o)=\sum_{c=1}^{10}\sum_{s=0}^{99}\sum_{\ell=1}^{13}w_{c,s,\ell}Q_j(o,c,s,\ell),\qquad \mathcal H(o)=-\sum_{c,s,\ell}w_{c,s,\ell}\log w_{c,s,\ell}. $$（50）

两个目标评论家记为 $\bar Q_1,\bar Q_2$，由它们得到 $\bar V_1,\bar V_2$。对回放样本 $(o,a,r,o',d_{\mathrm{end}})$，终止标志 $d_{\mathrm{end}}\in\{0,1\}$，目标值为：

$$ y=r+\gamma(1-d_{\mathrm{end}})\left[\min\left(\bar V_1(o'),\bar V_2(o')\right)+\alpha\mathcal H(o')\right],\qquad \gamma=0.99,\quad\alpha=0.01. $$（51）

即先对完整动作求期望，再取两个评论家较小者。它不同于逐动作取小后求期望，也不同于逐头取小再相加。评论家参数 $\psi_1,\psi_2$ 最小化两个均方误差之和；批量大小为 $B_{\mathrm{tr}}=256$：

$$ \mathcal L_Q=\frac1{B_{\mathrm{tr}}}\sum_{b=1}^{B_{\mathrm{tr}}}\left[\left(Q_1(o_b,a_b)-y_b\right)^2+\left(Q_2(o_b,a_b)-y_b\right)^2\right]. $$（52）

演员在每个观测处选择期望较小的整个评论家，令 $j^*(o)=1$ 当 $V_1(o)\le V_2(o)$，否则为 2。选出的评论家输出停止梯度，其演员损失为：

$$ \mathcal L_\pi=\frac1{B_{\mathrm{tr}}}\sum_{b=1}^{B_{\mathrm{tr}}}\sum_a\pi(a\mid o_b)\left[\alpha\log\pi(a\mid o_b)-Q_{j^*(o_b)}(o_b,a)\right]. $$（53）

实现复用本次评论家更新前前向所得的价值并分离梯度，选择分支本身也不反向传播。温度固定为 $0.01$，不更新温度。每次梯度更新后，两个目标评论家分别软更新：

$$ \bar\psi_j\leftarrow(1-\tau)\bar\psi_j+\tau\psi_j,\qquad\tau=0.005,\qquad j=1,2. $$（54）

最终配置采用演员学习率 $10^{-5}$、评论家学习率 $10^{-4}$、回放容量十万、热身两千步和梯度范数裁剪 10。热身后按每个有效环境步约一次更新的配置训练；评价时各头取最大概率动作。回合最后一束的目标不向下一场景自举。

对应代码：[完整 Q、联合权重与软目标:273–325](<D:/paper-s3lab/src/satellite_rl/s2lab/sac3.py#L273-L325>)；[评论家、演员与软更新:327–399](<D:/paper-s3lab/src/satellite_rl/s2lab/sac3.py#L327-L399>)；[掩码分数:12–13、242–243](<D:/paper-s3lab/src/satellite_rl/s2lab/nets.py#L242-L243>)；[回放保存:14–67](<D:/paper-s3lab/src/satellite_rl/s2lab/agents.py#L14-L67>)；[最终训练参数:36–50](<D:/paper/outputs/experiments/s2lab_round4_20261028/account_B/runs4/r4-ptr-s1/config.json#L36-L50>)。

## 六、第三阶段：MCTR-SCA 连续功率细调

### 6.1 匹配约束与服务量辅助变量

第二阶段冻结频率组、起点和实际槽数后，功率仍可在连续区间内调整。本研究以其功率 $\boldsymbol p^0$ 为起点，记 $F_0=F(\boldsymbol p^0)$。有槽束集合为 $\mathcal I=\{i:c_i>0\}$，$M=|\mathcal I|$，仅这些波束的功率作为变量：

$$ \max_{\boldsymbol p}W(\boldsymbol p)=\frac{\sum_{i=1}^{m}\min(C_i(\boldsymbol p),D_i)}{D_{\mathrm{tot}}},\qquad F(\boldsymbol p)\ge F_0-\delta,\qquad 0.5\le p_i\le50\ (i\in\mathcal I),\qquad p_i=0\ (i\notin\mathcal I),\qquad \sum_{i\in\mathcal I}p_i\le2000. $$（55）

其中 $D_{\mathrm{tot}}=\sum_iD_i$，功率以 W 计；主设定 $\delta=0$，放宽实验使用正值。SGM 对全部 $m$ 个真实波束平均，无槽束依然计入，不能改成只对 $M$ 个有槽束平均。

为避免直接优化服务量的折点，引入辅助服务量 $y_i$，单位 bit/s：

$$ \max_{\boldsymbol p,\boldsymbol y}\frac{\sum_{i\in\mathcal I}y_i}{D_{\mathrm{tot}}},\qquad y_i\le C_i(\boldsymbol p),\qquad y_i\le D_i. $$（56）

固定功率时目标对 $y_i$ 单调递增，最优值必取 $y_i=\min(C_i,D_i)$，故与式（55）等价。准确说这是服务量函数的下图约束：辅助变量被容量和需求从上方限制。式（56）对服务量变量线性，容量对功率仍是非线性的；将容量局部线性化后才得到完整线性子问题。代码未给辅助量设置零下界，在容量线性近似为负时仍允许子问题继续生成候选。

对应代码：[变量集合、上下限与总需求:42–82](<D:/paper-s3lab/src/satellite_rl/s3lab/problem.py#L42-L82>)；[目标与匹配指标:90–105](<D:/paper-s3lab/src/satellite_rl/s3lab/problem.py#L90-L105>)；[辅助量与线性规划边界:154–175](<D:/paper-s3lab/src/satellite_rl/s3lab/methods.py#L154-L175>)。

### 6.2 逐槽功率系数与容量雅可比

固定频率方案后，把每个占用槽 $q\in\mathcal S_i$ 的信号与干扰除以噪声，定义系数 $a_{iq},b_{iqj}$，单位均为 W$^{-1}$：

$$ a_{iq}=\frac{\zeta S_i\ell_{iq}}{c_iN_0},\qquad b_{iqj}=\frac{H_{ij}\rho^{\mathrm{pol}}_{ij}\ell_{iq}o_{jq}}{c_jN_0},\qquad b_{iqi}=0,\qquad i,j\in\mathcal I. $$（57）

其中 $\rho^{\mathrm{pol}}_{ij}$ 按极化取 1 或 $\chi$；在合法固定频率方案中，同组的 $o_{jq}$ 在接收束槽上为零。归一化干扰加噪声为 $\nu_{iq}$：

$$ \nu_{iq}=1+\sum_{j\in\mathcal I}b_{iqj}p_j,\qquad \Gamma_{iq}=\frac{a_{iq}p_i}{\nu_{iq}},\qquad \frac{\partial\Gamma_{iq}}{\partial p_j}=\frac{a_{iq}\mathbb{1}\{i=j\}}{\nu_{iq}}-\frac{\Gamma_{iq}b_{iqj}}{\nu_{iq}}. $$（58）

式（58）第一项给出本束功率的直接增益，第二项给出其它束增功率所带来的干扰损失。记槽速率导数权重为 $g_{iq}$，封顶及恰好到达封顶的槽按代码取零：

$$ g_{iq}=\begin{cases}\frac{B}{\log(2)\,(1+\Gamma_{iq})},&\log_2(1+\Gamma_{iq})<\eta,\\0,&\log_2(1+\Gamma_{iq})\ge\eta,\end{cases}\qquad J_{ij}=\frac{\partial C_i}{\partial p_j}=\sum_{q\in\mathcal S_i}g_{iq}\left(\frac{a_{iq}\mathbb{1}\{i=j\}}{\nu_{iq}}-\frac{\Gamma_{iq}b_{iqj}}{\nu_{iq}}\right). $$（59）

这里 $\log(2)$ 为 2 的自然对数。$J_{ii}$ 非负，非对角项非正；功率优化因而同时计算自身容量收益和对其它束的干扰影响。

对 SGM 约束，先求式（13）的容量导数：

$$ u_i'=\begin{cases}1/D_i,&C_i\ge D_i,\\D_i/C_i^2,&0<C_i<D_i,\end{cases}\qquad \Lambda_i'=-3(1-\exp(-h_i))^2\exp(-h_i)\frac{u_i u_i'+v_i/\beta}{h_i}. $$（60）

当 $h_i=0$ 时，代码把最后的模长导数置零，故 $C_i=D_i$ 时 $\Lambda_i'=0$；容量为零时也置零。由链式法则，匹配梯度与需求满足率的代码选定单侧梯度为：

$$ (\boldsymbol g_F)_j=\frac1m\sum_{i\in\mathcal I}\Lambda_i'J_{ij},\qquad (\boldsymbol g_W)_j=\frac1{D_{\mathrm{tot}}}\sum_{i\in\mathcal I}\mathbb{1}\{C_i<D_i\}J_{ij}. $$（61）

容量等于或超过需求的束在 $\boldsymbol g_W$ 中不贡献导数；但 MCTR-SCA 的主子问题使用式（56）的服务量辅助变量，而不是直接把 $\boldsymbol g_W^{\mathrm T}\boldsymbol\Delta$ 作为目标。这保留了线性容量模型跨越需求折点时的封顶结构。

对应代码：[归一化系数:63–80](<D:/paper-s3lab/src/satellite_rl/s3lab/problem.py#L63-L80>)；[SGM 导数:20–35](<D:/paper-s3lab/src/satellite_rl/s3lab/problem.py#L20-L35>)；[容量雅可比和梯度:108–125](<D:/paper-s3lab/src/satellite_rl/s3lab/problem.py#L108-L125>)。

### 6.3 匹配线性化与信赖域线性规划

容量含干扰耦合，SGM 又要求接近需求，式（55）整体为非凸问题。本研究在当前真实可行点 $\boldsymbol p^{(k)}$ 附近建立一阶模型，以功率增量 $\boldsymbol\Delta$ 为变量：

$$ \hat C_i(\boldsymbol\Delta)=C_i(\boldsymbol p^{(k)})+\sum_{j\in\mathcal I}J_{ij}^{(k)}\Delta_j,\qquad \hat F(\boldsymbol\Delta)=F(\boldsymbol p^{(k)})+(\boldsymbol g_F^{(k)})^{\mathrm T}\boldsymbol\Delta. $$（62）

记实际代码阈值为 $F_{\mathrm{thr}}=F_0-\delta-10^{-12}$，其中 $10^{-12}$ 为匹配比较容差。信赖域限制 $\|\boldsymbol\Delta\|_\infty\le R_k$，曲率余量为 $\mu_kR_k^2$，初值 $R_0=5$ W、$\mu_0=0$。令 $\tilde y_i=y_i/10^6$ 以 Mbps 计，线性子问题为：

$$ \max_{\boldsymbol\Delta,\tilde{\boldsymbol y}}\sum_{i\in\mathcal I}\tilde y_i,\qquad \tilde y_i\le\frac{C_i^{(k)}+\sum_jJ_{ij}^{(k)}\Delta_j}{10^6},\qquad \tilde y_i\le\frac{D_i}{10^6}. $$（63）

$$ F^{(k)}+(\boldsymbol g_F^{(k)})^{\mathrm T}\boldsymbol\Delta\ge F_{\mathrm{thr}}+\mu_kR_k^2,\qquad \sum_j\Delta_j\le2000-\sum_jp_j^{(k)},\qquad \max(0.5-p_j^{(k)},-R_k)\le\Delta_j\le\min(50-p_j^{(k)},R_k). $$（64）

辅助变量 $\tilde y_i$ 没有下界。代码另将 SGM 约束两边乘以 $1/\max(\|\boldsymbol g_F\|_\infty,10^{-12})$，用于改善数值尺度，不改变可行集。子问题是线性规划，使用 HiGHS 求解。其预测满足率提升为：

$$ \hat{\Delta W}_k=\frac{10^6\sum_i\tilde y_i^*-\sum_{i\in\mathcal I}\min(C_i^{(k)},D_i)}{D_{\mathrm{tot}}}. $$（65）

若子问题无解或预测提升小于 $10^{-7}$，且当前曲率余量系数为正，则先把 $\mu_k$ 清零重新求解；清零后仍无解则半径减半，已有解但预测提升不足则停止。半径低于 $10^{-3}$ W 时也停止。默认最多 200 轮、2000 次真实评价，并遵循调用端时限。

对应代码：[LP 的完整构造:154–175](<D:/paper-s3lab/src/satellite_rl/s3lab/methods.py#L154-L175>)；[初始化、余量清零及停止:178–204](<D:/paper-s3lab/src/satellite_rl/s3lab/methods.py#L178-L204>)。

### 6.4 真实值接受、可行性修复与单调性

线性模型只负责提出候选，接受由真实容量、真实 $W$ 和真实 SGM 决定。对子问题候选 $\boldsymbol p^{\mathrm c}=\boldsymbol p^{(k)}+\boldsymbol\Delta$，直接接受条件为：

$$ W(\boldsymbol p^{\mathrm c})>W(\boldsymbol p^{(k)})+10^{-12},\qquad F(\boldsymbol p^{\mathrm c})\ge F_{\mathrm{thr}}. $$（66）

直接接受后将信赖域扩大一倍，上限 50 W，同时减半余量系数：

$$ R_{k+1}=\min(2R_k,50),\qquad \mu_{k+1}=\mu_k/2. $$（67）

若候选 SGM 不达标，计算线性模型对匹配程度的高估量 $e_k$，用实测弯曲程度更新余量系数：

$$ e_k=F^{(k)}+(\boldsymbol g_F^{(k)})^{\mathrm T}\boldsymbol\Delta-F(\boldsymbol p^{\mathrm c}),\qquad \mu_k\leftarrow\max\left(2\mu_k,\frac{e_k}{\max(\|\boldsymbol\Delta\|_\infty,10^{-9})^2}\right). $$（68）

如果该被拒候选仍有 $W(\boldsymbol p^{\mathrm c})>W(\boldsymbol p^{(k)})$，算法在候选点重新计算容量雅可比和 SGM 梯度，使用半径 $R_k/2$、余量 $\mu_k(R_k/2)^2$ 再解一次同型线性规划。修复增量记为 $\boldsymbol d$，其匹配约束明确补足候选点的真实缺口：

$$ (\boldsymbol g_F^{\mathrm c})^{\mathrm T}\boldsymbol d\ge F_{\mathrm{thr}}-F(\boldsymbol p^{\mathrm c})+\mu_k(R_k/2)^2,\qquad \|\boldsymbol d\|_\infty\le R_k/2. $$（69）

修复点 $\boldsymbol p^{\mathrm{cc}}=\boldsymbol p^{\mathrm c}+\boldsymbol d$ 仍须相对于原接受点满足式（66），并在修复子问题中满足绝对功率边界与总预算。修复成功后保留原半径和更新后的余量系数；否则保留原功率，令半径减半。若首次候选只是不提高 $W$，但 SGM 达标，则不作上述修复，直接缩小半径。实现没有采用实际收益与预测收益之比来更新半径。

可行性与单调性可由归纳得到。初始功率 $\boldsymbol p^0$ 满足式（55）；每个 LP 候选满足功率盒约束和预算，只有真实 SGM 达标且真实 $W$ 提高才替换当前点；被拒时当前点不变。因此保留的接受序列始终满足匹配阈值，$W$ 单调不降，且 $0\le W\le1$。这一保证适用于接受点及返回点；试探候选可以违反 SGM，曲率余量也不是已证明的全局保守下界。严格匹配按代码的 $10^{-12}$ 容差实现，不由线性化本身保证。

对应代码：[真实接受、弯曲余量和修复:205–231](<D:/paper-s3lab/src/satellite_rl/s3lab/methods.py#L205-L231>)；[限时保留最好点:30–70](<D:/paper-s3lab/src/satellite_rl/s3lab/methods.py#L30-L70>)。

### 6.5 所有方法统一的保持匹配接入

为把各类功率方法接入同一三阶段链路，本研究以第二阶段原功率作为统一保底点。设方法原生输出为 $\boldsymbol p^{\mathrm m}$，先评价其真实指标；若满足以下条件，直接接入该输出：

$$ F(\boldsymbol p^{\mathrm m})\ge F_{\mathrm{thr}},\qquad W(\boldsymbol p^{\mathrm m})\ge W(\boldsymbol p^0). $$（70）

否则在两端连线上取二十一个等距点，插值系数 $\xi$ 与 SAC 的软更新率 $\tau$ 区分：

$$ \boldsymbol p(\xi)=\boldsymbol p^0+\xi(\boldsymbol p^{\mathrm m}-\boldsymbol p^0),\qquad \xi\in\{0,0.05,0.10,\ldots,1\}. $$（71）

从满足 $F(\boldsymbol p(\xi))\ge F_{\mathrm{thr}}$ 的点中按字典序最大化 $(W(\boldsymbol p(\xi)),-\sum_ip_i(\xi))$，即先取最高满足率，并列再取较省功率的点。$\xi=0$ 作为初始候选保留，循环评价其余二十点。

功率上下限和总预算构成凸集合，两个端点合法时连线上全部点均满足这些资源约束；原功率始终通过 SGM 筛选，故接入输出的 $W$ 不低于第二阶段。SGM 沿连线不一定单调，代码逐点真实评价，不使用二分搜索。式（70）的直接接入分支不比较内部点，也不在 $W$ 恰好相等时额外检查是否省功率；省功率并列规则只用于二十一点搜索分支。方法每产生新的最好原生输出就当场接入，接入评价耗时计入同一时限。

对应代码：[统一接入:537–553](<D:/paper-s3lab/src/satellite_rl/s3lab/methods.py#L537-L553>)；[接入记录时刻:30–56](<D:/paper-s3lab/src/satellite_rl/s3lab/methods.py#L30-L56>)。

## 代码与文字描述不一致之处

1. **贪心定组不直接计算容量。** 用户描述中的“容量感知贪心先定组”应改为“按平均干扰及空闲槽数贪心定组”。容量感知计算发生在规则基线的槽数和功率决策中。依据：[定组:261–277](<D:/paper-s3lab/src/satellite_rl/s2lab/env.py#L261-L277>)、[规则基线:43–81](<D:/paper-s3lab/src/satellite_rl/s2lab/rules.py#L43-L81>)。

2. **v2 收窄目标是张角总和，且与峰值二分分支不同。** “最小化总束宽”应改为式（26）；收窄在当前峰值和人数上限内求距离费用流，不是每次同时重新最小化峰值与平方和。已有原创性调研已指出这一点。依据：[收窄指派:152–202](<D:/paper-s3lab/src/satellite_rl/coverage/s1fix/skeleton.py#L152-L202>)、[收窄接受:531–566](<D:/paper-s3lab/src/satellite_rl/coverage/s1fix/skeleton.py#L531-L566>)、[调研说明:60–62](<D:/paper/docs/第一阶段/实验报告/30_FLOW-AO原创性与已有方法调研.md#L60-L62>)。

3. **K=26 有明确的保覆盖回退。** 布局参考容量仍为 $1.125$ Gbps，而非 v3 的 $1.375$ Gbps；叠放预算或时间用尽时可返回人数超限但几何合法的布局。不能把“每束人数上限”扩写成所有返回布局的硬保证。依据：[K 入口及超限记录:87–109](<D:/paper-s3lab/src/satellite_rl/s2fix/data.py#L87-L109>)、[叠放回退:579–595](<D:/paper-s3lab/src/satellite_rl/coverage/s1fix/skeleton.py#L579-L595>)、[最终检查:629–637](<D:/paper-s3lab/src/satellite_rl/coverage/s1fix/skeleton.py#L629-L637>)。

4. **基础讲解中的部分最优性与外层接受说法需要收窄。** “不加间隔就一定达到下界”不是代码保证，应改成“可行输出峰值达到下界时，该景峰值最优”；自适应初始化代码寻找预算内的最小 $k$，不是讲解所述“恰好用满预算的最大 $k$”；重定中心可仅凭合法性接受，拆束可在峰值不升时接受，也非每步三元组严格改善。依据：[原讲解:45、51–58、79–81](<D:/paper/docs/Claude维护/S1-FIX第一阶段纠偏/06_FLOW-AO覆盖骨架算法讲解.md#L45-L81>)、[初始化:255–263](<D:/paper-s3lab/src/satellite_rl/coverage/s1fix/skeleton.py#L255-L263>)、[外层接受:442–495](<D:/paper-s3lab/src/satellite_rl/coverage/s1fix/skeleton.py#L442-L495>)。

5. **重定中心是投影平面最小包围圆。** 应按式（24）描述，不能直接称为精确球面最小包络锥；热点同站点副本豁免相互间隔，不能照搬“任意两束均满足间隔”。依据：[几何更新:71–84](<D:/paper-s3lab/src/satellite_rl/coverage/s1fix/geom.py#L71-L84>)、[叠放豁免:152–159](<D:/paper-s3lab/src/satellite_rl/coverage/s1fix/geom.py#L152-L159>)。

6. **动态模块不在指定第三阶段工作树中。** 本文已核读原第一阶段工作树的动态源码，注明来源。其默认预算为 100 束，轻量动作是五轮、$\kappa=3$，没有当前静态 v2 的人数上限与叠放流程；且通用动态入口按动作选择修复，不能把所有动态运行都写成固定每帧轻量修复。依据：[动态预算:12–14](<C:/Users/Admin/.codex/worktrees/s1fix/paper/src/satellite_rl/coverage/s1fix/dynamic.py#L12-L14>)、[动作分支:353–377](<C:/Users/Admin/.codex/worktrees/s1fix/paper/src/satellite_rl/coverage/s1fix/dynamic.py#L353-L377>)。

7. **当前链路区别于开题和原论文的中心超级用户、完全极化隔离模型。** 有用增益采用逐机离轴线性平均，干扰与路径使用接收束中心；异极化有 20 dB 泄漏，逐槽频谱效率封顶 5.5；总功率按实际槽数均摊。原论文的雨衰、云衰分项在当前代码中没有独立实现，当前是固定 5 dB 其它损耗和 1 dB 实现损耗。依据：[物理实现:21–84、112–144、163–192](<D:/paper-s3lab/src/satellite_rl/s2lab/physics.py#L21-L192>)；文字对照：[原论文链路:176–253](<D:/paper/outputs/midterm_report/build/paper_text.txt#L176-L253>)。

8. **布局生成需求与资源分配需求不同。** $43.2$ Mbps/机用于冻结布局和 K；资源分配在同景使用统一的 $30$—$80$ Mbps 抽样值，不能写成所有阶段所有场景永远固定 $43.2$ Mbps。依据：[场景参数:21–34](<D:/paper-s3lab/src/satellite_rl/s2lab/scenes.py#L21-L34>)、[需求载入:157–173](<D:/paper-s3lab/src/satellite_rl/s2lab/env.py#L157-L173>)。

9. **最终策略为独立申请、执行截短。** 不能套用按槽数条件化的完整窗口策略，不能说申请槽数在选定起点一定完整执行；十三档功率也不是原论文的均匀分档。配置中的 `pi_power=count` 不改变当前独立离散功率头。依据：[独立分支:188–210](<D:/paper-s3lab/src/satellite_rl/s2lab/sac3.py#L188-L210>)、[执行逻辑:372–398](<D:/paper-s3lab/src/satellite_rl/s2lab/env.py#L372-L398>)；文字对照：[原论文动作:387–425](<D:/paper/outputs/midterm_report/build/paper_text.txt#L387-L425>)。

10. **奖励存在三种不同口径。** 原论文文字为 $\Lambda_i+C_i/D_i+\Psi_i$；环境 `paper` 分支为正负满足奖加 SGM 减容量偏差罚；最终配置为全部波束 $\sum_i(\Lambda_i+\sigma_i)$ 的增量。最终目标是 $m(F+\mathrm{SI})$，不是 $m(F+W)$，也不是单独最大化 SGM。依据：[论文奖励:427–435](<D:/paper/outputs/midterm_report/build/paper_text.txt#L427-L435>)、[目标和奖励:242–244、418–429](<D:/paper-s3lab/src/satellite_rl/s2lab/env.py#L418-L429>)。

11. **最终 SAC 采用固定温度和“期望后取小”。** 原论文的可学习温度与逐动作最小 Q 写法不能直接沿用。最终回放容量为十万，也不同于初版口径列出的两万。依据：[目标及演员选择:304–399](<D:/paper-s3lab/src/satellite_rl/s2lab/sac3.py#L304-L399>)、[最终配置:36–50](<D:/paper/outputs/experiments/s2lab_round4_20261028/account_B/runs4/r4-ptr-s1/config.json#L36-L50>)；文字对照：[原论文温度和更新:555–619](<D:/paper/outputs/midterm_report/build/paper_text.txt#L555-L619>)、[初版口径:54](<D:/paper-s2lab/docs/Claude维护/S2-LAB/00_口径.md#L54>)。

12. **第三阶段不是直接线性化 W 后优化其梯度。** 实际子问题线性化容量、引入受需求上限约束的服务量变量，同时线性化 SGM 并加曲率余量；辅助量无零下界。用户所称“上镜图”在此应写为“服务量下图约束”，完整约束在容量线性化后才成为 LP。依据：[实际 LP:154–175](<D:/paper-s3lab/src/satellite_rl/s3lab/methods.py#L154-L175>)；文字对照：[第三阶段方案表:41](<D:/paper-s3lab/docs/Claude维护/S3-LAB/03_S3-LAB方案定稿.md#L41>)。

13. **真实可行性保证针对保留点，带匹配数值容差。** 接受阈值为 $F_0-\delta-10^{-12}$；候选点可不满足 SGM，需真实核验。曲率余量来自被拒候选的实际高估，修复成功不执行直接接受时的扩大半径、减半余量。依据：[真实接受与修复:178–231](<D:/paper-s3lab/src/satellite_rl/s3lab/methods.py#L178-L231>)。

14. **二十一点接入有直接采用分支。** 方法输出已满足 SGM 且 W 不低于原值时直接用；否则才搜索二十一点，并列省功率也只在搜索分支执行。方案开头已更新这一点，正文“统一取二十一点”的旧说法及用户概述需按最终实现修订。依据：[接入实现:538–553](<D:/paper-s3lab/src/satellite_rl/s3lab/methods.py#L538-L553>)；文字对照：[方案更新:9](<D:/paper-s3lab/docs/Claude维护/S3-LAB/03_S3-LAB方案定稿.md#L9>)、[方案旧段落:51](<D:/paper-s3lab/docs/Claude维护/S3-LAB/03_S3-LAB方案定稿.md#L51>)。

## Claude 审阅与回填（2026-10-29）

**结论：推导与代码一致，全部采用；文末 14 处"代码与文字不一致"全部接受，正文按代码口径写。**

逐项核对了与本人已读代码重合的部分：S.672-4 三段方向图（LN=−20 dB 时主瓣边界 2.58、平台至 6.32、远旁瓣 0 dBi 地板）、逐槽封顶容量式（12）、SGM 式（13）—（14）、定组规则式（28）、独立三头与执行截短式（31）、块指针打分式（48）、"先求期望后取小"的目标式（51）与演员按期望选评论家式（53）、MCTR-SCA 子问题式（63）—（64）、弯曲余量式（68）与修复步式（69）、21 点接入式（70）—（71），均与 sac3.py、methods.py、env.py 一致，没有需要再讨论的分歧。

对正文的取舍（中期报告篇幅有限，只取主干，其余留在本文件备查）：

| 正文小节 | 采用的式子 | 说明 |
| --- | --- | --- |
| 4.1 系统模型 | （3）（4）（5）（6）（7）（10）（11）（12）（13）（14）（15）（16） | 几何式（1）（2）用文字带过；补"三阶段口径表" |
| 4.2 FLOW-AO | （17）（18）（19）（21）（22）（23）（25）（26）（27） | 下界只写"可行布局峰值等于下界即该景最优"；重定中心按投影平面最小包围圆描述 |
| 4.3 块指针 T-SAC | （28）（29）（31）（42）（45）（48）（49）（50）（51）（53） | 七项补充只列含义，不列缩放式（33）—（41）；奖励三种口径只在正文点明"初版为逐束奖励、最终为场景增量奖励" |
| 4.4 MCTR-SCA | （55）（56）（58）（59）（63）（64）（66）（68）（69）（70）（71） | 性质只写"接受点真实可行、W 单调不降" |

据本文修订的说法：①定组规则写"按平均干扰与空闲槽数选组"，不称"容量感知"；②v2"尽量窄"写"在峰值与人数上限内最小化飞机到所属中心的张角总和"；③K=26 写成 v2 的人数约束，并注明本数据集 1480 景全部满足（erify_s1.py 实测 0 个超限波束）；④初始化写"二分 k 使中心数不超过预算"；⑤最终奖励写 $\mathcal J=m(F+\mathrm{SI})$ 的增量，不写成 W；⑥"上镜图"改为"服务量辅助变量（下图约束）"。

另：Astra 指出第二阶段"初版"所用的 paper 奖励分支与已发表论文文字中的奖励式不同。正文把初版称为"逐束奖励"（只评价当前波束），不称"论文原版奖励"，避免与已发表论文混淆。
