"""提交信息（供 git commit -F 使用）。"""

MESSAGE = """feat: 实现 S2-P18 第二轮可变束数三条实验臂

按《第二阶段方案18 第二版》实现三条实验臂，全部改动在开发分支完成，未在 main 上
修改工作区，未创建训练、评价或工作台任务，未启动任何训练。

改动 1（奖励，V2 核心）：新增可选奖励预设 legacy_repaired_throughput，在保留原四项
（阈值满足率、本束 SGM、SGM 偏差惩罚、占用冲突比例）的前提下追加
w_throughput×ΔΣmin(C,D)/参考量 与 w_spectrum×(本步占用槽数/本组可用槽)。默认权重为 0
即关闭，legacy_repaired 逐位不变；原始奖励分量仍在 info.reward_terms 与
info.legacy_reward 中逐项保留。

改动 2（power_pressure 归一化）：参考功率由写死的 30 W 改为公平份额 budget/N；N=100 时
3000/100 与 30 是同一浮点数，100 束逐位不变。新增 group_remaining_obligation 与
remaining_slot_limit 两个标量，口径与动作掩码的 remaining_group/max_slots 一致。

改动 3（组号可见）：把 group 纳入 S2-P18 可变束数特征集（共 213 维），保留原 210 维冻结
档案，state_dim/config_summary/模型元数据同步。

改动 4（进度分层采样）：Replay 在数量桶之上增加回合内进度分位分层，批大小须能被
（数量桶×进度段）整除；每桶、每进度段的实际采样数与有放回次数写入
status.json/metrics.jsonl 并可随断点保存恢复；旧断点（无分层字段）按原方式加载。

改动 5（按回合长度设定折扣）：新增 training.gamma_schedule（none/constant/
by_beam_count），默认 none 即保持 0.99 且旧行为逐位不变；V3 使用
by_beam_count(=0.99)/default 0.995，按束数解析并写入环境与模型/断点元数据。

改动 6（跨束势能型奖励整形）：新增 w_shape×（γΦ(s')−Φ(s)），势函数
Φ(s)=mean_i min(C_i/D_i,1)（含未服务束，分母固定为场景束数）；权重默认 0，
由 tests/test_s2p18_arms.py 的确定性小例核对“只含端点势函数、不改变最优策略”。

改动 7（逐束实体编码 G0）：逐束上下文可按场景实际束数构造（不填充到 100），读出支持
attention_pooling（逐实体打分加权平均，对实体顺序严格不变）与原 first_token 对照，
并用 FiLM 做 N 条件调制；两者的可选开关写入配置、目录与模型元数据。

改动 8（实验编号集合化）：新增 workbench/experiments.py 定义
VARIABLE_BEAM_EXPERIMENTS=("S2-P17","S2-P18")；常规链路的场景来源校验接受集合成员，
而任务自身的 experiment_id（训练任务、案例合同、细调与评价记录）沿用请求值，
S2-P17 原路径逐位不变。recovery.py 的数值边界修复与 power_refinement/runner.py 的
失败 B 回退仍只认 S2-P17 精确匹配，刻意不放宽。

改动 9（前端同步）：workbench/static/app.js 与 index.html 增加奖励分量、折扣日程与进度
分层段数、逐束上下文/读出/N 条件调制的可编辑控件与三臂一键预设，运行详情新增
“按束数分档”面板（训练进度逐 N；评价节点到达后按 N 显示成绩）；catalog 同步登记
新预设、新特征、读出模式与三臂预设。

验证（本地，DQN 环境，CPU 与 RTX 4070 SUPER）：
- python -m unittest discover -s tests：704 项，失败 2 项（test_variable_beam_ops 的
  8781 端口占用检查，改动前即在 main 上同样失败）、跳过 4。
- 旧 100 束逐位对照（scripts/s2p18/reference_snapshot.py，改动前参照 vs 改动后）：
  210/207/204 维状态与奖励序列逐位相同，配置哈希全部一致；仅 N=150 的
  power_pressure 分量按新口径变化（0.397574→0.497470 等，修正而非新增信息）。
- 三臂配置核验（scripts/s2p18/verify_arms.py）：/api/validate 同路径校验通过，
  逐键回读证明配置未被静默丢弃；各做一次前向与单次参数更新冒烟（CUDA 亦通过）。
- G0 顺序置换一致性、N=50/100/200 前向与掩码构造、占位不参与池化：
  tests/test_s2p18_config_effect.py 与 tests/test_s2p18_arms.py。

未完成/未验证：未做任何训练、评价或共同考核；三臂的泛化效果未验证；N>100 的
gamma 日程只做了机制与元数据核对，未做学习效果对照。
"""

if __name__ == "__main__":
    print(MESSAGE)
