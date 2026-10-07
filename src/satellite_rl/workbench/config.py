"""Versioned, strict configuration and discoverable workbench capabilities."""

from copy import deepcopy
import math
from satellite_rl.envs.regions import get_profile, profile_catalog, profile_metadata
from satellite_rl.state_layouts import (NATURAL_CONTEXT_LEVELS, natural_dim,
                                        natural_shapes, natural_token_count,
                                        S2P18_OPTIONAL_FEATURES, S2P18_VARIABLE_FEATURES,
                                        VARIABLE_BEAM_FEATURES)
from satellite_rl.workbench.experiments import (S2P18, VARIABLE_BEAM_EXPERIMENTS,
                                                is_variable_beam_experiment)


FEATURES = [
    dict(id="beam", label="当前波束", description="经纬度与业务需求", shape=[3], unit="归一化", timing="动作前", source="当前波束", visibility="both"),
    dict(id="power", label="剩余功率", description="剩余功率占预算比例", shape=[1], unit="比例", timing="动作前", source="已分配功率", visibility="both"),
    dict(id="interference", label="逐槽干扰", description="当前组100槽的接收干扰归一化", shape=[100], unit="归一化dB", timing="动作前", source="已分配波束", visibility="both"),
    dict(id="occupancy", label="本组占用", description="当前组100槽的占用次数", shape=[100], unit="次数", timing="动作前", source="已分配波束", visibility="both"),
    dict(id="progress", label="处理进度", description="已处理波束占全部波束比例", shape=[1], unit="比例", timing="动作前", source="处理序号", visibility="both"),
    dict(id="remaining_count", label="剩余波束数", description="含当前波束的剩余数量比例", shape=[1], unit="比例", timing="动作前", source="处理序号", visibility="both"),
    dict(id="group", label="当前组号", description="组号除以7", shape=[1], unit="比例", timing="动作前", source="处理序号", visibility="both"),
    dict(id="power_per_remaining", label="每束可用功率", description="剩余功率除以剩余束数，再除以单束参考30W", shape=[1], unit="比例", timing="动作前", source="已分配功率", visibility="both"),
    dict(id="beta", label="平均业务需求", description="全场景平均需求除以600 Mbps", shape=[1], unit="比例", timing="动作前", source="全场景需求", visibility="known"),
    dict(id="remaining_demand", label="剩余业务需求", description="含当前波束的剩余总需求除以全部波束参考需求", shape=[1], unit="比例", timing="动作前", source="全场景需求", visibility="known"),
    dict(id="beam_count_normalized", label="实际波束数", description="实际目标束数除以200", shape=[1], unit="比例", timing="动作前", source="场景", visibility="both"),
    dict(id="remaining_beams_normalized", label="剩余波束数/200", description="含当前待分配束，终态为0", shape=[1], unit="比例", timing="动作前", source="处理序号", visibility="both"),
    dict(id="group_remaining_obligation", label="本组剩余预留义务", description="本复用组尚未处理束数（含当前束）除以总束数；与掩码的剩余预留口径一致", shape=[1], unit="比例", timing="动作前", source="处理序号", visibility="both"),
    dict(id="remaining_slot_limit", label="剩余合法槽上限", description="本组空槽数减去为同组其余束各预留1槽后，当前束可用的槽数除以100", shape=[1], unit="比例", timing="动作前", source="真实频率账", visibility="both"),
    dict(id="power_pressure", label="剩余功率压力", description="剩余功率/(剩余功率＋每束公平份额×剩余目标束数)，公平份额=预算/总束数，终态为0", shape=[1], unit="比例", timing="动作前", source="真实功率账", visibility="both"),
    dict(id="remaining_demand_mean", label="剩余平均需求", description="含当前波束的剩余平均需求除以600 Mbps", shape=[1], unit="比例", timing="动作前", source="全场景需求", visibility="known"),
    dict(id="served_shortfall", label="已服务平均欠供", description="已处理波束的平均(1-满足率)，无记录时为0", shape=[1], unit="比例", timing="动作前", source="已处理波束当前速率", visibility="both"),
    dict(id="served_met", label="已服务达标比例", description="已处理波束中满足需求者比例，无记录时为0", shape=[1], unit="比例", timing="动作前", source="已处理波束当前速率", visibility="both"),
    dict(id="served_min", label="已服务最低满足率", description="已处理波束最低截断满足率，无记录时为0", shape=[1], unit="比例", timing="动作前", source="已处理波束当前速率", visibility="both"),
    dict(id="other_occupancy", label="其余组占用", description="其余7组各100槽占用次数，按组号排序", shape=[7, 100], unit="次数", timing="动作前", source="已分配波束", visibility="both"),
    dict(id="longest_free", label="最长连续空槽", description="本组最长连续空槽数除以100", shape=[1], unit="比例", timing="动作前", source="已分配波束", visibility="both"),
    dict(id="global_context", label="分档全局信息", description="平均需求、进度、剩余需求及资源服务概况；按信息档位开放", shape=[10], unit="归一化", timing="动作前", source="全场景与当前分配", visibility="known"),
    dict(id="beam_context", label="分档全部波束信息", description="100束的几何需求与当前分配记录；10束一段，不含未来动作或最终容量", shape=[10, 10, 11], unit="归一化", timing="动作前", source="全场景与已执行动作", visibility="known"),
    dict(id="resource_context", label="分档全部频率组信息", description="8组资源服务概况及逐槽占用、当前束视角历史干扰", shape=[8, 204], unit="归一化", timing="动作前", source="真实资源账与历史波束", visibility="known"),
    dict(id="natural_global_context", label="自然全局信息", description="仅包含本档实际开放的全局信息", shape=[10], unit="归一化", timing="动作前", source="全场景与当前分配", visibility="known"),
    dict(id="natural_beam_context", label="自然波束信息", description="仅包含本档实际开放的波束字段", shape=[10, 10, 11], unit="归一化", timing="动作前", source="全场景与已执行动作", visibility="known"),
    dict(id="natural_resource_context", label="自然频率组信息", description="仅包含本档实际开放的资源字段", shape=[8, 204], unit="归一化", timing="动作前", source="真实资源账与历史波束", visibility="known"),
    dict(id="previous_actual_action", label="上一束实际分配", description="上一束实际起始槽/99、槽数/10、功率/50W与历史有效位；首步无历史", shape=[4], unit="归一化", timing="动作前", source="上一束实际执行动作", visibility="both"),
]


VARIABLE_BEAM_REWARD_PRESETS = ("legacy_repaired", "legacy_repaired_throughput")
# S2-P18（方案18 第二版）登记在案的实验臂与训练预算。臂身份写入配置后，校验器
# 才允许该臂声明的那几处差异；未声明臂的配置仍按 V0/V1 冻结方案逐项校验。
VARIABLE_BEAM_ARMS = ("V2", "V3", "G0")
VARIABLE_BEAM_STEP_BUDGETS = (100000, 400000)


def variable_beam_features(profile="legacy_210"):
    """可变束数状态字段表。默认 ``legacy_210`` 保持 V0/V1 冻结的 210 维顺序。

    ``s2p18`` 追加组号与本组预留/剩余槽上限三个标量（方案18 第 3 节 B、C 项），
    供第二轮 V2/V3 两条臂共用；G0 不启用额外标量，只改逐束编码。
    """
    if profile == "legacy_210":
        return list(VARIABLE_BEAM_FEATURES)
    if profile == "s2p18":
        return list(S2P18_VARIABLE_FEATURES)
    raise ValueError(f"未知可变束数状态档案: {profile!r}")


def _feature_shape(config, fid):
    if config["network"]["transformer_layout"] == "tsac_natural_context" and fid in natural_shapes(config["state"]["context_level"]):
        shape = list(natural_shapes(config["state"]["context_level"])[fid])
        if fid == "natural_beam_context" and network_beam_count_mode(config) == "actual":
            # 逐束编码档位：观测只含实际实体 [N, 写入通道数]，行数取场景束数上限。
            shape = [int(config["environment"]["beam_count_max"]), shape[-1]]
        return shape
    return next(f["shape"] for f in FEATURES if f["id"] == fid)


def _pos_slots(config):
    """逐束实体编码的位置槽位总数（按束数上界一次分配）。"""
    network, level = config["network"], config["state"]["context_level"]
    shapes = natural_shapes(level)
    if network.get("context_beam_count", "fixed_100") != "actual":
        return natural_token_count(level) - 13
    # 观测里逐束字段每实体只写本档声明的通道数（layout 档 5、full 档 11）。
    width = shapes["natural_beam_context"][-1]
    slots = width * int(config["environment"]["beam_count_max"])
    if "natural_resource_context" in shapes:
        slots += shapes["natural_resource_context"][0]
    return slots


def natural_state_width(config, count):
    """自然布局在给定场景束数下的观测宽度。

    逐束编码档位下，观测按``environment.beam_count_max``固定补零：网络输入宽度
    必须恒定，真实束数由观测内归一化束数还原并由编码器屏蔽补零实体。因此这里
    只要求 ``count`` 不超过上限，宽度与 ``count`` 无关。
    """
    base = sum(math.prod(_feature_shape(config, fid)) for fid in config["state"]["features"])
    if network_beam_count_mode(config) != "actual":
        return base
    shapes = natural_shapes(config["state"]["context_level"])
    if "natural_beam_context" not in shapes:
        return base
    reference = int(config["environment"]["beam_count_max"])
    if count > reference:
        raise ValueError("场景束数超过 environment.beam_count_max，无法构造固定宽度观测")
    return base


def expected_state_dim(config):
    """按 ``environment.beam_count_max`` 估算的观测宽度（训练脚本的一致性门槛）。"""
    return natural_state_width(config, int(config["environment"]["beam_count_max"]))


def network_beam_count_mode(config):
    return config["network"].get("context_beam_count", "fixed_100")

REWARD_TERMS = [
    ("si_delta", "需求满足率增量", "所有已处理波束当前满足率总和的变化", "step", "service"),
    ("basic_delta", "基本服务增量", "全部目标束基本服务程度之和的变化；逐束为min(速率/基本需求,1)", "step", "service"),
    ("sgm_delta", "供需匹配增量", "已处理波束SGM总和的变化", "step", "matching"),
    ("current_si", "本束满足率", "当前波束当时的截断满足率", "step", "service"),
    ("current_sgm", "本束供需匹配", "当前波束当时的SGM", "step", "matching"),
    ("throughput_delta", "总速率增量", "已处理波束当前总速率变化，单位Gbps", "step", "throughput"),
    ("useful_throughput_delta", "有效吞吐增量", "逐束速率按需求截断后的总量变化，单位Gbps", "step", "throughput"),
    ("jain_delta", "公平性增量", "全部100目标束含未服务零值的Jain指数变化；全零记0", "step", "fairness"),
    ("min_si_delta", "最低满足率增量", "全部100目标束含未服务零值的最低满足率变化", "step", "fairness"),
    ("shortfall_delta", "欠供减少", "按需求截断的欠供比例总量的减少", "step", "service"),
    ("excess_supply", "过量供给惩罚", "本束超过需求的速率，单位Gbps，取负", "step", "matching"),
    ("power_cost", "功率成本", "本步实际辐射功率占预算比例，取负", "step", "resource"),
    ("slot_cost", "频率槽成本", "实际占用槽数除以10，取负", "step", "resource"),
    ("interference_loss", "历史波束干扰损失", "本步造成此前波束速率减少，单位Gbps，取负", "step", "interference"),
    ("out_penalty", "越界裁剪", "越界槽数占请求槽数比例，取负", "step", "constraint"),
    ("occupy_penalty", "占用冲突", "冲突槽数占请求槽数比例，取负", "step", "constraint"),
    ("clip_penalty", "动作裁剪", "请求与实际槽数之差占请求槽数比例，取负", "step", "constraint"),
    ("overload_penalty", "超功率", "首次超限时超出功率占预算比例，取负", "step", "constraint"),
    ("zero_slot_penalty", "零槽", "实际槽数为零时取-1", "step", "constraint"),
    ("terminal_si", "终态需求满足率", "全部目标波束最终满足率的平均值", "terminal", "service"),
    ("terminal_basic", "终态基本服务", "全部目标波束基本服务达标程度平均值", "terminal", "service"),
    ("terminal_sgm", "终态供需匹配", "全部目标波束最终SGM平均值", "terminal", "matching"),
    ("terminal_throughput", "终态吞吐量", "全部目标波束最终总速率，单位Gbps", "terminal", "throughput"),
    ("terminal_jain", "终态公平性", "全部目标波束最终满足率的Jain指数", "terminal", "fairness"),
    ("terminal_min_si", "终态最低满足率", "全部目标波束最终最低满足率", "terminal", "fairness"),
    ("incomplete_penalty", "提前终止", "未处理束比例，取负", "terminal", "constraint"),
]

REWARD_FORMULAS = {
    "si_delta": "ΔΣmin(当前速率/需求,1)",
    "basic_delta": "ΔΣmin(当前速率/(基本比例×需求),1)",
    "sgm_delta": "ΔΣSGM(当前速率,需求,全场景平均需求)",
    "current_si": "min(本束当前速率/需求,1)",
    "current_sgm": "SGM(本束当前速率,需求,全场景平均需求)",
    "throughput_delta": "ΔΣ当前速率/10⁹",
    "useful_throughput_delta": "ΔΣmin(当前速率,需求)/10⁹",
    "jain_delta": "ΔJain(全部100束满足率，未服务为0)",
    "min_si_delta": "Δmin(全部100束满足率，未服务为0)",
    "shortfall_delta": "−ΔΣ(1−全部100束满足率)",
    "excess_supply": "−max(本束速率−需求,0)/10⁹",
    "power_cost": "−本步实际辐射功率/功率预算",
    "slot_cost": "−本步实际槽数/10",
    "interference_loss": "−本步历史束速率损失/10⁹",
    "out_penalty": "−越界槽数/请求槽数",
    "occupy_penalty": "−占用冲突槽数/请求槽数",
    "clip_penalty": "−(请求槽数−实际槽数)/请求槽数",
    "overload_penalty": "−首次超预算功率/功率预算",
    "zero_slot_penalty": "−1(实际槽数为0)",
    "terminal_si": "终态Σmin(全部100束最终速率/需求,1)/100",
    "terminal_basic": "终态Σmin(全部100束最终速率/(基本比例×需求),1)/100",
    "terminal_sgm": "终态ΣSGM(全部100束最终速率,需求,全场景平均需求)/100",
    "terminal_throughput": "终态Σ全部100束最终速率/10⁹",
    "terminal_jain": "终态Jain(全部100束满足率，未服务为0)",
    "terminal_min_si": "终态min(全部100束满足率，未服务为0)",
    "incomplete_penalty": "−终态未处理束数/100",
}

PRESETS = [
    dict(id="legacy_original", label="原公式与原记账", description="复现原奖励；零槽请求仍计功率"),
    dict(id="legacy_repaired", label="原公式与修正记账", description="复现原奖励；零槽不计功率"),
    dict(id="fused", label="融合奖励", description="2Δ基本服务＋Δ满足率＋0.2ΔSGM－实际功率/预算"),
    dict(id="legacy_repaired_throughput", label="原公式＋吞吐与频谱使用（S2-P18 V2）",
         description="保留原四项，另加 w_throughput×Δ有效承载/参考量与 w_spectrum×本步占用槽/本组可用槽；"
                     "权重默认0，必须显式配置才生效；原始奖励分量仍逐项记录"),
    dict(id="custom", label="自选奖励项", description="逐项计算并保存原值、缩放值与加权值"),
    dict(id="terminal_si_delta", label="全束终态SI增量（S2-P19）",
         description="每步100×全体目标波束当前SI均值变化；未服务束为0，γ=1时回合求和为100×终态SI"),
]


def default_config():
    return deepcopy(dict(
        schema_version=1, name="新实验", algorithm="sac",
        # S2-P18：实验臂身份必须由配置显式声明，校验器据此允许该臂声明的差异；
        # 未声明时（空串）完全按原冻结方案校验，默认配置语义不变。
        experiment=dict(id="", arm=""),
        network=dict(kind="mlp", transformer_layout="scalar_tokens",
                     hidden_sizes=[256, 256], activation="relu", dropout=0,
                     cnn_channels=[16, 32], kernel_size=3, embedding_dim=64,
                     num_heads=4, num_layers=2, feedforward_dim=128,
                     sac_policy_structure="factorized", sac_critic_structure="additive",
                     temporal_mode="none", temporal_window=16, graph_mode="none",
                     # S2-P18 手段G：逐束上下文按实际束数构造（默认保持固定100束），
                     # 读出方式与 N 条件调制都可选；默认值与改动前逐位一致。
                     context_beam_count="fixed_100", readout_mode="first_token",
                     beam_count_conditioning="none"),
        state=dict(features=["beam", "power", "interference", "occupancy"], demand_visibility="known", context_level="none"),
        reward=dict(preset="legacy_repaired", terms=[dict(id="si_delta", weight=1, scale=1, timing="step")],
                    # S2-P18 新增奖励分量。全部默认为 0（关闭），所以默认配置的语义、
                    # 数值与配置哈希都与改动前一致；启用时必须写入冻结配置。
                    shaping=dict(w_throughput=0, w_spectrum=0, w_shape=0,
                                 throughput_reference="mean_demand")),
        environment=dict(profile="china_legacy", beam_count_max=100, power_budget_w=3000, basic_fraction=0.1, action_mode="legacy_clip",
                         action_mask="none",
                         power_levels_w=[5, 10, 15, 20, 25, 30, 35, 40, 45, 50]),
        training=dict(seed=51001, scene_order="sequential", scene_order_seed=51001,
                      beam_count_mode="none", beam_count_min=50, beam_count_max=200,
                      held_out_beam_counts=[60, 90, 140, 180],
                      total_steps=10000, batch_size=256, actor_lr=0.00001,
                      critic_lr=0.0001, alpha_lr=0.0001, gamma=0.99, tau=0.005,
                      replay_capacity=20000, warmup_steps=2000, update_every=1,
                      gradient_steps=1, initial_alpha=0.1, target_entropy=6.9,
                      alpha_mode="adaptive", fixed_alpha=0.01,
                      # S2-P18 手段E：按回合内进度分位分层采样；1 表示关闭（旧行为逐位不变）。
                      progress_strata=1,
                      # S2-P18 手段D：按回合长度/束数设定折扣；mode="none" 时使用上面的 gamma。
                      gamma_schedule=dict(mode="none", default=0.99, by_beam_count=[]),
                      gradient_clip=10, rollout_steps=256, ppo_epochs=4, gae_lambda=0.95,
                      clip_ratio=0.2, entropy_coef=0.01, value_coef=0.5,
                      eps_start=1, eps_end=0.05, eps_decay_steps=5000,
                      target_update_interval=500, checkpoint_interval=1000,
                      log_interval=50, device="cuda"),
        evaluation=dict(validation_count=3, test_count=5, validation_seed=94000,
                        test_seed=95000, interval_steps=1000,
                        action_mode="deterministic", repeats=1, extra_steps=[]),
        telemetry=dict(mode="sync", flush_interval_seconds=2.0, queue_capacity=2048),
    ))


def _merge(template, raw, path=""):
    if not isinstance(raw, dict):
        raise ValueError(f"{path or 'config'} 必须是对象")
    result = deepcopy(template)
    for key, value in raw.items():
        if key not in result:
            raise ValueError(f"未知配置项: {path}{key}")
        result[key] = _merge(result[key], value, f"{path}{key}.") if isinstance(result[key], dict) else deepcopy(value)
    return result


def _number(value, name, lower=None, upper=None, integer=False, strictly=False):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise ValueError(f"{name} 必须是有限数字")
    if integer and not isinstance(value, int):
        raise ValueError(f"{name} 必须是整数")
    if lower is not None and (value <= lower if strictly else value < lower):
        raise ValueError(f"{name} 不得低于 {lower}")
    if upper is not None and value > upper:
        raise ValueError(f"{name} 不得高于 {upper}")


def _network_size(config):
    """Cheap conservative bound before the API constructs any PyTorch modules."""
    n = config["network"]
    state_dim = sum(math.prod(_feature_shape(config, fid))
                    for fid in config["state"]["features"])
    heads = 110 + len(config["environment"]["power_levels_w"])
    if n["kind"] == "mlp":
        widths = [state_dim, *n["hidden_sizes"]]
        encoder = sum((a + 1) * b for a, b in zip(widths, widths[1:]))
        output = widths[-1]
    elif n["kind"] == "cnn":
        widths = [1, *n["cnn_channels"]]
        encoder = sum((a * n["kernel_size"] + 1) * b for a, b in zip(widths, widths[1:]))
        output = widths[-1] * min(16, state_dim)
    else:
        embed, ff = n["embedding_dim"], n["feedforward_dim"]
        token_count = (natural_token_count(config["state"]["context_level"])
                       if n["transformer_layout"] == "tsac_natural_context" else
                       {"tsac_13_tokens": 13, "tsac_14_context": 14,
                        "tsac_14_variable_context": 14, "tsac_32_context": 32}.get(n["transformer_layout"], state_dim))
        if n["graph_mode"] != "none":
            token_count = 122
        elif n["temporal_mode"] != "none":
            token_count += 1
        if token_count ** 2 * n["num_heads"] * n["num_layers"] > 20_000_000:
            raise ValueError("Transformer 状态长度、注意力头数与层数的组合过大")
        projection = {"tsac_13_tokens": 42 * embed, "tsac_14_context": 47 * embed,
                      "tsac_14_variable_context": 50 * embed, "tsac_32_context": 388 * embed}.get(
            n["transformer_layout"], (state_dim + 2) * embed)
        if n["transformer_layout"] == "tsac_natural_context":
            shapes = natural_shapes(config["state"]["context_level"])
            projection = 42 * embed
            if "natural_global_context" in shapes:
                projection += (shapes["natural_global_context"][0] + 2) * embed
            if "natural_beam_context" in shapes:
                rows = (config["environment"].get("beam_count_max", 100)
                        if n.get("context_beam_count", "fixed_100") == "actual"
                        else shapes["natural_beam_context"][0])
                projection += (rows * shapes["natural_beam_context"][-1] + 1 + 10) * embed
            if "natural_resource_context" in shapes:
                projection += (shapes["natural_resource_context"][-1] + 1 + 8) * embed
        encoder = projection + n["num_layers"] * (
            4 * embed * embed + 2 * embed * ff + 9 * embed + ff)
        # Conservative admission bound; preview counts actual constructed modules.
        if n["graph_mode"] != "none":
            encoder += 122 * embed + 8 * embed * embed
        if n["temporal_mode"] != "none":
            encoder += 6 * embed
        if n["temporal_mode"] == "gru":
            encoder += n["num_layers"] * (6 * embed * embed + 6 * embed)
        elif n["temporal_mode"] == "causal_transformer":
            encoder += n["temporal_window"] * embed + n["num_layers"] * (
                4 * embed * embed + 2 * embed * ff + 9 * embed + ff)
        output = embed
    actor = encoder + (output + 1) * heads
    if config["algorithm"] == "sac":
        # The joint heads are bounded here before actual modules are built.
        # model_description supplies the exact per-network parameter counts.
        policy_outputs = (100 * 10 * (1 + len(config["environment"]["power_levels_w"]))
                          if n["sac_policy_structure"] == "conditional" else heads)
        critic_outputs = (100 * 10 * len(config["environment"]["power_levels_w"])
                          if n["sac_critic_structure"] == "joint" else heads)
        total = 5 * encoder + (output + 1) * (policy_outputs + 4 * critic_outputs) + 1
    elif config["algorithm"] == "dqn":
        total = 2 * actor
    else:
        total = actor + encoder + output + 1
    if total > 20_000_000:
        raise ValueError("模型预计参数超过2000万；请缩小网络后再预览或运行")
    return total


def resolve_gamma(training, beam_count=None):
    """S2-P18 手段D：按回合长度/束数解析实际折扣。

    ``gamma_schedule.mode``：
    - ``none``：使用 ``training.gamma``（默认 0.99，旧行为逐位不变）；
    - ``constant``：使用 ``default``，用于“统一取 0.995”这类单一新值；
    - ``by_beam_count``：按 ``by_beam_count`` 里升序的 ``[{"max_beam_count": N,
      "gamma": g}, ...]`` 取第一个 ``N <= max_beam_count`` 的 g，否则用 ``default``。
    """
    schedule = training.get("gamma_schedule") or {}
    mode = schedule.get("mode", "none")
    if mode == "none":
        return float(training["gamma"])
    if mode == "constant":
        return float(schedule["default"])
    if mode == "by_beam_count":
        if beam_count is None:
            raise ValueError("按束数设定折扣需要实际束数")
        for bucket in schedule["by_beam_count"]:
            if int(beam_count) <= int(bucket["max_beam_count"]):
                return float(bucket["gamma"])
        return float(schedule["default"])
    raise ValueError(f"未知折扣日程: {mode!r}")


def _validate_gamma_schedule(training):
    schedule = training["gamma_schedule"]
    if not isinstance(schedule, dict):
        raise ValueError("gamma_schedule 必须是对象")
    mode = schedule["mode"]
    if mode not in ("none", "constant", "by_beam_count"):
        raise ValueError("折扣日程仅支持 none、constant 或 by_beam_count")
    _number(schedule["default"], "折扣日程默认折扣", 0, 1)
    buckets = schedule["by_beam_count"]
    if not isinstance(buckets, list):
        raise ValueError("折扣分档必须是列表")
    limits = []
    for bucket in buckets:
        if not isinstance(bucket, dict) or set(bucket) != {"max_beam_count", "gamma"}:
            raise ValueError("折扣分档字段无效")
        _number(bucket["max_beam_count"], "折扣分档上限", 1, 200, integer=True)
        _number(bucket["gamma"], "折扣分档折扣", 0, 1)
        limits.append(bucket["max_beam_count"])
    if limits != sorted(set(limits)):
        raise ValueError("折扣分档上限必须升序且不重复")
    if mode == "by_beam_count" and not buckets:
        raise ValueError("按束数设定折扣需要至少一个分档")


def _validate_reward_shaping(reward):
    """S2-P18 新增奖励分量：默认全 0；启用时必须是有限、非负、可复现的冻结值。"""
    shaping = reward["shaping"]
    if not isinstance(shaping, dict):
        raise ValueError("奖励整形必须是对象")
    for key in ("w_throughput", "w_spectrum", "w_shape"):
        _number(shaping[key], f"奖励整形权重 {key}", 0)
    reference = shaping["throughput_reference"]
    if isinstance(reference, bool) or not (
            reference in ("mean_demand", "total_demand")
            or (isinstance(reference, (int, float)) and math.isfinite(reference) and reference > 0)):
        raise ValueError("吞吐归一化参考量须为 mean_demand、total_demand 或正数 Gbps")
    if reward["preset"] == "legacy_repaired_throughput" and not any(
            shaping[key] for key in ("w_throughput", "w_spectrum", "w_shape")):
        raise ValueError("吞吐与频谱使用预设要求至少给出一个正权重，否则请保留 legacy_repaired")


def validate_config(raw):
    # New research-only fields are injected only for S2-P19. Historical fully
    # materialized configurations keep their exact keys and checkpoint hashes.
    p19_requested = (isinstance(raw, dict) and isinstance(raw.get("experiment"), dict)
                     and raw["experiment"].get("id") == "S2-P19")
    template = default_config()
    if p19_requested:
        template["training"]["demonstrations"] = dict(
            mode="none", manifest_path="", manifest_sha256="", source_set_sha256="")
        template["training"]["pilot_stop_steps"] = 100000
    c = _merge(template, raw)
    p19 = c["experiment"]["id"] == "S2-P19"
    if type(c["schema_version"]) is not int or c["schema_version"] != 1:
        raise ValueError("不支持的配置版本")
    if not isinstance(c["name"], str) or not c["name"].strip() or len(c["name"]) > 120:
        raise ValueError("实验名称无效")
    if c["algorithm"] not in ("sac", "ppo", "dqn"):
        raise ValueError("不支持的算法")
    n = c["network"]
    if n["kind"] not in ("mlp", "cnn", "transformer") or n["activation"] not in ("relu", "gelu", "tanh"):
        raise ValueError("不支持的网络或激活函数")
    if n["sac_policy_structure"] not in ("factorized", "conditional"):
        raise ValueError("SAC 策略结构须为 factorized 或 conditional")
    if n["sac_critic_structure"] not in ("additive", "joint"):
        raise ValueError("SAC 价值结构须为 additive 或 joint")
    if n["temporal_mode"] not in ("none", "previous_action", "gru", "causal_transformer"):
        raise ValueError("不支持的跨波束历史模式")
    if n["graph_mode"] not in ("none", "beam_tokens", "interference_bias"):
        raise ValueError("不支持的波束图编码")
    temporal = n["temporal_mode"] != "none"
    graph = n["graph_mode"] != "none"
    if n["context_beam_count"] not in ("fixed_100", "actual"):
        raise ValueError("逐束上下文束数仅支持 fixed_100 或 actual")
    if n["readout_mode"] not in ("first_token", "attention_pooling"):
        raise ValueError("读出方式仅支持 first_token 或 attention_pooling")
    if n["beam_count_conditioning"] not in ("none", "film"):
        raise ValueError("N 条件调制仅支持 none 或 film")
    if (n["readout_mode"] != "first_token" or n["beam_count_conditioning"] != "none") and not (
            n["kind"] == "transformer" and n["transformer_layout"] == "tsac_natural_context"):
        raise ValueError("注意力池化读出与 N 条件调制仅用于自然状态的逐束布局")
    if n["context_beam_count"] == "actual" and not (
            n["kind"] == "transformer" and n["transformer_layout"] == "tsac_natural_context"
            and c["state"]["context_level"] in ("layout", "full")):
        raise ValueError("按实际束数构造逐束上下文需要自然状态的 layout 或 full 档")
    if n["context_beam_count"] == "actual" and n["transformer_layout"] == "tsac_natural_context" \
            and c["environment"]["beam_count_max"] > c["training"]["beam_count_max"]:
        # 观测宽度直接由场景束数决定，因此场景束数上限不得超过训练束数上限。
        raise ValueError("按实际束数构造要求场景束数上限不超过训练束数上限")
    _number(n["temporal_window"], "历史窗口", 1, 16, integer=True)
    temporal = n["temporal_mode"] != "none"
    graph = n["graph_mode"] != "none"
    if temporal and graph:
        raise ValueError("当前记忆与图编码分别对照，不能同时开启")
    if temporal or graph:
        if (c["algorithm"] != "sac" or n["kind"] != "transformer" or
                n["transformer_layout"] != "tsac_natural_context" or
                c["environment"]["action_mask"] != "joint_feasible" or
                n["sac_policy_structure"] != "factorized" or n["sac_critic_structure"] != "additive"):
            raise ValueError("记忆/图首版仅支持自然布局 Transformer SAC、联合掩码、原策略及加性Q")
        if temporal and c["state"]["context_level"] != "compact":
            raise ValueError("记忆首版要求compact的207维后追加上一实际动作，共211维")
        if graph and c["state"]["context_level"] != "full":
            raise ValueError("波束图编码要求full的2946维原始字段")
    if "previous_actual_action" in c["state"]["features"] and not temporal:
        raise ValueError("上一实际动作字段仅用于显式记忆模式")
    if (n["sac_policy_structure"] != "factorized" or n["sac_critic_structure"] != "additive") and (
            c["algorithm"] != "sac" or c["environment"]["action_mask"] != "joint_feasible"):
        raise ValueError("条件策略和联合价值结构仅支持 SAC＋联合可行动作掩码")
    if n["transformer_layout"] not in ("scalar_tokens", "tsac_13_tokens", "tsac_14_context", "tsac_14_variable_context", "tsac_32_context", "tsac_natural_context"):
        raise ValueError("不支持的 Transformer 输入布局")
    if n["transformer_layout"] == "tsac_13_tokens":
        if c["algorithm"] != "sac" or n["kind"] != "transformer":
            raise ValueError("原 T-SAC 13令牌布局仅支持 SAC 的 Transformer 网络")
        if c["state"]["features"] != ["beam", "power", "interference", "occupancy"]:
            raise ValueError("原 T-SAC 13令牌布局要求原204维状态特征及顺序：波束、功率、干扰、占用")
    if n["transformer_layout"] == "tsac_14_context":
        if c["algorithm"] != "sac" or n["kind"] != "transformer":
            raise ValueError("T-SAC全局状态布局仅支持SAC的Transformer网络")
        if c["state"]["features"] != ["beam", "power", "interference", "occupancy", "beta", "progress", "remaining_demand"]:
            raise ValueError("T-SAC全局状态布局要求原204维后依次追加平均需求、处理进度、剩余需求，共207维")
        if c["state"]["demand_visibility"] != "known":
            raise ValueError("T-SAC全局状态布局需要全场景需求已知")
    if n["transformer_layout"] == "tsac_14_variable_context":
        if c["algorithm"] != "sac" or n["kind"] != "transformer":
            raise ValueError("可变波束14令牌布局仅支持SAC Transformer")
        if (c["state"]["features"] != list(VARIABLE_BEAM_FEATURES)
                and c["state"]["features"] != list(S2P18_VARIABLE_FEATURES)):
            raise ValueError("可变波束布局要求冻结的210维字段顺序或S2-P18的213维字段顺序")
        if c["state"]["demand_visibility"] != "known":
            raise ValueError("可变波束布局要求已知全场景需求")
        if n["readout_mode"] != "first_token" or n["beam_count_conditioning"] != "none" or \
                n["context_beam_count"] != "fixed_100":
            raise ValueError("注意力池化读出、N条件调制与按实际束数构造上下文仅用于自然状态的逐束布局")
    if n["transformer_layout"] == "tsac_32_context":
        if c["algorithm"] != "sac" or n["kind"] != "transformer":
            raise ValueError("32令牌分档状态布局仅支持SAC的Transformer网络")
        if c["state"]["features"] != ["beam", "power", "interference", "occupancy", "global_context", "beam_context", "resource_context"]:
            raise ValueError("32令牌分档状态须为原204维后依次追加全局、全部波束、全部频率组信息，共2946维")
        if c["state"]["demand_visibility"] != "known" or c["state"]["context_level"] not in ("compact", "summary", "layout", "full"):
            raise ValueError("32令牌布局需要需求已知及有效的状态信息档位")
    elif n["transformer_layout"] == "tsac_natural_context":
        level = c["state"]["context_level"]
        if c["algorithm"] != "sac" or n["kind"] != "transformer" or level not in NATURAL_CONTEXT_LEVELS:
            raise ValueError("自然状态布局仅支持需求已知的SAC Transformer与有效档位")
        expected_features = list(natural_shapes(level)) + (["previous_actual_action"] if temporal else [])
        if c["state"]["demand_visibility"] != "known" or c["state"]["features"] != expected_features:
            raise ValueError("自然状态布局要求本档实际开放的状态特征及规范顺序")
        allowed_presets = (("terminal_si_delta",) if p19 else
                           ("legacy_repaired", "legacy_repaired_throughput"))
        # 已登记的 S2-P18 臂（G0）把自然逐束布局用于可变束数四桶抽样，因此对该臂
        # 额外允许 variable_beams 场景顺序；未声明臂的配置仍要求每轮打乱。
        allowed_orders = (("shuffle_each_cycle", "variable_beams")
                          if c["experiment"]["arm"] in VARIABLE_BEAM_ARMS or p19
                          else ("shuffle_each_cycle",))
        if n["dropout"] != 0 or c["reward"]["preset"] not in allowed_presets or \
                c["training"]["scene_order"] not in allowed_orders:
            raise ValueError("自然状态对照固定旧奖励、dropout=0与每轮打乱场景")
    elif c["state"]["context_level"] != "none" or any(fid in c["state"]["features"] for fid in ("global_context", "beam_context", "resource_context", "natural_global_context", "natural_beam_context", "natural_resource_context")):
        raise ValueError("状态信息档位与分档特征仅适用于32令牌布局，旧布局使用none")
    for key in ("hidden_sizes", "cnn_channels"):
        if not isinstance(n[key], list) or not n[key] or len(n[key]) > 8:
            raise ValueError(f"{key} 必须是非空列表")
        for v in n[key]: _number(v, key, 1, 1024 if key == "hidden_sizes" else 512, integer=True)
    for key, upper in (("kernel_size", 31), ("embedding_dim", 1024),
                       ("num_heads", 32), ("num_layers", 12), ("feedforward_dim", 8192)):
        _number(n[key], key, 1, upper, integer=True)
    if n["embedding_dim"] % n["num_heads"]:
        raise ValueError("embedding_dim 必须能被 num_heads 整除")
    _number(n["dropout"], "dropout", 0, 0.9)
    if c["algorithm"] == "ppo" and n["dropout"] > 0:
        raise ValueError("PPO 当前要求 dropout=0；采样与更新必须使用一致的动作概率")
    s = c["state"]
    if s["demand_visibility"] not in ("known", "online"):
        raise ValueError("demand_visibility 必须为 known 或 online")
    feature_ids = {f["id"] for f in FEATURES}
    if not isinstance(s["features"], list) or not s["features"] or not all(isinstance(fid, str) for fid in s["features"]) or len(s["features"]) != len(set(s["features"])):
        raise ValueError("状态特征必须非空且不重复")
    for fid in s["features"]:
        if fid not in feature_ids:
            raise ValueError(f"未知状态特征: {fid}")
        feature = next(f for f in FEATURES if f["id"] == fid)
        if feature["visibility"] == "known" and s["demand_visibility"] != "known":
            raise ValueError(f"在线需求模式不可读取未来需求: {fid}")
    e = c["environment"]
    if not isinstance(e["profile"], str):
        raise ValueError("environment.profile 必须是地区版本字符串")
    get_profile(e["profile"])
    _number(e["beam_count_max"], "beam_count_max", 1, 200, integer=True)
    _number(e["power_budget_w"], "power_budget_w", 0, strictly=True)
    _number(e["basic_fraction"], "basic_fraction", 0, 1, strictly=True)
    if e["action_mode"] not in ("legacy_clip", "feasible_projection"):
        raise ValueError("不支持的动作规则")
    if e["action_mask"] not in ("none", "joint_feasible"):
        raise ValueError("不支持的无效动作掩码方式")
    if e["action_mask"] == "joint_feasible":
        if c["algorithm"] != "sac":
            raise ValueError("联合无效动作掩码目前仅支持 SAC；PPO 和 DQN 暂不支持")
        if e["action_mode"] != "legacy_clip":
            raise ValueError("联合无效动作掩码不能与可行动作投影同时启用；请使用原动作裁剪作为物理校验")
    levels = e["power_levels_w"]
    if not isinstance(levels, list) or not levels or len(levels) > 100:
        raise ValueError("功率档位必须是非空列表")
    for level in levels: _number(level, "功率档位", 0, strictly=True)
    if sorted(set(levels)) != levels:
        raise ValueError("功率档位必须严格递增")
    if e["action_mask"] == "joint_feasible" and e["power_budget_w"] < e["beam_count_max"] * levels[0]:
        raise ValueError(f"联合无效动作掩码要求总功率预算至少覆盖{e['beam_count_max']}束的最低功率档位")
    r = c["reward"]
    if not isinstance(r["preset"], str) or r["preset"] not in {p["id"] for p in PRESETS}:
        raise ValueError("未知奖励预设")
    if not isinstance(r["terms"], list):
        raise ValueError("奖励项必须是列表")
    if r["preset"] == "custom" and not r["terms"]:
        raise ValueError("自选奖励至少需要一个奖励项")
    ids = [term.get("id") for term in r["terms"] if isinstance(term, dict)]
    if len(ids) != len(r["terms"]) or not all(isinstance(fid, str) for fid in ids) or len(ids) != len(set(ids)):
        raise ValueError("奖励项不得重复")
    definitions = {term[0]: term for term in REWARD_TERMS}
    for term in r["terms"]:
        if set(term) != {"id", "weight", "scale", "timing"} or term["id"] not in definitions:
            raise ValueError("奖励项字段或名称无效")
        _number(term["weight"], "奖励权重")
        _number(term["scale"], "奖励尺度", 0, strictly=True)
        if term["timing"] != definitions[term["id"]][3]:
            raise ValueError(f"奖励结算时点错误: {term['id']}")
    _validate_reward_shaping(r)
    t = c["training"]
    if t["alpha_mode"] not in ("adaptive", "fixed"):
        raise ValueError("温度模式须为adaptive或fixed")
    _number(t["fixed_alpha"], "固定温度", 0, strictly=True)
    if t["alpha_mode"] == "fixed" and c["algorithm"] != "sac":
        raise ValueError("固定温度仅支持SAC")
    ints = {"seed": (0, 2 ** 32 - 1), "scene_order_seed": (0, 2 ** 32 - 1),
            "total_steps": (1, None), "batch_size": (1, None), "replay_capacity": (1, None),
            "warmup_steps": (0, None), "update_every": (1, None), "gradient_steps": (1, None),
            "rollout_steps": (1, None), "ppo_epochs": (1, None), "eps_decay_steps": (1, None),
            "target_update_interval": (1, None), "checkpoint_interval": (1, None), "log_interval": (1, None),
            "progress_strata": (1, 8)}
    for key, (lo, hi) in ints.items(): _number(t[key], key, lo, hi, integer=True)
    if c["algorithm"] in ("sac", "dqn") and t["batch_size"] % (
            t["progress_strata"] * (1 if t["beam_count_mode"] != "mixed_four_buckets" else 4)):
        raise ValueError("进度分层采样要求批大小能被（数量桶×进度段）整除")
    if t["scene_order"] not in ("sequential", "shuffle_each_cycle", "variable_beams"):
        raise ValueError("训练场景顺序无效")
    if t["beam_count_mode"] not in ("none", "fixed_100", "mixed_four_buckets"):
        raise ValueError("波束数量训练模式无效")
    _number(t["beam_count_min"], "beam_count_min", 1, 200, integer=True)
    _number(t["beam_count_max"], "beam_count_max", 1, 200, integer=True)
    held_out = t["held_out_beam_counts"]
    if (not isinstance(held_out, list) or any(type(v) is not int or not 1 <= v <= 200 for v in held_out)
            or held_out != sorted(set(held_out))):
        raise ValueError("留出束数必须是升序、不重复的1至200整数")
    if t["beam_count_mode"] == "none":
        if t["scene_order"] == "variable_beams":
            raise ValueError("可变束数场景顺序只能用于V0/V1")
    elif not p19 and (t["scene_order"] != "variable_beams" or e["profile"] != "europe_v1"
          or e["beam_count_max"] != 200 or t["beam_count_min"] != 50
          or t["beam_count_max"] != 200 or held_out != [60, 90, 140, 180]
          or e["action_mask"] != "joint_feasible" or e["action_mode"] != "legacy_clip"
          or e["power_budget_w"] != 3000 or e["power_levels_w"] != [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
          or n["sac_policy_structure"] != "factorized" or n["sac_critic_structure"] != "additive"
          or n["dropout"] != 0
          or r["preset"] not in VARIABLE_BEAM_REWARD_PRESETS
          or t["alpha_mode"] != "fixed" or t["fixed_alpha"] != 0.01
          or t["seed"] != 51001
          or t["total_steps"] not in VARIABLE_BEAM_STEP_BUDGETS
          or t["batch_size"] != 256 or t["replay_capacity"] != 20000 or t["warmup_steps"] != 2000
          or t["update_every"] != 1 or t["gradient_steps"] != 1 or t["checkpoint_interval"] != 10000
          or n["embedding_dim"] != 128 or n["num_heads"] != 8 or n["num_layers"] != 2
          or n["feedforward_dim"] != 512 or t["gamma"] != 0.99 or t["tau"] != 0.005
          or t["actor_lr"] != 1e-5 or t["critic_lr"] != 1e-4 or t["gradient_clip"] != 10):
        raise ValueError("V0/V1配置必须与已批准的欧洲可变束数方案一致")
    # S2-P18 三条机制臂（方案18 第二版）需要与 V0/V1 不同的、但同样冻结的取值：
    # 试点预算 10 万步、逐束布局/读出/条件化、进度分层与折扣日程、奖励整形的
    # 非零权重。除这些「臂自身声明的差异」外，其余条件仍与 V0/V1 完全一致，
    # 并且只允许登记在案的取值集合，不做任意放宽。
    if c["experiment"]["arm"] and not p19:
        arm = c["experiment"]["arm"]
        if arm not in VARIABLE_BEAM_ARMS:
            raise ValueError("未知的S2-P18实验臂")
        if t["total_steps"] not in VARIABLE_BEAM_STEP_BUDGETS:
            raise ValueError("S2-P18臂的训练预算不在已批准集合内")
        if n["transformer_layout"] not in ("tsac_14_variable_context", "tsac_natural_context"):
            raise ValueError("S2-P18臂只允许可变束数布局或自然逐束布局")
        if (n["readout_mode"] != "first_token" or n["beam_count_conditioning"] != "none"
                or n["context_beam_count"] == "actual"):
            if (n["transformer_layout"] != "tsac_natural_context"
                    or c["state"]["context_level"] not in ("layout", "full")
                    or n["context_beam_count"] != "actual"):
                raise ValueError("逐束读出与条件化要求自然布局的 layout/full 档与实际束数上下文")
        if r["shaping"]["w_throughput"] < 0 or r["shaping"]["w_spectrum"] < 0 or r["shaping"]["w_shape"] < 0:
            raise ValueError("S2-P18奖励整形权重不得为负")
        if not 0 <= t["progress_strata"] <= 16:
            raise ValueError("进度分层段数超出允许范围")
        # 自然逐束布局原本只配「每轮打乱场景」；本轮 G0 臂把它用于可变束数四桶
        # 抽样，因此对该臂额外允许 variable_beams 顺序，其余条件不变。
        if (n["transformer_layout"] == "tsac_natural_context"
                and t["scene_order"] not in ("shuffle_each_cycle", "variable_beams")):
            raise ValueError("S2-P18自然布局臂只允许每轮打乱或可变束数四桶场景顺序")
    for key in ("actor_lr", "critic_lr", "alpha_lr", "tau", "initial_alpha", "gradient_clip"):
        _number(t[key], key, 0, strictly=True)
    if t["tau"] > 1:
        raise ValueError("tau 必须不大于1")
    if c["algorithm"] in ("sac", "dqn") and t["batch_size"] > t["replay_capacity"]:
        raise ValueError("批大小不能超过回放容量，否则无法进行参数更新")
    if t["seed"] + 100000 + t["total_steps"] > 2 ** 32 - 1:
        raise ValueError("训练种子与交互步预算之和超出场景随机数范围")
    for key in ("gamma", "gae_lambda", "clip_ratio", "eps_start", "eps_end"):
        _number(t[key], key, 0, 1)
    _validate_gamma_schedule(t)
    if p19:
        import re
        if c["experiment"]["arm"] not in ("C0", "C1"):
            raise ValueError("S2-P19 构造式SAC仅支持C0/C1")
        if (c["algorithm"] != "sac" or n["kind"] != "transformer" or
                n["transformer_layout"] != "tsac_natural_context" or
                n["context_beam_count"] != "actual" or
                c["state"]["context_level"] != "full" or
                c["state"]["features"] != list(natural_shapes("full")) or
                n["sac_policy_structure"] != "factorized" or
                n["sac_critic_structure"] != "additive" or
                n["temporal_mode"] != "none" or n["graph_mode"] != "none" or
                e["profile"] != "europe_v1" or e["beam_count_max"] != 200 or
                e["action_mask"] != "joint_feasible" or e["action_mode"] != "legacy_clip" or
                r["preset"] != "terminal_si_delta" or r["shaping"] != template["reward"]["shaping"] or
                t["gamma"] != 1 or t["gamma_schedule"]["mode"] not in ("none", "constant") or
                resolve_gamma(t, 200) != 1 or t["scene_order"] != "variable_beams" or
                t["beam_count_mode"] != "mixed_four_buckets" or
                t["beam_count_min"] != 50 or t["beam_count_max"] != 200 or
                t["held_out_beam_counts"] != [60, 90, 140, 180] or
                t["seed"] != 51001 or t["total_steps"] != 400000 or
                type(t["pilot_stop_steps"]) is not int or t["pilot_stop_steps"] != 100000 or
                t["batch_size"] != 256 or t["replay_capacity"] != 20000 or
                t["warmup_steps"] != 2000 or t["update_every"] != 1 or
                t["gradient_steps"] != 1 or t["progress_strata"] != 1 or
                t["checkpoint_interval"] != 10000):
            raise ValueError("S2-P19 C0/C1状态、动作、奖励与学习预算必须使用冻结合同")
        demo = t["demonstrations"]
        if demo["mode"] != ("half" if c["experiment"]["arm"] == "C1" else "none"):
            raise ValueError("C1要求半数示范回放，C0不得读取示范")
        if demo["mode"] == "half":
            if (not isinstance(demo["manifest_path"], str) or not demo["manifest_path"] or
                    not all(isinstance(demo[key], str) and re.fullmatch(r"[0-9a-f]{64}", demo[key])
                            for key in ("manifest_sha256", "source_set_sha256"))):
                raise ValueError("C1示范必须冻结绝对清单路径、清单SHA和训练场景集SHA")
            from pathlib import Path
            if not Path(demo["manifest_path"]).is_absolute():
                raise ValueError("C1示范清单须为绝对路径")
        elif any(demo[key] for key in ("manifest_path", "manifest_sha256", "source_set_sha256")):
            raise ValueError("C0不得声明示范来源")
    for key in ("entropy_coef", "value_coef"):
        _number(t[key], key, 0)
    _number(t["target_entropy"], "target_entropy")
    if e["action_mask"] == "joint_feasible" and not 0 <= t["target_entropy"] <= math.log(100 * 10 * len(levels)):
        raise ValueError("联合无效动作掩码的 SAC 目标熵须在0至离散动作理论上限之间")
    if t["eps_start"] < t["eps_end"] or t["device"] not in ("cuda", "cpu"):
        raise ValueError("探索率或设备配置无效")
    ev = c["evaluation"]
    for key in ("validation_count", "test_count", "validation_seed", "test_seed", "interval_steps", "repeats"):
        _number(ev[key], key, 0 if key != "repeats" else 1, integer=True)
    steps = ev["extra_steps"]
    if (not isinstance(steps, list) or any(type(step) is not int or step < 0 or
            step > t["total_steps"] for step in steps) or steps != sorted(set(steps))):
        raise ValueError("额外验证步数必须是不重复、升序的非负整数，且不超过交互步预算")
    for key in ("validation_count", "test_count"):
        if ev[key] > 1000:
            raise ValueError(f"{key} 不得超过1000")
    if ev["validation_seed"] + max(0, ev["validation_count"] - 1) > 2 ** 32 - 1 or ev["test_seed"] + max(0, ev["test_count"] - 1) > 2 ** 32 - 1:
        raise ValueError("场景种子超出NumPy允许范围")
    if ev["action_mode"] not in ("deterministic", "stochastic"):
        raise ValueError("不支持的评价动作方式")
    if t["beam_count_mode"] != "none" and not p19 and (
            ev["validation_count"] != 144 or ev["interval_steps"] != 20000
            or ev["extra_steps"] != [0] or ev["repeats"] != 1
            or ev["validation_seed"] != 53001 or ev["action_mode"] != "stochastic"
            or ev["test_count"] != 0):
        raise ValueError("V0/V1验证须为144场景、0至400000每20000步、每场景一次；训练任务不运行最终考核")
    telemetry = c["telemetry"]
    if telemetry["mode"] not in ("sync", "async"):
        raise ValueError("记录方式仅支持sync或async")
    _number(telemetry["flush_interval_seconds"], "异步记录刷新间隔", 0.1, 30)
    _number(telemetry["queue_capacity"], "异步记录队列容量", 16, 32768, integer=True)
    _network_size(c)
    return c


def catalog():
    features = deepcopy(FEATURES)
    for feature in features:
        feature["dim"] = math.prod(feature["shape"])
        feature["requires"] = "known_demand" if feature["visibility"] == "known" else None
    return dict(features=features,
                reward_terms=[dict(id=i, label=l, description=d, formula=REWARD_FORMULAS[i], kind="penalty" if i.endswith("penalty") or i in ("power_cost", "slot_cost", "interference_loss", "excess_supply") else "reward", default_weight=1, timing=t, group=g, implemented=True)
                              for i, l, d, t, g in REWARD_TERMS],
                environment_profiles=profile_catalog(),
                algorithms=[dict(id=i, label=i.upper(), description="新工作台可训练实现") for i in ("sac", "ppo", "dqn")],
                networks=[dict(id=i, label=i.upper(), description="可配置骨干网络") for i in ("mlp", "cnn", "transformer")],
                sac_policy_structures=[dict(id="factorized", label="原分头策略", description="频率起点、槽数与功率分头打分"),
                                       dict(id="conditional", label="条件策略", description="先选合法起点与槽数，再按已选频率槽选择功率")],
                sac_critic_structures=[dict(id="additive", label="原加性价值", description="三动作分量的价值相加"),
                                       dict(id="joint", label="联合动作价值", description="对完整频率槽与功率组合打分")],
                alpha_modes=[dict(id="adaptive", label="自动调节温度"), dict(id="fixed", label="固定温度")],
                temporal_modes=[dict(id="none", label="无跨步历史"),
                                dict(id="previous_action", label="只加上一束实际动作"),
                                dict(id="gru", label="同场景窗口GRU记忆"),
                                dict(id="causal_transformer", label="同场景窗口因果Transformer")],
                graph_modes=[dict(id="none", label="原状态令牌"),
                             dict(id="beam_tokens", label="逐束令牌（122）"),
                             dict(id="interference_bias", label="逐束令牌＋物理干扰关系")],
                scene_orders=[dict(id="sequential", label="原顺序循环", description="按固定场景集记录顺序循环", implemented=True),
                              dict(id="shuffle_each_cycle", label="每轮打乱", description="每轮将所有固定训练场景无放回排列；不改变场景内波束顺序", implemented=True),
                              dict(id="variable_beams", label="可变束数四桶", description="V0固定100束或V1四数量桶等概率抽样", implemented=True)],
                beam_count_modes=[dict(id="none", label="原固定束数"),
                                  dict(id="fixed_100", label="V0 固定100束"),
                                  dict(id="mixed_four_buckets", label="V1 四桶各25%")],
                presets=deepcopy(PRESETS),
                # S2-P18 三臂预设：前端按 id 套用对应状态/奖励/学习机制组合。
                s2p18_arms=[
                    dict(id="V2", label="V2 目标与观测修正",
                         description="奖励加吞吐与频谱使用项＋修正 power_pressure 归一化＋组号与预留义务可见＋进度分层采样",
                         state_features=list(S2P18_VARIABLE_FEATURES), reward_preset="legacy_repaired_throughput",
                         reward_shaping=dict(w_throughput=50, w_spectrum=1, w_shape=0,
                                             throughput_reference="mean_demand"),
                         progress_strata=4, gamma_schedule=dict(mode="none", default=0.99,
                                                                by_beam_count=[]),
                         network=dict(transformer_layout="tsac_14_variable_context",
                                      context_beam_count="fixed_100", readout_mode="first_token",
                                      beam_count_conditioning="none")),
                    dict(id="V3", label="V3 信用分配修正",
                         description="与 V2 同一状态定义；按束数设定折扣＋跨束势能型奖励整形，不改奖励主项",
                         state_features=list(S2P18_VARIABLE_FEATURES), reward_preset="legacy_repaired",
                         reward_shaping=dict(w_throughput=0, w_spectrum=0, w_shape=0.5,
                                             throughput_reference="mean_demand"),
                         progress_strata=1,
                         gamma_schedule=dict(mode="by_beam_count", default=0.995,
                                             by_beam_count=[dict(max_beam_count=100, gamma=0.99)]),
                         network=dict(transformer_layout="tsac_14_variable_context",
                                      context_beam_count="fixed_100", readout_mode="first_token",
                                      beam_count_conditioning="none")),
                    dict(id="G0", label="G0 逐束实体编码原型",
                         description="逐束上下文按实际束数构造＋注意力池化读出＋FiLM 式 N 条件调制，暂不加 A/D",
                         state_features=["beam", "power", "interference", "occupancy",
                                         "natural_global_context", "natural_beam_context",
                                         "natural_resource_context"],
                         reward_preset="legacy_repaired",
                         reward_shaping=dict(w_throughput=0, w_spectrum=0, w_shape=0,
                                             throughput_reference="mean_demand"),
                         progress_strata=1,
                         gamma_schedule=dict(mode="none", default=0.99, by_beam_count=[]),
                         network=dict(transformer_layout="tsac_natural_context",
                                      context_level="layout", context_beam_count="actual",
                                      readout_mode="attention_pooling",
                                      beam_count_conditioning="film")),
                ],
                optional_state_features=[
                    dict(id="group", label="当前组号", description="复用组号除以7；V2/V3 状态修正项"),
                    dict(id="group_remaining_obligation", label="本组剩余预留义务",
                         description="本复用组尚未处理束数（含当前束）除以总束数"),
                    dict(id="remaining_slot_limit", label="剩余合法槽上限",
                         description="本组空槽数减去为同组其余束各预留1槽后，当前束可用槽数除以100"),
                ],
                readout_modes=[dict(id="first_token", label="读取首个令牌（原读出）",
                                    description="与原 T-SAC 一致，读取位置0的令牌；已有对照都使用该模式"),
                               dict(id="attention_pooling", label="注意力池化读出",
                                    description="对逐束实体令牌做可学习打分的加权平均；实体置换后读出与决策一致，占位不参与池化")],
                beam_count_conditioning_modes=[dict(id="none", label="不加 N 条件",
                                                    description="与原结构一致，N 只经输入标量进入"),
                                               dict(id="film", label="FiLM 式 N 条件调制",
                                                    description="用当前处理序号除以束数上界，对读出结果做逐通道缩放与平移")],
                context_beam_counts=[dict(id="fixed_100", label="固定100束上下文",
                                          description="历史行为：逐束字段固定 100 行，观测宽度与 N 无关"),
                                     dict(id="actual", label="按实际束数构造",
                                          description="逐束实体编码：状态只含真实实体，宽度随 N 变化，不填充到100")],
                progress_strata_options=[dict(id="1", label="关闭（旧采样）", description="每批只在数量桶之间等分"),
                                          dict(id="2", label="2 段", description="每批同时覆盖数量桶×进度前半/后半"),
                                          dict(id="4", label="4 段", description="每批同时覆盖数量桶×进度四分位（V2 使用）")],
                gamma_schedule_modes=[dict(id="none", label="使用固定折扣（默认0.99）", description="旧行为逐位不变"),
                                      dict(id="constant", label="统一取新值", description="例如统一 0.995"),
                                      dict(id="by_beam_count", label="按束数分档",
                                           description="按实际束数取第一个 max_beam_count ≥ N 的档位，V3 使用")],
                transformer_layouts=[dict(id="scalar_tokens", label="逐标量令牌（现有工作台）", implemented=True),
                                     dict(id="tsac_13_tokens", label="原 T-SAC（13 令牌）", implemented=True,
                                          description="需 SAC、Transformer 和原204维状态顺序；旧权重不直接复用"),
                                     dict(id="tsac_14_context", label="T-SAC＋全局状态（14令牌）", implemented=True,
                                          description="207维：原204维后加平均需求、进度、剩余需求；独立权重版本"),
                                     dict(id="tsac_14_variable_context", label="可变束数T-SAC（14令牌）", implemented=True,
                                          description="210维：原207维后加实际束数、剩余束数、剩余功率压力；频率动作仍为100槽"),
                                     dict(id="tsac_32_context", label="状态分档对照（32令牌）", implemented=True,
                                          description="四档共用2946维封装与32令牌骨干，未开放信息严格置零"),
                                     dict(id="tsac_natural_context", label="自然维度状态对照", implemented=True,
                                          description="输入仅含实际开放信息；状态与令牌数随档位变化")],
                network_presets=[dict(id="tsac_13_tokens", label="原 T-SAC（13 令牌）",
                                      description="一键设置原网络尺寸和204维状态输入；新训练的权重与旧实验不逐位相同",
                                      algorithm="sac", state_features=["beam", "power", "interference", "occupancy"],
                                      network=dict(kind="transformer", transformer_layout="tsac_13_tokens",
                                                   embedding_dim=128, num_heads=8, num_layers=2,
                                                   feedforward_dim=512, activation="relu", dropout=0.1)),
                                 dict(id="tsac_14_context", label="T-SAC＋全局状态（14令牌）",
                                      description="补充平均业务需求、处理进度和剩余需求；保留原网络尺寸和奖励配置",
                                      algorithm="sac", demand_visibility="known",
                                      state_features=["beam", "power", "interference", "occupancy", "beta", "progress", "remaining_demand"],
                                      network=dict(kind="transformer", transformer_layout="tsac_14_context",
                                                   embedding_dim=128, num_heads=8, num_layers=2,
                                                   feedforward_dim=512, activation="relu", dropout=0.1)),
                                 dict(id="tsac_14_variable_context", label="可变束数T-SAC（14令牌）",
                                      description="V0/V1共同210维网络；需欧洲场景、3000W及四桶配置",
                                      algorithm="sac", demand_visibility="known", context_level="none",
                                      reward_preset="legacy_repaired", scene_order="variable_beams",
                                      state_features=list(VARIABLE_BEAM_FEATURES),
                                      network=dict(kind="transformer", transformer_layout="tsac_14_variable_context",
                                                   embedding_dim=128, num_heads=8, num_layers=2,
                                                   feedforward_dim=512, activation="relu", dropout=0)),
                                 dict(id="tsac_32_context", label="状态分档对照（32令牌）",
                                      description="同一结构比较四档可见信息，未开放信息归零；保留旧奖励设置",
                                      algorithm="sac", demand_visibility="known", context_level="compact",
                                      state_features=["beam", "power", "interference", "occupancy", "global_context", "beam_context", "resource_context"],
                                      network=dict(kind="transformer", transformer_layout="tsac_32_context",
                                                   embedding_dim=128, num_heads=8, num_layers=2,
                                                   feedforward_dim=512, activation="relu", dropout=0.1)),
                                 dict(id="tsac_natural_context", label="自然维度状态对照",
                                      description="状态输入及令牌随档位变化；旧奖励、关闭dropout、每轮打乱场景",
                                      algorithm="sac", demand_visibility="known", context_level="none",
                                      reward_preset="legacy_repaired", scene_order="shuffle_each_cycle",
                                      state_features=list(natural_shapes("none")),
                                      network=dict(kind="transformer", transformer_layout="tsac_natural_context",
                                                   embedding_dim=128, num_heads=8, num_layers=2,
                                                   feedforward_dim=512, activation="relu", dropout=0))],
                context_levels=[dict(id="compact", label="少量全局信息", description="原204维＋平均需求、进度、剩余需求"),
                                dict(id="summary", label="资源与服务概况", description="再开放全局及8组资源服务概况"),
                                dict(id="layout", label="完整布局与需求", description="再开放全部100束的位置、需求、分组与顺序"),
                                dict(id="full", label="布局、分配与干扰", description="再开放已执行动作、逐束当前服务、全部组逐槽占用与历史干扰")],
                natural_context_levels=[dict(id=level, label=label, description=description,
                                             state_dim=natural_dim(level), token_count=natural_token_count(level),
                                             features=list(natural_shapes(level)))
                                        for level, label, description in (
                                            ("none", "原204维", "仅原有波束、功率、干扰与占用"),
                                            ("compact", "少量全局信息", "增加平均需求、进度与剩余需求"),
                                            ("summary", "资源与服务概况", "增加全局10项与8组资源概况"),
                                            ("layout", "完整布局与需求", "增加100束位置、需求、分组与顺序"),
                                            ("full", "布局、分配与干扰", "增加已执行动作、当前服务、逐槽占用与历史干扰"))],
                action_masks=[dict(id="none", label="关闭", description="沿用原有动作选择与环境裁剪", implemented=True),
                              dict(id="joint_feasible", label="开启：联合掩码＋最低资源预留",
                                   description="仅支持 SAC；联合选择连续空槽和功率，为后续同组波束预留最低频率槽及最低功率", implemented=True,
                                   algorithms=["sac"])])


def config_summary(config):
    c = validate_config(config)
    layout, offset = [], 0
    definitions = {f["id"]: f for f in FEATURES}
    for fid in c["state"]["features"]:
        shape = _feature_shape(c, fid)
        dim = math.prod(shape)
        layout.append(dict(id=fid, offset=offset, dim=dim, shape=shape))
        offset += dim
    r = c["reward"]
    warnings = []
    environment_metadata = profile_metadata(c["environment"]["profile"])
    if c["environment"]["profile"] == "europe_v1":
        warnings.append("欧洲环境的几何与位置归一化独立；默认场景仅为合成参考，不能视作航空或真实业务。旧中国权重不构成欧洲已验证策略。")
    if r["preset"] == "custom":
        terms = r["terms"]
        formula = " + ".join(f"{x['weight']:g}×({REWARD_FORMULAS[x['id']]})/{x['scale']:g}" for x in terms)
        selected = {t["id"] for t in terms}
        groups = {g for i, _, _, _, g in REWARD_TERMS if i in selected}
        if "service" in groups and "matching" in groups:
            warnings.append("需求满足率与供需匹配项可能重复计分；请核对权重。")
        if {"si_delta", "shortfall_delta"} <= selected:
            warnings.append("欠供减少与需求满足率增量在本口径下数值相同，会重复计分。")
        if "clip_penalty" in selected and ("out_penalty" in selected or "occupy_penalty" in selected):
            warnings.append("动作裁剪已包含越界和占用导致的槽数减少，可能重复惩罚。")
        if "constraint" in groups and c["environment"]["action_mode"] == "feasible_projection":
            warnings.append("动作投影已处理部分约束；相应惩罚可能恒为零。")
        if c["state"]["demand_visibility"] == "online" and any(x["id"] in ("sgm_delta", "current_sgm", "terminal_sgm") for x in terms):
            warnings.append("供需匹配奖励使用全场景平均需求；在线观测不包含未来需求。")
    else:
        formula = {"legacy_original": "原公式：阈值满足率＋本束SGM－SGM偏差惩罚－占用冲突比例；零槽仍计功率",
                   "legacy_repaired": "原公式：阈值满足率＋本束SGM－SGM偏差惩罚－占用冲突比例；零槽不计功率",
                   "legacy_repaired_throughput":
                       "原公式四项 + w_throughput×ΔΣmin(C,D)/参考量 + w_spectrum×(本步占用槽/本组可用槽)"
                       " + w_shape×(γΦ(s')−Φ(s))；原始奖励分量仍逐项保存",
                   "terminal_si_delta": "100×[全部目标波束当前SI均值的逐步变化]；γ=1时回合和为100×终态SI",
                   "fused": "2×Δ基本服务＋Δ需求满足率＋0.2×ΔSGM－实际功率/预算"}[r["preset"]]
        if r["terms"] != default_config()["reward"]["terms"]:
            warnings.append("当前预设使用固定公式；奖励项列表仅在 custom 预设生效。")
        shaping = r["shaping"]
        if r["preset"] == "legacy_repaired_throughput":
            warnings.append(
                "奖励已追加吞吐与频谱使用项（w_throughput={:g}、w_spectrum={:g}，参考量 {}）；"
                "累计回报不能与旧奖励口径直接排名，报告须同时给出原始奖励口径的回报。".format(
                    shaping["w_throughput"], shaping["w_spectrum"], shaping["throughput_reference"]))
        if shaping["w_shape"]:
            warnings.append(
                "已启用跨束势能型奖励整形 w_shape={:g}；该项为 Φ(s)=mean_i min(C_i/D_i,1) 的差分，"
                "不改变最优策略，但会改变学习曲线的数值尺度。".format(shaping["w_shape"]))
    if c["state"]["demand_visibility"] == "online" and r["preset"] in ("legacy_original", "legacy_repaired", "fused"):
        warnings.append("在线需求模式的旧奖励仍按全场景需求计算SGM；观测不包含未来需求。")
    if c["algorithm"] == "sac" and not 0 <= c["training"]["target_entropy"] <= math.log(math.prod([100, 10, len(c["environment"]["power_levels_w"])])):
        warnings.append("SAC 目标熵位于当前离散动作的理论范围之外；请检查温度更新方向与取值。")
    transformer_layout = c["network"]["transformer_layout"] if c["network"]["kind"] == "transformer" else None
    variable_features = list(c["state"]["features"]) == list(S2P18_VARIABLE_FEATURES)
    beam_context_actual = c["network"].get("context_beam_count", "fixed_100") == "actual"
    layout_description = ({"scalar_tokens": "状态中每个标量各形成一个令牌，全部令牌汇总后输出",
                           "tsac_13_tokens": "位置、业务需求、功率各1个令牌，干扰与占用按10段合为10个频率槽令牌；读取首令牌输出",
                           "tsac_14_context": "保留原13令牌，另加平均需求、处理进度与剩余需求组成的全局令牌；207维输入，读取首令牌输出",
                           "tsac_14_variable_context": "保留原13令牌，全局令牌含原3项与束数、剩余束数、功率压力；210维输入，共14令牌",
                           "tsac_32_context": "四档同2946维封装：原13令牌＋全局1＋每10束一段共10＋8个资源组；未开放信息置零，共32令牌",
                           "tsac_natural_context": "状态与令牌仅包含本档实际开放的信息；无未开放字段占位"}
                          .get(transformer_layout))
    if transformer_layout == "tsac_14_variable_context" and variable_features:
        layout_description = ("保留原13令牌，全局令牌含原3项、束数、剩余束数、功率压力与组号、"
                              "本组剩余预留义务、剩余合法槽上限；共14令牌")
    if beam_context_actual:
        layout_description = (layout_description or "") + "；逐束上下文按场景实际束数构造，不填充到100束"
    temporal_mode = c["network"]["temporal_mode"]
    graph_mode = c["network"]["graph_mode"]
    token_count = (natural_token_count(c["state"]["context_level"])
                   if transformer_layout == "tsac_natural_context" else
                   {"tsac_13_tokens": 13, "tsac_14_context": 14,
                    "tsac_14_variable_context": 14, "tsac_32_context": 32}.get(transformer_layout, offset)
                   if transformer_layout else None)
    if graph_mode != "none":
        token_count = 122
        layout_description = "2946维原始字段：原13令牌＋全局1＋逐束100＋资源组8，共122令牌"
        if graph_mode == "interference_bias":
            layout_description += "；由动作前可见字段和固定物理模型计算有向干扰关系"
    elif temporal_mode != "none":
        token_count += 1
        layout_description = "211维：207维当前观测＋上一束实际动作与有效位；每步15个资源令牌"
        if temporal_mode in ("gru", "causal_transformer"):
            layout_description += f"；每个目标决策附最多{c['network']['temporal_window']}步同场景历史，仅目标位置计一个学习损失"
    return dict(state_dim=offset, state_layout=layout,
                state_version=("variable_beams_210_v1" if transformer_layout == "tsac_14_variable_context"
                               and not variable_features else
                               "variable_beams_s2p18_v1" if transformer_layout == "tsac_14_variable_context"
                               else "tsac_14_context_207_v1" if transformer_layout == "tsac_14_context"
                               else transformer_layout),
                beam_count_max=c["environment"]["beam_count_max"],
                beam_count_mode=c["training"]["beam_count_mode"],
                held_out_beam_counts=c["training"]["held_out_beam_counts"],
                environment_profile=c["environment"]["profile"],
                environment_metadata=environment_metadata,
                transformer_layout=transformer_layout,
                network_layout_description=layout_description,
                network_token_count=token_count,
                temporal_mode=temporal_mode,
                temporal_window=(c["network"]["temporal_window"] if temporal_mode in ("gru", "causal_transformer") else 1),
                graph_mode=graph_mode,
                alpha_mode=c["training"]["alpha_mode"],
                fixed_alpha=c["training"]["fixed_alpha"] if c["training"]["alpha_mode"] == "fixed" else None,
                progress_strata=c["training"]["progress_strata"],
                gamma=c["training"]["gamma"],
                gamma_schedule=c["training"]["gamma_schedule"],
                effective_gamma=({str(N): resolve_gamma(c["training"], N)
                                  for N in (50, 100, 120, 160, 200)}
                                 if c["training"]["gamma_schedule"]["mode"] != "none" else None),
                reward_shaping=c["reward"]["shaping"],
                reward_preset=c["reward"]["preset"],
                beam_context_count=c["network"].get("context_beam_count", "fixed_100"),
                readout_mode=c["network"].get("readout_mode", "first_token"),
                beam_count_conditioning=c["network"].get("beam_count_conditioning", "none"),
                context_level=c["state"]["context_level"],
                telemetry=c["telemetry"],
                action_dims=[100, 10, len(c["environment"]["power_levels_w"])],
                action_mask=c["environment"]["action_mask"],
                sac_policy_structure=c["network"]["sac_policy_structure"],
                sac_critic_structure=c["network"]["sac_critic_structure"],
                evaluation_extra_steps=c["evaluation"]["extra_steps"],
                reward_formula=formula, warnings=warnings)
