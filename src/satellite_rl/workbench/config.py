"""Versioned, strict configuration and discoverable workbench capabilities."""

from copy import deepcopy
import math
from satellite_rl.envs.regions import get_profile, profile_catalog, profile_metadata


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
    dict(id="remaining_demand_mean", label="剩余平均需求", description="含当前波束的剩余平均需求除以600 Mbps", shape=[1], unit="比例", timing="动作前", source="全场景需求", visibility="known"),
    dict(id="served_shortfall", label="已服务平均欠供", description="已处理波束的平均(1-满足率)，无记录时为0", shape=[1], unit="比例", timing="动作前", source="已处理波束当前速率", visibility="both"),
    dict(id="served_met", label="已服务达标比例", description="已处理波束中满足需求者比例，无记录时为0", shape=[1], unit="比例", timing="动作前", source="已处理波束当前速率", visibility="both"),
    dict(id="served_min", label="已服务最低满足率", description="已处理波束最低截断满足率，无记录时为0", shape=[1], unit="比例", timing="动作前", source="已处理波束当前速率", visibility="both"),
    dict(id="other_occupancy", label="其余组占用", description="其余7组各100槽占用次数，按组号排序", shape=[7, 100], unit="次数", timing="动作前", source="已分配波束", visibility="both"),
    dict(id="longest_free", label="最长连续空槽", description="本组最长连续空槽数除以100", shape=[1], unit="比例", timing="动作前", source="已分配波束", visibility="both"),
]

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
    dict(id="custom", label="自选奖励项", description="逐项计算并保存原值、缩放值与加权值"),
]


def default_config():
    return deepcopy(dict(
        schema_version=1, name="新实验", algorithm="sac",
        network=dict(kind="mlp", transformer_layout="scalar_tokens",
                     hidden_sizes=[256, 256], activation="relu", dropout=0,
                     cnn_channels=[16, 32], kernel_size=3, embedding_dim=64,
                     num_heads=4, num_layers=2, feedforward_dim=128),
        state=dict(features=["beam", "power", "interference", "occupancy"], demand_visibility="known"),
        reward=dict(preset="legacy_repaired", terms=[dict(id="si_delta", weight=1, scale=1, timing="step")]),
        environment=dict(profile="china_legacy", power_budget_w=3000, basic_fraction=0.1, action_mode="legacy_clip",
                         action_mask="none",
                         power_levels_w=[5, 10, 15, 20, 25, 30, 35, 40, 45, 50]),
        training=dict(seed=51001, total_steps=10000, batch_size=256, actor_lr=0.00001,
                      critic_lr=0.0001, alpha_lr=0.0001, gamma=0.99, tau=0.005,
                      replay_capacity=20000, warmup_steps=2000, update_every=1,
                      gradient_steps=1, initial_alpha=0.1, target_entropy=6.9,
                      gradient_clip=10, rollout_steps=256, ppo_epochs=4, gae_lambda=0.95,
                      clip_ratio=0.2, entropy_coef=0.01, value_coef=0.5,
                      eps_start=1, eps_end=0.05, eps_decay_steps=5000,
                      target_update_interval=500, checkpoint_interval=1000,
                      log_interval=50, device="cuda"),
        evaluation=dict(validation_count=3, test_count=5, validation_seed=94000,
                        test_seed=95000, interval_steps=1000,
                        action_mode="deterministic", repeats=1),
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
    state_dim = sum(math.prod(next(f["shape"] for f in FEATURES if f["id"] == fid))
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
        token_count = 13 if n["transformer_layout"] == "tsac_13_tokens" else state_dim
        if token_count ** 2 * n["num_heads"] * n["num_layers"] > 20_000_000:
            raise ValueError("Transformer 状态长度、注意力头数与层数的组合过大")
        projection = 42 * embed if n["transformer_layout"] == "tsac_13_tokens" else (state_dim + 2) * embed
        encoder = projection + n["num_layers"] * (
            4 * embed * embed + 2 * embed * ff + 9 * embed + ff)
        output = embed
    actor = encoder + (output + 1) * heads
    if config["algorithm"] == "sac":
        total = 5 * actor + 1
    elif config["algorithm"] == "dqn":
        total = 2 * actor
    else:
        total = actor + encoder + output + 1
    if total > 20_000_000:
        raise ValueError("模型预计参数超过2000万；请缩小网络后再预览或运行")
    return total


def validate_config(raw):
    c = _merge(default_config(), raw)
    if type(c["schema_version"]) is not int or c["schema_version"] != 1:
        raise ValueError("不支持的配置版本")
    if not isinstance(c["name"], str) or not c["name"].strip() or len(c["name"]) > 120:
        raise ValueError("实验名称无效")
    if c["algorithm"] not in ("sac", "ppo", "dqn"):
        raise ValueError("不支持的算法")
    n = c["network"]
    if n["kind"] not in ("mlp", "cnn", "transformer") or n["activation"] not in ("relu", "gelu", "tanh"):
        raise ValueError("不支持的网络或激活函数")
    if n["transformer_layout"] not in ("scalar_tokens", "tsac_13_tokens"):
        raise ValueError("不支持的 Transformer 输入布局")
    if n["transformer_layout"] == "tsac_13_tokens":
        if c["algorithm"] != "sac" or n["kind"] != "transformer":
            raise ValueError("原 T-SAC 13令牌布局仅支持 SAC 的 Transformer 网络")
        if c["state"]["features"] != ["beam", "power", "interference", "occupancy"]:
            raise ValueError("原 T-SAC 13令牌布局要求原204维状态特征及顺序：波束、功率、干扰、占用")
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
    if e["action_mask"] == "joint_feasible" and e["power_budget_w"] < 100 * levels[0]:
        raise ValueError("联合无效动作掩码要求总功率预算至少覆盖100束的最低功率档位")
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
    t = c["training"]
    ints = {"seed": (0, 2 ** 32 - 1), "total_steps": (1, None), "batch_size": (1, None), "replay_capacity": (1, None),
            "warmup_steps": (0, None), "update_every": (1, None), "gradient_steps": (1, None),
            "rollout_steps": (1, None), "ppo_epochs": (1, None), "eps_decay_steps": (1, None),
            "target_update_interval": (1, None), "checkpoint_interval": (1, None), "log_interval": (1, None)}
    for key, (lo, hi) in ints.items(): _number(t[key], key, lo, hi, integer=True)
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
    for key in ("validation_count", "test_count"):
        if ev[key] > 1000:
            raise ValueError(f"{key} 不得超过1000")
    if ev["validation_seed"] + max(0, ev["validation_count"] - 1) > 2 ** 32 - 1 or ev["test_seed"] + max(0, ev["test_count"] - 1) > 2 ** 32 - 1:
        raise ValueError("场景种子超出NumPy允许范围")
    if ev["action_mode"] not in ("deterministic", "stochastic"):
        raise ValueError("不支持的评价动作方式")
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
                presets=deepcopy(PRESETS),
                transformer_layouts=[dict(id="scalar_tokens", label="逐标量令牌（现有工作台）", implemented=True),
                                     dict(id="tsac_13_tokens", label="原 T-SAC（13 令牌）", implemented=True,
                                          description="需 SAC、Transformer 和原204维状态顺序；旧权重不直接复用")],
                network_presets=[dict(id="tsac_13_tokens", label="原 T-SAC（13 令牌）",
                                      description="一键设置原网络尺寸和204维状态输入；新训练的权重与旧实验不逐位相同",
                                      algorithm="sac", state_features=["beam", "power", "interference", "occupancy"],
                                      network=dict(kind="transformer", transformer_layout="tsac_13_tokens",
                                                   embedding_dim=128, num_heads=8, num_layers=2,
                                                   feedforward_dim=512, activation="relu", dropout=0.1))],
                action_masks=[dict(id="none", label="关闭", description="沿用原有动作选择与环境裁剪", implemented=True),
                              dict(id="joint_feasible", label="开启：联合掩码＋最低资源预留",
                                   description="仅支持 SAC；联合选择连续空槽和功率，为后续同组波束预留最低频率槽及最低功率", implemented=True,
                                   algorithms=["sac"])])


def config_summary(config):
    c = validate_config(config)
    layout, offset = [], 0
    definitions = {f["id"]: f for f in FEATURES}
    for fid in c["state"]["features"]:
        shape = definitions[fid]["shape"]
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
                   "fused": "2×Δ基本服务＋Δ需求满足率＋0.2×ΔSGM－实际功率/预算"}[r["preset"]]
        if r["terms"] != default_config()["reward"]["terms"]:
            warnings.append("当前预设使用固定公式；奖励项列表仅在 custom 预设生效。")
    if c["state"]["demand_visibility"] == "online" and r["preset"] in ("legacy_original", "legacy_repaired", "fused"):
        warnings.append("在线需求模式的旧奖励仍按全场景需求计算SGM；观测不包含未来需求。")
    if c["algorithm"] == "sac" and not 0 <= c["training"]["target_entropy"] <= math.log(math.prod([100, 10, len(c["environment"]["power_levels_w"])])):
        warnings.append("SAC 目标熵位于当前离散动作的理论范围之外；请检查温度更新方向与取值。")
    transformer_layout = c["network"]["transformer_layout"] if c["network"]["kind"] == "transformer" else None
    layout_description = ({"scalar_tokens": "状态中每个标量各形成一个令牌，全部令牌汇总后输出",
                           "tsac_13_tokens": "位置、业务需求、功率各1个令牌，干扰与占用按10段合为10个频率槽令牌；读取首令牌输出"}
                          .get(transformer_layout))
    return dict(state_dim=offset, state_layout=layout,
                environment_profile=c["environment"]["profile"],
                environment_metadata=environment_metadata,
                transformer_layout=transformer_layout,
                network_layout_description=layout_description,
                network_token_count=(13 if transformer_layout == "tsac_13_tokens" else offset if transformer_layout else None),
                action_dims=[100, 10, len(c["environment"]["power_levels_w"])],
                action_mask=c["environment"]["action_mask"],
                reward_formula=formula, warnings=warnings)
