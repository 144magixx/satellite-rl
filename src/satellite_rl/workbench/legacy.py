"""Explicit, strict inference adapters for the repository's historical weights."""

from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import torch

LEGACY_TYPES = {
    "sac_transformer": "旧 SAC Transformer", "sac_cnn": "旧 SAC CNN",
    "sac_mlp": "旧 SAC MLP", "sac_cnn_matched": "匹配规模 SAC CNN",
    "sac_mlp_matched": "匹配规模 SAC MLP", "ppo_direct": "旧 PPO direct",
    "dqn": "旧分量式 DQN",
}


def legacy_architecture(legacy_type):
    """Describe the historical actor itself, not the new workbench backbone."""
    if legacy_type not in LEGACY_TYPES:
        raise ValueError("请选择明确支持的旧模型类型")
    common = {
        "variant": legacy_type,
        "label": LEGACY_TYPES[legacy_type],
        "input_state_dim": 204,
        "action_dims": [100, 10, 10],
        "input_layout": "当前波束3＋剩余功率1＋逐槽干扰100＋本组占用100；固定原顺序与归一化",
        "training_provenance": "配置中的 training 字段仅供本次测试任务使用，不代表旧权重的真实训练参数或来源。",
        "implementation": "satellite_rl.agents." + legacy_type,
    }
    if legacy_type == "sac_transformer":
        common.update(layers=["地理2/需求1/功率1分别嵌入128维", "20维槽位片段嵌入128维×10",
                              "13个令牌的位置编码", "Transformer编码器128维×2层，8头，前馈512维",
                              "三动作头100/10/10"],
                      token_count=13, embedding_dim=128, num_heads=8, num_layers=2,
                      feedforward_dim=512, dropout=0.1,
                      note="旧模型将204维输入重排为13个令牌；与工作台逐标量令牌Transformer不是同一网络。")
    elif "cnn" in legacy_type:
        widths = [128, 128] if legacy_type == "sac_cnn_matched" else [256, 128]
        common.update(layers=["频谱200维重排为2×10×10", "Conv2d 2→16，3×3",
                              "Conv2d 16→32，3×3", f"拼接前4维后全连接{widths[0]}→{widths[1]}",
                              "三动作头100/10/10"],
                      input_image_shape=[2, 10, 10], conv_channels=[16, 32],
                      kernel_size=3, dense_hidden_sizes=widths,
                      note="旧模型使用二维卷积与3204维拼接；工作台CNN采用一维卷积，二者不能互换权重。")
    else:
        widths = {"sac_mlp": [128], "sac_mlp_matched": [512, 512],
                  "ppo_direct": [256, 128], "dqn": [512, 256]}[legacy_type]
        common.update(layers=[f"204→{'→'.join(map(str, widths))}", "三动作头100/10/10"],
                      dense_hidden_sizes=widths,
                      note="动作头与输入顺序固定；网络结构按旧实现加载和严格校验。")
    return common


def legacy_config(legacy_type):
    from .config import default_config
    if legacy_type not in LEGACY_TYPES:
        raise ValueError("请选择明确支持的旧模型类型")
    config = default_config()
    config["name"] = LEGACY_TYPES[legacy_type]
    config["algorithm"] = legacy_type.split("_")[0]
    network = config["network"]
    network["kind"] = "transformer" if "transformer" in legacy_type else "cnn" if "cnn" in legacy_type else "mlp"
    architecture = legacy_architecture(legacy_type)
    if network["kind"] == "transformer":
        network.update(embedding_dim=128, num_heads=8, num_layers=2,
                       feedforward_dim=512, dropout=0.1)
    elif network["kind"] == "cnn":
        network.update(cnn_channels=[16, 32], kernel_size=3,
                       hidden_sizes=architecture["dense_hidden_sizes"])
    else:
        network["hidden_sizes"] = architecture["dense_hidden_sizes"]
    config["reward"]["preset"] = "legacy_original"
    if legacy_type == "ppo_direct":
        config["environment"]["power_levels_w"] = [3 * index for index in range(1, 11)]
    return config


class LegacyPolicy:
    def __init__(self, network, legacy_type, device):
        self.network = network.to(device).eval()
        self.device = torch.device(device)
        self.legacy_type = legacy_type
        self.architecture = legacy_architecture(legacy_type)
        self.architecture["parameter_count"] = sum(parameter.numel() for parameter in network.parameters())

    @torch.no_grad()
    def act(self, state, deterministic=True):
        array = np.asarray(state, dtype=np.float32)
        if array.shape != (204,) or not np.isfinite(array).all():
            raise ValueError("旧模型只支持原始语义和顺序的204维输入")
        outputs = self.network(torch.as_tensor(array, device=self.device).unsqueeze(0))
        if deterministic or self.legacy_type == "dqn":
            return [int(value.argmax(-1).item()) for value in outputs]
        return [int(torch.distributions.Categorical(probs=value).sample().item()) for value in outputs]


def load_legacy_policy(path, legacy_type, device="cpu"):
    if legacy_type not in LEGACY_TYPES:
        raise ValueError("不支持的旧模型格式，请选择明确的算法与网络类型")
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("旧模型必须是权重字典，不能执行任意模型对象")
    state = payload
    for key in ("actor_state_dict", "actor", "online", "state_dict"):
        if isinstance(payload.get(key), dict):
            state = payload[key]
            break
    if isinstance(payload.get("networks"), dict):
        state = payload["networks"].get("actor", state)
    if not isinstance(state, dict):
        raise ValueError("旧权重内容必须是参数字典")
    for name, value in state.items():
        if isinstance(value, torch.Tensor) and (torch.is_floating_point(value) or torch.is_complex(value)):
            if not torch.isfinite(value).all().item():
                raise ValueError(f"旧权重包含NaN或Inf参数: {name}")
    if legacy_type.startswith("sac_"):
        module = importlib.import_module("satellite_rl.agents." + legacy_type)
        hidden = int(state["fc1.weight"].shape[0]) if "fc1.weight" in state else 128
        network = module.PolicyNet(204, hidden, 100, 10, 10)
    elif legacy_type == "ppo_direct":
        from satellite_rl.agents.ppo_direct import PolicyNet
        network = PolicyNet(204)
    else:
        from satellite_rl.agents.dqn import FactorisedDQN
        network = FactorisedDQN(204)
    network.load_state_dict(state, strict=True)
    policy = LegacyPolicy(network, legacy_type, device)
    policy.act(np.zeros(204, dtype=np.float32))
    return policy


def validate_legacy_config(config, legacy_type):
    expected = legacy_config(legacy_type)
    if config["state"] != expected["state"]:
        raise ValueError("旧模型状态定义不兼容：必须保留原204维内容、顺序与归一化")
    if config["algorithm"] != expected["algorithm"]:
        raise ValueError("旧权重算法不匹配")
    if config["network"] != expected["network"]:
        raise ValueError("旧权重网络结构不兼容：请使用该旧模型的固定结构与参数，工作台新网络不能冒充旧权重")
    if config["environment"]["power_levels_w"] != expected["environment"]["power_levels_w"]:
        raise ValueError("旧权重功率档位含义不匹配")
    if config["environment"]["action_mode"] != "legacy_clip":
        raise ValueError("旧权重导入使用其原动作含义；新的动作投影须作为独立环境对照")
