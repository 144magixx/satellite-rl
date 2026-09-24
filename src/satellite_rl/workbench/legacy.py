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


def legacy_config(legacy_type):
    from .config import default_config
    if legacy_type not in LEGACY_TYPES:
        raise ValueError("请选择明确支持的旧模型类型")
    config = default_config()
    config["name"] = LEGACY_TYPES[legacy_type]
    config["algorithm"] = legacy_type.split("_")[0]
    config["network"]["kind"] = "transformer" if "transformer" in legacy_type else "cnn" if "cnn" in legacy_type else "mlp"
    config["reward"]["preset"] = "legacy_original"
    if legacy_type == "ppo_direct":
        config["environment"]["power_levels_w"] = [3 * index for index in range(1, 11)]
    return config


class LegacyPolicy:
    def __init__(self, network, legacy_type, device):
        self.network = network.to(device).eval()
        self.device = torch.device(device)
        self.legacy_type = legacy_type

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
    if config["environment"]["power_levels_w"] != expected["environment"]["power_levels_w"]:
        raise ValueError("旧权重功率档位含义不匹配")
    if config["environment"]["action_mode"] != "legacy_clip":
        raise ValueError("旧权重导入使用其原动作含义；新的动作投影须作为独立环境对照")
