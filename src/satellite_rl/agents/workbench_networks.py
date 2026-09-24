"""Configurable workbench networks. These are new variants, not legacy weights."""

from __future__ import annotations

import torch
from torch import nn


def _activation(name):
    return {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh}[name]


class Encoder(nn.Module):
    def __init__(self, config, state_dim):
        super().__init__()
        spec = config["network"]
        kind = spec["kind"]
        self.kind = kind
        self.state_dim = state_dim
        activation = _activation(spec["activation"])
        dropout = float(spec["dropout"])
        if kind == "mlp":
            layers = []
            width = state_dim
            for hidden in spec["hidden_sizes"]:
                layers += [nn.Linear(width, hidden), activation()]
                if dropout:
                    layers.append(nn.Dropout(dropout))
                width = hidden
            self.net = nn.Sequential(*layers)
            self.output_dim = width
        elif kind == "cnn":
            channels = spec["cnn_channels"]
            kernel = spec["kernel_size"]
            layers = []
            width = 1
            for channel in channels:
                layers += [nn.Conv1d(width, channel, kernel, padding=kernel // 2), activation()]
                if dropout:
                    layers.append(nn.Dropout(dropout))
                width = channel
            bins = min(16, state_dim)
            self.net = nn.Sequential(*layers, nn.AdaptiveAvgPool1d(bins), nn.Flatten())
            self.output_dim = width * bins
        elif kind == "transformer":
            embed = spec["embedding_dim"]
            self.input_projection = nn.Linear(1, embed)
            self.position = nn.Parameter(torch.zeros(1, state_dim, embed))
            layer = nn.TransformerEncoderLayer(embed, spec["num_heads"],
                                               spec["feedforward_dim"], dropout,
                                               activation=torch.tanh if spec["activation"] == "tanh"
                                               else spec["activation"], batch_first=True)
            self.net = nn.TransformerEncoder(layer, spec["num_layers"], enable_nested_tensor=False)
            self.output_dim = embed
        else:
            raise ValueError(f"Unsupported network: {kind}")

    def forward(self, states):
        if states.ndim != 2 or states.shape[1] != self.state_dim:
            raise ValueError(f"State shape must be [batch,{self.state_dim}], got {tuple(states.shape)}")
        if self.kind == "mlp":
            return self.net(states)
        if self.kind == "cnn":
            return self.net(states.unsqueeze(1))
        return self.net(self.input_projection(states.unsqueeze(-1)) + self.position).mean(1)


class MultiHeadNetwork(nn.Module):
    def __init__(self, config, state_dim, action_dims):
        super().__init__()
        self.encoder = Encoder(config, state_dim)
        self.heads = nn.ModuleList([nn.Linear(self.encoder.output_dim, int(dim)) for dim in action_dims])

    def forward(self, states):
        x = self.encoder(states)
        return tuple(head(x) for head in self.heads)


class ScalarNetwork(nn.Module):
    def __init__(self, config, state_dim):
        super().__init__()
        self.encoder = Encoder(config, state_dim)
        self.head = nn.Linear(self.encoder.output_dim, 1)

    def forward(self, states):
        return self.head(self.encoder(states)).squeeze(-1)


def _model_description(config, state_dim, action_dims):
    """Trace the actual constructed layers and tensor shapes for the UI."""
    policy = MultiHeadNetwork(config, state_dim, action_dims)
    algorithm = config["algorithm"]
    actor_count = sum(p.numel() for p in policy.parameters())
    count = actor_count
    policy_layers = _trace_layers(policy, state_dim)
    if algorithm == "sac":
        count = actor_count * 5 + 1  # actor, twin critics/targets, log_alpha
        specifications = [(name, policy_layers, actor_count) for name in
                          ("策略", "价值网络1", "价值网络2", "目标价值网络1", "目标价值网络2")]
    elif algorithm == "ppo":
        value = ScalarNetwork(config, state_dim)
        value_count = sum(p.numel() for p in value.parameters())
        count += value_count
        specifications = [("策略", policy_layers, actor_count),
                          ("状态价值", _trace_layers(value, state_dim), value_count)]
    elif algorithm == "dqn":
        count *= 2
        specifications = [("在线Q网络", policy_layers, actor_count),
                          ("目标Q网络", policy_layers, actor_count)]
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")
    nodes = [{"id": "state", "label": "状态输入", "shape": [state_dim], "parameters": 0}]
    edges = []
    branches = {}
    for branch_index, (branch_name, layers, branch_count) in enumerate(specifications):
        prefix = f"branch_{branch_index}"
        nodes.append({"id": prefix, "label": branch_name, "shape": [state_dim],
                      "parameters": branch_count, "description": "独立参数网络"})
        edges.append({"from": "state", "to": prefix})
        previous_encoder = prefix
        branch_layers = []
        for index, layer in enumerate(layers):
            node_id = f"{prefix}_{index}"
            node = {"id": node_id, "label": f"{branch_name} · {layer['name']}",
                    "shape": layer["output_shape"], "input_shape": layer["input_shape"],
                    "parameters": layer["parameters"], "type": layer["type"],
                    "description": f"输入 {layer['input_shape']} → 输出 {layer['output_shape']}"}
            nodes.append(node)
            edges.append({"from": previous_encoder, "to": node_id})
            if not layer["name"].startswith("heads.") and layer["name"] != "head":
                previous_encoder = node_id
            branch_layers.append(node)
        branches[branch_name] = branch_layers
    if algorithm == "sac":
        nodes.append({"id": "temperature", "label": "SAC 温度 log_alpha", "shape": [1], "parameters": 1})
        edges.append({"from": "branch_0", "to": "temperature"})
    return {"graph": nodes, "nodes": nodes, "edges": edges, "branches": branches,
            "parameter_count": count,
            "variant": f"workbench_{algorithm}_{config['network']['kind']}_v1"}


def model_description(config, state_dim, action_dims):
    """Describe topology without consuming any training initialization randomness."""
    with torch.random.fork_rng(devices=[]):
        return _model_description(config, state_dim, action_dims)


def _trace_layers(model, state_dim):
    """Record leaf modules as they execute; no guessed dimensions or fake layers."""
    selected = (nn.Linear, nn.Conv1d, nn.ReLU, nn.GELU, nn.Tanh, nn.Dropout,
                nn.AdaptiveAvgPool1d, nn.Flatten, nn.MultiheadAttention, nn.LayerNorm)
    layers = []
    hooks = []

    def shape(value):
        if isinstance(value, torch.Tensor):
            return list(value.shape[1:])
        if isinstance(value, (tuple, list)):
            return shape(value[0])
        return None

    for name, module in model.named_modules():
        if name and isinstance(module, selected):
            def record(inner, inputs, output, layer_name=name):
                layers.append({"name": layer_name, "type": type(inner).__name__,
                               "input_shape": shape(inputs), "output_shape": shape(output),
                               "parameters": sum(p.numel() for p in inner.parameters(recurse=False))})
            hooks.append(module.register_forward_hook(record))
    try:
        # Training mode prevents PyTorch's fused Transformer inference fast path
        # from bypassing attention/feedforward module hooks in the diagram.
        model.train()
        with torch.no_grad():
            model(torch.zeros(1, state_dim))
    finally:
        for hook in hooks:
            hook.remove()
    if hasattr(model.encoder, "position"):
        position = model.encoder.position
        insert_at = next((i + 1 for i, layer in enumerate(layers)
                          if layer["name"] == "encoder.input_projection"), 0)
        layers.insert(insert_at, {"name": "encoder.position", "type": "PositionalParameter",
                                  "input_shape": list(position.shape[1:]),
                                  "output_shape": list(position.shape[1:]),
                                  "parameters": position.numel()})
    return layers
