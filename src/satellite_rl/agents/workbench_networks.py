"""Configurable workbench networks. These are new variants, not legacy weights."""

from __future__ import annotations

import torch
from torch import nn
from satellite_rl.state_layouts import natural_dim, natural_shapes, natural_token_count


def _activation(name):
    return {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh}[name]


class Encoder(nn.Module):
    def __init__(self, config, state_dim):
        super().__init__()
        spec = config["network"]
        kind = spec["kind"]
        self.kind = kind
        self.state_dim = state_dim
        self.transformer_layout = spec.get("transformer_layout", "scalar_tokens")
        self.previous_actual_action = "previous_actual_action" in config.get("state", {}).get("features", [])
        self.graph_mode = spec.get("graph_mode", "none")
        if self.graph_mode != "none":
            from satellite_rl.agents.graph_encoder import GraphEncoder
            self.graph = GraphEncoder(config, state_dim, self.graph_mode)
            self.output_dim = self.graph.output_dim
            self.transformer_layout = "graph_" + self.graph_mode
            return
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
            if self.transformer_layout == "scalar_tokens":
                self.input_projection = nn.Linear(1, embed)
                self.position = nn.Parameter(torch.zeros(1, state_dim, embed))
            elif self.transformer_layout in ("tsac_13_tokens", "tsac_14_context", "tsac_32_context", "tsac_natural_context"):
                expected_dim = {"tsac_13_tokens": 204, "tsac_14_context": 207,
                                "tsac_32_context": 2946}.get(self.transformer_layout)
                if self.transformer_layout == "tsac_natural_context":
                    self.natural_level = config["state"]["context_level"]
                    self.natural_shapes = natural_shapes(self.natural_level)
                    expected_dim = natural_dim(self.natural_level) + (4 if self.previous_actual_action else 0)
                if state_dim != expected_dim:
                    raise ValueError(f"T-SAC {self.transformer_layout} layout requires the ordered {expected_dim}-dimensional state")
                self.geo_embed = nn.Linear(2, embed)
                self.rate_embed = nn.Linear(1, embed)
                self.power_embed = nn.Linear(1, embed)
                self.slot_embed = nn.Linear(20, embed)
                self.pos_embed = nn.Parameter(torch.zeros(1, 13, embed))
            else:
                raise ValueError(f"Unsupported transformer layout: {self.transformer_layout}")
            layer = nn.TransformerEncoderLayer(embed, spec["num_heads"],
                                               spec["feedforward_dim"], dropout,
                                               activation=torch.tanh if spec["activation"] == "tanh"
                                               else spec["activation"],
                                               batch_first=self.transformer_layout == "scalar_tokens")
            self.net = nn.TransformerEncoder(layer, spec["num_layers"], enable_nested_tensor=False)
            if self.transformer_layout == "tsac_14_context":
                # Keep the 13 original token and encoder initializations in their
                # original RNG order. The extra projection cannot shift the heads.
                with torch.random.fork_rng(devices=[]):
                    self.context_embed = nn.Linear(3, embed)
                self.context_pos_embed = nn.Parameter(torch.zeros(1, 1, embed))
            elif self.transformer_layout == "tsac_32_context":
                # Four information levels share exactly the same parameters.
                # Keep the original 13 token and encoder initialization order.
                with torch.random.fork_rng(devices=[]):
                    self.global_embed = nn.Linear(10, embed)
                    self.beam_embed = nn.Linear(110, embed)
                    self.resource_embed = nn.Linear(204, embed)
                self.extra_pos_embed = nn.Parameter(torch.zeros(1, 19, embed))
            elif self.transformer_layout == "tsac_natural_context":
                # Preserve the common 13-token and encoder initialization. Only
                # actual visible fields receive a projection and position.
                with torch.random.fork_rng(devices=[]):
                    if "natural_global_context" in self.natural_shapes:
                        self.global_embed = nn.Linear(self.natural_shapes["natural_global_context"][0], embed)
                    if "natural_beam_context" in self.natural_shapes:
                        self.beam_embed = nn.Linear(10 * self.natural_shapes["natural_beam_context"][-1], embed)
                    if "natural_resource_context" in self.natural_shapes:
                        self.resource_embed = nn.Linear(self.natural_shapes["natural_resource_context"][-1], embed)
                extra = natural_token_count(self.natural_level) - 13
                if extra:
                    self.extra_pos_embed = nn.Parameter(torch.zeros(1, extra, embed))
                if self.previous_actual_action:
                    with torch.random.fork_rng(devices=[]):
                        self.previous_action_embed = nn.Linear(4, embed)
                    self.previous_action_pos = nn.Parameter(torch.zeros(1, 1, embed))
            self.output_dim = embed
        else:
            raise ValueError(f"Unsupported network: {kind}")

    def forward(self, states):
        if states.ndim != 2 or states.shape[1] != self.state_dim:
            raise ValueError(f"State shape must be [batch,{self.state_dim}], got {tuple(states.shape)}")
        if self.graph_mode != "none":
            return self.graph(states)
        if self.kind == "mlp":
            return self.net(states)
        if self.kind == "cnn":
            return self.net(states.unsqueeze(1))
        if self.transformer_layout == "scalar_tokens":
            return self.net(self.input_projection(states.unsqueeze(-1)) + self.position).mean(1)
        batch = states.shape[0]
        row = states[:, 4:104].reshape(batch, 10, 10)
        occupancy = states[:, 104:204].reshape(batch, 10, 10)
        slot_features = torch.cat((row, occupancy), dim=-1)
        tokens = torch.cat((
            self.geo_embed(states[:, :2]).unsqueeze(1),
            self.rate_embed(states[:, 2:3]).unsqueeze(1),
            self.power_embed(states[:, 3:4]).unsqueeze(1),
            self.slot_embed(slot_features),
        ), dim=1) + self.pos_embed
        if self.transformer_layout == "tsac_14_context":
            context = self.context_embed(states[:, 204:207]).unsqueeze(1) + self.context_pos_embed
            tokens = torch.cat((tokens, context), dim=1)
        elif self.transformer_layout == "tsac_32_context":
            context = torch.cat((
                self.global_embed(states[:, 204:214]).unsqueeze(1),
                self.beam_embed(states[:, 214:1314].reshape(batch, 10, 110)),
                self.resource_embed(states[:, 1314:2946].reshape(batch, 8, 204)),
            ), dim=1) + self.extra_pos_embed
            tokens = torch.cat((tokens, context), dim=1)
        elif self.transformer_layout == "tsac_natural_context":
            offset = 204
            parts = []
            if "natural_global_context" in self.natural_shapes:
                width = self.natural_shapes["natural_global_context"][0]
                parts.append(self.global_embed(states[:, offset:offset + width]).unsqueeze(1))
                offset += width
            if "natural_beam_context" in self.natural_shapes:
                width = 100 * self.natural_shapes["natural_beam_context"][-1]
                parts.append(self.beam_embed(states[:, offset:offset + width].reshape(batch, 10, width // 10)))
                offset += width
            if "natural_resource_context" in self.natural_shapes:
                width = 8 * self.natural_shapes["natural_resource_context"][-1]
                parts.append(self.resource_embed(states[:, offset:offset + width].reshape(batch, 8, width // 8)))
                offset += width
            if parts:
                tokens = torch.cat((tokens, torch.cat(parts, dim=1) + self.extra_pos_embed), dim=1)
            if self.previous_actual_action:
                previous = (self.previous_action_embed(states[:, -4:]).unsqueeze(1) +
                            self.previous_action_pos)
                tokens = torch.cat((tokens, previous), dim=1)
        encoded = self.net(tokens.permute(1, 0, 2)).permute(1, 0, 2)
        return encoded[:, 0]


class MultiHeadNetwork(nn.Module):
    def __init__(self, config, state_dim, action_dims):
        super().__init__()
        self.encoder = Encoder(config, state_dim)
        self.heads = nn.ModuleList([nn.Linear(self.encoder.output_dim, int(dim)) for dim in action_dims])

    def forward(self, states):
        x = self.encoder(states)
        return tuple(head(x) for head in self.heads)


class TemporalEncoder(nn.Module):
    """Encode each visible resource state, then only its finite causal history."""

    def __init__(self, config, state_dim):
        super().__init__()
        spec = config["network"]
        self.mode = spec["temporal_mode"]
        self.window = int(spec["temporal_window"])
        self.spatial = Encoder(config, state_dim)
        width = self.spatial.output_dim
        with torch.random.fork_rng(devices=[]):
            if self.mode == "gru":
                self.memory = nn.GRU(width, width, num_layers=int(spec["num_layers"]),
                                     batch_first=True)
            elif self.mode == "causal_transformer":
                layer = nn.TransformerEncoderLayer(
                    width, spec["num_heads"], spec["feedforward_dim"],
                    float(spec["dropout"]), batch_first=True,
                    activation=torch.tanh if spec["activation"] == "tanh" else spec["activation"])
                self.memory = nn.TransformerEncoder(
                    layer, spec["num_layers"], enable_nested_tensor=False)
                self.time_position = nn.Parameter(torch.zeros(1, self.window, width))
            else:
                raise ValueError("TemporalEncoder requires gru or causal_transformer")
        self.output_dim = width

    def forward(self, sequence, lengths):
        if (sequence.ndim != 3 or sequence.shape[1] > self.window or
                lengths.shape != (sequence.shape[0],) or
                not bool(((lengths >= 1) & (lengths <= sequence.shape[1])).all())):
            raise ValueError("Temporal input must have valid [batch,window,state] and lengths")
        batch, steps, width = sequence.shape
        encoded = self.spatial(sequence.reshape(batch * steps, width)).reshape(batch, steps, -1)
        if self.mode == "gru":
            packed = nn.utils.rnn.pack_padded_sequence(
                encoded, lengths.detach().cpu(), batch_first=True, enforce_sorted=False)
            _, hidden = self.memory(packed)
            return hidden[-1]
        positions = self.time_position[:, :steps]
        causal = torch.ones((steps, steps), device=sequence.device, dtype=torch.bool).triu(1)
        padding = torch.arange(steps, device=sequence.device)[None, :] >= lengths[:, None]
        contextual = self.memory(encoded + positions, mask=causal,
                                 src_key_padding_mask=padding)
        return contextual[torch.arange(batch, device=sequence.device), lengths - 1]


class TemporalMultiHeadNetwork(nn.Module):
    def __init__(self, config, state_dim, action_dims):
        super().__init__()
        self.encoder = TemporalEncoder(config, state_dim)
        self.heads = nn.ModuleList([nn.Linear(self.encoder.output_dim, int(dim))
                                    for dim in action_dims])

    def forward(self, input_with_lengths):
        states, lengths = input_with_lengths
        encoded = self.encoder(states, lengths)
        return tuple(head(encoded) for head in self.heads)


class ConditionalPolicyNetwork(nn.Module):
    """Full frequency/length logits and power logits for every pair."""

    def __init__(self, config, state_dim, action_dims):
        super().__init__()
        self.action_dims = tuple(action_dims)
        self.encoder = Encoder(config, state_dim)
        self.frequency_slots = nn.Linear(self.encoder.output_dim, action_dims[0] * action_dims[1])
        self.conditional_power = nn.Linear(self.encoder.output_dim,
                                           action_dims[0] * action_dims[1] * action_dims[2])

    def forward(self, states):
        encoded = self.encoder(states)
        batch = states.shape[0]
        return (self.frequency_slots(encoded).reshape(batch, *self.action_dims[:2]),
                self.conditional_power(encoded).reshape(batch, *self.action_dims))


class JointQNetwork(nn.Module):
    """One learned value per complete frequency/length/power action."""

    def __init__(self, config, state_dim, action_dims):
        super().__init__()
        self.action_dims = tuple(action_dims)
        self.encoder = Encoder(config, state_dim)
        self.joint_q = nn.Linear(self.encoder.output_dim,
                                 action_dims[0] * action_dims[1] * action_dims[2])

    def forward(self, states):
        return self.joint_q(self.encoder(states)).reshape(states.shape[0], *self.action_dims)


class ScalarNetwork(nn.Module):
    def __init__(self, config, state_dim):
        super().__init__()
        self.encoder = Encoder(config, state_dim)
        self.head = nn.Linear(self.encoder.output_dim, 1)

    def forward(self, states):
        return self.head(self.encoder(states)).squeeze(-1)


def _model_description(config, state_dim, action_dims):
    """Trace the actual constructed layers and tensor shapes for the UI."""
    algorithm = config["algorithm"]
    policy_structure = config["network"].get("sac_policy_structure", "factorized")
    critic_structure = config["network"].get("sac_critic_structure", "additive")
    temporal_mode = config["network"].get("temporal_mode", "none")
    policy = (TemporalMultiHeadNetwork(config, state_dim, action_dims)
              if algorithm == "sac" and temporal_mode in ("gru", "causal_transformer")
              else ConditionalPolicyNetwork(config, state_dim, action_dims)
              if algorithm == "sac" and policy_structure == "conditional"
              else MultiHeadNetwork(config, state_dim, action_dims))
    actor_count = sum(p.numel() for p in policy.parameters())
    count = actor_count
    policy_layers = _trace_layers(policy, state_dim)
    if algorithm == "sac":
        critic = (TemporalMultiHeadNetwork(config, state_dim, action_dims)
                  if temporal_mode in ("gru", "causal_transformer") else
                  JointQNetwork(config, state_dim, action_dims)
                  if critic_structure == "joint" else MultiHeadNetwork(config, state_dim, action_dims))
        critic_count = sum(p.numel() for p in critic.parameters())
        critic_layers = _trace_layers(critic, state_dim)
        count = actor_count + 4 * critic_count + 1
        specifications = [("策略", policy_layers, actor_count)] + [
            (name, critic_layers, critic_count) for name in
            ("价值网络1", "价值网络2", "目标价值网络1", "目标价值网络2")]
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
    layout = config["network"].get("transformer_layout", "scalar_tokens")
    variant = f"workbench_{algorithm}_{config['network']['kind']}_v1"
    if config["network"]["kind"] == "transformer" and layout == "tsac_13_tokens":
        variant = f"workbench_{algorithm}_transformer_tsac_13_tokens_v1"
    elif config["network"]["kind"] == "transformer" and layout == "tsac_14_context":
        variant = f"workbench_{algorithm}_transformer_tsac_14_context_v1"
    elif config["network"]["kind"] == "transformer" and layout == "tsac_32_context":
        variant = f"workbench_{algorithm}_transformer_tsac_32_context_v1"
    elif config["network"]["kind"] == "transformer" and layout == "tsac_natural_context":
        variant = f"workbench_{algorithm}_transformer_tsac_natural_context_{config['state']['context_level']}_v1"
    if algorithm == "sac" and (policy_structure != "factorized" or critic_structure != "additive"):
        variant += f"_{policy_structure}_{critic_structure}"
    if temporal_mode != "none":
        variant += f"_temporal_{temporal_mode}_w{config['network'].get('temporal_window', 16)}"
    graph_mode = config["network"].get("graph_mode", "none")
    if graph_mode != "none":
        variant += f"_graph_{graph_mode}"
    return {"graph": nodes, "nodes": nodes, "edges": edges, "branches": branches,
            "parameter_count": count,
            "variant": variant}


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

    spatial = (model.encoder.spatial if isinstance(model, TemporalMultiHeadNetwork)
               else model.encoder)
    spatial_prefix = "encoder.spatial." if isinstance(model, TemporalMultiHeadNetwork) else "encoder."
    layout = getattr(spatial, "transformer_layout", None)
    tsac_layout = layout in ("tsac_13_tokens", "tsac_14_context", "tsac_32_context", "tsac_natural_context")
    token_count = (natural_token_count(spatial.natural_level) +
                   (1 if spatial.previous_actual_action else 0)
                   if layout == "tsac_natural_context"
                   else {"tsac_14_context": 14, "tsac_32_context": 32}.get(layout, 13))

    def shape(value, layer_name):
        if isinstance(value, torch.Tensor):
            if tsac_layout and layer_name.startswith(spatial_prefix + "net.") and value.ndim == 3:
                # T-SAC's encoder is sequence first: [tokens, batch, channels].
                return [value.shape[0], value.shape[2]]
            return list(value.shape[1:])
        if isinstance(value, (tuple, list)):
            return shape(value[0], layer_name)
        return None

    for name, module in model.named_modules():
        if name and isinstance(module, selected):
            def record(inner, inputs, output, layer_name=name):
                layers.append({"name": layer_name, "type": type(inner).__name__,
                               "input_shape": shape(inputs, layer_name),
                               "output_shape": shape(output, layer_name),
                               "parameters": sum(p.numel() for p in inner.parameters(recurse=False))})
            hooks.append(module.register_forward_hook(record))
    try:
        # Training mode prevents PyTorch's fused Transformer inference fast path
        # from bypassing attention/feedforward module hooks in the diagram.
        model.train()
        with torch.no_grad():
            if isinstance(model, TemporalMultiHeadNetwork):
                model((torch.zeros(1, model.encoder.window, state_dim),
                       torch.as_tensor([model.encoder.window], dtype=torch.long)))
            else:
                model(torch.zeros(1, state_dim))
    finally:
        for hook in hooks:
            hook.remove()
    if hasattr(spatial, "position"):
        position = spatial.position
        insert_at = next((i + 1 for i, layer in enumerate(layers)
                          if layer["name"] == spatial_prefix + "input_projection"), 0)
        layers.insert(insert_at, {"name": spatial_prefix + "position", "type": "PositionalParameter",
                                  "input_shape": list(position.shape[1:]),
                                  "output_shape": list(position.shape[1:]),
                                  "parameters": position.numel()})
    if hasattr(spatial, "pos_embed"):
        embed = spatial.pos_embed.shape[-1]
        before_net = next((i for i, layer in enumerate(layers)
                           if layer["name"].startswith(spatial_prefix + "net.")), len(layers))
        layers.insert(before_net, {"name": spatial_prefix + "pos_embed", "type": "PositionalParameter",
                                   "input_shape": [13, embed], "output_shape": [13, embed],
                                   "parameters": spatial.pos_embed.numel()})
        if layout == "tsac_14_context":
            layers.insert(before_net + 1, {"name": spatial_prefix + "context_pos_embed", "type": "PositionalParameter",
                                           "input_shape": [1, embed], "output_shape": [1, embed],
                                           "parameters": spatial.context_pos_embed.numel()})
            before_net += 1
        elif hasattr(spatial, "extra_pos_embed"):
            layers.insert(before_net + 1, {"name": spatial_prefix + "extra_pos_embed", "type": "PositionalParameter",
                                           "input_shape": list(spatial.extra_pos_embed.shape[1:]),
                                           "output_shape": list(spatial.extra_pos_embed.shape[1:]),
                                           "parameters": spatial.extra_pos_embed.numel()})
            before_net += 1
        if hasattr(spatial, "previous_action_pos"):
            layers.insert(before_net + 1, {"name": spatial_prefix + "previous_action_pos", "type": "PositionalParameter",
                                           "input_shape": [1, embed], "output_shape": [1, embed],
                                           "parameters": spatial.previous_action_pos.numel()})
            before_net += 1
        layers.insert(before_net + 1, {"name": spatial_prefix + "tokens", "type": "TokenAssembly",
                                       "input_shape": [state_dim], "output_shape": [token_count, embed],
                                       "parameters": 0})
        after_net = next((i + 1 for i, layer in reversed(list(enumerate(layers)))
                          if layer["name"].startswith(spatial_prefix + "net.")), before_net + 2)
        layers.insert(after_net, {"name": spatial_prefix + "first_token", "type": "FirstTokenPool",
                                  "input_shape": [token_count, embed], "output_shape": [embed],
                                  "parameters": 0})
    if getattr(spatial, "graph_mode", "none") != "none":
        graph = spatial.graph
        embed = graph.output_dim
        before_attention = next((i for i, layer in enumerate(layers)
                                 if layer["name"].startswith("encoder.graph.layers.")), len(layers))
        extras = []
        for name in ("base_position", "global_position", "resource_position", "edge_weight"):
            if hasattr(graph, name):
                parameter = getattr(graph, name)
                extras.append({"name": "encoder.graph." + name, "type": "PositionalParameter"
                               if name != "edge_weight" else "PhysicalEdgeParameter",
                               "input_shape": list(parameter.shape),
                               "output_shape": list(parameter.shape),
                               "parameters": parameter.numel()})
        extras.append({"name": "encoder.graph.tokens", "type": "BeamTokenAssembly",
                       "input_shape": [state_dim], "output_shape": [122, embed],
                       "parameters": 0})
        layers[before_attention:before_attention] = extras
    if isinstance(model, TemporalMultiHeadNetwork):
        width = model.encoder.output_dim
        position = next((index for index, layer in enumerate(layers)
                         if layer["name"].startswith("heads.")), len(layers))
        if model.encoder.mode == "gru":
            layers.insert(position, {"name": "encoder.temporal_gru", "type": "GRU",
                                     "input_shape": [model.encoder.window, width],
                                     "output_shape": [width],
                                     "parameters": sum(p.numel() for p in model.encoder.memory.parameters())})
        else:
            layers.insert(position, {"name": "encoder.time_position", "type": "PositionalParameter",
                                     "input_shape": [model.encoder.window, width],
                                     "output_shape": [model.encoder.window, width],
                                     "parameters": model.encoder.time_position.numel()})
            layers.insert(position + 1, {"name": "encoder.temporal_causal_mask", "type": "CausalWindow",
                                         "input_shape": [model.encoder.window, width],
                                         "output_shape": [width], "parameters": 0})
    return layers
