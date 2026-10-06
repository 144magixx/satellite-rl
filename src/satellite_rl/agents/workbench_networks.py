"""Configurable workbench networks. These are new variants, not legacy weights."""

from __future__ import annotations

import torch
from torch import nn
from satellite_rl.state_layouts import (S2P18_VARIABLE_FEATURES, natural_dim, natural_shapes,
                                        natural_token_count)


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
        # 逐束实体编码的只读标记：只有自然布局的 actual 档会置真，其余一律为假，
        # 这样 forward 的统一分支可以直接查询而无需按 kind 分支判断。
        self.actual_count = False
        self.beam_rows = 0
        self.max_beam_count = 0
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
            self.readout_mode = spec.get("readout_mode", "first_token")
            self.beam_count_conditioning = spec.get("beam_count_conditioning", "none")
            self.context_beam_count = spec.get("context_beam_count", "fixed_100")
            if self.readout_mode not in ("first_token", "attention_pooling"):
                raise ValueError(f"Unsupported readout mode: {self.readout_mode}")
            if self.beam_count_conditioning not in ("none", "film"):
                raise ValueError(f"Unsupported beam-count conditioning: {self.beam_count_conditioning}")
            if self.context_beam_count not in ("fixed_100", "actual"):
                raise ValueError(f"Unsupported context beam count mode: {self.context_beam_count}")
            if self.transformer_layout == "scalar_tokens":
                self.input_projection = nn.Linear(1, embed)
                self.position = nn.Parameter(torch.zeros(1, state_dim, embed))
            elif self.transformer_layout in ("tsac_13_tokens", "tsac_14_context", "tsac_14_variable_context", "tsac_32_context", "tsac_natural_context"):
                expected_dim = {"tsac_13_tokens": 204, "tsac_14_context": 207,
                                "tsac_14_variable_context": (
                                    213 if config.get("state", {}).get("features")
                                    == list(S2P18_VARIABLE_FEATURES) else 210),
                                "tsac_32_context": 2946}.get(self.transformer_layout)
                if self.transformer_layout == "tsac_natural_context":
                    self.natural_level = config["state"]["context_level"]
                    self.natural_shapes = natural_shapes(self.natural_level)
                    expected_dim = natural_dim(self.natural_level) + (
                        4 if self.previous_actual_action else 0)
                    self.actual_count = (self.context_beam_count == "actual"
                                         and "natural_beam_context" in self.natural_shapes)
                    if self.actual_count:
                        # 逐束实体编码（S2-P18 手段G）：逐束字段只含真实实体，观测宽度
                        # 随场景实际束数 N 变化，不填充到 100；配置期的状态维数上界用
                        # environment.beam_count_max，实际维数由场景束数决定。
                        self.max_beam_count = int(config["environment"].get("beam_count_max", 100))
                        rows = self.max_beam_count
                        expected_dim += ((rows - 100)
                                         * self.natural_shapes["natural_beam_context"][-1])
                    else:
                        rows = self.natural_shapes.get("natural_beam_context", [10, 10, 5])[0]
                        self.max_beam_count = 100
                else:
                    self.actual_count = False
                    rows = 10
                    self.beam_rows = 100
                    self.max_beam_count = 100
                if state_dim != expected_dim:
                    raise ValueError(f"T-SAC {self.transformer_layout} layout requires the ordered {expected_dim}-dimensional state")
                self.beam_rows = rows
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
            if self.transformer_layout in ("tsac_14_context", "tsac_14_variable_context"):
                # Keep the 13 original token and encoder initializations in their
                # original RNG order. The extra projection cannot shift the heads.
                with torch.random.fork_rng(devices=[]):
                    self.context_embed = nn.Linear(6 if self.transformer_layout == "tsac_14_variable_context" else 3, embed)
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
                        if self.actual_count:
                            # 逐束实体编码：每个实体行共享一个投影，参数形状只与“每实体
                            # 通道数”有关，行数不进参数，因此同一份权重可用于任意 N。
                            self.beam_embed = nn.Linear(
                                self.natural_shapes["natural_beam_context"][-1], embed)
                            self.beam_embed_features = self.natural_shapes["natural_beam_context"][-1]
                        else:
                            # 固定 100 束档位：状态里逐束块是 (100, 每段通道数)，
                            # 拼接后按展平宽度做一次投影（与历史行为一致）。
                            self.beam_embed_features = 100 * self.natural_shapes["natural_beam_context"][-1]
                            self.beam_embed = nn.Linear(self.beam_embed_features, embed)
                    if "natural_resource_context" in self.natural_shapes:
                        self.resource_embed = nn.Linear(self.natural_shapes["natural_resource_context"][-1], embed)
                if self.actual_count:
                    # 位置参数按束数上界一次分配；每个场景只用前 N 行，不因 N 大小
                    # 改变参数形状，所以同一份权重可用于任意 N 而无需重新初始化。
                    pos_slots = (self.natural_shapes["natural_beam_context"][-1] * self.max_beam_count
                                 + self.natural_shapes["natural_resource_context"][0]
                                 if "natural_resource_context" in self.natural_shapes
                                 else self.natural_shapes["natural_beam_context"][-1] * self.max_beam_count)
                else:
                    pos_slots = natural_token_count(self.natural_level) - 13
                self.pos_slots = pos_slots
                if pos_slots:
                    self.extra_pos_embed = nn.Parameter(torch.zeros(1, pos_slots, embed))
                if self.actual_count and self.readout_mode == "attention_pooling":
                    # 注意力池化读出：一个可学习的可加性打分向量（无额外参数形状），
                    # 对逐束实体令牌做掩码加权平均。该参数只在池化模式下创建，
                    # 因此 first_token 模式的参数与初始化顺序逐位不变。
                    with torch.random.fork_rng(devices=[]):
                        self.pool_query = nn.Linear(embed, 1)
                if self.beam_count_conditioning == "film":
                    # N 条件调制（FiLM）：把场景实际束数作为条件，对读出结果做
                    # 逐通道缩放与平移，使同一套权重按规模调整内部表示。
                    with torch.random.fork_rng(devices=[]):
                        self.film = nn.Linear(1, 2 * embed)
                        nn.init.zeros_(self.film.weight)
                        nn.init.zeros_(self.film.bias)
                if self.previous_actual_action:
                    with torch.random.fork_rng(devices=[]):
                        self.previous_action_embed = nn.Linear(4, embed)
                    self.previous_action_pos = nn.Parameter(torch.zeros(1, 1, embed))
            self.output_dim = embed
        else:
            raise ValueError(f"Unsupported network: {kind}")

    def _resolve_lengths(self, states, lengths):
        """逐束实体编码的“有效令牌长度”与单样本实际状态宽度。

        返回 ``(lengths, actual_width)``：``lengths`` 是长度张量，``actual_width`` 是
        单个样本未填充时的状态宽度。``lengths`` 为整数时表示单样本（不填充）。
        """
        if lengths is None:
            return None, self.state_dim
        if torch.is_tensor(lengths):
            return lengths.reshape(-1), int(lengths.max().item())
        return torch.as_tensor([int(lengths)], device=states.device), int(lengths)

    def _beam_count(self, states, actual_width):
        """从状态里取出“当前是第几束”（即处理序号 idx+1）。

        逐束实体的第 5 个通道是 ``i/(N−1)``；已处理束在实体里是连续前缀
        （layout 档第 5 通道、full 档第 6 通道为 1），它们的个数就是当前处理序号
        idx。行数是补零后的固定行数，真实束数由 ``lengths`` 给出，因此这里同时
        接受长度张量（批）与整数（单样本）。该标量同时用于 N 条件调制与规模统计。
        """
        if not self.actual_count:
            return None
        width = self.natural_shapes["natural_beam_context"][-1]
        served_channel = 5 if width == 11 else 4
        offset = 204
        if "natural_global_context" in self.natural_shapes:
            offset += self.natural_shapes["natural_global_context"][0]
        trailing = 4 * self.previous_actual_action
        if "natural_resource_context" in self.natural_shapes:
            trailing += 8 * self.natural_shapes["natural_resource_context"][0]
        rows = (states.shape[1] - offset - trailing) // width
        if torch.is_tensor(actual_width):
            count = actual_width.reshape(-1).to(states.device)
        elif isinstance(actual_width, int) and actual_width < rows:
            count = torch.full((states.shape[0],), int(actual_width),
                               device=states.device, dtype=torch.long)
        else:
            count = torch.full((states.shape[0],), rows, device=states.device, dtype=torch.long)
        served = states[:, offset + served_channel:offset + rows * width:width] > 0.5
        index = served.to(torch.int64).sum(dim=1)
        position = torch.round(
            states[:, offset + 4:offset + rows * width:width]
            * torch.clamp(count - 1, min=1).reshape(-1, 1).to(states.dtype)).long()
        gathered = position.gather(1, index.clamp(max=rows - 1).reshape(-1, 1)).reshape(-1)
        return (gathered.to(states.dtype) + 1.0).clamp(min=1.0, max=float(self.max_beam_count))

    def forward(self, states, lengths=None):
        if states.ndim != 2:
            raise ValueError(f"State must be [batch,width], got {tuple(states.shape)}")
        resolved, actual_width = self._resolve_lengths(states, lengths)
        if self.actual_count:
            # 逐束实体档位：观测宽度固定为「按场景束数上限补零」的宽度，而
            # ``lengths`` 给出的是每个样本的真实束数，两者不是同一个量。
            if states.shape[1] != self.state_dim:
                raise ValueError(f"State shape must be [batch,{self.state_dim}], got {tuple(states.shape)}")
        else:
            if actual_width != states.shape[1]:
                raise ValueError(f"State width {states.shape[1]} differs from declared {actual_width}")
            if states.shape[1] != self.state_dim:
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
        padding_mask = None
        entity_start = 13
        entity_count = 0
        if self.transformer_layout in ("tsac_14_context", "tsac_14_variable_context"):
            context_width = 6 if self.transformer_layout == "tsac_14_variable_context" else 3
            context = self.context_embed(states[:, 204:204 + context_width]).unsqueeze(1) + self.context_pos_embed
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
            extra_count = 0
            entity_tokens = None
            if "natural_global_context" in self.natural_shapes:
                width = self.natural_shapes["natural_global_context"][0]
                parts.append(self.global_embed(states[:, offset:offset + width]).unsqueeze(1))
                offset += width
                extra_count += 1
            if "natural_beam_context" in self.natural_shapes:
                # 观测里的逐束字段是 [实体数, 通道数]：通道数取本档声明宽度，
                # 行数在逐束编码档位下由状态宽度反推（不填充到 100）。
                width = self.natural_shapes["natural_beam_context"][-1]
                trailing = 4 * self.previous_actual_action
                if "natural_resource_context" in self.natural_shapes:
                    resource = self.natural_shapes["natural_resource_context"]
                    trailing += resource[0] * resource[-1]
                available = states.shape[1] - offset - trailing
                if available < 0:
                    raise ValueError("逐束字段宽度与状态维数不一致")
                if self.actual_count:
                    # 逐束实体编码档位：行数由状态宽度反推，先按“每实体通道数”
                    # 切成单实体令牌，再对每个实体做同一个投影。
                    rows, remainder = divmod(available, width)
                    if remainder:
                        raise ValueError("逐束字段宽度与状态维数不一致")
                    if not 1 <= rows <= self.max_beam_count:
                        raise ValueError(f"实际束数 {rows} 超出 1–{self.max_beam_count}")
                    entity_count = rows
                    block = states[:, offset:offset + rows * width].reshape(batch, rows, width)
                    entity_tokens = self.beam_embed(block)
                else:
                    # 固定档位：逐束信息按展平块整体投影（与历史实现一致）。
                    if available != self.beam_embed_features:
                        raise ValueError("逐束字段宽度与状态维数不一致")
                    entity_count = 10
                    block = states[:, offset:offset + available]
                    parts.append(self.beam_embed(block).unsqueeze(1))
                    extra_count += 1
                offset += available
            if "natural_resource_context" in self.natural_shapes:
                width = 8 * self.natural_shapes["natural_resource_context"][-1]
                parts.append(self.resource_embed(states[:, offset:offset + width].reshape(batch, 8, width // 8)))
                offset += width
                extra_count += 8
            if parts:
                extra = torch.cat(parts, dim=1) + self.extra_pos_embed[:, :extra_count]
                tokens = torch.cat((tokens, extra), dim=1)
            if self.actual_count and entity_tokens is not None:
                # 合并顺序：13 个基础令牌 + 1 个全局 + N 个逐束实体 + 8 个资源组。
                # 实体块不带位置参数：位置参数会把实体顺序写进注意力结果，而逐束
                # 实体本身无序，加了会破坏置换一致性。
                entity_start = tokens.shape[1]
                tokens = torch.cat((tokens, entity_tokens), dim=1)
            if self.previous_actual_action:
                previous = (self.previous_action_embed(states[:, -4:]).unsqueeze(1) +
                            self.previous_action_pos)
                tokens = torch.cat((tokens, previous), dim=1)
            if self.actual_count:
                # 单样本调用（无批次填充）时所有实体都有效：等价于“每样本有效令牌长度
                # 等于实体数”。批量采样时由调用方传入真实长度，用于屏蔽批内填充。
                lengths = (torch.full((states.shape[0],), entity_count, device=states.device,
                                      dtype=torch.long) if resolved is None
                           else (resolved if torch.is_tensor(resolved) else resolved.reshape(-1)))
                entity_positions = torch.arange(tokens.shape[1], device=states.device)[None, :]
                padding_mask = ~((entity_positions >= entity_start)
                                 & (entity_positions < entity_start + entity_count)
                                 & (entity_positions - entity_start < lengths[:, None]))
                # 用零替换被屏蔽的令牌，避免位置参数读出非有限值。
                tokens = tokens.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        sequence = tokens.permute(1, 0, 2)
        # 上一次前向实际喂给骨干的令牌数与实体数（只读诊断，供前端/测试核对，
        # 不影响任何计算）。
        self.last_token_count = int(sequence.shape[0])
        self.last_entity_count = int(entity_count)
        if padding_mask is None:
            encoded = self.net(sequence)
        else:
            # 直接调用底层 forward：某些 PyTorch 版本在注册了 forward pre-hook
            # （例如结构追踪）时无法用关键字传 src_key_padding_mask，这里绕开
            # hook 派发，行为与 TransformerEncoder.forward 完全一致。
            encoded = type(self.net).forward(
                self.net, sequence, None, padding_mask.to(torch.bool))
        encoded = encoded.permute(1, 0, 2)
        result = encoded[:, 0]
        if self.transformer_layout == "tsac_natural_context" and self.actual_count and entity_count:
            if self.readout_mode == "attention_pooling":
                # 注意力池化读出：对“逐束实体令牌”做掩码加权平均，权重由每个实体自身
                # 的可学习打分决定（本档每实体 5 个通道）。因为打分逐实体独立、聚合是
                # 对实体的求和，读出对实体顺序严格不变（等变性由构造保证）；公开的
                # 基础令牌不参与池化，只由 first_token 通道提供全局上下文。
                entity = tokens[:, entity_start:entity_start + entity_count]
                query = self.pool_query.weight.reshape(-1)
                scale = float(query.shape[0] * max(entity_count, 1)) ** 0.5
                scores = (entity * query).sum(-1) / scale
                valid = ((torch.arange(entity_count, device=states.device)[None, :]
                          < lengths[:, None]) if padding_mask is not None
                         else torch.ones(1, entity_count, dtype=torch.bool, device=states.device))
                scores = scores.masked_fill(~valid, float("-inf"))
                weights = scores.softmax(dim=-1)
                if not bool(torch.isfinite(weights).all()):
                    raise ValueError("注意力池化的有效实体为空，无法计算读出")
                result = torch.einsum("bn,bnd->bd", weights, entity)
            if self.beam_count_conditioning == "film":
                count = self._beam_count(states, resolved if resolved is not None else actual_width).reshape(-1, 1)
                scale, shift = self.film(count / float(self.max_beam_count)).chunk(2, dim=-1)
                result = result * (1.0 + scale) + shift
        return result


class MultiHeadNetwork(nn.Module):
    def __init__(self, config, state_dim, action_dims):
        super().__init__()
        self.encoder = Encoder(config, state_dim)
        self.heads = nn.ModuleList([nn.Linear(self.encoder.output_dim, int(dim)) for dim in action_dims])

    def forward(self, states, lengths=None):
        x = self.encoder(states, lengths=lengths)
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

    def forward(self, states, lengths=None):
        encoded = self.encoder(states, lengths=lengths)
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

    def forward(self, states, lengths=None):
        return (self.joint_q(self.encoder(states, lengths=lengths))
                .reshape(states.shape[0], *self.action_dims))


class ScalarNetwork(nn.Module):
    def __init__(self, config, state_dim):
        super().__init__()
        self.encoder = Encoder(config, state_dim)
        self.head = nn.Linear(self.encoder.output_dim, 1)

    def forward(self, states, lengths=None):
        return self.head(self.encoder(states, lengths=lengths)).squeeze(-1)


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
    elif config["network"]["kind"] == "transformer" and layout == "tsac_14_variable_context":
        variant = f"workbench_{algorithm}_transformer_tsac_14_variable_context_v1"
    elif config["network"]["kind"] == "transformer" and layout == "tsac_32_context":
        variant = f"workbench_{algorithm}_transformer_tsac_32_context_v1"
    elif config["network"]["kind"] == "transformer" and layout == "tsac_natural_context":
        variant = f"workbench_{algorithm}_transformer_tsac_natural_context_{config['state']['context_level']}_v1"
        if config["network"].get("context_beam_count", "fixed_100") == "actual":
            variant += "_actual_beams"
        if config["network"].get("readout_mode", "first_token") != "first_token":
            variant += "_" + config["network"]["readout_mode"]
        if config["network"].get("beam_count_conditioning", "none") != "none":
            variant += "_" + config["network"]["beam_count_conditioning"]
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
    tsac_layout = layout in ("tsac_13_tokens", "tsac_14_context", "tsac_14_variable_context", "tsac_32_context", "tsac_natural_context")
    token_count = (natural_token_count(spatial.natural_level) +
                   (1 if spatial.previous_actual_action else 0) +
                   ((getattr(spatial, "beam_rows", 100) - 100)
                    if getattr(spatial, "actual_count", False) else 0)
                   if layout == "tsac_natural_context"
                   else {"tsac_14_context": 14, "tsac_14_variable_context": 14, "tsac_32_context": 32}.get(layout, 13))

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
        if layout in ("tsac_14_context", "tsac_14_variable_context"):
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
