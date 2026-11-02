"""Candidate-scoring networks for the I0 improvement SAC.

All networks map (state, candidate features) to one score per candidate:
    state    (B, STATE_WIDTH)   = 200 beam rows x 14 features, then 12 global features
    features (B, A, action_w)   = up to 4 changed beams x 5 values, kind one-hot, change count, STOP flag
    output   (B, A)

mlp          the original I0 network: flattened state through an MLP
deepsets     per-beam MLP, masked mean/max pooling (no interaction between beams)
transformer  beam tokens plus one global token through a Transformer encoder

With pointer=True the candidate is also described by the encoded rows of the beams it changes
(the first 5-value block of each change holds beam_index/200 and slot_count/10 > 0).
"""
import torch
from torch import nn
from .environment import ACTION_WIDTH

MAX_BEAMS = 200
BEAM_FEATURES = 14
GLOBAL_FEATURES = 12
MAX_CHANGES = 4


def split_state(state):
    beams = state[:, :MAX_BEAMS * BEAM_FEATURES].reshape(-1, MAX_BEAMS, BEAM_FEATURES)
    return beams, state[:, MAX_BEAMS * BEAM_FEATURES:], beams[:, :, 0] > 0.5


def changed_beams(features):
    """Beam indices (B, A, 4) and presence mask (B, A, 4) of each candidate's changes."""
    blocks = features[:, :, :5 * MAX_CHANGES].reshape(*features.shape[:2], MAX_CHANGES, 5)
    index = torch.round(blocks[..., 0] * MAX_BEAMS).long().clamp(0, MAX_BEAMS - 1)
    return index, blocks[..., 2] > 0


def gather_rows(rows, index, present):
    """Mean of rows (B, N, d) at index (B, A, 4) over present changes -> (B, A, d)."""
    batch, actions, changes = index.shape
    flat = index.reshape(batch, actions * changes, 1).expand(-1, -1, rows.shape[-1])
    picked = rows.gather(1, flat).reshape(batch, actions, changes, -1)
    weight = present.float().unsqueeze(-1)
    return (picked * weight).sum(2) / weight.sum(2).clamp(min=1.)


class CandidateNetwork(nn.Module):
    """Original I0 network (kept identical so S2-FIX L1 weights load unchanged)."""

    def __init__(self, width, action_width=ACTION_WIDTH, state_width=MAX_BEAMS * BEAM_FEATURES + GLOBAL_FEATURES):
        super().__init__()
        self.state = nn.Sequential(nn.Linear(state_width, width), nn.LayerNorm(width), nn.ReLU(),
                                   nn.Linear(width, width), nn.ReLU())
        self.action = nn.Sequential(nn.Linear(action_width, width // 2), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(width + width // 2, width), nn.ReLU(), nn.Linear(width, 1))

    def forward(self, state, features):
        common = self.state(state).unsqueeze(1).expand(-1, features.shape[1], -1)
        return self.head(torch.cat((common, self.action(features)), dim=-1)).squeeze(-1)


class SetCandidateNetwork(nn.Module):
    """DeepSets or Transformer state encoder with an optional pointer to the changed beams."""

    def __init__(self, kind, action_width, dim, layers, heads, ffn_mult, pointer):
        super().__init__()
        self.kind, self.pointer = kind, pointer
        self.beam_embed = nn.Linear(BEAM_FEATURES, dim)
        self.global_embed = nn.Linear(GLOBAL_FEATURES, dim)
        if kind == "transformer":
            layer = nn.TransformerEncoderLayer(dim, heads, ffn_mult * dim, dropout=0.0,
                                               batch_first=True, norm_first=True)
            self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
            self.final_norm = nn.LayerNorm(dim)
        else:
            self.beam_mlp = nn.Sequential(nn.ReLU(), nn.Linear(dim, dim), nn.ReLU())
            self.pool = nn.Sequential(nn.Linear(3 * dim, dim), nn.ReLU())
        self.action = nn.Sequential(nn.Linear(action_width, dim), nn.ReLU())
        parts = 3 if pointer else 2
        self.head = nn.Sequential(nn.Linear(parts * dim, dim), nn.ReLU(), nn.Linear(dim, 1))

    def encode(self, state):
        # 编码部分在 GPU 上用 bf16 混合精度（更新快一倍），输出转回 fp32 再进打分头
        with torch.autocast(device_type=state.device.type, dtype=torch.bfloat16, enabled=state.is_cuda):
            summary, rows = self._encode(state)
        return summary.float(), rows.float()

    def _encode(self, state):
        beams, global_part, mask = split_state(state)
        # 有效波束总在前面；截到本批最长的有效长度，省掉全是填充的位置（输出不变）
        length = int(mask.sum(1).max().clamp(min=1))
        beams, mask = beams[:, :length], mask[:, :length]
        if self.kind == "transformer":
            tokens = torch.cat((self.global_embed(global_part).unsqueeze(1), self.beam_embed(beams)), dim=1)
            padding = torch.cat((torch.zeros_like(mask[:, :1]), ~mask), dim=1)
            out = self.final_norm(self.encoder(tokens, src_key_padding_mask=padding))
            return out[:, 0], out[:, 1:]
        rows = self.beam_mlp(self.beam_embed(beams))
        weight = mask.to(rows.dtype).unsqueeze(-1)
        mean = (rows * weight).sum(1) / weight.sum(1).clamp(min=1.)
        peak = rows.masked_fill(~mask.unsqueeze(-1), -1e4).max(1).values
        summary = self.pool(torch.cat((mean, peak, self.global_embed(global_part)), dim=-1))
        return summary, rows

    def forward(self, state, features):
        summary, rows = self.encode(state)
        parts = [summary.unsqueeze(1).expand(-1, features.shape[1], -1), self.action(features)]
        if self.pointer:
            parts.append(gather_rows(rows, *changed_beams(features)))
        return self.head(torch.cat(parts, dim=-1)).squeeze(-1)


def build_network(config, action_width):
    if config.get("network", "mlp") == "mlp":
        return CandidateNetwork(config["hidden_width"], action_width)
    return SetCandidateNetwork(config["network"], action_width, config["embed_dim"], config["num_layers"],
                               config["num_heads"], config["ffn_mult"], config["pointer"])


def parameter_count(module):
    return sum(p.numel() for p in module.parameters())
