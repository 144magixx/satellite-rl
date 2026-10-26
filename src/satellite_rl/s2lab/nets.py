"""网络：三种编码器（Transformer 块令牌/每束令牌、CNN、MLP）与因子化动作头（分类或指针）。

观测字段：scal[B,10]、intf[B,2,T]、occ[B,G,T]、allow[B,G]，可选 beam[B,N,F]、beam_valid[B,N]。
动作头输出三组 logits/Q：组×起点 [B,G*T]、槽数 [B,10]、功率档 [B,K]。
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

BLOCK = 10
NEG = -1e4


def block_features(obs):
    """[B, G*nb, 29]：每个 10 槽块的干扰（本块极化）、占用、组 one-hot、是否允许组。"""
    occ = obs['occ']
    B, G, T = occ.shape
    nb = T // BLOCK
    parity = torch.arange(G, device=occ.device) % 2
    intf = obs['intf'][:, parity]                                   # [B,G,T]
    onehot = torch.eye(G, device=occ.device)[None, :, None, :].expand(B, G, nb, G)
    allow = obs['allow'][:, :, None, None].expand(B, G, nb, 1).float()
    x = torch.cat([intf.reshape(B, G, nb, BLOCK), occ.float().reshape(B, G, nb, BLOCK), onehot, allow], -1)
    return x.reshape(B, G * nb, 2 * BLOCK + G + 1)


class TransformerEncoder(nn.Module):
    """令牌 = [全局, 当前束标量, 80 个频率块(, N 个波束)]。返回 pooled 与块令牌输出。
    legacy_blocks=True 时只取当前允许组的 10 个频率块，共 12 个令牌（全局＋标量＋10 块）。这是历史 T-SAC
    14 令牌（位置、需求、剩余功率各一令牌＋10 块＋全局上下文令牌，取位置令牌读出）的近似，不是原样复现。
    ff/activation/norm_first 可配：历史最好版为 512 / relu / 后置 LayerNorm。"""

    def __init__(self, d=128, layers=2, heads=8, ff=None, beam_tokens=False, beam_dim=20, G=8, T=100,
                 activation='relu', norm_first=False, legacy_blocks=False):
        super().__init__()
        self.d, self.G, self.T, self.beam_tokens, self.legacy_blocks = d, G, T, beam_tokens, legacy_blocks
        nb = T // BLOCK
        self.block_proj = nn.Linear(2 * BLOCK + G + 1, d)
        self.block_pos = nn.Parameter(torch.randn(G * nb, d) * .02)
        self.scal_proj = nn.Linear(10, d)
        self.global_token = nn.Parameter(torch.randn(1, 1, d) * .02)
        self.type_emb = nn.Parameter(torch.randn(4, d) * .02)
        if beam_tokens:
            self.beam_proj = nn.Linear(beam_dim, d)
        layer = nn.TransformerEncoderLayer(d, heads, ff or 4 * d, dropout=0., batch_first=True, norm_first=norm_first, activation=activation)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d) if norm_first else nn.Identity()
        self.out_dim = d

    def forward(self, obs):
        B = obs['scal'].shape[0]
        blocks = self.block_proj(block_features(obs)) + self.block_pos[None] + self.type_emb[2]
        if self.legacy_blocks:
            nb = self.T // BLOCK
            group = obs['allow'].float().argmax(1)                                     # 唯一允许组
            index = (group[:, None] * nb + torch.arange(nb, device=blocks.device)[None, :])
            blocks = blocks.gather(1, index[:, :, None].expand(-1, -1, blocks.shape[-1]))
        tokens = [self.global_token.expand(B, 1, -1) + self.type_emb[0],
                  (self.scal_proj(obs['scal']) + self.type_emb[1])[:, None],
                  blocks]
        pad = None
        if self.beam_tokens:
            beams = self.beam_proj(obs['beam']) + self.type_emb[3]
            tokens.append(beams)
            pad = torch.zeros(B, 2 + blocks.shape[1] + beams.shape[1], dtype=torch.bool, device=beams.device)
            pad[:, 2 + blocks.shape[1]:] = ~obs['beam_valid']
        x = torch.cat(tokens, 1)
        x = self.norm(self.encoder(x, src_key_padding_mask=pad))
        return x[:, 0], x[:, 2:2 + blocks.shape[1]]


class TSAC14Encoder(nn.Module):
    """原样 14 令牌（历史 T-SAC，t0_width/power30 那条线）：位置（纬度、经度、宽度）、需求、剩余功率各 1 令牌，
    本组 10 个频率块（每块 10 槽干扰＋10 槽占用＝20 维），全局令牌（平均需求、进度、剩余总需求），取位置令牌读出。
    intf_mode='in'：干扰用 I/N 映射；'dbw'：历史的绝对 dBW 映射。需要规则定组（只有一个允许组）。"""

    def __init__(self, d=128, layers=2, heads=8, ff=512, activation='relu', norm_first=False, intf_mode='in', T=100,
                 feats=(), no_global=False, intf_zero=False):
        """第三轮选项：feats 为补充特征（features.FEATURE_SLOT 决定进哪个令牌）；no_global 去掉全局令牌（13 令牌）；
        intf_zero 把频率块里的干扰列置零（演员与评论家都用同一编码器，故两者同时去掉）。"""
        from .features import FEATURE_SLOT
        super().__init__()
        self.T, self.intf_key = T, ('intf_abs' if intf_mode == 'dbw' else 'intf')
        self.feats, self.no_global, self.intf_zero = tuple(feats), no_global, intf_zero
        extra = {k: sum(FEATURE_SLOT[f][1] for f in self.feats if FEATURE_SLOT[f][0] == k) for k in ('rate', 'power', 'slot', 'glob')}
        self.geo, self.rate, self.power = nn.Linear(3, d), nn.Linear(1 + extra['rate'], d), nn.Linear(1 + extra['power'], d)
        self.slot = nn.Linear((2 + extra['slot']) * BLOCK, d)
        if not no_global:
            self.glob = nn.Linear(3 + extra['glob'], d)
        self.pos = nn.Parameter(torch.zeros(1, 13 if no_global else 14, d))
        layer = nn.TransformerEncoderLayer(d, heads, ff, dropout=0., batch_first=True, norm_first=norm_first, activation=activation)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.out_dim = d

    def forward(self, obs):
        scal = obs['scal']
        B = scal.shape[0]
        rows = torch.arange(B, device=scal.device)
        group = obs['allow'].float().argmax(1)
        intf = obs[self.intf_key][rows, group % 2]                            # [B,T]
        if self.intf_zero:
            intf = torch.zeros_like(intf)
        occ = obs['occ'][rows, group].float()                                  # [B,T]
        nb = self.T // BLOCK
        cols = [intf, occ]
        rate_in, power_in, glob_in = [scal[:, 0:1]], [scal[:, 6:7]], [obs['glob']] if not self.no_global else []
        for f in self.feats:
            v = obs['f_' + f].float()
            if f == 'pneed':
                rate_in.append(v)
            elif f == 'share':
                power_in.append(v)
            elif f == 'snr':
                cols += [v[:, 0], v[:, 1]]
            elif f in ('outint', 'sens'):
                cols.append(v)
            else:
                glob_in.append(v)
        blocks = torch.cat([c.reshape(B, nb, BLOCK) for c in cols], -1)
        geo = torch.stack([scal[:, 2], scal[:, 3], scal[:, 1]], 1)
        tokens = [self.geo(geo)[:, None], self.rate(torch.cat(rate_in, 1))[:, None], self.power(torch.cat(power_in, 1))[:, None],
                  self.slot(blocks)]
        if not self.no_global:
            tokens.append(self.glob(torch.cat(glob_in, 1))[:, None])
        x = self.encoder(torch.cat(tokens, 1) + self.pos)
        return x[:, 0], x[:, 3:3 + nb]


class MLP14Encoder(nn.Module):
    """与 14 令牌同字段的全连接编码器（论文对照用）：位置 3、需求 1、剩余功率 1、本组 100 槽干扰与占用、全局 3。"""

    def __init__(self, d=128, hidden=512, T=100):
        super().__init__()
        self.T = T
        self.net = nn.Sequential(nn.Linear(3 + 1 + 1 + 2 * T + 3, hidden), nn.ReLU(), nn.Linear(hidden, d), nn.ReLU())
        self.out_dim = d

    def forward(self, obs):
        scal = obs['scal']
        B = scal.shape[0]
        rows = torch.arange(B, device=scal.device)
        group = obs['allow'].float().argmax(1)
        x = torch.cat([scal[:, 2:4], scal[:, 1:2], scal[:, 0:1], scal[:, 6:7], obs['intf'][rows, group % 2],
                       obs['occ'][rows, group].float(), obs['glob']], 1)
        return self.net(x), None


class CNNEncoder(nn.Module):
    def __init__(self, d=128, channels=64, G=8, T=100, layernorm=False):
        super().__init__()
        self.G, self.T = G, T
        self.conv = nn.Sequential(
            nn.Conv1d(3 * G, channels, 5, padding=2), nn.ReLU(),
            nn.Conv1d(channels, channels, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv1d(channels, channels, 5, stride=2, padding=2), nn.ReLU())
        self.fc = nn.Sequential(nn.Linear(channels * ((T + 3) // 4) + 10, 2 * d), nn.ReLU(), nn.Linear(2 * d, d), nn.ReLU())
        self.norm = nn.LayerNorm(d) if layernorm else nn.Identity()
        self.out_dim = d

    def forward(self, obs):
        occ = obs['occ']
        B, G, T = occ.shape
        parity = torch.arange(G, device=occ.device) % 2
        x = torch.cat([obs['intf'][:, parity], occ.float(), obs['allow'][:, :, None].float().expand(B, G, T)], 1)
        h = self.conv(x).flatten(1)
        return self.norm(self.fc(torch.cat([h, obs['scal']], 1))), None


class MLPEncoder(nn.Module):
    def __init__(self, d=128, hidden=512, G=8, T=100, layernorm=False):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2 * T + G * T + G + 10, hidden), nn.ReLU(), nn.Linear(hidden, d), nn.ReLU())
        self.norm = nn.LayerNorm(d) if layernorm else nn.Identity()
        self.out_dim = d

    def forward(self, obs):
        x = torch.cat([obs['intf'].flatten(1), obs['occ'].float().flatten(1), obs['allow'].float(), obs['scal']], 1)
        return self.norm(self.net(x)), None


class Heads(nn.Module):
    """三组输出。pointer=True 时组×起点由块令牌打分（块分数 + 块内偏移），其余为线性分类。"""

    def __init__(self, d, G=8, T=100, K=13, pointer=False):
        super().__init__()
        self.G, self.T, self.pointer = G, T, pointer
        if pointer:
            self.query = nn.Linear(d, d)
            self.offset = nn.Linear(d, BLOCK)
        else:
            self.gs = nn.Linear(d, G * T)
        self.cnt = nn.Linear(d, 10)
        self.pw = nn.Linear(d, K)

    def forward(self, pooled, blocks=None):
        if self.pointer:
            q = self.query(pooled)
            score = torch.einsum('bkd,bd->bk', blocks, q) / math.sqrt(blocks.shape[-1])
            gs = (score[:, :, None] + self.offset(blocks)).reshape(pooled.shape[0], self.G * self.T)
        else:
            gs = self.gs(pooled)
        return gs, self.cnt(pooled), self.pw(pooled)


def build_encoder(kind, size, beam_dim=20, G=8, T=100):
    size = dict(size or {})
    if kind in ('tf_blocks', 'tf_beams', 'tf_legacy'):
        return TransformerEncoder(size.get('d', 128), size.get('layers', 2), size.get('heads', 8), size.get('ff'), kind == 'tf_beams',
                                  beam_dim, G, T, size.get('activation', 'relu'), size.get('norm_first', False), kind == 'tf_legacy')
    if kind == 'tsac14':
        return TSAC14Encoder(size.get('d', 128), size.get('layers', 2), size.get('heads', 8), size.get('ff', 512),
                             size.get('activation', 'relu'), size.get('norm_first', False), size.get('intf_mode', 'in'), T,
                             size.get('feats', ()), size.get('no_global', False), size.get('intf_zero', False))
    if kind == 'mlp14':
        return MLP14Encoder(size.get('d', 128), size.get('hidden', 512), T)
    if kind == 'cnn':
        return CNNEncoder(size.get('d', 128), size.get('channels', 64), G, T, size.get('layernorm', False))
    if kind == 'mlp':
        return MLPEncoder(size.get('d', 128), size.get('hidden', 512), G, T, size.get('layernorm', False))
    raise ValueError(kind)


class PolicyNet(nn.Module):
    """编码器 + 三头；value=True 时另出一个标量价值头（PPO）。"""

    def __init__(self, kind, size, K, pointer=False, value=False, beam_dim=20, G=8, T=100):
        super().__init__()
        self.encoder = build_encoder(kind, size, beam_dim, G, T)
        self.heads = Heads(self.encoder.out_dim, G, T, K, pointer and kind.startswith('tf'))
        self.value = nn.Linear(self.encoder.out_dim, 1) if value else None

    def forward(self, obs):
        pooled, blocks = self.encoder(obs)
        gs, cnt, pw = self.heads(pooled, blocks)
        if self.value is not None:
            return (gs, cnt, pw), self.value(pooled).squeeze(-1)
        return (gs, cnt, pw), None


def masked(logits, mask):
    return logits.masked_fill(~mask, NEG)


def head_masks(obs):
    return obs['mask_gs'], obs['mask_cnt'], obs['mask_pw']


def count_parameters(module):
    return sum(p.numel() for p in module.parameters())
