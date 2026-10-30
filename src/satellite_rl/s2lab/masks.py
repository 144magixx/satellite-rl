"""第三轮：按槽数条件化的合法动作掩码（环境与智能体共用同一实现）。

记号：c＝槽数（1—10），x＝本组内起点（0—T-1），组 g 由规则先定（14 令牌只有一个允许组）。
"""
import torch


def free_run(occ):
    """occ[B,G,T] 布尔占用 → free_run[B,G,T]：从该槽起连续空闲槽数（已占为 0）。"""
    T = occ.shape[-1]
    t = torch.arange(T, device=occ.device)
    idx = torch.where(occ, t[None, None, :], torch.full_like(occ, T, dtype=torch.long))
    nxt = torch.flip(torch.cummin(torch.flip(idx, [2]), 2)[0], [2])
    return nxt - t[None, None, :]


def group_index(allow):
    """允许组 one-hot → 组号 [B]（第三轮各组都由规则定组，只有一个允许组）。"""
    return allow.float().argmax(1)


def window_cost(intf_rel, c_max=10):
    """intf_rel[B,2,T] 线性 I/N → cost[B,10,2,T]：从起点 s 起 c 个槽的平均干扰（越界处为 inf）。"""
    B, P, T = intf_rel.shape
    cs = torch.cat([torch.zeros(B, P, 1, device=intf_rel.device), intf_rel.float().cumsum(-1)], -1)
    s = torch.arange(T, device=intf_rel.device)
    c = torch.arange(1, c_max + 1, device=intf_rel.device)
    end = s[None, :] + c[:, None]                                     # [10,T]
    ok = end <= T
    endc = end.clamp(max=T)
    cost = (cs[:, :, endc] - cs[:, :, s][:, :, None, :]) / c[None, None, :, None].float()   # [B,2,10,T]
    cost = torch.where(ok[None, None], cost, torch.full_like(cost, float('inf')))
    return cost.permute(0, 2, 1, 3)                                   # [B,10,2,T]


def conditional_masks(occ, allow, cmax, topk=0, intf_rel=None, c_max=10):
    """返回 mask_c[B,10]、mask_x[B,10,T]（只在允许组内）。
    合法：c ≤ cmax（含给后续波束预留），且本组存在从起点起 c 个槽全空的完整窗口；
    topk>0 时，在每个 c 的合法完整窗口里只留窗口平均干扰最低的 topk 个。"""
    B, G, T = occ.shape
    g = group_index(allow)
    fr = free_run(occ)[torch.arange(B, device=occ.device), g]          # [B,T]
    c = torch.arange(1, c_max + 1, device=occ.device)
    mask_x = fr[:, None, :] >= c[None, :, None]                        # [B,10,T]
    if topk:
        cost = window_cost(intf_rel, c_max)[torch.arange(B, device=occ.device), :, g % 2]   # [B,10,T]
        cost = torch.where(mask_x, cost, torch.full_like(cost, float('inf')))
        k = min(topk, T)
        keep = torch.zeros_like(mask_x)
        keep.scatter_(2, cost.topk(k, dim=2, largest=False).indices, True)
        mask_x = mask_x & keep
    mask_c = (c[None, :] <= cmax[:, None]) & mask_x.any(-1)
    return mask_c, mask_x
