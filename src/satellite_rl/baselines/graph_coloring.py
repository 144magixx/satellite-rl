from satellite_rl.paths import project_path
# -*- coding: utf-8 -*-
# graph_coloring.py
# 基于图着色（DSATUR + First-Fit）的启发式频谱分配与环境对接

import os
import math
import csv
import numpy as np
from datetime import datetime

from satellite_rl.envs.satellite import BeamInfo, SatelliteFreqState, CurrentBeamState, Env  # :contentReference[oaicite:1]{index=1}


# =========================
# 工具：地理距离（km）
# =========================
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1 = np.radians(lat1); lon1 = np.radians(lon1)
    lat2 = np.radians(lat2); lon2 = np.radians(lon2)
    dlat = lat2 - lat1; dlon = lon2 - lon1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2.0)**2
    return 2.0 * R * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))


# =========================
# 构建邻接图（冲突边）
# - 距离小于阈值 → 视为邻居（不能重叠频段）
# =========================
def build_adjacency(beam_locs, dist_threshold_km=800.0):
    n = beam_locs.shape[0]
    adj = np.zeros((n, n), dtype=bool)
    for i in range(n):
        lat_i, lon_i = beam_locs[i]
        # 向量化计算与 i 的距离
        d = haversine_km(lat_i, lon_i, beam_locs[:, 0], beam_locs[:, 1])
        # 自身不算邻居；小于阈值定义为邻接
        mask = (d > 0) & (d <= dist_threshold_km)
        adj[i, mask] = True
    # 无向图
    adj = np.logical_or(adj, adj.T)
    return adj


# =========================
# 图着色启发式分配（连续频段）
# - demands: 每个波束需要的“连续槽数”（整数）
# - adj_matrix: 邻接矩阵（True=有冲突边）
# - total_bandwidth: 可用频点数量（如 100）
# - degrees: 图度数
# 返回：allocations: [n_beams, 2]，每行是 [start, end)
# =========================
def graph_coloring_allocator_with_groups(demands, adj_matrix, total_bandwidth, degrees, num_groups=8):
    """
    基于图着色 + 组复用的启发式分配器：
    - 先为每个波束选择 group（0..7），再在该 group 内做 First-Fit 连续区间分配；
    - 只禁止与“同组且为邻居”的已分配区间重叠；
    - 放不下就换组；所有组都紧张时，选“最大间隙”给部分需求。
    返回:
      groups      : [n_beams]，每个波束选到的 group
      allocations : [n_beams, 2]，每个波束在其组内的区间 [start, end)
    """
    n_beams = len(demands)
    allocations = np.zeros((n_beams, 2), dtype=int)
    groups = -np.ones(n_beams, dtype=int)
    allocated_mask = np.zeros(n_beams, dtype=bool)

    # DSATUR 风格优先级：高需求 + 高度优先
    scores = (degrees + 1) * demands
    order = np.argsort(scores)[::-1]

    # 每个组各自维护“已占用的邻居区间”仅为查询之便（可按需构造）
    for i in order:
        demand = int(demands[i])
        neigh = np.where(adj_matrix[i])[0]

        best_choice = None  # (group, start, end, wasted_space)
        # 组选择优先级：先尝试 i%8、i^1（同/异极化），再试其它组
        candidate_groups = [i % num_groups, (i ^ 1) % num_groups] + \
                           [g for g in range(num_groups) if g not in {(i % num_groups), ((i ^ 1) % num_groups)}]

        for g in candidate_groups:
            # 只看“同组且已分配且为邻居”的区间
            same_group_allocated_neighbors = [j for j in neigh if allocated_mask[j] and groups[j] == g]
            forb = sorted([tuple(allocations[j]) for j in same_group_allocated_neighbors
                           if allocations[j,1] > allocations[j,0]], key=lambda x: x[0])

            # —— First-Fit 查找 —— #
            cur = 0
            placed = None
            if not forb:
                if cur + demand <= total_bandwidth:
                    placed = (cur, cur + demand)
            else:
                for s_forb, e_forb in forb:
                    gap = s_forb - cur
                    if gap >= demand:
                        placed = (cur, cur + demand)
                        break
                    cur = max(cur, e_forb)
                if placed is None and cur + demand <= total_bandwidth:
                    placed = (cur, cur + demand)

            if placed is not None:
                # 评价一下“浪费空间”以便在多个可行组中挑更紧凑的（可换成更复杂准则）
                start, end = placed
                wasted = (forb[0][0] - 0) if forb else start  # 简单启发：越靠前越好
                if (best_choice is None) or (wasted < best_choice[3]):
                    best_choice = (g, start, end, wasted)
                continue

            # 若放不下：找最大间隙给“部分需求”
            best_gap, best_start = 0, -1
            cur = 0
            first_start = forb[0][0] if forb else total_bandwidth
            gap = first_start - cur
            if gap > best_gap:
                best_gap, best_start = gap, cur
            for k in range(len(forb)):
                cur = forb[k][1]
                nxt = forb[k+1][0] if k+1 < len(forb) else total_bandwidth
                gap = nxt - cur
                if gap > best_gap:
                    best_gap, best_start = gap, cur
            if best_start != -1 and best_gap > 0:
                alloc_size = min(demand, best_gap)
                # 把“部分分配”也作为备选，但劣于完整放下
                placed = (best_start, best_start + alloc_size)
                wasted = 1e6  # 劣化权重，优先选择完整分配
                if best_choice is None:
                    best_choice = (g, placed[0], placed[1], wasted)

        # 应用本波束的最佳组与区间
        if best_choice is None:
            # 完全无处可放：留空（也可以回退到最空组的最小区间给1槽）
            groups[i] = candidate_groups[0]
            allocations[i] = [0, 0]
        else:
            g, s, e, _ = best_choice
            groups[i] = g
            allocations[i] = [s, e]

        allocated_mask[i] = True

    return groups, allocations



# =========================
# 将着色分配结果 → 环境动作并评估
# =========================
def demand_to_power_w(required_bps):
    # 与 greedy/fixed 一致的三档功率
    if required_bps >= 8e8:
        return 30
    elif required_bps >= 4e8:
        return 25
    return 20


def run_graph_coloring(dist_threshold_km=800.0, total_bandwidth=100, log_csv=True):
    # ---- 1) 初始化环境（与 fixed/greedy 保持一致） ----
    beam_info = BeamInfo()
    sat = SatelliteFreqState()
    cur = CurrentBeamState(beam_info.beam_info)
    env = Env(sat, beam_info, cur)

    # ---- 2) 构建邻接图与“需求 → 连续槽数” ----
    locs = beam_info.beam_info['beam_location']              # (N, 2)
    rates = beam_info.beam_info['rate'].reshape(-1)          # (N,)
    # 将速率需求转化为“时隙需求”（每槽带宽 25 MHz × 谱效 ~ 2）
    # 这里给出一个保守映射：每槽可承载 ~ 25e6 * 2 = 50 Mbps
    slot_capacity_bps = 10e7
    demands = np.clip(np.ceil(rates / slot_capacity_bps).astype(int), 1, 5)  # 每波束最多 10 槽（与环境一致）
    adj = build_adjacency(locs, dist_threshold_km=dist_threshold_km)
    degrees = adj.sum(axis=1)

    # ---- 3) 着色分配：得到每个波束的连续区间 [start, end) ----
    groups, allocations = graph_coloring_allocator_with_groups(demands, adj, total_bandwidth=100, degrees=degrees)

    # ---- 4) 生成动作流并与环境交互评估 ----
    state = env.reset(BeamInfo())   # 环境按内部顺序（按需降序）逐个波束推进
    done = False
    ep_ret = 0.0
    ep_satis = 0.0

    # 日志（可选）
    if log_csv:
        os.makedirs(project_path("logs"), exist_ok=True)
        log_path = os.path.join(project_path("logs"), f"gc_log_{datetime.now().strftime('%Y%m%d%H%M')}.csv")
        fields = [
            'beam_id','group','freq','slots','power_dB',
            'required_rate','allocate_rate','spectral_efficiency_avg',
            'current_satisfaction','total_satisfaction','is_interfere'
        ]
        f = open(log_path, "w", newline=""); writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
    else:
        f = None; writer = None

    while not done:
        i = env.beam_idx
        g = int(groups[i])  # 这里用分配器挑出来的 group
        start, end = allocations[i]
        slots = int(max(1, min(10, end - start)))
        freq = int(np.clip(start, 0, 99))  # 该“组内列”对应环境的列索引
        pw_w = demand_to_power_w(rates[i])

        action = [g, freq, slots, 10 * math.log10(float(pw_w))]

        next_state, reward, done, info = env.step(action)
        ep_ret += float(reward)
        ep_satis = float(info['total_satisfaction'])
        state = next_state

        if writer is not None:
            writer.writerow({
                'beam_id': i,
                'group': g,
                'freq': freq,
                'slots': slots,
                'power_dB': 10*math.log10(float(pw_w)),
                'required_rate': float(info['required_rate']),
                'allocate_rate': float(info['spectral_efficiency_avg'] * 2.5e7 * max(1, slots)),
                'spectral_efficiency_avg': float(info['spectral_efficiency_avg']),
                'current_satisfaction': float(info['current_satisfaction']),
                'total_satisfaction': float(info['total_satisfaction']),
                'is_interfere': int(info['is_interfere']),
            })

    if f is not None:
        f.close()
        print(f"[Saved] {log_path}")

    print(f"[GraphColoring] Return={ep_ret:.2f} | Satisfaction={ep_satis:.3f}")
    return ep_ret, ep_satis, allocations


if __name__ == "__main__":
    run_graph_coloring(dist_threshold_km=800.0, total_bandwidth=100, log_csv=True)
