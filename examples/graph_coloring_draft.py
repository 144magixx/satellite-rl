def graph_coloring_allocator(demands, adj_matrix, total_bandwidth, degrees):
    """
    基于图着色的启发式分配器 (DSATUR + First-Fit 思想)
    策略：
    1. 排序：按 (度 + 1) * 需求量 降序。
    2. 分配：遍历排序后的波束，使用 First-Fit 策略为其寻找一个连续槽块。
    3. 寻找：只检查已分配的 *邻居*，在它们的间隙中寻找 *第一个* 空间。
    """
    n_beams = len(demands)
    allocations = np.zeros((n_beams, 2), dtype=int)
    scores = (degrees + 1) * demands
    sorted_beams = np.argsort(scores)[::-1]  # 降序
    allocated_mask = np.zeros(n_beams, dtype=bool)

    for i in sorted_beams:
        demand = demands[i]
        neighbors = adj_matrix[i]
        allocated_neighbors = np.where(neighbors & allocated_mask)[0]
        forbidden_ranges = sorted(
            [tuple(allocations[j]) for j in allocated_neighbors if allocations[j, 1] > allocations[j, 0]],
            key=lambda x: x[0])

        current_slot = 0
        found_slot = False

        if not forbidden_ranges:
            if current_slot + demand <= total_bandwidth:
                allocations[i] = [current_slot, current_slot + demand]
                found_slot = True
        else:
            # 遍历“间隙” (First-Fit)
            for start_forbidden, end_forbidden in forbidden_ranges:
                gap_size = start_forbidden - current_slot
                if gap_size >= demand:
                    allocations[i] = [current_slot, current_slot + demand]
                    found_slot = True
                    break
                current_slot = max(current_slot, end_forbidden)

            if not found_slot:
                # 检查最后一个禁用区间到 B 之间的空间
                if current_slot + demand <= total_bandwidth:
                    allocations[i] = [current_slot, current_slot + demand]
                    found_slot = True

        # ---- 资源稀缺处理 ----
        if not found_slot:
            best_gap = 0
            best_start = -1
            current_slot = 0

            first_start = forbidden_ranges[0][0] if forbidden_ranges else total_bandwidth
            gap = first_start - current_slot
            if gap > best_gap:
                best_gap = gap
                best_start = current_slot

            for k in range(len(forbidden_ranges)):
                current_slot = forbidden_ranges[k][1]
                next_start = forbidden_ranges[k + 1][0] if k + 1 < len(forbidden_ranges) else total_bandwidth
                gap = next_start - current_slot
                if gap > best_gap:
                    best_gap = gap
                    best_start = current_slot

            if best_start != -1:
                alloc_size = min(demand, best_gap)
                if alloc_size > 0:
                    # 稀缺时，仍然使用 First-Fit (放在块的开头)
                    allocations[i] = [best_start, best_start + alloc_size]

        allocated_mask[i] = True
    return allocations