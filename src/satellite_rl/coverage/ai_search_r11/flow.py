"""R11 Gh SSP: R10 deterministic solver with convex histogram marginal arcs.
No scene generation, native processes, files or scientific execution at import.
Bigint counters cover explicit cost arithmetic, not all index/capacity/limb work.
"""
import copy
from ..ai_search_r10.flow import Ledger, ResourceUnknown, _Heap

def solve(aircraft_ids, beam_ids, adjacency, costs, q, ledger, *, ncap):
    """Return full minimum-cost assignment or a finite-graph infeasibility.

Rows/columns follow the caller's original ID order; internal nodes and insertion
order follow sorted stable string IDs. Caller arrays are never mutated. No
anchors are pinned, and duplicate anchor identities are irrelevant to this API.
"""
    if type(ncap) is not int or not len(beam_ids) <= ncap <= 200:
        raise ValueError('original integer Ncap bounds targetN')
    aids, bids = tuple(aircraft_ids), tuple(beam_ids)
    n, b = len(aids), len(bids)
    ledger.scan('flow_input_ID_validation', n+b)
    if (n != ledger.n or not n or not b or
        any(type(x) is not str or not x for x in aids+bids) or
        len(set(aids)) != n or len(set(bids)) != b or
        type(q) is not int or not 1 <= q <= n or b > n or n > q*b):
        raise ValueError('qualified distinct IDs and feasible integer count bound')
    if (len(adjacency) != n or len(costs) != n or
        any(len(row) != b for row in adjacency) or
        any(len(row) != b for row in costs)):
        raise ValueError('complete supplied adjacency/cost matrix shapes')
    ledger.scan('flow_validate_matrix_entries', n*b)
    for i in range(n):
        for j in range(b):
            if type(adjacency[i][j]) is not bool:
                raise ValueError('explicit boolean adjacency')
            if adjacency[i][j] and (type(costs[i][j]) is not int or not 0 <= costs[i][j] <= n*250001+1+250001):
                raise ValueError('eligible costs are nonnegative Python integers')
    def observe(value):
        if type(value) is not int:raise TypeError('Python exact integer')
        ledger.data['observed_max_integer_bits']=max(ledger.data.get('observed_max_integer_bits',0),abs(value).bit_length())
        return value
    ledger.tick('flow_queries')
    ai = ledger.ordered(list(range(n)), lambda i: aids[i], 'flow_aircraft_ID_sort')
    bj = ledger.ordered(list(range(b)), lambda j: bids[j], 'flow_beam_ID_sort')
    source, unode, sink = 0, n+b+1, n+b+2
    nodes = sink+1
    ledger.tick('full_vector_node_visits', nodes)
    graph = [[] for _ in range(nodes)]

    forward = []

    def edge(u, v, cap, cost):
        observe(cost)
        ledger.tick('graph_arc_initialization_writes', 2)
        ledger.tick('bigint_negations')
        e = [v, len(graph[v]), cap, cost]
        graph[u].append(e)
        graph[v].append([u, len(graph[u])-1, 0, -cost])
        forward.append((u, v, cap, e))
        return e

    for ii in range(n):
        edge(source, 1+ii, 1, 0)
    assignments = []
    for ii, i in enumerate(ai):
        for jj, j in enumerate(bj):
            if adjacency[i][j]:
                e = edge(1+ii, 1+n+jj, 1, costs[i][j])
                assignments.append((i, j, e))
    ledger.tick('bigint_multiplications',2)
    ledger.tick('bigint_additions_subtractions',4)
    base = ncap+1
    wd = n*250001+1
    wh = n*(wd+250001)+1
    powers=[1]
    ledger.scan('histogram_power_construction',q)
    for r in range(1,q+1):
        ledger.tick('bigint_multiplications')
        powers.append(observe(powers[-1]*base))
    mandatory = []
    marginal_arcs = []
    for jj in range(b):
        ledger.tick('bigint_subtractions')
        ledger.tick('bigint_multiplications')
        mandatory.append(edge(1+n+jj, sink, 1, observe(wh*(base-1))))
        for r in range(2,q+1):
            ledger.tick('bigint_subtractions')
            ledger.tick('bigint_multiplications')
            cost=observe(wh*(powers[r]-powers[r-1]))
            marginal_arcs.append((bj[jj],r,edge(1+n+jj,unode,1,cost)))
    edge(unode, sink, n-b, 0)
    ledger.data['flow_graphs'].append(dict(nodes=nodes, aircraft=n, beams=b,
        residual_arcs=2*len(forward), assignment_edges=len(assignments), q=q))
    ledger.tick('full_vector_node_visits', nodes)
    potential = [0]*nodes
    delivered, total_cost = 0, 0
    active=dict(required=n,delivered=0,last_completed_total_cost='0',
                augmentation_in_progress=False,observed_augmentation_cost='0',augmentation_costs=[])
    ledger.data['active_flow']=active
    while delivered < n:
        # dist, predecessor, settled are three complete initializations.
        ledger.tick('full_vector_node_visits', 3*nodes)
        dist, prev, settled = [None]*nodes, [None]*nodes, [False]*nodes
        dist[source] = 0
        heap = _Heap(ledger)
        heap.push((0, source))
        while heap.items:
            distance, u = heap.pop()
            if distance != dist[u]:
                ledger.tick('stale_heap_pops')
                continue
            if settled[u]:
                ledger.tick('settled_heap_pops')
                continue
            settled[u] = True
            for ei, arc in enumerate(graph[u]):
                ledger.tick('residual_edge_inspections')
                v, _, capacity, cost = arc
                if capacity == 0:
                    continue
                ledger.tick('bigint_additions_subtractions',2)
                interim = observe(cost+potential[u])
                reduced = observe(interim-potential[v])
                if reduced < 0:
                    raise AssertionError('negative reachable reduced cost')
                ledger.tick('bigint_additions_subtractions')
                candidate = observe(distance+reduced)
                if dist[v] is None or candidate < dist[v]:
                    if settled[v]:
                        raise AssertionError('settled node strictly improved')
                    dist[v], prev[v] = candidate, (u, ei)
                    heap.push((candidate, v))
        if dist[sink] is None:
            # Reuse the final exhaustive Dijkstra finite set; no extra BFS.
            ledger.tick('full_vector_node_visits',nodes)
            reachable=[v for v in range(nodes) if dist[v] is not None]
            cut=[]
            capacity=0
            for u,v,original_capacity,arc in forward:
                ledger.tick('residual_edge_inspections')
                if dist[u] is not None and dist[v] is None:
                    cut.append(dict(u=u,v=v,capacity=original_capacity))
                    capacity+=original_capacity
                    if arc[2]!=0:raise AssertionError('reachable cut residual capacity')
            if capacity!=delivered or capacity>=n:raise AssertionError('finite cut certificate')
            ledger.scan('cut_certificate_ID_and_mask_output',n*b+n+b)
            certificate=dict(reachable_nodes=reachable,forward_cut=cut,cut_capacity=capacity,
                required=n,node_order=dict(aircraft_ids=[aids[i] for i in ai],beam_ids=[bids[j] for j in bj]),
                adjacency=[[adjacency[i][j] for j in bj] for i in ai],q=q,N=b)
            row = dict(status='graph_infeasible', feasible=False,certificate=certificate,
                       potentials=[str(x) for x in potential],distances=[None if x is None else str(x) for x in dist], delivered=delivered,
                       required=n, total_cost=str(total_cost), assignment=None,
                       augmentation_costs=list(active['augmentation_costs']))
            ledger.data['flow_results'].append(copy.deepcopy(row))
            ledger.data.pop('active_flow',None)
            return row
        ledger.tick('full_vector_node_visits', nodes)
        for v in range(nodes):
            if dist[v] is not None:
                ledger.tick('bigint_additions_subtractions')
                potential[v] = observe(potential[v]+dist[v])
        chain, seen, v = [], {sink}, sink
        while v != source:
            if prev[v] is None:
                raise AssertionError('missing augment predecessor')
            u, ei = prev[v]
            ledger.tick('residual_edge_inspections')
            arc = graph[u][ei]
            if arc[0] != v or arc[2] < 1 or u in seen:
                raise AssertionError('invalid or cyclic augment chain')
            chain.append((u, ei))
            seen.add(u)
            v = u
            if len(chain) >= nodes:
                raise AssertionError('augment chain exceeds V-1')
        active['augmentation_in_progress']=True
        active['observed_augmentation_cost']='0'
        cycle_cost=0
        for u, ei in chain:
            ledger.tick('residual_edge_inspections')
            arc = graph[u][ei]
            ledger.tick('augmentation_capacity_writes', 2)
            arc[2] -= 1
            graph[arc[0]][arc[1]][2] += 1
            ledger.tick('bigint_additions_subtractions',2)
            total_cost = observe(total_cost+arc[3])
            cycle_cost = observe(cycle_cost+arc[3])
            active['observed_augmentation_cost']=str(cycle_cost)
        delivered += 1
        active.update(delivered=delivered,last_completed_total_cost=str(total_cost),augmentation_in_progress=False)
        active['augmentation_costs'].append(str(cycle_cost))
        ledger.tick('augmentations')
    owners = [None]*n
    counts = [0]*b
    for i, j, arc in assignments:
        ledger.tick('residual_edge_inspections')
        if arc[2] == 0:
            if owners[i] is not None:
                raise AssertionError('multiple aircraft owners')
            owners[i] = j
            counts[j] += 1
    for arc in mandatory:
        ledger.tick('residual_edge_inspections')
        if arc[2] != 0:
            raise AssertionError('mandatory lower-one arc not saturated')
    ledger.scan('flow_complete_owner_and_capacity_audit', n+b)
    if any(x is None for x in owners) or any(not 1 <= x <= q for x in counts):
        raise AssertionError('full ownership/lower-upper capacity')
    used=[]
    for j,r,arc in marginal_arcs:
        ledger.tick('residual_edge_inspections')
        if arc[2]==0:used.append(dict(beam_index=j,r=r,cost=str(arc[3])))
    ledger.scan('histogram_used_prefix_and_secondary_audit',n+b+len(used))
    bybeam=[[] for _ in range(b)]
    for item in used:bybeam[item['beam_index']].append(item['r'])
    for j,count in enumerate(counts):
        if bybeam[j]!=list(range(2,count+1)):
            raise AssertionError('strict convex cheapest marginal prefix')
    secondary=0
    for i,j in enumerate(owners):
        ledger.tick('bigint_additions_subtractions')
        secondary=observe(secondary+costs[i][j])
    ledger.scan('histogram_final_count_objective',b)
    histogram=0
    for count in counts:
        ledger.tick('bigint_additions_subtractions')
        histogram=observe(histogram+powers[count])
    ledger.tick('bigint_subtractions')
    ledger.tick('bigint_multiplications')
    ledger.tick('bigint_additions_subtractions')
    expected=observe(wh*(histogram-b)+secondary)
    if expected!=total_cost:raise AssertionError('histogram network objective equality')
    row = dict(status='optimal' , feasible=True, delivered=n, required=n,
               total_cost=str(total_cost), assignment=owners, counts=counts,
               histogram=str(histogram),secondary_cost=str(secondary),WH=str(wh),WD=str(wd),base=base,used_marginal_slots=used,
               distances=[None if x is None else str(x) for x in dist],
               potentials=[str(x) for x in potential],augmentation_costs=list(active['augmentation_costs']))
    ledger.data['flow_results'].append({k:copy.deepcopy(v) for k,v in row.items()
                                       if k not in ('assignment','potentials')})
    ledger.data.pop('active_flow',None)
    return row
