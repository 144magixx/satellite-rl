"""Exact deterministic SSP for the R10 lower-one transportation network.

This module knows no scenes, geometry, models, files or remote execution.
All integer arithmetic is Python integer arithmetic. Resource interruption is
reported separately from a proof that the finite supplied graph is infeasible.
"""
import copy


class ResourceUnknown(RuntimeError):
    def __init__(self, reason, ledger=None):
        super().__init__(reason)
        self.partial_ledger = None if ledger is None else ledger.record()


class Ledger:
    CAPS = {
        'raypairs': None, 'flow_queries': 256,
        'residual_edge_inspections': 2**30,
        'heap_pushes_plus_pops': 2**31,
        'heap_key_comparisons': 2**37,
        'graph_arc_initialization_writes': 2**20,
        'augmentation_capacity_writes': 2**24,
        'full_vector_node_visits': 2**26,
        'sort_key_comparisons': 2000000,
    }

    def __init__(self, n, caps=None, stop=None):
        if type(n) is not int or n < 1:
            raise ValueError('positive full-scene Python integer n')
        self.n = n
        self.caps = dict(self.CAPS, raypairs=8192*n)
        if caps is not None:
            if any(k not in self.caps or type(v) is not int or v < 0 or
                   v > self.caps[k] for k, v in caps.items()):
                raise ValueError('caps may only tighten registered bounds')
            self.caps.update(caps)
        self.data = dict.fromkeys(self.caps, 0)
        self.data.update(augmentations=0, heap_pushes=0, heap_pops=0,
                         stale_heap_pops=0, settled_heap_pops=0,
                         scan_input_count=0, sort_input_count=0,
                         scan_passes={}, sort_passes={}, flow_graphs=[],
                         flow_results=[])
        self.stop = stop

    def tick(self, key, count=1):
        if type(count) is not int or count < 0:
            raise ValueError('exact nonnegative Python integer count')
        if self.stop is not None and self.stop():
            raise ResourceUnknown('external_guard_or_pause', self)
        value = self.data.get(key, 0) + count
        if key in self.caps and value > self.caps[key]:
            raise ResourceUnknown(key+'_cap', self)
        self.data[key] = value

    def pairs(self, count, audit=False):
        self.tick('raypairs', count)

    def _pass(self, kind, label, count):
        self.tick(kind+'_input_count', count)
        row = self.data[kind+'_passes'].setdefault(label, {'calls': 0, 'items': 0})
        row['calls'] += 1
        row['items'] += count

    def scan(self, label, count):
        self._pass('scan', label, count)

    def sort(self, label, count):
        self._pass('sort', label, count)

    def record(self):
        return copy.deepcopy(dict(self.data, caps=self.caps, n=self.n))

    def ordered(self, values, key, label):
        """Count actual Python sort key comparisons, separate from input size."""
        self.sort(label,len(values))
        ledger=self
        class Key:
            __slots__=('value',)
            def __init__(self,value):self.value=value
            def __lt__(self,other):
                ledger.tick('sort_key_comparisons')
                return self.value<other.value
        return sorted(values,key=lambda value:Key(key(value)))


class _Heap:
    """Owned binary min heap; each integer-tuple key comparison is metered."""
    def __init__(self, ledger):
        self.items = []
        self.ledger = ledger

    def less(self, left, right):
        self.ledger.tick('heap_key_comparisons')
        return left < right

    def push(self, key):
        self.ledger.tick('heap_pushes_plus_pops')
        self.ledger.tick('heap_pushes')
        items = self.items
        items.append(key)
        child = len(items)-1
        while child:
            parent = (child-1)//2
            if not self.less(items[child], items[parent]):
                break
            items[parent], items[child] = items[child], items[parent]
            child = parent

    def pop(self):
        self.ledger.tick('heap_pushes_plus_pops')
        self.ledger.tick('heap_pops')
        items = self.items
        first = items[0]
        last = items.pop()
        if items:
            items[0] = last
            parent = 0
            while 2*parent+1 < len(items):
                child = 2*parent+1
                if child+1 < len(items) and self.less(items[child+1], items[child]):
                    child += 1
                if not self.less(items[child], items[parent]):
                    break
                items[parent], items[child] = items[child], items[parent]
                parent = child
        return first


def solve(aircraft_ids, beam_ids, adjacency, costs, q, ledger):
    """Return full minimum-cost assignment or a finite-graph infeasibility.

Rows/columns follow the caller's original ID order; internal nodes and insertion
order follow sorted stable string IDs. Caller arrays are never mutated. No
anchors are pinned, and duplicate anchor identities are irrelevant to this API.
"""
    aids, bids = tuple(aircraft_ids), tuple(beam_ids)
    n, b = len(aids), len(bids)
    ledger.scan('flow_input_ID_validation', n+b)
    if (n != ledger.n or not n or not b or
        any(type(x) is not str or not x for x in aids+bids) or
        len(set(aids)) != n or len(set(bids)) != b or
        type(q) is not int or q < 1 or b > n or n > q*b):
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
            if adjacency[i][j] and (type(costs[i][j]) is not int or costs[i][j] < 0):
                raise ValueError('eligible costs are nonnegative Python integers')
    ledger.tick('flow_queries')
    ai = ledger.ordered(list(range(n)), lambda i: aids[i], 'flow_aircraft_ID_sort')
    bj = ledger.ordered(list(range(b)), lambda j: bids[j], 'flow_beam_ID_sort')
    source, unode, sink = 0, n+b+1, n+b+2
    nodes = sink+1
    ledger.tick('full_vector_node_visits', nodes)
    graph = [[] for _ in range(nodes)]

    def edge(u, v, cap, cost):
        ledger.tick('graph_arc_initialization_writes', 2)
        e = [v, len(graph[v]), cap, cost]
        graph[u].append(e)
        graph[v].append([u, len(graph[u])-1, 0, -cost])
        return e

    for ii in range(n):
        edge(source, 1+ii, 1, 0)
    assignments = []
    for ii, i in enumerate(ai):
        for jj, j in enumerate(bj):
            if adjacency[i][j]:
                e = edge(1+ii, 1+n+jj, 1, costs[i][j])
                assignments.append((i, j, e))
    mandatory = []
    for jj in range(b):
        mandatory.append(edge(1+n+jj, sink, 1, 0))
        edge(1+n+jj, unode, q-1, 0)
    edge(unode, sink, n-b, 0)
    ledger.data['flow_graphs'].append(dict(nodes=nodes, aircraft=n, beams=b,
        residual_arcs=2*(n+len(assignments)+2*b+1), assignment_edges=len(assignments), q=q))
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
                reduced = cost+potential[u]-potential[v]
                if reduced < 0:
                    raise AssertionError('negative reachable reduced cost')
                candidate = distance+reduced
                if dist[v] is None or candidate < dist[v]:
                    if settled[v]:
                        raise AssertionError('settled node strictly improved')
                    dist[v], prev[v] = candidate, (u, ei)
                    heap.push((candidate, v))
        if dist[sink] is None:
            row = dict(status='graph_infeasible', feasible=False, delivered=delivered,
                       required=n, total_cost=str(total_cost), assignment=None,
                       augmentation_costs=list(active['augmentation_costs']))
            ledger.data['flow_results'].append(copy.deepcopy(row))
            ledger.data.pop('active_flow',None)
            return row
        ledger.tick('full_vector_node_visits', nodes)
        for v in range(nodes):
            if dist[v] is not None:
                potential[v] += dist[v]
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
            total_cost += arc[3]
            cycle_cost += arc[3]
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
    row = dict(status='optimal', feasible=True, delivered=n, required=n,
               total_cost=str(total_cost), assignment=owners, counts=counts,
               potentials=[str(x) for x in potential],augmentation_costs=list(active['augmentation_costs']))
    ledger.data['flow_results'].append({k:copy.deepcopy(v) for k,v in row.items()
                                       if k not in ('assignment','potentials')})
    ledger.data.pop('active_flow',None)
    return row
