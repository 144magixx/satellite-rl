"""R9 pinned-anchor BFS recipe, with R10 shared primitive metering added.

Node/edge order, BFS early sink exit, anchor pinning and capacity search match
the frozen R9 implementation. Added counters do not change mathematical work.
"""
from collections import deque


def feasible(member_ids,anchor_ids,adjacency,q,ledger):
    ids,anchors=tuple(member_ids),tuple(anchor_ids)
    m,k=len(ids),len(anchors)
    ledger.scan('flow_ID_shape_validation',m+k)
    sorted_ids=tuple(ledger.ordered(list(ids),lambda a:a,'flow_verify_member_order'))
    sorted_anchors=tuple(ledger.ordered(list(anchors),lambda a:a,'flow_verify_anchor_order'))
    if not (all(type(x) is str and x for x in ids+anchors) and 1<=k<=3 and k<=m<=128
            and ids==sorted_ids and anchors==sorted_anchors
            and len(set(ids))==m and len(set(anchors))==k and set(anchors)<=set(ids)):
        raise ValueError('R9 stable sorted distinct member/anchor IDs')
    if len(adjacency)!=m or any(len(row)!=k for row in adjacency) or type(q) is not int or not 1<=q<=m:
        raise ValueError('bounded adjacency and capacity')
    ledger.tick('flow_queries')
    ledger.scan('flow_remove_pinned_anchors',m)
    free=[(i,aid) for i,aid in enumerate(ids) if aid not in anchors]
    f=len(free)
    sink=1+f+k
    nodes=sink+1
    ledger.outer.tick('full_vector_node_visits',nodes)
    graph=[[] for _ in range(nodes)]
    def edge(u,v,cap):
        ledger.outer.tick('graph_arc_initialization_writes',2)
        graph[u].append([v,len(graph[v]),cap])
        graph[v].append([u,len(graph[u])-1,0])
    target_edges={}
    ledger.scan('flow_build_free_member_edges',f)
    ledger.scan('flow_adjacency_entries',f*k)
    for j,(i,aid) in enumerate(free):
        node=1+j
        edge(0,node,1)
        target_edges[aid]=[]
        for target in range(k):
            if adjacency[i][target]:
                index=len(graph[node])
                edge(node,1+f+target,1)
                target_edges[aid].append((target,node,index))
    for target in range(k):edge(1+f+target,sink,q-1)
    ledger.scan('flow_graph_arc_count',nodes)
    arcs=sum(map(len,graph))
    if nodes>133 or arcs>1030:raise ValueError('R9 declared graph bound')
    ledger.data['flow_graphs'].append(dict(q=q,nodes=nodes,residual_arcs=arcs,aircraft_anchor_edges=sum(len(v) for v in target_edges.values())))
    delivered=0
    while delivered<f:
        ledger.scan('flow_BFS_fresh_visited_array',nodes)
        ledger.outer.tick('full_vector_node_visits',nodes)
        prev=[None]*nodes
        prev[0]=(-1,-1)
        queue=deque([0])
        while queue and prev[sink] is None:
            u=queue.popleft()
            for ei,e in enumerate(graph[u]):
                ledger.tick('residual_edge_inspections')
                if e[2]>0 and prev[e[0]] is None:
                    prev[e[0]]=(u,ei)
                    if e[0]==sink:break
                    queue.append(e[0])
        if prev[sink] is None:
            ledger.data['flow_results'].append(dict(q=q,feasible=False,delivered=delivered,required=f))
            return None
        v=sink
        while v:
            u,ei=prev[v]
            # Frozen local recipe uses one combined trace/update pass.
            ledger.outer.tick('residual_edge_inspections')
            ledger.outer.tick('augmentation_capacity_writes',2)
            e=graph[u][ei]
            e[2]-=1
            graph[v][e[1]][2]+=1
            v=u
        delivered+=1
        ledger.tick('augmentations')
    result={aid:anchors.index(aid) for aid in anchors}
    ledger.scan('flow_pinned_assignment',k)
    ledger.scan('flow_extract_matching_edges',sum(len(v) for v in target_edges.values()))
    for aid,edges in target_edges.items():
        ledger.outer.tick('residual_edge_inspections',len(edges))
        targets=[t for t,u,ei in edges if graph[u][ei][2]==0]
        if len(targets)!=1:raise AssertionError('one complete assignment')
        result[aid]=targets[0]
    ledger.data['flow_results'].append(dict(q=q,feasible=True,delivered=delivered,required=f))
    ledger.scan('flow_complete_assignment_vector',m)
    return [result[aid] for aid in ids]


def minimum_capacity(member_ids,anchor_ids,adjacency,ledger):
    m,k=len(member_ids),len(anchor_ids)
    if not 1<=k<=3 or not k<=m<=128:raise ValueError('bounded R9 region')
    lo,hi=(m+k-1)//k,m
    assignment=feasible(member_ids,anchor_ids,adjacency,hi,ledger)
    if assignment is None:return None
    best={hi:assignment}
    while lo<hi:
        mid=(lo+hi)//2
        assignment=feasible(member_ids,anchor_ids,adjacency,mid,ledger)
        if assignment is None:lo=mid+1
        else:
            hi=mid
            best[mid]=assignment
    return dict(capacity=lo,assignment=best[lo])
