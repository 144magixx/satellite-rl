"""Deterministic bounded integer flow; no geometry/model/producer imports."""
from collections import deque
import copy

class ResourceUnknown(RuntimeError):
    """A guard/counter interruption is missing work, never infeasibility."""
    def __init__(self,reason,partial_ledger=None):
        super().__init__(reason);self.partial_ledger=copy.deepcopy(partial_ledger)

class Ledger:
    CAPS={'flow_queries':8,'residual_edge_inspections':1048576,'anchor_selection_passes':4}
    def __init__(self,n):
        if not isinstance(n,int) or n<1:raise ValueError('full positive scene n')
        self.n=n;self.data=dict(raypairs=0,flow_queries=0,residual_edge_inspections=0,anchor_selection_passes=0,augmentations=0,flow_graphs=[],flow_results=[],center_constructions=0,cached_member_scans=0,mean_input_count=0,sort_input_count=0,scan_input_count=0,scan_passes={},sort_passes={})
    def tick(self,key,count=1):
        if not isinstance(count,int) or count<0:raise ValueError('nonnegative exact count')
        value=self.data.get(key,0)+count
        if key in self.CAPS and value>self.CAPS[key]:raise ResourceUnknown(key+'_cap',self.record())
        self.data[key]=value
    def pairs(self,count,audit=False):
        limit=32*self.n if audit else 29*self.n
        if self.data['raypairs']+count>limit:raise ResourceUnknown('raypairs_cap_audit_reserve',self.record())
        self.tick('raypairs',count)
    def _pass(self,kind,label,count):
        """Logical source traversal input items, not machine instructions.

        NumPy primitives are one named logical pass each. Their compiler-level
        internal loops are not claimed measured. Aggregate by label, not BFS log.
        """
        self.tick(kind+'_input_count',count);value=self.data[kind+'_passes'].setdefault(label,dict(calls=0,items=0));value['calls']+=1;value['items']+=count
    def scan(self,label,count):self._pass('scan',label,count)
    def sort(self,label,count):self._pass('sort',label,count)
    def record(self):return {**self.data,'raypairs_max':32*self.n,'audit_reserved_raypairs':3*self.n,'caps':dict(self.CAPS)}

def feasible(member_ids,anchor_ids,adjacency,q,ledger):
    """Fresh graph each query; own anchors pinned, remaining capacities q-1."""
    ids=tuple(member_ids);anchors=tuple(anchor_ids);m=len(ids);k=len(anchors)
    ledger.scan('flow_ID_shape_validation',m+k);ledger.sort('flow_verify_member_order',m);ledger.sort('flow_verify_anchor_order',k)
    if not (all(isinstance(x,str) and x for x in ids+anchors) and 1<=k<=3 and k<=m<=128 and ids==tuple(sorted(ids)) and anchors==tuple(sorted(anchors)) and len(set(ids))==m and len(set(anchors))==k and set(anchors)<=set(ids)):raise ValueError('stable sorted distinct string member/anchor IDs')
    if len(adjacency)!=m or any(len(row)!=k for row in adjacency) or not isinstance(q,int) or not 1<=q<=m:raise ValueError('bounded adjacency and capacity')
    ledger.tick('flow_queries');ledger.scan('flow_remove_pinned_anchors',m);free=[(i,aid) for i,aid in enumerate(ids) if aid not in anchors];f=len(free);sink=1+f+k;graph=[[] for _ in range(sink+1)]
    def edge(u,v,cap):
        graph[u].append([v,len(graph[v]),cap]);graph[v].append([u,len(graph[u])-1,0])
    target_edges={}
    ledger.scan('flow_build_free_member_edges',f);ledger.scan('flow_adjacency_entries',f*k)
    for j,(i,aid) in enumerate(free):
        node=1+j;edge(0,node,1);target_edges[aid]=[]
        for target in range(k):
            if adjacency[i][target]:
                index=len(graph[node]);edge(node,1+f+target,1);target_edges[aid].append((target,node,index))
    for target in range(k):edge(1+f+target,sink,q-1)
    ledger.scan('flow_graph_arc_count',len(graph));arcs=sum(map(len,graph));nodes=len(graph)
    if nodes>133 or arcs>1030:raise ValueError('flow graph declared bound')
    ledger.data['flow_graphs'].append(dict(q=q,nodes=nodes,residual_arcs=arcs,aircraft_anchor_edges=sum(len(v) for v in target_edges.values())))
    delivered=0
    while delivered<f:
        ledger.scan('flow_BFS_fresh_visited_array',nodes)
        prev=[None]*nodes;prev[0]=(-1,-1);queue=deque([0])
        while queue and prev[sink] is None:
            u=queue.popleft()
            for ei,e in enumerate(graph[u]):
                ledger.tick('residual_edge_inspections')
                if e[2]>0 and prev[e[0]] is None:
                    prev[e[0]]=(u,ei)
                    if e[0]==sink:break
                    queue.append(e[0])
        if prev[sink] is None:
            ledger.data['flow_results'].append(dict(q=q,feasible=False,delivered=delivered,required=f));return None
        v=sink
        while v:
            u,ei=prev[v];e=graph[u][ei];e[2]-=1;graph[v][e[1]][2]+=1;v=u
        delivered+=1;ledger.tick('augmentations')
    result={aid:anchors.index(aid) for aid in anchors}
    ledger.scan('flow_pinned_assignment',k);ledger.scan('flow_extract_matching_edges',sum(len(v) for v in target_edges.values()))
    for aid,edges in target_edges.items():
        targets=[t for t,u,ei in edges if graph[u][ei][2]==0]
        if len(targets)!=1:raise AssertionError('one complete assignment')
        result[aid]=targets[0]
    ledger.data['flow_results'].append(dict(q=q,feasible=True,delivered=delivered,required=f))
    ledger.scan('flow_complete_assignment_vector',m);return [result[aid] for aid in ids]

def minimum_capacity(member_ids,anchor_ids,adjacency,ledger):
    m=len(member_ids);k=len(anchor_ids)
    if not 1<=k<=3 or not k<=m<=128:raise ValueError('bounded region')
    lo=(m+k-1)//k;hi=m;best={};assignment=feasible(member_ids,anchor_ids,adjacency,hi,ledger)
    if assignment is None:return None
    best[hi]=assignment
    while lo<hi:
        mid=(lo+hi)//2;assignment=feasible(member_ids,anchor_ids,adjacency,mid,ledger)
        if assignment is None:lo=mid+1
        else:hi=mid;best[mid]=assignment
    return dict(capacity=lo,assignment=best[lo])
