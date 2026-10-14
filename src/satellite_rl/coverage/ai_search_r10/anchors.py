"""Frozen R9 anchor mathematics with owned sort comparison observation."""
import numpy as np
from ..ai_search_r9.core import angles


def choose_anchors(scene,member_ids,k,ledger):
    ledger.scan('anchor_plane_index',len(scene.aircraft_ids))
    ledger.scan('anchor_member_ray_gather',len(member_ids))
    ledger.scan('anchor_mean_ray_sum',len(member_ids))
    pindex={a:i for i,a in enumerate(scene.aircraft_ids)}
    ids=tuple(ledger.ordered(list(member_ids),lambda a:a,'anchor_member_ID_order'))
    rays=scene.rays[[pindex[a] for a in ids]]
    mean=np.sum(rays,axis=0,dtype=np.float64)
    norm=float(np.linalg.norm(mean))
    ledger.tick('mean_input_count',len(ids))
    if norm<1e-12:return None
    ledger.tick('anchor_selection_passes')
    v=angles(rays,mean/norm,ledger)
    ledger.scan('anchor_mean_argmin',len(ids))
    meanidx=min(range(len(ids)),key=lambda i:(float(v[i]),ids[i]))
    if k==1:return [ids[meanidx]]
    ledger.tick('anchor_selection_passes')
    v=angles(rays,rays[meanidx],ledger)
    ledger.scan('anchor_first_argmax',len(ids))
    first=min(range(len(ids)),key=lambda i:(-float(v[i]),ids[i]))
    ledger.tick('anchor_selection_passes')
    ledger.scan('anchor_second_domain',len(ids))
    remaining=[i for i in range(len(ids)) if i!=first]
    v=angles(rays[remaining],rays[first],ledger)
    ledger.scan('anchor_second_argmax',len(remaining))
    second=remaining[min(range(len(remaining)),key=lambda j:(-float(v[j]),ids[remaining[j]]))]
    selected=[first,second]
    if k==3:
        ledger.tick('anchor_selection_passes')
        ledger.scan('anchor_third_domain',len(ids))
        remaining=[i for i in range(len(ids)) if i not in selected]
        v=angles(rays[remaining,None,:],rays[None,selected,:],ledger)
        ledger.scan('anchor_nearest_selected_min',2*len(remaining))
        ledger.scan('anchor_third_argmax',len(remaining))
        third=min(range(len(remaining)),key=lambda j:(-float(v[j].min()),ids[remaining[j]]))
        selected.append(remaining[third])
    return ledger.ordered([ids[i] for i in selected],lambda a:a,'anchor_final_target_ID_order')
