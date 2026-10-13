"""Typed finite-domain labels and input-only normalization; no solver calls."""
import numpy as np
import time
from .evaluate import physical_key
from .postprocess import metric_key,polish_envelope,lex_better
from ..ai_search.core import Layout

def build_teacher_label(p,reference_result):
    started=time.monotonic()
    if not reference_result.get('valid_reference_labels',reference_result.get('valid_reference')):return dict(status='label_unavailable',reason='no_native_reference_or_certified_start')
    points=[];seen=set()
    for raw in reference_result.get('reference_labels',[]):
        layout=Layout.from_record(raw['layout']);key=physical_key(layout)
        if key in seen:continue
        seen.add(key);final,_=polish_envelope(p,layout);points.append((raw,metric_key(p.scene,final)))
    if not points or not p.scene.demands.sum():return dict(status='label_unavailable',reason='no_label_or_zero_total_request')
    best=points[0]
    for point in points[1:]:
        if lex_better(point[1],best[1]):best=point
    best_peak=min(k[0] for _,k in points)
    accepted=[]
    for raw,key in points:
        tail_ok=key[1]<=1.02*best[1][1]+1e-6
        cv_ok=key[2]<=1.02*best[1][2]+1e-9
        if key[0]<=1.01*best_peak+1e-6 and tail_ok and cv_ok:accepted.append(raw)
    accepted=accepted[:2]
    targets=np.zeros((len(p.centers),3,3),np.float32)
    for raw in accepted:
        count=np.zeros((len(p.centers),3),np.int64)
        ids=np.asarray(raw['active_slot_ids'],np.int64)
        np.add.at(count,(ids//6,(ids//2)%3),1)
        if (count>2).any():raise ValueError('teacher_copy_limit')
        targets+=np.eye(3,dtype=np.float32)[count]/len(accepted)
    return dict(status='label_available',targets=targets.tolist(),active_slot_ids=best[0]['active_slot_ids'],
        accepted_points=accepted,unique_point_count=len(accepted),reference_type=best[0]['reference_type'],
        label_generation_seconds=time.monotonic()-started,offline_cost_separate=True)

def fit_normalization(feature_arrays):
    # Caller supplies only frozen training inputs, no development/test features.
    values=np.concatenate(feature_arrays,axis=0).astype(np.float64)
    mean=values.mean(0);std=values.std(0);constant=std<1e-8
    return mean.astype(np.float32),np.maximum(std,1e-8).astype(np.float32),constant

def weighted_soft_loss(logits,targets):
    import torch
    weights=logits.new_tensor([.1,1.,1.]);weighted=targets*weights
    return -(weighted*torch.log_softmax(logits,-1)).sum()/weighted.sum().clamp_min(1e-30)
