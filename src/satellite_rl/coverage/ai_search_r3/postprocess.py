"""Shared fixed-center assignment polish then conservative width envelope."""
import time
import numpy as np
from .pool import omega
from ..ai_search.core import Layout,metrics,minimum_enclosing_width
from ..ai_search_r2.validation import validate_owned
from ..geometry import satellite_rays_numpy

def metric_key(scene,layout):
    m=metrics(scene,layout,evidence=validate_owned(scene,layout))
    return (m['maxload'],m['top10_mean'],m['CV'] or 0,m['N'],m['solid_angle_sum_sr'])

def lex_better(first,second):
    tolerances=(1e-6,1e-6,1e-9,0,1e-12)
    for a,b,t in zip(first,second,tolerances):
        delta=a-b;tol=t+1e-12*max(abs(a),abs(b))
        if delta < -tol:return True
        if delta > tol:return False
    return False

def polish_envelope(p,layout):
    started=time.monotonic();owner=layout.assignment.copy();n=len(layout.centers);proposals=0
    rays=satellite_rays_numpy(layout.centers,p.scene.profile)
    best=metric_key(p.scene,layout)
    for _ in range(2):
        loads=np.bincount(owner,weights=p.scene.demands,minlength=n)
        for b in sorted(range(n),key=lambda j:(-loads[j],layout.beam_ids[j])):
            members=sorted(np.flatnonzero(owner==b),key=lambda i:(-p.scene.demands[i],p.scene.aircraft_ids[i]))
            for i in members:
                source=int(owner[i])
                if proposals>=512 or time.monotonic()-started>=.25:break
                if np.count_nonzero(owner==source)<=1:continue
                chord=np.sum((rays-p.scene.rays[i])**2,axis=1)
                legal=chord<=4*np.sin(np.deg2rad(layout.widths/2)/2)**2+1e-14
                for dest in sorted(np.flatnonzero(legal),key=lambda j:(loads[j],layout.beam_ids[j])):
                    if dest==source:continue
                    if proposals>=512 or time.monotonic()-started>=.25:break
                    proposals+=1;owner[i]=dest
                    candidate=Layout(layout.centers,layout.widths,owner,layout.beam_ids);key=metric_key(p.scene,candidate)
                    if lex_better(key,best):best=key;loads=np.bincount(owner,weights=p.scene.demands,minlength=n);break
                    owner[i]=source
    menu=Layout(layout.centers,layout.widths,owner,layout.beam_ids);widths=[]
    for b in range(n):
        width=minimum_enclosing_width(p.scene,np.flatnonzero(owner==b),layout.centers[b])
        if width is None or width>layout.widths[b]+1e-12:raise ValueError('envelope_exceeds_selected_menu')
        widths.append(min(width,layout.widths[b]))
    final=Layout(layout.centers,widths,owner,layout.beam_ids);audit=validate_owned(p.scene,final)
    if not audit['valid']:raise ValueError('final_independent_validation:'+str(audit['reasons']))
    return final,dict(proposals=proposals,seconds=time.monotonic()-started,menu_layout=menu.to_record(),
        menu_Omega=float(omega(menu.widths).sum()),actual_Omega=float(omega(final.widths).sum()),audit=audit)
