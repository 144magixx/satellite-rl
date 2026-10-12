"""Extract existing signed R1 points; never constructs or solves a layout."""
from dataclasses import dataclass
from pathlib import Path
from functools import cmp_to_key
import json,hashlib
import numpy as np
from ..ai_search.core import Layout,layout_sha256,metrics
from .validation import validate_owned

def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def digest(x):return hashlib.sha256(canonical(x)).hexdigest()
def file_sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def read_pinned(ref):
    if file_sha(ref['path'])!=ref['sha256']:raise ValueError('source SHA mismatch')
    return json.loads(Path(ref['path']).read_text(encoding='utf8'))

@dataclass(frozen=True)
class CommonStart:
    frozen_bytes: bytes
    @property
    def record(self):return json.loads(self.frozen_bytes)
    @property
    def archive(self):return self.record['archive']
    @property
    def primary_parent(self):return Layout.from_record(self.record['primary_parent']['layout'])
    @property
    def slice_ref(self):return self.record['slice_ref']
    @property
    def input_sha(self):return self.record['input_sha']
    @property
    def archive_sha(self):return self.record['archive_sha']
    @property
    def parent_sha(self):return self.record['parent_sha']
    @property
    def acquisition_cost_refs(self):return self.record['acquisition_cost_refs']
    def to_record(self):return self.record
    @classmethod
    def from_record(cls,record):
        if digest(record['archive'])!=record['archive_sha']:raise ValueError('archive SHA mismatch')
        for row in record['archive']:
            if layout_sha256(Layout.from_record(row['layout']))!=row['sha256']:raise ValueError('archive layout SHA mismatch')
        if layout_sha256(Layout.from_record(record['primary_parent']['layout']))!=record['parent_sha']:raise ValueError('parent SHA mismatch')
        if not any(canonical(x)==canonical(record['primary_parent']) for x in record['archive']):raise ValueError('parent outside archive or metrics differ')
        if layout_sha256(Layout.from_record(record['slice_ref']['layout']))!=record['slice_ref']['sha256']:raise ValueError('slice layout SHA mismatch')
        if digest(record['slice_ref'])!=record['slice_content_sha']:raise ValueError('slice content SHA mismatch')
        return cls(canonical(record))

def freeze_common(scene,source_refs,geometry=None):
    from ..ai_search.config import GeometryConfig,SearchConfig
    from ..ai_search.search import compare_metrics,metric_tolerance
    geometry=geometry or GeometryConfig();sources=[];points=[];slices=[]
    allowed={'07-CV','07-LEX','JOINT-CV','JOINT-LEX','POST_SHRINK'}
    if len(source_refs)!=5 or {x['method'] for x in source_refs}!=allowed:raise ValueError('all five signed final sources required')
    for ref in source_refs:
        method=ref['method']
        if method not in allowed:raise ValueError('unsupported R1 source')
        raw=read_pinned(ref)
        if raw.get('method')!=method:raise ValueError('method pin differs')
        if raw.get('input_sha256')!=scene.input_sha256:raise ValueError('source scene differs')
        if not ref.get('audit_path') or not ref.get('audit_sha256'):raise ValueError('signed source audit required')
        audit=read_pinned({'path':ref['audit_path'],'sha256':ref['audit_sha256']})
        if not audit.get('passed') or audit.get('issues') or audit.get('source_case_sha256')!=ref['sha256']:raise ValueError('source audit not closed')
        r=raw['result'];nodes=[]
        if r.get('layout') is not None:nodes.append({'layout':r['layout']})
        else:raise ValueError('source final absent')
        sr=r.get('slice_reference')
        if sr is not None:slices.append(sr)
        for node in nodes:
            rec=dict(node['layout'])
            if 'widths_deg' not in rec and 'widths' not in rec:rec['widths_deg']=[.5]*len(rec['centers'])
            layout=Layout.from_record(rec);e=validate_owned(scene,layout,geometry)
            if not e['valid']:raise ValueError('signed layout invalid: '+str(e['reasons']))
            sh=layout_sha256(layout);values=metrics(scene,layout,geometry,e)
            points.append(dict(layout=layout.to_record(),metrics=values,sha256=sh,sources=[{'method':method,'path':ref['path'],'sha256':ref['sha256']}]))
        sources.append({k:v for k,v in ref.items() if k not in ('rows',)})
    if not points:raise ValueError('no complete existing initial')
    if not slices:raise ValueError('R1 unshrunk .5 slice reference required')
    slice_layouts=[Layout.from_record(x['layout']) for x in slices]
    if len({layout_sha256(x) for x in slice_layouts})!=1:raise ValueError('slice reference not common')
    sl=slice_layouts[0];ev=validate_owned(scene,sl,geometry)
    if not ev['valid']:raise ValueError('invalid resource slice')
    sm=metrics(scene,sl,geometry,ev);cfg=SearchConfig();archive=sorted(points,key=lambda x:(x['sha256'],x['sources'][0]['method']))
    feasible=[p for p in archive if p['metrics']['N']<=sm['N'] and p['metrics']['solid_angle_sum_sr']<=sm['solid_angle_sum_sr']+metric_tolerance('solid_angle_sum_sr',p['metrics']['solid_angle_sum_sr'],sm['solid_angle_sum_sr'],cfg)]
    if not feasible:raise ValueError('common resource slice empty')
    def compare(a,b):
        for f in ('maxload','top10_mean','CV','N','solid_angle_sum_sr'):
            x,y=a['metrics'][f],b['metrics'][f]
            if x is None or y is None:continue
            tol=metric_tolerance(f,x,y,cfg)
            if x<y-tol:return -1
            if x>y+tol:return 1
        return (a['sha256']>b['sha256'])-(a['sha256']<b['sha256'])
    parent=min(feasible,key=cmp_to_key(compare))
    slice_ref=dict(layout=sl.to_record(),sha256=layout_sha256(sl),metrics=sm)
    return CommonStart.from_record(dict(schema='s1_ai18_common_start_v1',input_sha=scene.input_sha256,archive=archive,archive_sha=digest(archive),primary_parent=parent,parent_sha=parent['sha256'],slice_ref=slice_ref,slice_content_sha=digest(slice_ref),acquisition_cost_refs=sources,acquisition_scope='reuse_R1_audited_points_no_new_solve'))
