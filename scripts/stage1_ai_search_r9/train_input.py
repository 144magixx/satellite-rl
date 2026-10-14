"""New train-only acquisition caller for unchanged original RULE and D5 APIs."""
import time
from .contracts import pin,digest

def load_original(case):
 if case['identity'].get('role')!='train' or case.get('uniform_request_bps')!=21600000:raise ValueError('new train-only RULE acquisition')
 import numpy as np
 from satellite_rl.coverage.ai_search.core import Scene
 from satellite_rl.coverage.ai_search_r5.core import PreparedCase
 from satellite_rl.workbench.r6_io import read_asset
 began=time.monotonic()
 with np.load(pin(case['scene_ref']),allow_pickle=False) as z:
  rawids=z['aircraft_ids'].tolist()
  if not all(isinstance(x,str) and x for x in rawids) or len(set(rawids))!=len(rawids):raise ValueError('original stable strings, no coercion')
  if not np.all(z['demands']==21600000):raise ValueError('uniform original request')
  scene=Scene(z['positions'],z['demands'],tuple(rawids),str(z['profile']))
 if scene.input_sha256!=case['identity']['input_sha256']:raise ValueError('original raw input SHA')
 prepared=PreparedCase.from_dict(scene,read_asset(case['prepared_ref'])['result'])
 if prepared.identity!=case['identity'] or prepared.status!='available' or prepared.Ncap!=case['Ncap'] or prepared.Omega_cap!=case['Omega_cap']:raise ValueError('original identity/caps/prepared publication')
 sizes={k:pin(case[k]).stat().st_size for k in ('scene_ref','prepared_ref','shared_case_ref')}
 return scene,prepared,dict(original_input_read_scene_ray_seconds=time.monotonic()-began,scene_ray_constructions=len(rawids),input_logical_bytes=sizes,source_refs={k:case[k] for k in ('scene_ref','prepared_ref','shared_case_ref')},IO_and_scene_geometry_included=True,pin_SHA_reads_included=True,logical_input_bytes_not_total_OS_read_bytes=True)
def rule(prepared,prepared_ref):
 from satellite_rl.coverage.ai_search_r5.core import build_proposal
 value=build_proposal(prepared,'RULE',producer_sha256=digest(__file__),evidence_refs=[prepared_ref])
 if value.get('status')!='ready' or value['construction_cost']['raypairs']>200*len(prepared.scene.demands):raise ValueError('original RULE publication or200n limit')
 return value
def d5(prepared,proposal):
 from satellite_rl.coverage.ai_search_r5.core import decode_groups
 return decode_groups(prepared,proposal,time_limit=5.)
def layout_delta(layout,origin):
 if layout is None:return None
 old={bid:i for i,bid in enumerate(origin['beam_ids'])};new={bid:i for i,bid in enumerate(layout['beam_ids'])};beams=[]
 for bid,j in new.items():
  row=dict(beam_id=bid,anchor_aircraft_id=layout['anchor_aircraft_ids'][j],center=layout['centers'][j],width_deg=layout['widths_deg'][j])
  k=old.get(bid)
  if k is None or row!=dict(beam_id=bid,anchor_aircraft_id=origin['anchor_aircraft_ids'][k],center=origin['centers'][k],width_deg=origin['widths_deg'][k]):beams.append(row)
 owners=[]
 for i,j in enumerate(layout['assignment']):
  bid=layout['beam_ids'][j]
  if bid!=origin['beam_ids'][origin['assignment'][i]]:owners.append([i,bid])
 return dict(schema='R9ExactLayoutDeltaV1',beam_order=list(layout['beam_ids']),changed_beams=beams,changed_aircraft_owners=owners,aircraft_count=len(layout['assignment']))
def restore_layout(delta,origin):
 import copy
 if delta is None:return None
 old={bid:i for i,bid in enumerate(origin['beam_ids'])};changes={x['beam_id']:x for x in delta['changed_beams']};ids=delta['beam_order'];new={bid:i for i,bid in enumerate(ids)};centers=[];widths=[];anchors=[]
 for bid in ids:
  if bid in changes:row=changes[bid];centers.append(row['center']);widths.append(row['width_deg']);anchors.append(row['anchor_aircraft_id'])
  else:j=old[bid];centers.append(origin['centers'][j]);widths.append(origin['widths_deg'][j]);anchors.append(origin['anchor_aircraft_ids'][j])
 patches=dict(delta['changed_aircraft_owners']);owners=[]
 for i,j in enumerate(origin['assignment']):owners.append(new[patches[i] if i in patches else origin['beam_ids'][j]])
 if len(owners)!=delta['aircraft_count']:raise ValueError('delta original denominator')
 return copy.deepcopy(dict(beam_ids=ids,anchor_aircraft_ids=anchors,centers=centers,widths_deg=widths,assignment=owners))
