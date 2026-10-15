"""Admitted independent full12 collection, no producer numerical imports."""
import math
import json
from datetime import datetime,timezone
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from .persistence import Journal, content_sha, identity
from .reader import Reader, Incomplete, gate
from .geometry_reader import GeometryAuditor, need
from .exact_order import key as exact_key,counts_widths_evidence
from .region_v2 import reader_pair_reservation

def reconstruct_rays(positions, physical):
    """Independent original lat/lon projection using frozen numeric constants."""
    a=np.asarray(positions,dtype=np.float64)
    need(a.ndim==2 and a.shape[1]==2 and np.isfinite(a).all(),'full latlon')
    er=float(physical['earth_radius_km']);orbit=float(physical['orbit_radius_km'])
    slon=np.deg2rad(float(physical['satellite_longitude_deg']))
    need(0<er<orbit and all(math.isfinite(v) for v in (er,orbit,float(slon))),'physical projection constants')
    lat,lon=np.deg2rad(a[:,0]),np.deg2rad(a[:,1])
    surface=er*np.stack((np.cos(lat)*np.cos(lon),np.cos(lat)*np.sin(lon),np.sin(lat)),axis=-1)
    satellite=orbit*np.array([np.cos(slon),np.sin(slon),0.])
    rays=surface-satellite
    return rays/np.linalg.norm(rays,axis=-1,keepdims=True)

def objective(m):return tuple(m[k] for k in ('maxload','top10_mean','CV','N','actual_Omega'))
def hit(a,b):
    return b['maxload']>0 and a['maxload']<=.99*b['maxload']+1e-6+1e-12*abs(b['maxload']) and all(a[k]<=b[k]+(1e-9 if k=='CV' else 1e-6)+1e-12*abs(b[k]) for k in ('top10_mean','CV'))

def controls(origin,tree,saved,origin_exact):
    paths=tree['paths'];nodes={n['semantic_SHA']:n for n in tree['nodes']}
    best=lambda choices:min(choices,key=lambda p:exact_key(paths[p]['exact_evidence'])+(p,))
    g1=best(['0','1']);g=best([g1+'0',g1+'1']);h=''
    for _ in range(2):
        children=[h+'0',h+'1']
        eligible=[p for p in children if nodes[paths[p]['semantic_SHA']]['descriptor']['structurally_eligible']]
        h=min(eligible,key=lambda p:tuple(nodes[paths[p]['semantic_SHA']]['descriptor']['rank'])) if eligible else children[0]
    S=min(('S2','SH-C'),key=lambda k:objective(saved[k]['metrics'])+(('S2','SH-C').index(k),))
    arms=dict(S=saved[S],H=paths[h],G=paths[g])
    q=min(('S','H','G'),key=lambda k:exact_key(arms[k]['exact_evidence'])+(('S','H','G').index(k),))
    o1=min([dict(metrics=origin['metrics'],terminal=origin['layout'],exact_evidence=origin_exact)]+[paths[k] for k in ('0','1')],key=lambda x:exact_key(x['exact_evidence']))
    def logical(keys):
        semantics=sorted({paths[p]['semantic_SHA'] for p in keys})
        return dict(path_references=keys,unique_semantic_SHAs=semantics,
                    kernels=sum(int(nodes[s]['kernel_entered']) for s in semantics),
                    flows=sum(nodes[s]['ledger']['flow_queries'] for s in semantics))
    costs=dict(H=logical([h[:1],h]),G=logical(['0','1',g1+'0',g1+'1']))
    costs.update({'F'+p:logical([p[:1],p]) for p in ('00','01','10','11')})
    return dict(**arms,Q=arms[q],O1=o1,O2=paths[best(['00','01','10','11'])],
                F={p:paths[p] for p in ('00','01','10','11')},logical_costs=costs)

def collect(plan,archive):
    from .worker import pinned,load_case,audited_S_rows,_markers
    from .runtime_epoch import stopped
    from .historical_s import audited_selected
    if stopped(plan,force=True):raise RuntimeError('pause before independent reader')
    def stop():return stopped(plan)
    root=Path(archive);journal=Journal(root/'reader');reader=Reader(journal,prior_seconds=plan['reader_prior_seconds'],stop=stop);rows=[]
    production=dict(kernels=0,flow_queries=0,raypairs=0,new_rays=0,seconds=0.,aircraft_records=0)
    try:
        control_ref=plan['historical_control_ref']
        need(control_ref['sha256']=='804dd2e1cbaa264ceda230d153bb2e5b29fb10ab0703c7745aeef1cac1ec49e5','sole old complete804 source')
        historical=json.loads(pinned(control_ref))
        audited_S=audited_S_rows(plan)
        from .admission import INDICES
        need([r['original_index'] for r in historical['rows']]==list(INDICES),'complete original oldcontrol12')
        for row_position,case in enumerate(plan['cases']):
            reader.boundary()
            index=case['original_index'];record=load_case(case['input_adapter_ref'])
            raw=record['scene'];n=len(raw['aircraft_ids'])
            need('rays' not in raw,'IO-only normalization must not precompute rays')
            need(record.get('original_index')==index and raw.get('input_sha256')==record['origin']['identity']['input_sha256'],'original case identity')
            need(record['origin']['identity'].get('role')=='train' and record['origin']['identity'].get('profile')=='uniform21600000','train diagnostic fixed request role')
            production['aircraft_records']+=n
            old=historical['rows'][row_position]
            need(old['common_state']=='complete' and old['origin_metrics']==record['origin']['metrics'] and old['original_aircraft_count']==n,'original full old control row binding')
            for method in ('S2','SH-C'):
                extracted=record['saved_controls'][method];actual=old['arms'][method]
                need(actual['state']=='complete' and actual['whole_method_unknown'] is False,'complete old strongcontrol')
                need(extracted['metrics']==actual['metrics'] and extracted['state']==actual['state'] and extracted['layout_SHA']==actual['layout_SHA'],'extracted original metrics/state/layoutSHA exact')
                need(identity(extracted['source_control_ref'])==identity(control_ref) and extracted['original_index']==index and extracted['original_method']==method,'old control original source content binding')
                need(extracted['historical_cost_ref']['json_path']==['rows',row_position,'arms',method,'ledger'] and 'ledger' in actual,'old paid cost source retained without replay')
            selected_S=audited_selected(audited_S[index],record)
            reader.boundary()
            selected_S['exact_evidence']=counts_widths_evidence(selected_S['counts'],selected_S['widths_deg'])
            record['saved_controls'][selected_S['method']]=selected_S
            # Charge ray projection BEFORE execution and persist as an inflight
            # operation: a failed projection/IO cannot be silently redone.
            key='scene_'+str(index)
            binding=dict(input_content_identity=identity(record['source_original_input_ref']),
                         source_content_identity=identity(plan['source_manifest_ref']),
                         physical_content_SHA=content_sha(record['physical_parameters']),
                         source_scene_content_SHA=content_sha(raw))
            if journal.has(key):
                checkpoint=journal.read(key)
                need(checkpoint.get('checkpoint_content_SHA')==content_sha({k:v for k,v in checkpoint.items() if k!='checkpoint_content_SHA'}),'scene checkpoint content identity')
                need(content_sha(checkpoint.get('binding'))==content_sha(binding),'scene checkpoint original source/input/physics binding')
                scene_data=checkpoint['scene']
                need(all(scene_data[k]==raw[k] for k in ('aircraft_ids','positions','demands','input_sha256')),'scene checkpoint full original arrays')
            else:
                if reader.budget['inflight'] or reader.budget['rays']+n>4715:raise Incomplete('reader ray budget/inflight')
                reader.budget['rays']+=n;reader.budget['inflight']=key;reader._persist_budget()
                reader.boundary()
                rays=reconstruct_rays(raw['positions'],record['physical_parameters'])
                scene_data=dict(raw,rays=rays.tolist())
                checkpoint=dict(schema='R12IndependentSceneGeometryV1',binding=binding,scene=scene_data)
                checkpoint['checkpoint_content_SHA']=content_sha(checkpoint)
                journal.write(key,checkpoint)
                reader.budget['inflight']=None;reader._persist_budget()
            for k in ('positions','rays','demands'):scene_data[k]=np.asarray(scene_data[k],dtype=np.float64)
            scene_data['aircraft_ids']=tuple(scene_data['aircraft_ids']);scene=SimpleNamespace(**scene_data)
            producer_geometry=json.loads((root/('mother_'+str(index))/'scene_preparation_completed.json').read_text(encoding='utf8'))
            need(producer_geometry.get('new_rays')==n and producer_geometry.get('physical_parameters')==record['physical_parameters'],'producer one charged original ray construction')
            need(identity(producer_geometry['original_source_input_ref'])==identity(record['source_original_input_ref']),'producer original scene input content')
            need(np.array_equal(scene.rays,np.asarray(producer_geometry['rays'])),'independent rays equal production rays')
            production['new_rays']+=producer_geometry['new_rays']
            native=json.loads((root/('mother_'+str(index))/'native_exit.json').read_text(encoding='utf8'))
            need(native.get('complete') is True and native.get('exit_code')==0,'native exit evidence')
            receipt=json.loads((root/'batch'/('mother_'+str(index)+'.json')).read_text(encoding='utf8'))
            need(receipt.get('exit_code')==0 and receipt.get('reason') is None,'parent reaped native exit')
            binding=json.loads((root/'batch'/'runtime_binding.json').read_text(encoding='utf8'))
            need(receipt.get('RSS_samples',0)>=1 and 0<receipt.get('peak_parent_children_RSS_bytes',0)<=2147483648,'observed parent children RSS')
            need(receipt.get('native_birth_unix')==native.get('native_birth_unix') and receipt.get('native_birth_unix') is not None,'exact native process birth')
            need(receipt.get('native_pid')==native.get('native_pid') and receipt.get('parent_pid')==native.get('parent_pid')==binding['actual_pid'],'native PID and parent binding')
            need(receipt.get('boot_id')==native.get('boot_id')==binding['boot_id'],'actual boot binding')
            need(0<=receipt['seconds']<=180,'native mother180s including imports/input/save')
            production['seconds']+=receipt['seconds']
            tree=json.loads((root/('mother_'+str(index))/'tree.json').read_text(encoding='utf8'))
            need(tree.get('schema')=='R12PersistedTreeIndexV2' and tree.get('original_tree_schema')=='R12TreeV2','canonical versioned raw-node tree index')
            need(tree.get('complete') is True and tree.get('identity')==record['origin']['identity'] and len(tree['nodes'])<=6 and set(tree['paths'])=={'0','1','00','01','10','11'},'bounded original full tree index before node reads')
            hydrated=[]
            for ref in tree['nodes']:
                need(type(ref.get('semantic_SHA')) is str and len(ref['semantic_SHA'])==64 and all(c in '0123456789abcdef' for c in ref['semantic_SHA']),'bounded canonical semantic reference')
                raw_node=json.loads((root/('mother_'+str(index))/('node_'+ref['semantic_SHA']+'.json')).read_text(encoding='utf8'))
                need(raw_node.get('event')=='completed','canonical durable node complete before index')
                node=raw_node['payload']['node']
                need(node['semantic_SHA']==ref['semantic_SHA'] and node['source_content_id']==ref['source_content_id'],'canonical full node source reference')
                hydrated.append(node)
            tree['nodes']=hydrated
            indexed={n['semantic_SHA']:n for n in hydrated}
            tree['paths']={p:dict(v,terminal=indexed[v['semantic_SHA']]['terminal'],metrics=indexed[v['semantic_SHA']]['metrics']) for p,v in tree['paths'].items()}
            need(tree.get('complete') is True and tree.get('identity')==record['origin']['identity'],'full tree identity')
            need(set(tree['paths'])=={'0','1','00','01','10','11'} and len(tree['nodes'])<=6,'fixed two-step tree')
            bykey={v['semantic_SHA']:v for v in tree['nodes']}
            need(len(bykey)==len(tree['nodes']),'unique semantic nodes')
            limits=dict(flow_queries=48,kernels=6,raypairs=98504+28*n,
                residual_edge_inspections=6*8388608,anchor_selection_passes=6*34,
                graph_arc_initialization_writes=6*65536,augmentation_capacity_writes=6*524288,
                full_vector_node_visits=6*1048576,sort_key_comparisons=6*2000000,scan_input_count=6*16777216)
            for name,limit in limits.items():
                value=tree['ledger'].get(name)
                need(type(value) is int and 0<=value<=limit,'producer mother atomic budget '+name)
            for name in ('kernels','flow_queries','raypairs'):production[name]+=tree['ledger'][name]
            need(production['kernels']<=72 and production['flow_queries']<=576 and production['raypairs']<=1314068 and production['seconds']<=2160,'global production finite budget')
            for node in tree['nodes']:
                for name,limit in limits.items():
                    if name=='kernels':continue
                    maximum=16384+4*n if name=='raypairs' else limit//6
                    value=node['ledger'].get(name)
                    need(type(value) is int and 0<=value<=maximum,'per-node atomic budget '+name)
            current={'':dict(layout=record['origin']['layout'],metrics=record['origin']['metrics'])}
            for path in ('0','1','00','01','10','11'):
                entry=tree['paths'][path];node=bykey[entry['semantic_SHA']]
                need(entry['terminal']==node['terminal'] and entry['metrics']==node['metrics'],'shared path exact reference')
                parent=current[path[:-1]]
                need(node['descriptor']['action']=='J'+path[-1],'path/action binding')
                m=len(node['descriptor']['member_ids']);k=node['descriptor']['target_k']
                # Two uncached layouts at most, region anchor pairs plus mean
                # and singleton neighbors. Reserved totals are conservative;
                # unsupported regions never enter m*k geometry.
                eligible=node['descriptor']['structurally_eligible']
                cache_id=content_sha(record['origin']['identity']);cache_key='geometry_'+cache_id
                old=journal.read(cache_key)['layouts'] if journal.has(cache_key) else {}
                candidates=[parent['layout'],node['terminal']]
                if node.get('raw_layout') is not None and node.get('raw_metrics') is not None:candidates.append(node['raw_layout'])
                new_layouts=len({content_sha(v) for v in candidates}-set(old))
                old_regions=journal.read(cache_key).get('regions',{}) if journal.has(cache_key) else {}
                region_key=content_sha([node['descriptor']['state_SHA'],node['descriptor']['member_ids']])
                region_pairs=m*m if eligible and region_key not in old_regions else 0
                contacts=journal.read(cache_key).get('contacts',{}) if journal.has(cache_key) else {}
                contact_pairs=5100 if node['descriptor']['state_SHA'] not in contacts else 0
                allowance=dict(rays=0,layouts=new_layouts,raypairs=reader_pair_reservation(n,m,
                    new_contact=bool(contact_pairs),new_region=bool(region_pairs),eligible=eligible))
                auditor=GeometryAuditor(scene,record['origin'],content_sha(record['origin']['identity']),parent,stop=stop)
                if stopped(plan,force=True):raise RuntimeError('pause before independent reader node')
                audit_receipt=reader.audit(node,source_ref=plan['source_manifest_ref'],input_ref=record['source_original_input_ref'],
                    parent_cache_SHA=content_sha(record['origin']['identity']),static_allowance=allowance,independent_auditor=auditor)
                tree['paths'][path]['exact_evidence']=audit_receipt['result']['terminal_exact']
                current[path]=dict(layout=node['terminal'],metrics=node['metrics'],exact_evidence=audit_receipt['result']['terminal_exact'])
            origin_exact=journal.read(cache_key)['exact'][content_sha(record['origin']['layout'])]
            derived=controls(record['origin'],tree,record['saved_controls'],origin_exact)
            for arm in ('S','H','G','Q','O1','O2'):
                saved=tree['controls'][arm]
                need(saved['metrics']==derived[arm]['metrics'] and saved['exact_evidence']==derived[arm]['exact_evidence'],'independent exact control ordering '+arm)
            for seq in ('00','01','10','11'):
                saved=tree['fixed_controls'][seq]
                need(saved['metrics']==derived['F'][seq]['metrics'] and saved['exact_evidence']==derived['F'][seq]['exact_evidence'],'independent fixed exact endpoint '+seq)
            for arm,cost in derived['logical_costs'].items():
                need(all(tree['logical_costs'][arm][k]==cost[k] for k in ('path_references','unique_semantic_SHAs','kernels','flows')),'separate original logical path cost '+arm)
            row=dict(original_index=index,complete=True,controls=derived,O2_hit_S=hit(derived['O2']['metrics'],derived['S']['metrics']),
                O2_hit_Q=hit(derived['O2']['metrics'],derived['Q']['metrics']),O2_hit_O1=hit(derived['O2']['metrics'],derived['O1']['metrics']))
            row.update({'F'+p+'_hit_Q':hit(derived['F'][p]['metrics'],derived['Q']['metrics']) for p in ('00','01','10','11')})
            journal.write('mother_'+str(index),row);rows.append(row)
        need(production['aircraft_records']==production['new_rays']==4715,'all original4715 aircraft and new production rays')
        result=gate(rows)
        journal.write('renderer_small_table',dict(schema='R12RendererTableV1',rows=rows,gate=result,producer_budget=production))
        reader._persist_budget()
        return result
    finally:journal.close()
