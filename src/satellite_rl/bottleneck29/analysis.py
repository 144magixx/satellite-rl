"""Read-only A0 numerical review. No core, model, PHY, solver or ledger writer.

Pickles must be the user's own verified worker outputs, never untrusted uploads.
The reader hashes every consumed source, checks journal ownership, and treats
missing/invalid/unknown rows as missing evidence rather than zero performance.
"""
import csv
import hashlib
import json
import pickle
from pathlib import Path
import numpy as np

from satellite_rl.t0_width.environment import sgm_values
from satellite_rl.t0_width.logging import content_sha
from satellite_rl.shared_interface.spherical_gaussian import PHYSICS_SHA
from satellite_rl.shared_interface.content import read_wire


class EvidenceError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise EvidenceError(message)


def _digest(raw):
    return hashlib.sha256(raw).hexdigest()


def read_ledger(path, read):
    """Parse only; constructing a live Ledger would mutate missing journals."""
    events = read(path).splitlines(keepends=True)
    artifacts = {}; operations = {}; sessions = {}; binding = None
    for raw in events:
        require(raw.endswith(b'\n'), 'torn journal')
        event = json.loads(raw); item = event['payload']
        encoded = json.dumps(item, sort_keys=True, separators=(',', ':')).encode()
        require(_digest(encoded) == event['sha256'], 'journal checksum')
        if item['type'] == 'binding':
            require(binding is None or binding == item['value'], 'changed journal binding')
            binding = item['value']
        elif item['type'] == 'artifact':
            artifacts[item['id']] = item
        elif item['type'] == 'operation':
            operations[item['id']] = item
        elif item['type'] == 'session':
            sessions[item['id']] = item
    require(binding is not None, 'missing journal binding')
    unresolved = [key for key, event in operations.items() if event.get('status') != 'done']
    unresolved += ['session:'+str(key) for key,event in sessions.items() if 'elapsed' not in event]
    return dict(binding=binding, artifacts=artifacts, operations=operations,
                unresolved=unresolved, sessions=sessions)


def check_lane_artifacts(journal, root, expected_artifacts, read):
    """Normal cursor/completion and retained stop metadata are journal-owned too."""
    root=Path(root);metadata_ids={'cursor','completed','stopped'}
    require(set(journal['artifacts'])-metadata_ids==expected_artifacts,'unexpected or incomplete lane artifact IDs')
    require({'cursor','completed'}<=journal['artifacts'].keys(),'missing committed lane metadata')
    for key in metadata_ids & journal['artifacts'].keys():
        artifact=journal['artifacts'][key];raw=read(root/artifact['path'])
        require(_digest(raw)==artifact['sha'],'lane metadata checksum')
        require((root/artifact['path']).resolve()==(root/(key+'.pkl')).resolve(),'lane metadata path')
        value=pickle.loads(raw)
        if key=='completed': require(value.get('status')=='completed','lane incomplete')


def resolve_rows(rows, snapshot_sha, binding):
    """Resolve cache chains only to identical physical points in the same state."""
    resolved = {}
    def visit(row_id, stack):
        require(row_id in rows, 'cache reference missing')
        require(row_id not in stack, 'cache cycle')
        if row_id in resolved:
            return resolved[row_id]
        row = rows[row_id]; spec = row['spec']; result = row['result']
        require(row['row_id'] == row_id and row['snapshot_sha'] == snapshot_sha and row['binding'] == binding,
                'row state/source binding')
        status = result['status']
        if not spec['valid']:
            require(status == 'invalid' and result.get('physical_calls') == 0 and bool(spec['invalid_reasons']),
                    'invalid row masquerades as result')
            value = None
        elif status == 'cache':
            target_id = result['reference']; require(target_id in rows, 'external state cache reference')
            target = rows[target_id]
            require(tuple(spec[k] for k in ('start','count','power_w')) == tuple(target['spec'][k] for k in ('start','count','power_w')),
                    'cache physical point mismatch')
            require(result.get('physical_calls') == 0, 'cache physical count')
            value = visit(target_id, stack | {row_id})
            require(value is not None, 'cache references invalid point')
        else:
            require(status == 'completed' and result.get('physical_calls') == 1, 'unknown candidate result')
            require(result['spec'] == spec, 'completed candidate spec mismatch')
            value = result
        resolved[row_id] = value
        return value
    for key in rows:
        visit(key, set())
    return resolved


def check_result(result, snapshot, expected_spec, n, prefix, demands, expected_indices):
    """Recompute scalar scores from saved full-N arrays, with no physical calls."""
    require(result['base_snapshot_SHA'] == content_sha(snapshot), 'numeric base snapshot mismatch')
    rate = np.asarray(result['capacity_bps'], dtype=np.float64)
    request = np.asarray(result['demand_bps'], dtype=np.float64)
    require(rate.shape == request.shape == (n,) and np.isfinite(rate).all() and (rate >= 0).all(), 'bad capacity arrays')
    require(np.array_equal(request, demands) and (request > 0).all(), 'candidate demand mismatch')
    order = snapshot['identity']['order']; bid = order[prefix]
    require(result['beam_id'] == bid, 'current beam identity')
    # snapshot arrays use layout order. Saved prefix.indices binds that order.
    indices = np.asarray(result['prefix']['indices'], dtype=np.int64)
    require(np.array_equal(indices,expected_indices[:prefix+1]), 'prefix index identity')
    require((rate[expected_indices[prefix+1:]]==0).all(), 'unprocessed beam service')
    si = np.minimum(rate/request, 1); sgm = sgm_values(rate, request, float(request.mean()))
    require(np.allclose(si, result['SI'], rtol=0, atol=1e-12) and np.allclose(sgm, result['SGM'], rtol=0, atol=1e-12), 'saved service score mismatch')
    require(abs(float(si[indices].sum())-result['prefix']['SI_sum']) <= 1e-10 and abs(float(sgm[indices].sum())-result['prefix']['SGM_sum']) <= 1e-10, 'prefix aggregate mismatch')
    current = int(indices[-1]); require(abs(rate[current]-result['current']['capacity_bps']) <= 1e-8 and request[current] == result['current']['demand_bps'], 'current capacity mismatch')
    action = result['actual_action']; requested = result['requested_action']
    require(action == requested and action['beam_id'] == bid and all(action[k] == expected_spec[k] for k in ('start','count','power_w')), 'actual/requested candidate mismatch')
    require(result['constraints']['requested_execution_equal'] is True and result['constraints']['slot_conflicts'] == 0, 'candidate constraint violation')
    require(result['constraints']['total_power_w']+result['constraints']['future_reserved_power_w'] <= 3000, 'candidate power reserve violation')
    require(result['constraints']['total_power_w']==float(snapshot['powers'].sum())+expected_spec['power_w'] and
            result['constraints']['future_reserved_power_w']==5*(n-prefix-1),'saved resource totals mismatch')
    return dict(prefix_SI=float(si[indices].sum())/n, prefix_SGM=float(sgm[indices].sum())/n,
                current_capacity_bps=float(rate[current]), demand_bps=float(request[current]),
                current_SI=float(si[current]), current_SGM=float(sgm[current]),
                total_power_w=float(result['constraints']['total_power_w']), reward=float(result['reward']))


def summarize_state(rows, snapshot, mother, snapshot_sha, binding):
    n = mother['N']; prefix = snapshot['beam_idx']; case = mother['case_id']; state_id = case+'/'+str(prefix)
    require(snapshot['identity']['layout_SHA'] == mother['layout_content_SHA'] and snapshot['identity']['physics_SHA'] == PHYSICS_SHA,
            'snapshot layout/physics source')
    require(snapshot['identity']['order'] == mother['processing_index_to_beam_id'], 'snapshot processing order')
    expected_indices=np.asarray([mother['_layout_beam_ids'].index(b) for b in mother['processing_index_to_beam_id']],dtype=np.int64)
    pairs = [(f,l) for f in range(100) for l in range(1,11) if f+l <= 100]
    original = mother['actions_indices_by_processing_order'][prefix]
    f0,l0,p0 = int(original[0]), int(original[1])+1, float((original[2]+1)*5)
    expected = {}
    for table, values in [('minimum5',[(f,l,5.) for f,l in pairs]),
                          ('fine_power',[(f0,l0,k/2.) for k in range(1,101)])]:
        for sequence, point in enumerate(values):
            key = state_id+'/'+table+'/'+str(sequence); expected[key] = (table,point)
    require(set(rows) == set(expected), 'state does not have exactly 955+100 rows')
    for key,(table, point) in expected.items():
        spec = rows[key]['spec']
        require(tuple(spec[k] for k in ('start','count','power_w')) == point and spec['family'] == table, 'table ordering/point mismatch')
        f,l,p=point;free=~snapshot['occupied'][prefix%8]
        reserve=sum(j%8==prefix%8 for j in range(prefix+1,n))
        valid=bool(free[f:f+l].all()) and l<=int(free.sum())-reserve and float(snapshot['powers'].sum())+p+5*(n-prefix-1)<=3000
        require(spec['valid']==valid and (not spec['invalid_reasons'])==valid, 'candidate admissibility mismatch')
        domain='original' if p in range(5,51,5) else ('lower_bound_relaxation' if p<5 else 'granularity_relaxation')
        require(spec['domain']==domain,'power intervention domain mismatch')
    numeric = resolve_rows(rows,snapshot_sha,binding)
    baseline_id = state_id+'/fine_power/'+str(int(p0*2)-1)
    require(numeric[baseline_id] is not None, 'missing original-action baseline')
    demands = np.asarray(mother['_demands'], dtype=np.float64)
    candidates=[]; scores={}
    for key,(table,point) in expected.items():
        row=rows[key]; value=numeric[key]
        score = None if value is None else check_result(value,snapshot,row['spec'],n,prefix,demands,expected_indices)
        scores[key]=score
        candidates.append(dict(mother=mother['mother'],case_id=case,prefix=prefix,row_id=key,table=table,
            start=point[0],count=point[1],power_w=point[2],domain=row['spec']['domain'],
            status=row['result']['status'],invalid_reasons='|'.join(row['spec']['invalid_reasons']),
            **({k:None for k in ('prefix_SI','prefix_SGM','current_capacity_bps','demand_bps','current_SI','current_SGM','total_power_w','reward')} if score is None else score)))
    baseline=scores[baseline_id]
    legal5=[(key,scores[key]) for key,(table,_) in expected.items() if table=='minimum5' and scores[key] is not None]
    require(bool(legal5), 'no legal original 5W action')
    minimum_id,minimum=min(legal5,key=lambda item:(item[1]['current_capacity_bps'],item[0]))
    tolerance=1e-6
    pareto=[key for key,score in legal5 if score['prefix_SI']>=baseline['prefix_SI']-tolerance and score['prefix_SGM']>=baseline['prefix_SGM']-tolerance
            and (score['prefix_SI']>baseline['prefix_SI']+tolerance or score['prefix_SGM']>baseline['prefix_SGM']+tolerance)]
    groups={}
    for name,predicate in [('original_discrete',lambda p:p in range(5,51,5)),
                           ('fine_same_floor',lambda p:p>=5 and p not in range(5,51,5)),
                           ('relaxed_lower_floor',lambda p:p<5)]:
        pool=[(key,scores[key]) for key,(table,(_,_,p)) in expected.items() if table=='fine_power' and predicate(p) and scores[key] is not None]
        groups[name]=dict(valid_rows=len(pool),best_prefix_SGM=None if not pool else max(score['prefix_SGM'] for _,score in pool),
            best_prefix_SI=None if not pool else max(score['prefix_SI'] for _,score in pool),
            best_SGM_row=None if not pool else max(pool,key=lambda item:item[1]['prefix_SGM'])[0])
    original_group=groups['original_discrete']
    for name in ('fine_same_floor','relaxed_lower_floor'):
        for metric in ('prefix_SI','prefix_SGM'):
            value=groups[name]['best_'+metric];reference=original_group['best_'+metric]
            groups[name]['best_'+metric+'_minus_original_discrete_best']=None if value is None or reference is None else value-reference
        groups[name]['comparison_caution']='Separate metric maxima may be different candidates; inspect candidate CSV for joint trade-offs.'
    d=baseline['demand_bps']; cap_tol=max(1.,1e-6*d)
    booleans=dict(original_domain_space=bool(pareto),
        early_under_supply_residual=prefix==n//8 and baseline['current_capacity_bps']<d-cap_tol,
        nonfloor_unresolved_residual=baseline['current_SGM']<1-1e-6 and minimum['current_capacity_bps']<=d+cap_tol)
    summary=dict(mother=mother['mother'],case_id=case,prefix=prefix,N=n,
        interpretation='current prefix, all-N normalization; not terminal',baseline_row=baseline_id,
        baseline=baseline,minimum5_capacity_row=minimum_id,minimum5_capacity_bps=minimum['current_capacity_bps'],
        minimum5_minus_demand_bps=minimum['current_capacity_bps']-d,pareto5_rows=pareto,
        power_groups=groups,gate_booleans=booleans,rows=1055,invalid_rows=sum(v is None for v in numeric.values()))
    return summary,candidates


def analyze_lanes(lane_roots, manifest_path, contract_path, output_dir=None, *, expected_binding=None):
    """Read complete four-lane output; partial/unknown evidence never passes gate."""
    sources={}
    def read(path):
        path=Path(path).resolve(); raw=path.read_bytes(); sources[str(path)]=dict(sha256=_digest(raw),bytes=len(raw)); return raw
    manifest=json.loads(read(manifest_path)); contract=json.loads(read(contract_path))
    require(contract['schema']=='S2W29A0DesignContractV2' and contract['inputs']['expected_states']==60, 'wrong scientific contract')
    require(len(manifest['rows'])==20 and len(lane_roots)==4, 'full four-lane matrix required')
    require([r['mother'] for r in manifest['rows']]==list(range(20)) and len({r['case_id'] for r in manifest['rows']})==20,'fixed ordered mother identities')
    for row in manifest['rows']:
        n=row['N']
        require(type(n) is int and 0<n<=200 and row['snapshot_positions']==sorted(set([n//8,n//2,7*n//8])) and len(row['snapshot_positions'])==3,'fixed prefix positions')
    states=[]; candidates=[]; problems=[]
    if expected_binding is None:
        problems.append(dict(lane='all',error='MissingRootBinding',message='root-accepted source/data/contract/authority binding required for complete gate'))
    else:
        require({'source_SHA','data_SHA','contract_SHA','authority_SHA'}<=expected_binding.keys(),'missing root scientific source binding')
        require(expected_binding['contract_SHA']==sources[str(Path(contract_path).resolve())]['sha256'],'root contract SHA mismatch')
        require(expected_binding['data_SHA']==sources[str(Path(manifest_path).resolve())]['sha256'],'root input manifest SHA mismatch')
    common_binding=None
    for lane,root in enumerate(lane_roots):
        root=Path(root)
        try:
            journal=read_ledger(root/'ledger.jsonl',read)
            require(not journal['unresolved'], 'unresolved journal intent/session')
            counts={}
            for event in journal['operations'].values(): counts[event['kind']]=counts.get(event['kind'],0)+1
            limits=journal['binding']['limits']
            for kind,count in counts.items(): require(kind in limits and count<=limits[kind], 'lane operation cap exceeded')
            require(sum(v for k,v in counts.items() if k.startswith('phy_'))<=18825 and counts.get('kernel',0)<=15,'lane scientific cap exceeded')
            require(sum(event['elapsed'] for event in journal['sessions'].values())<=900,'lane active time cap exceeded')
            binding=journal['binding']['binding']
            if expected_binding is not None: require(binding==expected_binding,'lane differs from root-accepted source binding')
            if common_binding is None: common_binding=binding
            require(binding==common_binding,'mixed lane scientific sources')
            expected_artifacts=set()
            for row in manifest['rows']:
                if row['mother']%4!=lane: continue
                for position in row['snapshot_positions']:
                    sid=row['case_id']+'/'+str(position);expected_artifacts.add(sid+'/snapshot')
                    expected_artifacts.update(sid+'/'+table+'/'+str(i) for table,count in [('minimum5',955),('fine_power',100)] for i in range(count))
            check_lane_artifacts(journal,root,expected_artifacts,read)
            for mother in manifest['rows']:
                if mother['mother']%4!=lane: continue
                require(mother['lane']==lane, 'mother lane mismatch')
                mother=dict(mother)
                layout_raw=read(mother['layout_ref'].get('path',mother['layout_ref'].get('local_path')))
                require(_digest(layout_raw)==mother['layout_ref']['sha256'],'layout bytes mismatch')
                layout=read_wire(layout_raw)
                require(layout['content_sha256']==mother['layout_content_SHA'],'layout content mismatch')
                mother['_layout_beam_ids']=layout['beam_ids'];mother['_demands']=layout['request_bps_by_beam_int64']
                for prefix in mother['snapshot_positions']:
                    state_id=mother['case_id']+'/'+str(prefix)
                    snap_raw=read(root/'snapshots'/(_digest(state_id.encode())+'.pkl'))
                    snapshot=pickle.loads(snap_raw); snapshot_sha=_digest(snap_raw)
                    require(snapshot['beam_idx']==prefix,'snapshot prefix cursor mismatch')
                    artifact=journal['artifacts'][state_id+'/snapshot']
                    require(artifact['sha']==snapshot_sha and (root/artifact['path']).resolve()==(root/'snapshots'/(_digest(state_id.encode())+'.pkl')).resolve(),'snapshot not journal-owned')
                    rows={}
                    for table,count in [('minimum5',955),('fine_power',100)]:
                        for sequence in range(count):
                            row_id=state_id+'/'+table+'/'+str(sequence)
                            path=root/'rows'/(_digest(row_id.encode())+'.pkl'); raw=read(path)
                            artifact=journal['artifacts'].get(row_id)
                            require(artifact is not None and artifact['sha']==_digest(raw) and (root/artifact['path']).resolve()==path.resolve(), 'row not journal-owned')
                            rows[row_id]=pickle.loads(raw)
                    state,table=summarize_state(rows,snapshot,mother,snapshot_sha,binding)
                    states.append(state);candidates.extend(table)
        except (OSError,KeyError,ValueError,TypeError,pickle.UnpicklingError) as error:
            problems.append(dict(lane=lane,error=type(error).__name__,message=str(error)))
    complete=not problems and len(states)==60 and len(candidates)==63300
    booleans={name:any(s['gate_booleans'][name] for s in states) for name in
              ('original_domain_space','early_under_supply_residual','nonfloor_unresolved_residual')}
    mothers=[]
    for mother in manifest['rows']:
        selected=[s for s in states if s['mother']==mother['mother']]
        mothers.append(dict(mother=mother['mother'],case_id=mother['case_id'],states_complete=len(selected),
            **{key:any(s['gate_booleans'][key] for s in selected) for key in booleans}))
    summary=dict(schema='W29A0NumericalReadbackV1',complete=complete,states_complete=len(states),
        candidate_rows=len(candidates),problems=problems,gate_booleans=booleans,
        eligible_for_A1_admission=complete and any(booleans.values()),training_admitted=False,
        interpretation='A0 existence diagnostic on existing development prefixes; not terminal improvement or significance',
        calls=dict(PHY=0,kernel=0,model_forward=0,parameter_updates=0),states=states,mothers=mothers,sources=sources)
    if output_dir is not None:
        out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
        (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
        for name,table in [('mothers',mothers),('states',[dict(mother=s['mother'],case_id=s['case_id'],prefix=s['prefix'],N=s['N'],
            minimum5_capacity_bps=s['minimum5_capacity_bps'],minimum5_minus_demand_bps=s['minimum5_minus_demand_bps'],
            baseline_prefix_SI=s['baseline']['prefix_SI'],baseline_prefix_SGM=s['baseline']['prefix_SGM'],
            invalid_rows=s['invalid_rows'],**s['gate_booleans']) for s in states]),('candidates',candidates)]:
            with (out/(name+'.csv')).open('w',encoding='utf8',newline='') as stream:
                if table:
                    writer=csv.DictWriter(stream,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
        (out/'sources.json').write_text(json.dumps(sources,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    return summary
