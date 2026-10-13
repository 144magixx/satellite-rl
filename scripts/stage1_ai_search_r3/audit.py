"""Independent raw R3 validation; never imports producer decisions or metrics."""
import hashlib
import json
from pathlib import Path
import numpy as np
from satellite_rl.envs.regions import get_profile

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def canonical_sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def rays_visible(points,profile='europe_v1'):
    p=get_profile(profile); points=np.asarray(points,float); lat,lon=np.deg2rad(points.T)
    surface=p.earth_radius_km*np.column_stack((np.cos(lat)*np.cos(lon),np.cos(lat)*np.sin(lon),np.sin(lat)))
    slon=np.deg2rad(p.satellite_longitude_deg); sat=p.orbit_radius_km*np.array([np.cos(slon),np.sin(slon),0.])
    rays=surface-sat; rays/=np.linalg.norm(rays,axis=1)[:,None]
    cosine=np.cos(lat)*np.cos(lon-slon)
    elev=np.rad2deg(np.arctan2(cosine-p.earth_radius_km/p.orbit_radius_km,np.sqrt(np.maximum(0.,1-cosine**2))))
    visible=((points[:,0]>=p.latitude_bounds_deg[0])&(points[:,0]<=p.latitude_bounds_deg[1])&
             (points[:,1]>=p.longitude_bounds_deg[0])&(points[:,1]<=p.longitude_bounds_deg[1])&(elev>=p.min_elevation_deg))
    return rays,visible

def solid_angle(widths):
    return float(np.sum(2*np.pi*(1-np.cos(np.deg2rad(np.asarray(widths,float)/2)))))

def audit_layout(scene,layout,menu_widths=None,ncap=200,omega_cap=None):
    failures=[]
    def fail(code,detail=''):failures.append({'code':code,'detail':str(detail)})
    try:
        pos=np.asarray(scene['positions'],float);d=np.asarray(scene['demands'],float)
        c=np.asarray(layout['centers'],float).reshape(-1,2);w=np.asarray(layout.get('widths_deg',layout.get('widths')),float)
        a=np.asarray(layout['assignment']);n=len(d);b=len(c)
        if pos.shape!=(n,2) or d.shape!=(n,) or not n or not np.isfinite(pos).all() or not np.isfinite(d).all() or not np.isfinite(d.sum()) or (d<0).any():raise ValueError('invalid_scene')
        ids=scene['aircraft_ids']
        if len(ids)!=n or len(set(ids))!=n:fail('aircraft_identity')
        if b<1 or b>min(200,n,ncap):fail('N_cap')
        if w.shape!=(b,) or not np.isfinite(w).all() or not np.isfinite(c).all():raise ValueError('layout_shape_finite')
        if a.shape!=(n,) or not np.isfinite(a).all() or not np.equal(a,np.floor(a)).all():raise ValueError('nonintegral_unique_ownership')
        a=a.astype(np.int64)
        if (a<0).any() or (a>=b).any():raise ValueError('incomplete_ownership')
        if (np.bincount(a,minlength=b)==0).any():fail('empty_beam')
        menu=np.asarray(menu_widths if menu_widths is not None else w,float)
        if menu.shape!=(b,) or not np.isfinite(menu).all() or not np.isin(menu,[.3,.5,1.]).all():fail('menu_identity')
        elif (w<.3-1e-12).any() or (w>menu+1e-12).any():fail('envelope_bounds')
        ur,uv=rays_visible(pos,scene.get('profile','europe_v1'));br,bv=rays_visible(c,scene.get('profile','europe_v1'))
        if not uv.all() or not bv.all():fail('visibility')
        bad=np.flatnonzero(np.sum((ur-br[a])**2,axis=1)>4*np.sin(np.deg2rad(w[a]/2)/2)**2+1e-14)
        if len(bad):fail('full_ray_geometry',bad.tolist())
        loads=np.bincount(a,weights=d,minlength=b);total=float(d.sum())
        if not np.isclose(loads.sum(),total,rtol=1e-12,atol=1e-6):fail('request_conservation')
        mo=solid_angle(menu);ao=solid_angle(w)
        if omega_cap is not None and mo>omega_cap+1e-12:fail('menu_omega_cap')
        if ao>mo+1e-12:fail('actual_omega_exceeds_menu')
        metrics={'N':b,'menu_Omega':mo,'actual_Omega':ao,'total_request':total,'loads':loads.tolist(),
                 'maxload':float(loads.max()) if total else None,'top10_mean':float(np.sort(loads)[-max(1,int(np.ceil(.1*b))):].mean()) if total else None,
                 'CV':float(loads.std()/loads.mean()) if total else None}
        return {'valid':not failures,'failures':failures,'metrics':metrics,'layout_sha256':canonical_sha(layout),'scene_sha256':canonical_sha(scene)}
    except (ValueError,TypeError,KeyError,IndexError) as exc:
        fail('malformed_raw_record',exc);return {'valid':False,'failures':failures,'metrics':None}

def audit_problem_vector(problem,vector,tolerance=1e-6):
    p=problem;v=np.asarray(vector,float);failures=[]
    if v.shape!=np.asarray(p['costs']).shape or not np.isfinite(v).all():return {'valid':False,'failures':[{'code':'original_vector_shape_finite'}]}
    ix=np.asarray(p['integer'],bool);res=float(np.max(np.abs(v[ix]-np.rint(v[ix])),initial=0))
    if res>tolerance:failures.append({'code':'original_integrality'})
    if (v<np.asarray(p['lower'])-tolerance).any() or (v>np.asarray(p['upper'])+tolerance).any():failures.append({'code':'original_bounds'})
    starts=np.asarray(p['starts'],int);rows=np.repeat(np.arange(len(starts)-1),np.diff(starts))
    act=np.bincount(rows,weights=np.asarray(p['values'])*v[np.asarray(p['indices'],int)],minlength=len(starts)-1)
    if (act<np.asarray(p['row_lower'])-tolerance).any() or (act>np.asarray(p['row_upper'])+tolerance).any():failures.append({'code':'original_rows'})
    return {'valid':not failures,'failures':failures,'max_integer_residual':res}

def audit_native(problem,result,stage='regular',start_log_text=None):
    seconds,nodes={'regular':(2,2000),'reference_stage1':(12,20000),'reference_stage2':(8,10000)}[stage]
    expected={'threads':1,'random_seed':51001,'time_limit':seconds,'mip_max_nodes':nodes,'mip_rel_gap':.001,'presolve':'on'}
    failures=[]
    if any(result.get('options',{}).get(k)!=v for k,v in expected.items()):failures.append({'code':'native_option_readback'})
    if result.get('backend')!='highspy1.15.1':failures.append({'code':'native_backend'})
    start=audit_problem_vector(problem,problem['seed'])
    if not start['valid']:failures.append({'code':'full_start_invalid'})
    accepted=bool(start_log_text and ('MIP start solution is feasible' in start_log_text or 'User-supplied values of discrete variables are feasible' in start_log_text))
    if not accepted:failures.append({'code':'native_start_acceptance_unproved'})
    vector=result.get('vector');primal=audit_problem_vector(problem,vector)
    failures.extend(primal['failures']);same=primal['valid'] and np.allclose(vector,problem['seed'],rtol=0,atol=1e-6)
    gap=result.get('mip_gap');bound=result.get('mip_dual_bound')
    fun=result.get('fun')
    cert=gap is not None and bound is not None and fun is not None and np.isfinite([gap,bound,fun]).all() and 0<=gap<=.001+1e-12 and bound<=fun+1e-8
    valid=not failures and bool(result.get('value_valid'))
    return {'valid':valid,'failures':failures,'start_audit':start,'primal_audit':primal,'solution_same_as_W':same,
            'native_bound_certificate':bool(cert),'native_reference_valid':bool(valid and (not same or cert))}

def audit_integral_semantics(scene,pool,slots,edge_aircraft,edge_slot,vector,stage=1,stage1_peak=None):
    """Independent model constraints from raw geometry and slot numbering.

    This checks the mathematical meaning as well as audit_problem_vector's CSR
    algebra. The producer's sparse rows cannot define their own correctness.
    """
    failures=[]
    def fail(code):failures.append({'code':code})
    slots=np.asarray(slots,int);ia=np.asarray(edge_aircraft,int);bs=np.asarray(edge_slot,int);v=np.asarray(vector,float)
    nb=len(slots);ne=len(ia);zi=nb+ne
    if v.shape!=(zi+1+(nb if stage==2 else 0),) or not np.isfinite(v).all():return {'valid':False,'failures':[{'code':'semantic_full_vector_shape'}]}
    if len(set(slots.tolist()))!=nb or len(bs)!=ne or (ia<0).any() or (ia>=len(scene['demands'])).any() or (bs<0).any() or (bs>=nb).any():return {'valid':False,'failures':[{'code':'semantic_edge_identity'}]}
    y=v[:nb];x=v[nb:zi];z=v[zi];tol=1e-6
    if np.max(np.abs(v[:zi]-np.rint(v[:zi])),initial=0)>tol:fail('semantic_integrality')
    if (v< -tol).any() or (v[:zi]>1+tol).any():fail('semantic_bounds')
    n=len(scene['demands']);ownership=np.bincount(ia,weights=x,minlength=n)
    if not np.allclose(ownership,1,rtol=0,atol=tol):fail('semantic_unique_ownership')
    if (x>y[bs]+tol).any():fail('semantic_activation')
    count=np.bincount(bs,weights=x,minlength=nb)
    if (y>count+tol).any():fail('semantic_nonempty')
    demand=np.asarray(scene['demands'],float);total=demand.sum();shares=demand/total if total else np.zeros(n)
    loads=np.bincount(bs,weights=shares[ia]*x,minlength=nb)
    if (loads>z+tol).any():fail('semantic_peak')
    if y.sum()>pool['ncap']+tol:fail('semantic_N_cap')
    widths=np.asarray([.3,.5,1.])[(slots//2)%3]
    omegas=2*np.pi*(1-np.cos(np.deg2rad(widths/2)))
    if np.dot(y,omegas)/pool['omega_cap']>1+tol:fail('semantic_menu_Omega')
    ur,uv=rays_visible(scene['positions'],scene.get('profile','europe_v1'))
    br,bv=rays_visible(np.asarray(pool['centers'])[slots//6],scene.get('profile','europe_v1'))
    distance=np.sum((ur[ia]-br[bs])**2,axis=1)
    legal=(distance<=4*np.sin(np.deg2rad(widths[bs]/2)/2)**2+1e-14)&uv[ia]&bv[bs]
    if not legal.all():fail('semantic_illegal_edge_in_model')
    expected_edges={(i,j) for j in range(nb) for i in range(n) if uv[i] and bv[j] and np.sum((ur[i]-br[j])**2)<=4*np.sin(np.deg2rad(widths[j]/2)/2)**2+1e-14}
    if set(zip(ia.tolist(),bs.tolist()))!=expected_edges or ne!=len(expected_edges):fail('semantic_missing_or_duplicate_legal_edges')
    lookup={int(s):j for j,s in enumerate(slots)}
    for s,j in lookup.items():
        if s%2:
            if s-1 not in lookup:fail('semantic_copy_prefix');continue
            first=lookup[s-1]
            if y[j]>y[first]+tol or loads[j]>loads[first]+tol:fail('semantic_copy_symmetry')
    if stage==2:
        if stage1_peak is None or z>stage1_peak+1e-8+tol:fail('semantic_stage2_peak_bound')
        if (v[zi+1:]<np.abs(loads-y/pool['ncap'])-tol).any():fail('semantic_stage2_L1')
    return {'valid':not failures,'failures':failures,'actual_peak_share':float(loads.max(initial=0))}

def load_prepared_raw(path):
    """Read pinned NPZ directly, without instantiating PreparedCase/Scene."""
    with np.load(path,allow_pickle=False) as d:
        m=json.loads(str(d['metadata']))
        scene={'positions':d['positions'].tolist(),'demands':d['demands'].tolist(),'aircraft_ids':m['aircraft_ids'],'profile':m['profile']}
        pool={'centers':d['centers'].tolist(),'origins':d['origins'].tolist(),'ncap':m['ncap'],'omega_cap':m['omega_cap'],
              'witness':m['witness'],'coverage':d['coverage'].copy(),'content_sha':m['content_sha'],'ledger':m['ledger'],'case_id':m['case_id']}
        h=hashlib.sha256(canonical_sha(scene).encode())
        for key in ('centers','origins','coverage'):h.update(np.ascontiguousarray(d[key]).tobytes())
        h.update(json.dumps([m['witness'],m['ncap'],m['omega_cap']],sort_keys=True).encode())
        if h.hexdigest()!=m['content_sha']:raise ValueError('raw_pool_content_sha_mismatch')
    return scene,pool

def audit_pool(scene,pool):
    failures=[];centers=np.asarray(pool['centers'],float);origins=np.asarray(pool['origins'])
    n=len(scene['demands']);c=len(centers);w=pool['witness']
    wa=audit_layout(scene,w,w.get('widths_deg'),200,None)
    if not wa['valid']:failures.append({'code':'invalid_streaming_witness'})
    nw=len(w['centers']);expected_cap=min(200,n,int(np.ceil(1.25*nw)))
    if pool['ncap']!=expected_cap or abs(pool['omega_cap']-solid_angle(w['widths_deg']))>1e-12:failures.append({'code':'input_resource_caps'})
    if c>512 or origins.shape!=(c,4) or not np.isin(origins,[0,1]).all():failures.append({'code':'candidate_count_origins'})
    if len(set(map(tuple,centers)))!=c:failures.append({'code':'candidate_exact_dedup'})
    for point in w['centers']:
        match=np.flatnonzero(np.all(centers==point,axis=1))
        if len(match)!=1 or not origins[match[0],0]:failures.append({'code':'missing_mandatory_witness_center'})
    ur,uv=rays_visible(scene['positions'],scene.get('profile','europe_v1'));br,bv=rays_visible(centers,scene.get('profile','europe_v1'))
    reconstructed=None
    for width in (.5,1.):
        remaining=np.ones(n,bool);assignment=np.full(n,-1,int);chosen=[]
        for i in sorted(range(n),key=lambda j:scene['aircraft_ids'][j]):
            if not remaining[i]:continue
            if len(chosen)>=200:break
            members=remaining&(np.sum((ur-ur[i])**2,axis=1)<=4*np.sin(np.deg2rad(width/2)/2)**2+1e-14)
            assignment[members]=len(chosen);chosen.append(scene['positions'][i]);remaining[members]=False
        if not remaining.any():reconstructed=(chosen,[width]*len(chosen),assignment.tolist());break
    if reconstructed is None or reconstructed!=(w['centers'],w['widths_deg'],w['assignment']):failures.append({'code':'streaming_witness_rule_mismatch'})
    distance=np.sum((br[:,None]-ur[None])**2,axis=2)
    expected=distance[:,None,:]<=4*np.sin(np.deg2rad(np.asarray([.3,.5,1.])[None,:,None]/2)/2)**2+1e-14
    expected &= bv[:,None,None]&uv[None,None,:]
    if not np.array_equal(expected,pool['coverage']):failures.append({'code':'raw_coverage_not_actual_rays'})
    ledger=pool.get('ledger',{})
    if ledger.get('raypairs',float('inf'))>1424*n+262144:failures.append({'code':'pool_geometry_ledger_cap'})
    return {'valid':not failures,'failures':failures,'witness_audit':wa,'Ncap':expected_cap,'candidate_centers':c,'possible_slots':6*c}

def audit_selection(scene,pool,result,oracle_slots=None):
    """Independently reconstruct shared domain and deterministic non-AI ranking."""
    if result['method']=='WITNESS':return {'valid':True,'failures':[]}
    coverage=np.asarray(pool['coverage'],bool);origins=np.asarray(pool['origins'])
    available=np.flatnonzero(np.repeat(coverage.any(2).reshape(-1),2))
    mandatory=np.flatnonzero(np.repeat(origins[:,0].astype(bool),6));extra=np.setdiff1d(available,mandatory)
    m=min(pool['ncap'],len(extra));selected=np.asarray(result.get('selected_slot_ids',[]),int)
    failures=[];method=result['method'];expected=None
    if method in ('FULL','REF20'):expected=available
    else:
        demand=np.asarray(scene['demands'],float)
        scores=np.asarray([demand[coverage[s//6,(s//2)%3]].sum()/(solid_angle([[.3,.5,1.][(s//2)%3]])/pool['omega_cap']) for s in extra])
        if method=='RANDOM':
            seed=int.from_bytes(hashlib.sha256((pool['case_id']+':51001').encode()).digest()[:8],'little')
            priorities=np.random.default_rng(seed).random((len(origins),3,2));priorities.sort(axis=2)
            scores=priorities[:,:,::-1].reshape(-1)[extra]
        if method=='ORACLE_RESTRICT':
            if oracle_slots is None:return {'valid':False,'failures':[{'code':'oracle_reference_label_not_supplied'}]}
            required=set(oracle_slots).intersection(extra.tolist())
            if len(required)>m:failures.append({'code':'oracle_budget'})
            scores+=np.asarray([1e100 if s in required else 0 for s in extra])
        if method!='AI':
            order=sorted(range(len(extra)),key=lambda j:(-scores[j],int(extra[j])))[:m]
            expected=np.sort(np.r_[mandatory,extra[order]])
        elif len(selected)!=len(mandatory)+m or not set(mandatory).issubset(selected) or not set(selected).issubset(available):failures.append({'code':'AI_domain_size_membership'})
    if expected is not None and not np.array_equal(selected,expected):failures.append({'code':'method_domain_ranking_mismatch'})
    for s in selected:
        if s%2 and s-1 not in selected:failures.append({'code':'selected_copy_prefix'})
    return {'valid':not failures,'failures':failures,'mandatory_slots':len(mandatory),'extra_budget':m}

def audit_method(scene,pool,result,log_loader=None,raw_loader=None,publication_cost=None):
    """Produce gate row from raw final/menu/native data, preserving typed failures."""
    failures=[];layout=result.get('layout');post=result.get('postprocessing',{})
    menu_layout=post.get('menu_layout')
    menu=menu_layout.get('widths_deg',menu_layout.get('widths')) if menu_layout else None
    if layout is None:
        return apply_publication_cost({'valid':False,'published':False,'inner_published':False,'metrics':None,'native_reference_valid':False,
                'publication_seconds':result.get('comparable_seconds'),'failures':[{'code':result.get('status','missing_layout')}]},result,publication_cost)
    if menu is None:failures.append({'code':'missing_original_menu_layout'})
    elif menu_layout.get('centers')!=layout.get('centers') or menu_layout.get('assignment')!=layout.get('assignment'):
        failures.append({'code':'envelope_changed_center_or_members'})
    final=audit_layout(scene,layout,menu,pool['ncap'],pool['omega_cap']);failures.extend(final['failures'])
    if final['metrics'] is not None:
        saved=result.get('metrics',{})
        aliases={'actual_Omega':'solid_angle_sum_sr'}
        tolerances={'N':0,'actual_Omega':1e-12,'maxload':1e-6,'top10_mean':1e-6,'CV':1e-9}
        for key,tol in tolerances.items():
            x=final['metrics'][key];y=saved.get(aliases.get(key,key))
            if x is None:
                if y is not None:failures.append({'code':'saved_metric_mismatch','metric':key})
            elif y is None or not np.isfinite(y) or abs(x-y)>tol+1e-12*abs(x):failures.append({'code':'saved_metric_mismatch','metric':key})
    raw_slots=np.asarray(result.get('selected_slot_ids',[]));native=[];previous_peak=None
    if raw_slots.ndim!=1 or not np.isfinite(raw_slots).all() or not np.equal(raw_slots,np.floor(raw_slots)).all():
        failures.append({'code':'nonintegral_selected_slot_identity'});raw_slots=np.asarray([],int)
    slots=raw_slots.astype(int)
    if slots.ndim!=1 or len(set(slots.tolist()))!=len(slots) or (slots<0).any() or (slots>=6*len(pool['centers'])).any():
        failures.append({'code':'invalid_selected_slot_identity'});slots=np.asarray([],int)
    if len(slots):
        ur,uv=rays_visible(scene['positions'],scene.get('profile','europe_v1'))
        br,bv=rays_visible(np.asarray(pool['centers'])[slots//6],scene.get('profile','europe_v1'))
        widths=np.asarray([.3,.5,1.])[(slots//2)%3]
        distance=np.sum((ur[:,None]-br[None])**2,axis=2)
        legal=(distance<=4*np.sin(np.deg2rad(widths[None]/2)/2)**2+1e-14)&uv[:,None]&bv[None]
        ia,bs=np.nonzero(legal)
        for j,call in enumerate(result.get('solver_calls',[])):
            if call.get('native_output_ref'):
                text=raw_loader(call['native_output_ref']) if raw_loader else Path(call['native_output_ref']).read_text(encoding='utf8')
                raw_native=json.loads(text)
                actual_sha=hashlib.sha256(text.encode('utf8')).hexdigest()
                if actual_sha!=call.get('native_output_sha256'):
                    failures.append({'code':'native_output_sha_mismatch'});continue
                call={**raw_native,**call}
            stage=1 if j==0 else 2
            raw=call.get('vector');seed=call.get('seed_vector')
            seed_audit=audit_integral_semantics(scene,pool,slots,ia,bs,seed,stage,previous_peak)
            primal=audit_integral_semantics(scene,pool,slots,ia,bs,raw,stage,previous_peak) if raw is not None else {'valid':False,'failures':[{'code':'no_native_primal'}]}
            log=log_loader(call['start_native_log_ref']) if log_loader and call.get('start_native_log_ref') else ''
            accepted=('MIP start solution is feasible' in log or 'User-supplied values of discrete variables are feasible' in log)
            seconds,nodes=(12,20000) if result['method']=='REF20' and j==0 else (8,10000) if j else (2,2000)
            expected={'threads':1,'random_seed':51001,'time_limit':seconds,'mip_max_nodes':nodes,'mip_rel_gap':.001,'presolve':'on'}
            options_valid=all(call.get('options',{}).get(k)==v for k,v in expected.items())
            status_valid=call.get('status') in ('HighsModelStatus.kOptimal','HighsModelStatus.kTimeLimit','HighsModelStatus.kSolutionLimit','HighsModelStatus.kIterationLimit')
            valid=primal['valid'] and seed_audit['valid'] and accepted and options_valid and status_valid and call.get('value_valid') and call.get('backend')=='highspy1.15.1' and call.get('start_set_status')=='HighsStatus.kOk' and call.get('highspy_version')=='1.15.1' and call.get('native_library_version')=='1.15.1'
            fun,bound,gap=call.get('fun'),call.get('mip_dual_bound'),call.get('mip_gap')
            cert=all(x is not None and np.isfinite(x) for x in (fun,bound,gap)) and 0<=gap<=.001+1e-12 and bound<=fun+1e-8
            if j and native:cert=native[0]['native_bound_certificate']
            same=False
            if primal['valid']:
                active=np.flatnonzero(np.asarray(raw)[:len(slots)]>.5)
                groups=[]
                for k in active:
                    members=tuple(ia[(bs==k)&(np.asarray(raw)[len(slots):len(slots)+len(ia)]>.5)])
                    sid=slots[k];groups.append((tuple(pool['centers'][sid//6]),float(widths[k]),members))
                w=pool['witness'];wg=[(tuple(c),float(width),tuple(np.flatnonzero(np.asarray(w['assignment'])==b))) for b,(c,width) in enumerate(zip(w['centers'],w['widths_deg']))]
                same=sorted(groups)==sorted(wg)
                if j==0:previous_peak=primal['actual_peak_share']
            provenance_centers=None;provenance_widths=None
            if primal['valid']:
                provenance_ids=slots[np.flatnonzero(np.asarray(raw)[:len(slots)]>.5)]
                provenance_centers=np.asarray(pool['centers'])[provenance_ids//6].tolist()
                provenance_widths=np.asarray([.3,.5,1.])[(provenance_ids//2)%3].tolist()
            witness_d=np.asarray(scene['demands'],float);witness_total=witness_d.sum()
            witness_load=np.bincount(pool['witness']['assignment'],weights=witness_d)
            witness_peak=float(witness_load.max()/witness_total) if witness_total>0 else None
            certifies_W=bool(j==0 and valid and witness_peak is not None and bound is not None and np.isfinite(bound) and bound<=witness_peak+1e-8 and 0<=max(0.,witness_peak-bound)/witness_peak<=.001+1e-12)
            if j and native:certifies_W=native[0]['certifies_external_W']
            native.append({'valid':bool(valid),'seed_audit':seed_audit,'primal_audit':primal,'native_log_accepted':accepted,
                           'menu_centers':provenance_centers,'menu_widths':provenance_widths,
                           'certifies_external_W':certifies_W,
                           'native_bound_certificate':bool(cert),'solution_same_as_W':same,'native_reference_valid':bool(valid and (not same or cert))})
    source=result.get('chosen_source')
    if source=='external_W':
        if menu_layout and (menu_layout['centers']!=pool['witness']['centers'] or menu!=pool['witness']['widths_deg']):failures.append({'code':'fallback_not_common_W'})
    elif source in ('native_stage1','native_stage2'):
        j=0 if source=='native_stage1' else 1
        if len(native)<=j or not native[j]['valid']:failures.append({'code':'published_rejected_native_source'})
        elif menu_layout and (menu_layout['centers']!=native[j]['menu_centers'] or menu!=native[j]['menu_widths']):failures.append({'code':'published_menu_not_native_source'})
    else:failures.append({'code':'missing_chosen_source'})
    target_valid=bool(native and native[0]['certifies_external_W']) if source=='external_W' else bool(source=='native_stage1' and native and native[0]['native_reference_valid']) or bool(source=='native_stage2' and len(native)>1 and native[1]['native_reference_valid'])
    row={'valid':bool(final['valid'] and not failures),'published':False,'inner_published':result.get('status')=='published','metrics':final['metrics'],
            'publication_seconds':result.get('comparable_seconds'),'native_reference_valid':target_valid,
            'layout_sha256':final.get('layout_sha256'),'failures':failures,'native_audits':native}
    return apply_publication_cost(row,result,publication_cost)

def apply_publication_cost(row,result,receipt,required_provider_seconds=None):
    """Only the full row/index/status endpoint is a formal publication."""
    row=dict(row);row['publication_audited']=False;row['published']=False
    errors=[]
    if receipt is None:
        row['publication_seconds']=None;errors.append({'code':'publication_cost_unknown'})
    else:
        keys=('core_comparable_seconds','required_provider_seconds','wrapper_publication_seconds','comparable_publication_seconds')
        values=[receipt.get(k) for k in keys]
        if any(v is None or not isinstance(v,(float,int)) or not np.isfinite(v) or v<0 for v in values):errors.append({'code':'publication_cost_invalid'})
        else:
            core,provider,wrapper,total=values
            if result.get('comparable_seconds') is None or abs(core-result['comparable_seconds'])>1e-9:errors.append({'code':'publication_core_cost_mismatch'})
            if abs(total-core-provider-wrapper)>1e-9:errors.append({'code':'publication_sum_mismatch'})
            if required_provider_seconds is not None and abs(provider-required_provider_seconds)>1e-9:errors.append({'code':'publication_provider_mismatch'})
            if receipt.get('case_id')!=result.get('case_id') or receipt.get('method')!=result.get('method') or receipt.get('endpoint')!='row_case_index_status_visible':errors.append({'code':'publication_identity_endpoint'})
            row['publication_seconds']=float(total)
            row['publication_audited']=not errors
            row['published']=bool(not errors and row.get('inner_published',result.get('status')=='published') and (result['method']=='REF20' or total<=30))
            if result['method']!='REF20' and total>30:errors.append({'code':'censored_publication'})
    row['publication_failures']=errors
    return row
