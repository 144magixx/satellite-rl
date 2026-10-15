"""Concrete metadata-only engineering catalog; construction invokes no PHY."""
from satellite_rl.t0_width.logging import content_sha

FIXTURES=[dict(id='w28_n1_w03',N=1,widths=[.3]),dict(id='w28_n1_w05',N=1,widths=[.5]),
    dict(id='w28_n1_w10',N=1,widths=[1.]),dict(id='w28_n200_mixed',N=200,widths=[1.,.5,.5]+[.3]*197)]
REQUIRED_COVERAGE=('N1','N200','width03','width05','width10','uniform21600000','unique_owner','Omega',
    'P1','P4','R1','R4','probability_original_masked','fork_readonly_no_pollution','trial_commit_exact',
    'mask_domain_shrink','dedup_no_refill','earliest_tie','pause_before_proposal','pause_after_query',
    'pause_after_selection','resume_after_commit','unknown_query_blocked','unknown_commit_blocked','binding_no_PHY_replay')

def catalog():
    rows=[]
    def add(kind,fixture,method,commit,query,forward,coverage):
        rows.append(dict(check_index=len(rows),check_id='check_%02d_%s'%(len(rows),kind),kind=kind,fixture_id=fixture,method=method,
            worst_case=dict(commit=commit,query=query,forward=forward,kernel_preparation=1,restore_PHY=0),coverage=coverage))
    add('method','w28_n1_w03','P1',1,0,1,['N1','width03','uniform21600000','unique_owner','Omega','P1'])
    add('method','w28_n1_w05','P4',1,4,1,['width05','P4','trial_commit_exact'])
    add('method','w28_n1_w10','R1',1,0,0,['width10','R1'])
    add('method','w28_n200_mixed','R4',1,4,0,['N200','R4','trial_commit_exact'])
    add('method','w28_n200_mixed','P4',1,4,1,['P4','trial_commit_exact'])
    add('dedup_tie','w28_n1_w03','R4',1,1,0,['dedup_no_refill','earliest_tie'])
    add('fork','w28_n200_mixed','R1',0,0,0,['fork_readonly_no_pollution'])
    add('probability','w28_n200_mixed','P1',0,0,2,['probability_original_masked'])
    add('mask_shrink','w28_n200_mixed','R4',1,5,0,['mask_domain_shrink','trial_commit_exact'])
    add('pause_before','w28_n1_w03','P4',1,4,1,['pause_before_proposal'])
    add('pause_query','w28_n200_mixed','P4',1,4,1,['pause_after_query'])
    add('pause_selection','w28_n200_mixed','R4',1,4,0,['pause_after_selection'])
    add('pause_commit','w28_n1_w05','P4',1,4,1,['resume_after_commit'])
    add('unknown_query','w28_n1_w03','R4',0,1,0,['unknown_query_blocked'])
    add('unknown_commit','w28_n1_w03','R1',1,0,0,['unknown_commit_blocked'])
    add('binding','w28_n200_mixed','R1',0,0,0,['binding_no_PHY_replay'])
    return rows

def fixture_manifest():
    return dict(schema='W28EngineeringFixturesV1',rows=FIXTURES,scientific_data=False,
        generator='satellite_rl.shared_interface.fixtures.object_recipe; only static arrays/hash/Omega expression during object construction',
        uniform_request_bps=21600000,ownership='one aircraft per unique beam; aircraft located at assigned beam centre',
        geometry='[52deg,8deg] shared GEO-visible centre; Ncap=N; common Omega cap=100*cone(0.5deg); N200 widths 197x0.3+2x0.5+1x1.0',
        qualification_execution_counts=dict(kernel_preparations=0,capacity_reward_calls=0,model_forwards=0))

def validate_catalog(checks,fixtures):
    if checks!=catalog() or fixtures!=fixture_manifest():raise ValueError('exact finite engineering catalog/fixture recipe required')
    covered={tag for row in checks for tag in row['coverage']}
    if covered!=set(REQUIRED_COVERAGE):raise ValueError('required engineering coverage incomplete')
    totals={key:sum(row['worst_case'][key] for row in checks) for key in ('commit','query','forward','kernel_preparation','restore_PHY')}
    if totals['commit']>64 or totals['query']+totals['restore_PHY']>256 or totals['commit']+totals['query']+totals['restore_PHY']>320 or totals['forward']>64 or totals['kernel_preparation']>64:raise ValueError('engineering finite caps exceeded')
    return dict(catalog_SHA=content_sha(checks),fixture_manifest_SHA=content_sha(fixtures),worst_case=totals,
        capacity_reward_calls=totals['commit']+totals['query']+totals['restore_PHY'],required_coverage=list(REQUIRED_COVERAGE))

def construct_fixture(spec):
    """Future runtime only. Static recipe, then actual root engineering geometry/kernel."""
    if spec not in FIXTURES:raise ValueError('unregistered engineering fixture')
    from satellite_rl.shared_interface.fixtures import object_recipe
    from satellite_rl.shared_interface.contracts import Authority
    from satellite_rl.shared_interface.stages import UniformStage2Input
    from satellite_rl.shared_interface.spherical_gaussian import PHYSICS
    from satellite_rl.t0_width.environment import T0WidthEnv
    from satellite_rl.shared_interface.content import seal,content_sha
    import numpy as np
    objects=object_recipe(dict(spec,expected={}))
    objects['layout']=seal(dict(objects['layout'],geometry_Omega_cap_sr=float(100*2*np.pi*(1-np.cos(np.deg2rad(.5)/2)))))
    registry=dict(objects['authority_registry'],layout_shas=[objects['layout']['content_sha256']])
    objects['authority_registry']=registry;objects['authority_registry_sha']=content_sha(registry)
    stage=UniformStage2Input(objects['case'],objects['layout'],Authority(objects['authority_registry'],objects['authority_registry_sha']),PHYSICS)
    if (stage.case['fixed_request_bps']!=21600000 or len(stage.layout['beam_ids'])!=spec['N']
            or int(stage.layout['request_bps_by_beam_int64'].sum())!=21600000*spec['N']):raise ValueError('fixture data invariants differ')
    return T0WidthEnv(stage,'S10')

def prefix_recipe():
    """Pure arithmetic legal 196-action constructed prefix, not 196 real steps."""
    offsets=[0]*8;ordinals=[0]*8;actions=[]
    for index in range(196):
        group=index%8;count=10 if ordinals[group]<8 else 1
        power=50 if index<44 else 5
        actions.append(dict(processing_index=index,group=group,start=offsets[group],count=count,power_w=power))
        offsets[group]+=count;ordinals[group]+=1
    return dict(schema='W28EngineeredPrefixV1',N=200,constructed_prefix_count=196,real_steps_used=0,
        actions=actions,total_power_w=sum(a['power_w'] for a in actions),remaining_beams=4,
        current_group=4,expected_free_slots_current_group=4,expected_FL_legal=10,expected_P_legal=5,
        prefix_capacity_calls=1,subsequent_candidate_queries_max=4,subsequent_commit_max=1)
