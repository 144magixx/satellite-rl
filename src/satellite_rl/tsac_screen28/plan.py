"""Metadata-only finite matrix and single irreversible development gate."""
METHODS=('P1','P4','R1','R4')

def matrix(case_metadata,phase):
    expected=20 if phase=='development' else 40 if phase=='known' else None
    if expected is None or len(case_metadata)!=expected:raise ValueError('exact frozen phase mothers required')
    if len({r['case_id'] for r in case_metadata})!=expected:raise ValueError('duplicate case ID')
    jobs=[]
    for mother,row in enumerate(case_metadata):
        n=row['N']
        if type(n) is not int or not 1<=n<=200:raise ValueError('actual metadata N required')
        methods=list(METHODS)
        # Predeclared cyclic order reduces systematic method-position effects.
        methods=methods[mother%4:]+methods[:mother%4]
        for repeat in range(3):
            for method in methods:jobs.append(dict(case_id=row['case_id'],mother=mother,method=method,repeat=repeat,N=n))
        if phase=='known':
            for method in ('fixed','greedy'):jobs.append(dict(case_id=row['case_id'],mother=mother,method=method,repeat=0,N=n))
    return dict(schema='W28FiniteMatrixV1',phase=phase,jobs=jobs,episodes=len(jobs),
        commits=sum(j['N'] for j in jobs),queries=sum(4*j['N'] for j in jobs if j['method'] in ('P4','R4')),
        model_forwards=sum(j['N'] for j in jobs if j['method'].startswith('P')),
        new_training=0,new_parameter_updates=0,new_data=0,S3_calls=0)

def development_gate(means,qualified):
    p1,p4,r4=(means[k] for k in ('P1','P4','R4'))
    dsi=p4['SI']-p1['SI'];dsgm=p4['SGM']-p1['SGM']
    rsi=p4['SI']-r4['SI'];rsgm=p4['SGM']-r4['SGM']
    passed=(qualified is True and dsi>=-.002 and dsgm>=-.005 and (dsi>=.003 or dsgm>=.005)
        and p4['power_w']<=1.1*p1['power_w'] and rsi>=-.002 and rsgm>=-.005 and (rsi>=0 or rsgm>=0))
    return dict(schema='W28DevelopmentDecisionV1',passed=bool(passed),known_episodes=560 if passed else 0,
        delta_P4_P1=dict(SI=dsi,SGM=dsgm),delta_P4_R4=dict(SI=rsi,SGM=rsgm),not_independent=True)
