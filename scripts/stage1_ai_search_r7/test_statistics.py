"""Frozen40 mother paired confirmation; single pre-data PCG64 draw matrix."""
import math
from pathlib import Path
import numpy as np
from satellite_rl.workbench.coverage_ai_search import pin
from satellite_rl.workbench.storage import digest,read_json
from satellite_rl.training.run_monitor import atomic_json

def prepare(output):
    """Explicit root statistical freeze before test reads; no aircraft/model access."""
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    matrix=np.random.Generator(np.random.PCG64(51001)).integers(0,40,size=(10000,40),dtype=np.int64)
    path=output/'draw_matrix.npy';np.save(path,matrix,allow_pickle=False)
    spec=dict(schema='R7FrozenTestBootstrapV1',seed=51001,draws=10000,mothers=40,numpy_version=np.__version__,numpy_source_ref=dict(path=np.__file__,sha256=digest(np.__file__)),
              statistical_source_sha256=digest(__file__),draw_matrix_ref=dict(path=str(path),sha256=digest(path)),matrix_generated_once_before_test_reads=True,shared_two_comparisons_three_metrics=True,quantile=.975,quantile_method='linear')
    atomic_json(output/'statistical_plan.json',spec);return spec

def log_ratio(ai,reference,metric):
    if type(ai) not in (float,int) or type(reference) not in (float,int) or not math.isfinite(ai) or not math.isfinite(reference) or ai<0 or reference<0:raise ValueError('negative/nonfinite/missing paired metric; no case drop')
    if metric=='CV':
        if reference==0:
            if ai>1e-9+1e-12*max(abs(ai),abs(reference)):raise ValueError('zero reference CV absolute noninferiority failed')
            return 0.
        if ai==0:return 0. # frozen conservative convention
    if ai<=0 or reference<=0:raise ValueError('positive peak/top10 required; no epsilon')
    return math.log(ai)-math.log(reference)

def confirm(cases,statistics_ref,output=None):
    if len(cases)!=40 or [c['case_index'] for c in cases]!=list(range(40)) or len({c['mother_id'] for c in cases})!=40:raise ValueError('original ordered40 unique equal-weight mothers')
    spec=read_json(pin(statistics_ref),{})
    if spec.get('schema')!='R7FrozenTestBootstrapV1' or (spec.get('seed'),spec.get('draws'),spec.get('mothers'),spec.get('quantile'),spec.get('quantile_method'))!=(51001,10000,40,.975,'linear') or spec.get('statistical_source_sha256')!=digest(__file__) or spec.get('numpy_version')!=np.__version__ or spec.get('matrix_generated_once_before_test_reads')is not True:raise ValueError('pre-test fixed statistical source/version/draw plan')
    pin(spec['numpy_source_ref']);matrix=np.load(pin(spec['draw_matrix_ref']),allow_pickle=False)
    if matrix.shape!=(10000,40) or matrix.dtype.kind not in 'iu' or (matrix<0).any() or (matrix>39).any():raise ValueError('one fixed10000x40 mother draw matrix')
    result={};failed=[]
    for reference in ('RULE','RANDOM'):
        for metric in ('maxload','top10_mean','CV'):
            key=reference+':'+metric
            try:
                values=[]
                for case in cases:
                    ai=case['methods']['AI'];base=case['methods'][reference]
                    if not all(v.get('published') and v.get('valid') and v.get('identity_closed') and v.get('budget_closed') for v in (ai,base)):raise ValueError('required method missing/invalid; preserve40 denominator')
                    values.append(log_ratio(ai['metrics'][metric],base['metrics'][metric],metric))
                values=np.asarray(values,dtype=np.float64);drawmeans=np.mean(values[matrix],axis=1);ordered=np.sort(drawmeans);h=(10000-1)*.975;lo=math.floor(h);hi=math.ceil(h);upper=float(ordered[lo]+(h-lo)*(ordered[hi]-ordered[lo]));threshold=math.log(.99 if metric=='maxload' else 1.02)
                passed=upper<threshold if metric=='maxload' else upper<=threshold
                result[key]=dict(passed=bool(passed),mean_log_ratio=float(values.mean()),one_sided_97_5_upper=upper,threshold_log_ratio=threshold,denominator=40)
                if not passed:failed.append(key)
            except (ValueError,KeyError,TypeError) as e:result[key]=dict(passed=False,reason=str(e),denominator=40);failed.append(key)
    record=dict(passed=not failed,comparisons=result,failed_confirmations=failed,statistics_ref=statistics_ref,draw_matrix_ref=spec['draw_matrix_ref'],single_shared_draw_matrix=True,no_omitted_mothers=True,one_seed_limit=True)
    if output:atomic_json(output,record)
    return record
