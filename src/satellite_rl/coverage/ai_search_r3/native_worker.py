"""Isolated pinned highspy worker. This module is the only native run owner."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import numpy as np

def main():
    a=argparse.ArgumentParser();a.add_argument('--input',required=True);a.add_argument('--output',required=True)
    a.add_argument('--vendor',required=True);a.add_argument('--seconds',type=float,required=True);a.add_argument('--nodes',type=int,required=True)
    args=a.parse_args();sys.path.insert(0,str(Path(args.vendor).resolve()))
    import highspy
    if highspy.Highs().version()!='1.15.1':raise RuntimeError('unpinned_highspy_version')
    from .integer import IntegerProblem
    p=IntegerProblem.load(args.input);h=highspy.Highs()
    native_log=str(Path(args.output).with_suffix('.highs.log'))
    options={'threads':1,'random_seed':51001,'time_limit':args.seconds,'mip_max_nodes':args.nodes,'mip_rel_gap':.001,'presolve':'on','output_flag':True,
        'log_to_console':False,'log_file':native_log,'mip_feasibility_tolerance':1e-6}
    for key,value in options.items():
        if h.setOptionValue(key,value)!=highspy.HighsStatus.kOk:raise RuntimeError('option_rejected:'+key)
    actual={k:h.getOptionValue(k)[1] for k in options}
    if actual['threads']!=1 or actual['random_seed']!=51001:raise RuntimeError('native_configuration_unverified')
    lp=highspy.HighsLp();lp.num_col_=len(p.costs);lp.num_row_=len(p.row_lower)
    lp.col_cost_=p.costs;lp.col_lower_=p.lower;lp.col_upper_=p.upper;lp.row_lower_=p.row_lower;lp.row_upper_=p.row_upper
    lp.integrality_=[highspy.HighsVarType.kInteger if x else highspy.HighsVarType.kContinuous for x in p.integer]
    lp.a_matrix_.format_=highspy.MatrixFormat.kRowwise;lp.a_matrix_.start_=p.starts;lp.a_matrix_.index_=p.indices;lp.a_matrix_.value_=p.values
    if h.passModel(lp)!=highspy.HighsStatus.kOk:raise RuntimeError('pass_model_failed')
    from .integer import validate_vector
    start_valid,start_reason=validate_vector(p,p.seed)
    if not start_valid:raise RuntimeError('independent_start_audit_failed:'+start_reason)
    start_status=h.setSolution(len(p.seed),np.arange(len(p.seed),dtype=np.int32),p.seed)
    if start_status!=highspy.HighsStatus.kOk:raise RuntimeError('complete_mip_start_rejected')
    started=time.monotonic();h.run();elapsed=time.monotonic()-started;info=h.getInfo();solution=h.getSolution()
    result=dict(status=str(h.getModelStatus()),message=h.modelStatusToString(h.getModelStatus()),
        fun=float(info.objective_function_value),mip_gap=float(info.mip_gap),mip_dual_bound=float(info.mip_dual_bound),
        mip_node_count=int(info.mip_node_count),native_elapsed=elapsed,options=actual,pid=os.getpid(),
        value_valid=bool(solution.value_valid),vector=list(solution.col_value) if solution.value_valid else None,
        backend='highspy1.15.1',highspy_version=h.version(),native_library_version=h.version(),
        complete_seed_submitted=True,start_set_status=str(start_status),start_independent_audit=dict(valid=start_valid,reason=start_reason),
        start_native_log_ref=native_log,seed_vector=p.seed.tolist())
    h.clear()
    log_text=Path(native_log).read_text(encoding='utf8',errors='replace') if Path(native_log).exists() else ''
    result['native_start_feasibility_accepted']=('MIP start solution is feasible' in log_text or 'Solution supplied by user is feasible' in log_text)
    # Null for native unavailable quantities; never non-standard NaN JSON.
    for k,v in list(result.items()):
        if isinstance(v,float) and not np.isfinite(v):result[k]=None
    target=Path(args.output);tmp=target.with_suffix('.tmp');tmp.write_text(json.dumps(result,allow_nan=False),encoding='utf-8');os.replace(tmp,target)

if __name__=='__main__':main()
