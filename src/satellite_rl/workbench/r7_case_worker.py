"""One exclusive case subprocess; no API publication and no new run identity."""
def main():
    from .r7_parent import bind_parent
    bind_parent('R7_CASE_PARENT_PID')
    import argparse
    from pathlib import Path
    from .storage import read_json,digest,Store
    from .coverage_ai_search import pin
    from . import frozen_round_tasks,coverage_ai_search_r7 as policy
    from satellite_rl.training.run_monitor import atomic_json
    p=argparse.ArgumentParser();p.add_argument('--parent-dir',required=True);p.add_argument('--case-index',type=int,required=True);p.add_argument('--directory',required=True);args=p.parse_args()
    parent=Path(args.parent_dir);directory=Path(args.directory);i=args.case_index
    job=read_json(parent/'job.json',{});config=read_json(parent/'config.json',{})
    frozen_round_tasks.verify(Store(job['archive_root'],source_root=job['source_root']),job,policy)
    if i not in job['case_indices'] or directory.resolve()!=(parent/'case_workers'/str(i)).resolve():raise ValueError('exact original parent case/source directory')
    mini={**job,'case_indices':[i]}
    if job['mode']=='labels':
        from .r7_labels import labels
        result=labels(mini,config,directory);ref=result['cases'][str(i)]
    elif job['mode']=='data_train_dev':
        from .r7_uniform_data import derive
        result=derive(job['model_inputs']['recipe'],[i],directory/'derivation',job['model_inputs'].get('transport_map'))
        path=directory/'data_row.json';atomic_json(path,result['cases'][str(i)]);ref=dict(path=str(path),sha256=digest(path))
    else:raise ValueError('parallel case does not admit learning or timing node')
    atomic_json(directory/'case_result.json',dict(case_index=i,parent_job_sha256=digest(parent/'job.json'),mode=job['mode'],result_ref=ref))
if __name__=='__main__':main()
