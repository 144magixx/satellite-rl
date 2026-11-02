"""External operations bootstrap; all scientific imports come from frozen137."""
import argparse,sys,json,importlib.util
from pathlib import Path

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=('worker','case'),required=True);parser.add_argument('--contract',required=True);parser.add_argument('--contract-sha256',required=True);parser.add_argument('--review',required=True);parser.add_argument('--review-sha256',required=True);parser.add_argument('--job-dir');parser.add_argument('--parent-dir');parser.add_argument('--case-index',type=int);parser.add_argument('--directory');args=parser.parse_args()
    directory=Path(args.job_dir or args.parent_dir);job=json.loads((directory/'job.json').read_text());source=Path(job['source_root'])
    sys.path[:0]=[str(source/'src'),str(source)]
    from satellite_rl.workbench.coverage_ai_search import pin
    metadata=dict(contract_ref=dict(path=args.contract,sha256=args.contract_sha256),root_review_ref=dict(path=args.review,sha256=args.review_sha256))
    contract=json.loads(pin(metadata['contract_ref']).read_text())
    if Path(__file__).resolve()!=pin(contract['operations_source_refs']['bootstrap']).resolve():raise ValueError('actual operations bootstrap path/source differs')
    module_path=pin(contract['operations_source_refs']['module']);name='satellite_rl.workbench.r7_recovery_ops'
    spec=importlib.util.spec_from_file_location(name,module_path);ops=importlib.util.module_from_spec(spec);sys.modules[name]=ops;spec.loader.exec_module(ops)
    ops.validate(job,metadata)
    from satellite_rl.workbench import r4_monitor_repair
    r4_monitor_repair.rss_tree=ops.rss_tree
    if args.mode=='worker':
        ops.install_worker(job,metadata,directory)
        from satellite_rl.workbench.r7_runner import run_job
        run_job(directory,resume=False)
    else:
        if args.case_index not in ops.REMAINING:raise ValueError('no original attempted case may be relaunched')
        sys.argv=['original_r7_case_worker','--parent-dir',str(directory),'--case-index',str(args.case_index),'--directory',args.directory]
        from satellite_rl.workbench.r7_case_worker import main as case_main
        case_main()

if __name__=='__main__':main()
