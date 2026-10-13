"""Two isolated CPU cases per lane, parent serial publication and aggregate guard."""
import os,sys,time,subprocess,shutil
from pathlib import Path
from satellite_rl.training.run_monitor import atomic_json
from .storage import read_json,digest,now
from .r5_stage_guard import birth,rss_tree,terminate_owned

def parallel_cases(job,config,directory):
    directory=Path(directory);mode=job['mode'];parallelism=config['case_parallelism']
    if mode not in ('data_train_dev','labels') or parallelism not in (1,2):raise ValueError('CPU first wave finite case concurrency only')
    index=read_json(directory/'case_index.json',{});queue=[i for i in job['case_indices'] if str(i) not in index];active={};started=time.monotonic();peak=0
    try:
        while queue or active:
            while queue and len(active)<parallelism:
                i=queue.pop(0);case_dir=directory/'case_workers'/str(i);case_dir.mkdir(parents=True,exist_ok=True)
                if (case_dir/'started.json').exists():raise ValueError('unknown original isolated case preserved; root reconciliation required')
                command=[sys.executable,'-B','-u','-m','satellite_rl.workbench.r7_case_worker','--parent-dir',str(directory),'--case-index',str(i),'--directory',str(case_dir)]
                atomic_json(case_dir/'started.json',dict(case_index=i,parent_job_sha256=digest(directory/'job.json'),command=command,science_identity_reserved=True))
                env=os.environ.copy();env.update(R7_CASE_PARENT_PID=str(os.getpid()),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',NUMEXPR_NUM_THREADS='2')
                log=(case_dir/'worker.log').open('ab',buffering=0)
                process=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=os.name!='nt',creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
                actual_birth=birth(process.pid);active[i]=(process,actual_birth,log,case_dir)
                atomic_json(case_dir/'launch.json',dict(pid=process.pid,birth=actual_birth,at=now(),case_parallelism=parallelism,threads=2))
            peak=max(peak,rss_tree(os.getpid()))
            if peak>4294967296:raise MemoryError('whole lane worker+all case/stage children4GiB guard')
            if any(rss_tree(process.pid)>2147483648 for process,_,_,_ in active.values()):raise MemoryError('individual case+all stages2GiB guard')
            if shutil.disk_usage(directory).free<8589934592:raise OSError('8GiB reserve reached')
            if (directory/'stop.request').exists():raise RuntimeError('lane requested stop; retain unknown spend')
            for i,(process,actual_birth,log,case_dir) in list(active.items()):
                if process.poll() is None:continue
                log.close()
                if process.returncode!=0:raise RuntimeError('isolated case failed; preserve spend and stop successors')
                result_path=case_dir/'case_result.json';result=read_json(result_path,{})
                if result.get('case_index')!=i or result.get('parent_job_sha256')!=digest(directory/'job.json') or result.get('mode')!=mode:raise ValueError('isolated case source/identity publication')
                index[str(i)]=dict(path=str(result_path),sha256=digest(result_path));atomic_json(directory/'case_index.json',index);del active[i]
            atomic_json(directory/'parallel_ledger.json',dict(case_parallelism=parallelism,active_case_indices=sorted(active),completed_cases=len(index),pending_cases=len(queue),peak_lane_parent_children_rss_bytes=peak,elapsed_seconds=time.monotonic()-started,case_throughput_per_second=len(index)/max(time.monotonic()-started,1e-9),case_threads=2,wall_includes_waits=True))
            if active:time.sleep(.05)
    finally:
        for process,actual_birth,log,case_dir in active.values():
            # Every level binds PR_SET_PDEATHSIG, including each science stage.
            # Killing owned case groups cannot leave a detached algorithm.
            terminate_owned(process,actual_birth);log.close()
    cases={}
    for i in job['case_indices']:
        result=read_json(index[str(i)]['path'],{})
        if digest(index[str(i)]['path'])!=index[str(i)]['sha256']:raise ValueError('published isolated case changed')
        cases[str(i)]=result['result_ref']
    manifest=dict(group=job['experiment_id'],phase=mode,scientific_commit=job['scientific_commit'],cases=cases,
                  case_parallelism=parallelism,source_dataset_sha256=config['datasets'].get('train',{}).get('sha256'),science_updates=0,future_dispatch_allowed=False)
    atomic_json(directory/'result_manifest.json',manifest);return manifest
