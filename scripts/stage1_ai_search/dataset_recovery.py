"""Bounded read-only AI17 dataset-failure preservation; no scientific jobs."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import hashlib,json
from pathlib import Path
import shutil
from scripts.stage1_ai_search import ops

TARGETS={20418:["prepare_ai_search_20261013_020009_9d488437","prepare_ai_search_20261013_020013_1d5b1cd2"],
         20424:["prepare_ai_search_20261013_020010_8ee9f195"],
         20473:["prepare_ai_search_20261013_020012_e0eafd35"]}


def snapshot(port,root):
    if port not in TARGETS: raise ValueError("recovery only permits original dataset owners, never 74")
    info=ops.instance(port);rows=[]
    all_tasks=ops.api(port,"/api/ai-search/tasks")
    target=root/str(port);target.mkdir(parents=True,exist_ok=True)
    (target/"actual_api_tasks.json").write_text(json.dumps(all_tasks,indent=2),encoding="utf-8")
    client=ops.connect(port)
    try:
        sftp=client.open_sftp()
        for run_id in TARGETS[port]:
            actual=ops.api(port,"/api/ai-search/tasks/"+run_id)
            local=target/run_id;local.mkdir(exist_ok=True)
            (local/"actual_api_detail.json").write_text(json.dumps(actual,indent=2),encoding="utf-8")
            directory=info["archive"]+"/outputs/workbench/jobs/"+run_id
            saved=[]
            def pull(path,relative):
                stat=sftp.stat(path)
                if stat.st_size>8*(1<<20): return
                data=sftp.open(path,"rb").read()
                destination=local/relative;destination.parent.mkdir(parents=True,exist_ok=True)
                destination.write_bytes(data)
                saved.append(dict(remote_path=path,path=str(destination),sha256=hashlib.sha256(data).hexdigest(),bytes=len(data)))
            listing=sftp.listdir_attr(directory)
            for item in listing:
                if item.filename.endswith((".json",".jsonl",".log")): pull(directory+"/"+item.filename,item.filename)
            try:
                for item in sftp.listdir_attr(directory+"/prepared_dataset"):
                    if item.filename.endswith((".json",".jsonl",".log")):
                        pull(directory+"/prepared_dataset/"+item.filename,"prepared_dataset/"+item.filename)
            except FileNotFoundError:pass
            job=actual["job"]
            for key,ref in job["references"].items():
                if isinstance(ref,dict) and ref.get("path"):
                    pull(ref["path"],"frozen_references/"+key+"/"+Path(ref["path"]).name)
            pid=actual["status"].get("pid")
            code=f'''import pathlib,json,datetime
p=pathlib.Path('/proc/{int(pid or 0)}');r={{'pid':{pid!r},'exists':p.exists(),'observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}}
if p.exists():
 text=(p/'stat').read_text();r['birth_ticks']=int(text[text.rfind(')')+2:].split()[19]);r['cmdline']=(p/'cmdline').read_bytes().replace(b'\\0',b' ').decode()
root=pathlib.Path({job['dataset_spec']['raw_shared_root']!r});r['raw_members']=[{{'name':q.name,'bytes':q.stat().st_size}} for q in root.iterdir()] if root.exists() else []
print(json.dumps(r))'''
            observed=ops.remote_python(code,port,timeout=30)
            if observed["exit_code"]:raise RuntimeError(observed["stderr"][-700:])
            process=json.loads(observed["stdout"].splitlines()[-1])
            (local/"process_and_raw_observation.json").write_text(json.dumps(process,indent=2))
            rows.append(dict(port=port,run_id=run_id,stage=actual["status"]["stage"],error=actual["status"].get("error"),
                             process=process,preserved=saved))
        sftp.close()
    finally:client.close()
    return rows


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--output",required=True);args=parser.parse_args()
    root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
    with ThreadPoolExecutor(max_workers=3) as pool:rows=sum(list(pool.map(lambda p:snapshot(p,root),TARGETS)),[])
    record=dict(observed_utc=datetime.now(timezone.utc).isoformat(),mode="read_only_failure_preservation",rows=rows,
                original_scientific_commit="fbd3a743dc93ac14a7352e4f66a30db9194dc300",port74_accessed=False)
    (root/"snapshot.json").write_text(json.dumps(record,indent=2),encoding="utf-8")
    for row in rows:print(json.dumps({k:row[k] for k in ("port","run_id","stage","error")}))


if __name__=="__main__":main()
