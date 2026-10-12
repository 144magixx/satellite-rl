"""Frozen bounded observer of exactly four original dataset IDs; no dispatch."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import hashlib,json,sys,time
from pathlib import Path


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--config",required=True)
    parser.add_argument("--frozen-control",required=True);args=parser.parse_args()
    sys.path.insert(0,args.frozen_control)
    from scripts.stage1_ai_search import ops
    from scripts.stage1_ai_search.controller import refresh_route
    from satellite_rl.workbench.geoload_runner import exclusive_lock
    from satellite_rl.training.run_monitor import atomic_json
    config_path=Path(args.config);raw=config_path.read_bytes();config=json.loads(raw)
    source=Path(__file__);source_sha=hashlib.sha256(source.read_bytes()).hexdigest()
    if source_sha!=config["observer_source_sha256"]:raise ValueError("observer source pin differs")
    if set(map(int,config["targets"]))!={20418,20424,20473}:raise ValueError("observer scope differs; 74 forbidden")
    root=config_path.parent;start=time.monotonic();iteration=0
    with exclusive_lock(root/"observer.lock"):
        while True:
            if config_path.read_bytes()!=raw or hashlib.sha256(source.read_bytes()).hexdigest()!=source_sha:
                raise ValueError("observer source/config changed")
            if (root/"stop.request").exists() or time.monotonic()-start>=config["maximum_seconds"]:
                atomic_json(root/"state.json",{"phase":"observer_stopped_workers_preserved","observed_utc":datetime.now(timezone.utc).isoformat()});return
            def lane(port):
                actual=ops.api(port,"/api/ai-search/tasks")
                targets=config["targets"][str(port)];found=[]
                for item in actual["tasks"]:
                    job=item["job"];status=item["status"]
                    if job["run_id"] in targets:
                        if job["scientific_commit"]!=config["scientific_commit"] or job["mode"]!="dataset":
                            raise ValueError("original dataset identity/epoch changed")
                        found.append(dict(port=port,run_id=job["run_id"],status=status,source_root=job["source_root"],
                            source_manifest_sha256=job["source_manifest_sha256"],config_sha256=job["config_sha256"]))
                    elif job.get("scientific_commit")!=config["scientific_commit"] and status.get("stage") in ("running","queued","queued_resources"):
                        raise ValueError("new science epoch exists; original observer refuses route mutation")
                if len(found)!=len(targets):raise ValueError("original dataset task missing")
                route=refresh_route(port,config["scientific_commit"])
                return dict(port=port,route=route,targets=found)
            try:
                with ThreadPoolExecutor(max_workers=3) as pool:lanes=list(pool.map(lane,map(int,config["targets"])))
                rows=[row for lane in lanes for row in lane["targets"]]
                stages=[r["status"]["stage"] for r in rows]
                phase="completed" if all(s=="completed" for s in stages) else (
                    "attention_required" if any(s in ("failed","stopped","interrupted","cancelled") for s in stages) else "observing_download_or_preparation")
                state=dict(phase=phase,observed_utc=datetime.now(timezone.utc).isoformat(),iteration=iteration,
                    elapsed_seconds=time.monotonic()-start,rows=rows,routes=[l["route"] for l in lanes],
                    config_sha256=hashlib.sha256(raw).hexdigest(),observer_source_sha256=source_sha,
                    science_direct_calls=0,new_job_creations=0,resume_api_calls=0,port74_accessed=False)
                atomic_json(root/"state.json",state)
                with (root/"progress.jsonl").open("a",encoding="utf-8") as f:
                    f.write(json.dumps(dict(observed_utc=state["observed_utc"],phase=phase,rows=rows))+"\n")
                if phase in ("completed","attention_required"):return
            except Exception as exc:
                atomic_json(root/"observer_error.json",dict(observed_utc=datetime.now(timezone.utc).isoformat(),error=repr(exc),
                    iteration=iteration,workers_preserved=True))
                # A failed route/control refresh is attention, never authority
                # to reissue resume or manufacture another job.
                return
            iteration+=1;time.sleep(config["poll_seconds"])


if __name__=="__main__":main()
