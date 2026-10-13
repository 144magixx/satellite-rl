"""Bounded, read-only original74 monitor; no task creation or resume methods."""
import argparse,datetime,json,os,time,urllib.request
from pathlib import Path

RUN='r7_labels_20261013_234807_312cdce7'
def atomic(path,data):
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(data,indent=2)+'\n');os.replace(temporary,path)

def main():
    p=argparse.ArgumentParser();p.add_argument('--output-root',required=True);p.add_argument('--deadline-utc',default='2026-10-13T23:00:00+00:00');p.add_argument('--interval-seconds',type=int,default=90);args=p.parse_args()
    if args.interval_seconds!=90 or args.deadline_utc!='2026-10-13T23:00:00+00:00':raise ValueError('root fixed90 seconds and07:00 Beijing deadline')
    root=Path(args.output_root);root.mkdir(parents=True,exist_ok=True);deadline=datetime.datetime.fromisoformat(args.deadline_utc)
    state=dict(schema='R7RecoveryReadOnlyMonitorV1',run_id=RUN,interval_seconds=90,deadline_utc=args.deadline_utc,maximum_GET_calls=256,actual_POST=0,science_dispatch=False,compute_deadline_guard_is_separate=True)
    errors=0
    for count in range(256):
        now=datetime.datetime.now(datetime.timezone.utc)
        if now>=deadline:state.update(stage='deadline_reached',checked_utc=now.isoformat());break
        try:
            request=urllib.request.Request('http://127.0.0.1:9015/api/ai-search-r7/tasks/'+RUN,method='GET')
            with urllib.request.urlopen(request,timeout=15) as response:detail=json.loads(response.read())
            if detail['job']['run_id']!=RUN:raise ValueError('foreign task response')
            entry=dict(checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),GET_index=count+1,status=detail['status'])
            errors=0;state.update(stage='observing',latest=entry,actual_GET_calls=count+1)
            with (root/'observations.jsonl').open('a') as output:output.write(json.dumps(entry)+'\n');output.flush()
            if detail['status']['stage'] in ('completed','failed','stopped','cancelled','interrupted'):
                state.update(stage='terminal',terminal_status=detail['status']);atomic(root/'state.json',state);break
        except Exception as error:
            errors+=1;state.update(stage='GET_error',consecutive_GET_errors=errors,last_error=dict(type=type(error).__name__,message=str(error)),actual_GET_calls=count+1)
            if errors>=3:state['stage']='attention_required';atomic(root/'state.json',state);break
        atomic(root/'state.json',state)
        time.sleep(max(0,min(90,(deadline-datetime.datetime.now(datetime.timezone.utc)).total_seconds())))
    else:state['stage']='bounded_GET_limit'
    atomic(root/'state.json',state)

if __name__=='__main__':main()
