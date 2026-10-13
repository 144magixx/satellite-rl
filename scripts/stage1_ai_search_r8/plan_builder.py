"""Prepare an unexecutable exact-plan draft; never register or POST a task."""
import argparse,json
from pathlib import Path
from .contracts import read,digest,atomic,GROUP

def draft(qualification_ref,design_ref,root_design_ref,archive_root,source_root,executor_port=20474):
 q=read(qualification_ref)
 if q.get('passed')is not True or len(q['cases'])!=20:raise ValueError('full qualified original20 required')
 own=Path(source_root);files={}
 for name in ('__init__.py','contracts.py','deadline.py','server.py','worker.py','child.py','plan_builder.py'):
  p=own/'scripts/stage1_ai_search_r8'/name;files['scripts/stage1_ai_search_r8/'+name]=dict(path=str(p),sha256=digest(p))
 return dict(schema='R8ExactCachedOwnerPilotPlanV1',group=GROUP,run_id=None,request_id=None,status='DRAFT_NOT_EXECUTABLE',executor_port=executor_port,archive_root=archive_root,scientific_commit=None,design_ref=design_ref,root_design_acceptance_ref=root_design_ref,input_qualification_ref=qualification_ref,case_indices=list(range(20)),arms=['G','A','R'],cases=q['cases'],checkpoint_ref=q['checkpoint_ref'],original_source_manifest_ref=q['original_source_manifest_ref'],source_files=files,core_source_root=str(own/'src'),core_ref=None,actual_resource_route_ref=None,deadline_registration_ref=None,scientific_new_calls=dict(local=60,D5=60,W=0,T=0,forward=0,updates=0),local_resume_admissions={},automatic_future_dispatch=False)
def main():
 p=argparse.ArgumentParser();p.add_argument('--qualification',required=True);p.add_argument('--design',required=True);p.add_argument('--root-design',required=True);p.add_argument('--archive-root',required=True);p.add_argument('--source-root',required=True);p.add_argument('--output',required=True);a=p.parse_args()
 refs=[dict(path=x,sha256=digest(x)) for x in (a.qualification,a.design,a.root_design)];value=draft(*refs,a.archive_root,a.source_root);out=Path(a.output)
 if out.exists():raise ValueError('new draft only')
 atomic(out,value)
if __name__=='__main__':main()
