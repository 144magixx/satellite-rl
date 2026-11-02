"""Local candidate graph only. No SSH, scientific calls, signatures or dispatch."""
import argparse,copy,hashlib,json,pathlib,subprocess
BASE=pathlib.Path('D:/paper/outputs/experiments/first_stage_ai_search_20261013')
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def ref(p):return dict(path=str(pathlib.Path(p).resolve()),sha256=sha(p),bytes=pathlib.Path(p).stat().st_size)
def pair(r):return {k:r[k] for k in ('path','sha256')}
def read(p):return json.loads(pathlib.Path(p).read_bytes())
def build(output,source_repo,commit,repo):
 output=pathlib.Path(output);output.mkdir(parents=True,exist_ok=False);exports=output/'git_blob_exports';exports.mkdir();bundlepath=BASE/'round10/scientific_preparation_01/scientific_final_02/root_scientific_metadata_bundle.json';assert sha(bundlepath)=='ee7a49f942fa4213c5aa0cf287797c48c24ff4376537d4b20ca5b5993fa2e419';oldbundle=read(bundlepath);files=copy.deepcopy(oldbundle['files']);lookup={v['runtime_ref']['path']:v for v in files};aliases=[]
 actual=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
 if actual!=commit:raise ValueError('exact committed operations epoch')
 def add(path,runtime,origin=None):
  item=dict(local_ref=ref(path),runtime_ref=dict(path=runtime,sha256=sha(path),bytes=pathlib.Path(path).stat().st_size))
  if runtime in lookup:
   if lookup[runtime]['runtime_ref']['sha256']!=item['runtime_ref']['sha256']:raise ValueError('ambiguous runtime byte mapping')
  else:files.append(item);lookup[runtime]=item
  if origin is not None:aliases.append(dict(runtime_ref=pair(item['runtime_ref']),origin_ref=origin))
  return pair(item['runtime_ref'])
 def emit(name,value,meta):
  path=output/name
  with path.open('x',encoding='utf8',newline='\n') as f:json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n')
  return add(path,meta+'/'+name)
 first=read(lookup[oldbundle['plans'][0]['plan_ref']['path']]['local_ref']['path']);keys=list(first['source_files'])+['scripts/stage1_ai_search_r10_recovery/'+n for n in ('__init__.py','contracts.py','storage.py','worker.py','server.py','controller.py','plan_builder.py')];sources={}
 for key in keys:
  raw=subprocess.check_output(['git','show',commit+':'+key],cwd=repo);digest=hashlib.sha256(raw).hexdigest();path=exports/(digest+'.raw')
  if not path.exists():path.write_bytes(raw)
  sources[key]=add(path,source_repo+'/'+key)
 shared='/home/u2024110224/first_stage_satellite_rl/s1_ai17/20473/round10/recovery_preparation_01/metadata/shared_'+commit[:12]
 ownrefs={}
 paths=dict(design_ref=BASE/'round10/recovery_design_01/same_id_storage_race_recovery_contract.json',root_design_acceptance_ref=BASE/'round10/recovery_design_01/root_recovery_design_acceptance.json',clarification_ref=BASE/'round10/recovery_design_01/root_private_temp_and_quota_clarification.json',independent_original_reuse_review_ref=BASE/'round10/recovery_independent_review_01/independent_original_reuse_and_zero_entry_review.json',reuse_map_ref=BASE/'round10/recovery_preparation_01/original_stage_reuse_and_remaining_map.json',zero_entry_source_fence_ref=BASE/'round10/recovery_preparation_01/original_failed_S2_before_algorithm_source_fence.json')
 # Clarification filename is resolved by its exact acceptedSHA, never guessed.
 if not paths['clarification_ref'].exists():
  found=[p for p in (BASE/'round10/recovery_design_01').glob('*.json') if sha(p)=='74eecbddd6e191fb7bde5b47a2f4d250e10d3ecb10bade9f2f42b7ed5d5eccc9']
  if len(found)!=1:raise ValueError('exact quota/private-temp clarification absent')
  paths['clarification_ref']=found[0]
 for key,path in paths.items():ownrefs[key]=add(path,shared+'/'+key+'.json',ref(path))
 reuse=read(paths['reuse_map_ref']);lanes=[]
 for oldlane in oldbundle['plans']:
  port=oldlane['executor_port'];oldp=read(lookup[oldlane['plan_ref']['path']]['local_ref']['path']);oldr=read(lookup[oldlane['root_review_ref']['path']]['local_ref']['path']);old_ca=read(lookup[oldlane['controller_admission_ref']['path']]['local_ref']['path']);old_e=read(lookup[oldp['human_epoch_ref']['path']]['local_ref']['path']);old_s=read(lookup[oldp['storage_inventory_ref']['path']]['local_ref']['path']);host=next(h for h in reuse['hosts'] if h['executor_port']==port);home=BASE/'round10/scientific_preparation_01/actual_scientific_collection_01'/('host_'+str(port));collection=read(home/'collection_manifest.json');extpath=home/'external_sameboot_exit.json';ext=read(extpath);epoch_root=str(pathlib.PurePosixPath(oldp['archive_root']).parent)+'/operations_recovery/storage_race_01';meta='/home/u2024110224/first_stage_satellite_rl/s1_ai17/'+str(port)+'/round10/recovery_preparation_01/metadata/'+oldp['run_id']+'/candidate_01';archive=epoch_root+'/archive';outer=epoch_root+'/control/outer';worker=epoch_root+'/control/worker'
  external_ref=add(extpath,meta+'/original_external_sameboot_exit.json',host['old_external_sameboot_ref']);collection_ref=add(home/'collection_manifest.json',meta+'/original_collection_manifest.json',ref(home/'collection_manifest.json'));inventory_ref=add(home/'canonical_file_inventory.json',meta+'/original_canonical_file_inventory.json',ref(home/'canonical_file_inventory.json'))
  # Immutable status copy has its own canonical bytes; original mutable path is
  # retained in the collected manifest and never used as a future status target.
  statuspath=home/'originals/output/status.json';status_snapshot=add(statuspath,meta+'/original_status_snapshot.json',ref(statuspath));protected=[v['remote_ref'] for v in collection['files']]
  remaining=[r for r in reuse['rows'] if r['executor_port']==port and not r['typed_output_exists']];native=sum(30 if r['stage']=='common' else (300-r['actual_stage_wall_seconds'] if r['native_stage_wal_exists'] else 300) for r in remaining);debit=ext['conservative_upper_seconds'];epochid='r10_storage_race_same_id_recovery_01_'+str(port)
  contract=dict(schema='R10StorageRaceSameIDRecoveryContractV1',**ownrefs,run_id=oldp['run_id'],executor_port=port,execution_epoch_id=epochid,original_plan_ref=oldlane['plan_ref'],original_root_review_ref=oldlane['root_review_ref'],original_archive_root=oldp['archive_root'],recovery_archive_root=archive,outer_control_root=outer,worker_control_root=worker,original_external_sameboot_ref=external_ref,original_collection_ref=collection_ref,original_canonical_inventory_ref=inventory_ref,original_status_snapshot_ref=status_snapshot,protected_original_refs=protected,lifecycle_dependencies=old_ca['lifecycle_dependencies'],old_outer_debit_seconds=debit,outer_hard_seconds=10800-debit,native_ceiling_seconds=min(native,9180-debit),new_native_max=len(remaining),new_distinct_reservations=len(remaining)-(1 if port==20473 else 0),old_algorithm_spend_exact=None,failed_S2_new_hard_seconds=298.376676261425 if port==20473 else None,core_commit='3e608d077c09209c436dc4581fb11310e4bbe141',original_operations_commit='0fcdefe875bbbae4a1b7a140bf30fe8c1e5d05d9',recovery_operations_commit=commit,execution_authorized_by_this_contract=False)
  c_ref=emit('recovery_contract_'+str(port)+'.json',contract,meta)
  e=copy.deepcopy(old_e);e.update(root_accepted=False,run_id=oldp['run_id'],request_id=oldp['request_id'],archive_root=archive,execution_epoch_id=epochid,new_deadline_unix_seconds=None,watched_pause_paths=sorted(set(old_e['watched_pause_paths'])|{oldp['archive_root']+'/stop.request',oldp['archive_root']+'/pause_state.json',archive+'/stop.request',archive+'/pause_state.json',outer+'/stop.request',outer+'/pause_state.json',outer+'/storage_pause.json',worker+'/stop.request',worker+'/pause_state.json',worker+'/storage_pause.json'}))
  for key in ('resource_snapshot_ref','root_accepted_utc'):e.pop(key,None)
  e_ref=emit('recovery_human_epoch_'+str(port)+'.json',e,meta)
  storage=copy.deepcopy(old_s);storage.update(root_accepted=False,archive_root=archive,personal_quota='unknown',observed_roots_include_old_new_source_metadata_controls=True)
  for key in ('actual_space_observation_ref','actual_mount_free_bytes'):storage.pop(key,None)
  s_ref=emit('recovery_storage_'+str(port)+'.json',storage,meta)
  route_ref=emit('recovery_route_'+str(port)+'.json',dict(group='S1-AI17-R10',ssh_port=port,ready=False,root_accepted=False,own_task_absence_verified=False,old_owner_handoff_verified=False,observed_unix_seconds=None,boot_id=None,python='/home/u2024110224/first_stage_satellite_rl/envs/DQN/bin/python'),meta)
  p=copy.deepcopy(oldp);p.update(source_repo=source_repo,source_files={k:sources[k] for k in first['source_files']},operations_commit=commit,original_operations_commit=oldp['operations_commit'],recovery_operations_commit=commit,recovery_source_files={k:sources[k] for k in keys if k.startswith('scripts/stage1_ai_search_r10_recovery/')},recovery_contract_ref=c_ref,original_plan_ref=oldlane['plan_ref'],original_archive_root=oldp['archive_root'],archive_root=archive,execution_epoch_id=epochid,human_epoch_ref=e_ref,storage_inventory_ref=s_ref,actual_resource_route_ref=route_ref,source_input_budget_frozen=False,controller_source_ref=sources['scripts/stage1_ai_search_r10_recovery/controller.py'],controller_lifecycle_dependencies=old_ca['lifecycle_dependencies'],outer_controller_hard_seconds=10800-debit,parent_IO_and_control_costs_separate=True)
  p['document_aliases']=copy.deepcopy(oldp.get('document_aliases',[]))+copy.deepcopy(aliases)
  p['original_controller_source_acceptance_ref']=p.pop('root_controller_source_acceptance_ref',None)
  p_ref=emit('recovery_plan_'+str(port)+'.json',p,meta)
  r=copy.deepcopy(oldr);r.update(exact_plan_ref=p_ref,human_epoch_ref=e_ref,storage_inventory_ref=s_ref,recovery_contract_ref=c_ref,passed=False,execution_authorized=False,scientific_execution_authorized=False,runtime_accepted=False,recovery_execution_authorized=False,old_reuse_accepted=False,cumulative_budget_accepted=False,zero_entry_reentry_accepted=False,reason='Candidate same-ID ops recovery; root exact source/proof/runtime/budget admission still required')
  r_ref=emit('recovery_root_review_'+str(port)+'.json',r,meta);lanes.append(dict(executor_port=port,run_id=p['run_id'],workbench_port=p['workbench_port'],plan_ref=p_ref,root_review_ref=r_ref,contract_ref=c_ref,human_epoch_ref=e_ref,storage_ref=s_ref,route_ref=route_ref,metadata_root=meta,archive_root=archive,outer_control_root=outer,worker_control_root=worker))
 graph=dict(schema='R10SameIDRecoveryCandidateGraphV1',operations_commit=commit,source_repo=source_repo,files=files,plans=lanes,root_dispatch_authorized=False,services_started=0,POST=0,scientific_calls=0,original72stage_denominator_preserved=True)
 path=output/'recovery_candidate_graph.json'
 with path.open('x',encoding='utf8',newline='\n') as f:json.dump(graph,f,ensure_ascii=False,indent=2);f.write('\n')
 print(json.dumps(ref(path)));return path
def main():
 a=argparse.ArgumentParser();a.add_argument('--output',required=True);a.add_argument('--source-repo',required=True);a.add_argument('--operations-commit',required=True);a.add_argument('--repo',default=str(pathlib.Path(__file__).resolve().parents[2]));v=a.parse_args();build(v.output,v.source_repo,v.operations_commit,v.repo)
if __name__=='__main__':main()
