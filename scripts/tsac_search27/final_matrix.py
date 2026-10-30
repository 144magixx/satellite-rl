"""Offline W27 final known-matrix contract and audited result export.

No deployment, HTTP, SSH, model loading, or optimizer update occurs here.
Root stages exact terminal inventory on one qualified card, freezes evaluation
source, reviews this candidate, publishes the lock, and launches one worker.
"""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
from scripts.tsac_search27.remote_ops import write_once,sha
from scripts.t0_width.report_analysis import audit_terminal,action_mismatch,decode_array,bootstrap
from satellite_rl.tsac_search27.known_evaluation import jobs_for, same_schema


def read(path):return json.loads(Path(path).read_text(encoding='utf8'))

def policy_reference_catalog(science,science_plans,w26_science_plan,w26_confirmation_plan,node):
    """Read plan metadata only; never load policies, known aircraft or layouts."""
    from satellite_rl.tsac_search27.agent import schema_identity as new_schema
    from satellite_rl.t0_width.agent import schema_identity as old_schema
    from satellite_rl.t0_width.logging import content_sha
    if node not in (50000,150000):raise ValueError('only exact registered common nodes')
    catalog={}
    for variant,job in science['jobs'].items():
        plan=science_plans[str(job['port'])]
        if plan['source_commit']!=science['source_commit']:raise ValueError('plan source commit mismatch')
        config=plan['tasks'][job['task_key']]['config']
        if config['variant']!=variant or config['group']!='S10' or config['run_id']!=job['run_id']:
            raise ValueError('actual scientific config variant/run mismatch')
        catalog[variant]=dict(policy_ref=dict(path=config['model_dir']+'/policy_%06d.pt'%node,sha256='REQUIRE_ACTUAL_FILE_SHA'),
            schema=new_schema(variant),config_SHA=content_sha(config),source_SHA=config['source_sha'],data_manifest_SHA=config['data_manifest_sha'],
            env_step=node,origin_run_id=job['run_id'],source_commit=science['source_commit'],origin_executor_port=job['port'],
            executor_port='SET_SINGLE_EVALUATION_PORT',root_approved=False,fixed_node_not_best=True,owner_exit_verified=False,
            policy_file_SHA_verified=False,stage='completed' if node==150000 else 'pilot_complete')
    old=[t['config'] for t in w26_science_plan['tasks'].values() if t.get('mode')=='train' and t['config'].get('group')=='S10']
    if len(old)!=1:raise ValueError('exact original W26 S10 config required')
    old=old[0];confirm=[t['config'] for t in w26_confirmation_plan['tasks'].values() if t.get('mode')=='evaluate']
    if len(confirm)!=1:raise ValueError('original W26 confirmation metadata required')
    confirm=confirm[0];old_ref=dict(confirm['terminal_policies']['S10'])
    # W26 workbench.create injects these fields before hashing/saving policies.
    # Preserve the original plan; reconstruct its exact effective producer config.
    from pathlib import PurePosixPath
    model_dir=PurePosixPath(old_ref['path']).parent
    old=dict(old,run_id=model_dir.name,model_dir=str(model_dir))
    if node==50000:old_ref=dict(path=old_ref['path'].replace('policy_150000.pt','policy_050000.pt'),sha256='REQUIRE_ACTUAL_FILE_SHA')
    catalog['HIST']=dict(policy_ref=old_ref,schema=old_schema('S10'),config_SHA=content_sha(old),source_SHA=old['source_sha'],
        data_manifest_SHA=old['data_manifest_sha'],env_step=node,group='S10',historical_W26=True,fixed_node_not_best=True,
        root_approved=False,owner_exit_verified=False,policy_file_SHA_verified=False,executor_port='SET_SINGLE_EVALUATION_PORT')
    manifest=confirm['confirmation_manifest_ref'];unlock=dict(confirm['confirmation_unlock'],manifest_ref=manifest)
    return dict(schema='W27PolicyReferenceCatalogV1',metadata_only=True,models_loaded=0,known40_arrays_read=0,node=node,
                inventory_candidates=catalog,known_manifest_ref=manifest,w26_unlock=unlock,
                unresolved=['new policy real file SHAs','all scientific actual endpoint and owner exits','single GPU staged file references','root inventory approval','future evaluation source freeze and terminal lock'])


def build_candidate(ready,decision,science,inventory,*,evaluation_commit,executor_port,rootpauseepoch,run_id,decision_ref,terminal_lock_ref,known_manifest_ref,w26_unlock):
    if ready.get('stage')!='ready_for_known_matrix' or ready.get('evaluation_started') is not False:
        raise ValueError('controller not ready; known scenes remain unopened')
    passed=decision.get('passed_variants')
    if decision.get('root_approved') is not True or decision.get('node')!=50000 or decision.get('known40_opened') is not False:
        raise ValueError('exact locked original pilot decision required')
    node=ready['node'];jobs=jobs_for(passed,node)
    if ready.get('passed_variants')!=passed:raise ValueError('ready state differs from gate decision')
    methods={m for m,r in jobs}-{'random','fixed','greedy'}
    if set(inventory)!=methods or executor_port not in (20418,20424,20627):raise ValueError('exact common-budget inventory and authorized single card required')
    if len(evaluation_commit)!=40 or any(c not in '0123456789abcdef' for c in evaluation_commit):raise ValueError('actual frozen evaluation Git commit required')
    if not inventory['HIST'].get('original_plan_ref'):raise ValueError('history original W26 plan SHA ref required')
    policies={};core=None;logical=None
    from satellite_rl.tsac_search27.agent import schema_identity as new_schema
    from satellite_rl.t0_width.agent import schema_identity as old_schema
    for method,item in inventory.items():
        expected_schema=old_schema('S10') if method=='HIST' else new_schema(method)
        if not same_schema(item.get('schema'),expected_schema):raise ValueError('strict LP/new variant or original history schema mismatch')
        if (item.get('root_approved') is not True or item.get('env_step')!=node or item.get('fixed_node_not_best') is not True
                or item.get('owner_exit_verified') is not True or item.get('policy_file_SHA_verified') is not True
                or item.get('executor_port')!=executor_port):
            raise ValueError('immutable fixed node policy staged and owner exited required')
        if method!='HIST':
            expected='completed' if node==150000 else 'pilot_complete'
            if item.get('stage')!=expected or item.get('origin_run_id')!=science['jobs'][method]['run_id'] or item.get('source_commit')!=science['source_commit']:
                raise ValueError('new policy actual original run/source/common endpoint differs')
            if core is None:core=item['source_SHA'];logical=item['data_manifest_SHA']
            elif core!=item['source_SHA'] or logical!=item['data_manifest_SHA']:raise ValueError('new arms do not share scientific/core data identity')
        elif item.get('group')!='S10' or item.get('historical_W26') is not True:
            raise ValueError('history must be exact W26 S10 same node')
        policies[method]={k:item[k] for k in ('policy_ref','schema','config_SHA','source_SHA','data_manifest_SHA','env_step')}
    if policies['HIST']['source_SHA']!=core:raise ValueError('historical geometry scientific CORE differs')
    if w26_unlock.get('manifest_ref')!=known_manifest_ref or w26_unlock.get('manifest_SHA')!=known_manifest_ref['sha256']:
        raise ValueError('original known40 W26 unlock exact partition required')
    config=dict(task_kind='known_evaluation',phase='known_scenes',run_id=run_id,executor_port=executor_port,
                rootpauseepoch=rootpauseepoch,device='cuda',source_sha=core,training_source_sha=core,data_manifest_sha=logical,
                training_source_commit=science['source_commit'],evaluation_source_commit=evaluation_commit,node=node,
                budget_episodes=40*len(jobs),known_manifest_ref=known_manifest_ref,w26_original_unlock=w26_unlock,
                terminal_lock_ref=terminal_lock_ref,policies=policies,historical_training_plan_ref=inventory['HIST']['original_plan_ref'])
    lock=dict(schema='W27KnownMatrixLockV1',root_approved=False,experiment_id='S2-W27',run_id=run_id,node=node,
              manifest_ref=known_manifest_ref,policies=policies,source_SHA=core,data_manifest_SHA=logical,
              training_source_commit=science['source_commit'],evaluation_source_commit=evaluation_commit,
              not_independent_test=True,all_continuation_decisions_locked=True,decision_ref=decision_ref,passed_variants=passed,
              soft_UTC='2026-10-14T22:58:00+00:00',hard_UTC='2026-10-14T23:00:00+00:00',
              historical_training_plan_ref=inventory['HIST']['original_plan_ref'],single_gpu_single_worker=True,algorithm_updates=0,new_training=0,new_data=0,new_seeds=0,S3_calls=0)
    return dict(schema='W27KnownMatrixCandidateV1',root_review_required=True,submitted=False,config=config,lock_candidate=lock,
                jobs=[dict(method=m,repeat=r) for m,r in jobs],known_scene_use='already used W26 scene review only',budget_episodes=40*len(jobs))


def export_results(config,run_dir,output,known_manifest_metadata=None):
    """Arithmetic-only full final-matrix audit/export; never claim independent test."""
    d=Path(run_dir);out=Path(output);out.mkdir(parents=True,exist_ok=True);passed=sorted(set(config['policies']) & {'E','LP'}) if config['node']==150000 else []
    expected=jobs_for(passed,config['node'])
    budget=40*len(expected)
    if config['budget_episodes']!=budget:raise ValueError('result contract budget differs')
    cursor=read(d/'known_cursor.json')
    if cursor['cursor']!=budget:raise ValueError('matrix not complete; no formal result export')
    from satellite_rl.t0_width.logging import content_sha
    if cursor['task_sha']!=content_sha(config):raise ValueError('immutable evaluation task identity differs')
    for name,offset in cursor['log_offsets'].items():
        p=d/'logs'/(name+'.jsonl.gz')
        if (p.stat().st_size if p.exists() else 0)!=offset:raise ValueError('unaccounted unique log suffix')
    if (d/'known_wal.pt').exists():raise ValueError('matrix WAL not closed')
    metadata=Path(known_manifest_metadata or config['known_manifest_ref']['path'])
    if sha(metadata)!=config['known_manifest_ref']['sha256']:raise ValueError('exact known40 metadata manifest required')
    manifest=read(metadata);mother_rows=manifest['rows']
    if len(mother_rows)!=40 or len({row['case_id'] for row in mother_rows})!=40:raise ValueError('known mother metadata not40 unique')
    actual={};tables=[];terminal_rows={};identities={};demand_signatures={}
    with gzip.open(d/'logs/evaluations.jsonl.gz','rt',encoding='utf8') as stream:
        for line in stream:
            row=json.loads(line);phase=row['phase']
            if not phase.startswith('known_') or row['node']!=config['node']:raise ValueError('phase/node mismatch')
            method=phase[len('known_'):];key=(row['mother'],method,row['repeat'])
            if key in actual or not 0<=row['mother']<40 or (method,row['repeat']) not in expected:raise ValueError('duplicate/unexpected evaluation identity')
            parent=mother_rows[row['mother']];identity=row['identity']
            if (identity['case_id']!=parent['case_id'] or identity['layout_SHA']!=parent['layout_content_SHA']
                    or identity['aircraft_SHA']!=parent['case_content_SHA']
                    or row['action_seed']!=275000+row['mother']+row['repeat']*1000000):raise ValueError('known mother/order/parent/seed mapping differs')
            signature=tuple(identity[k] for k in ('case_id','layout_SHA','aircraft_SHA','physics_SHA'))
            if row['mother'] in identities and identities[row['mother']]!=signature:raise ValueError('cross-method paired mother physics/identity differs')
            identities[row['mother']]=signature
            metrics,demand_signature,_=audit_terminal(row)
            if row['mother'] in demand_signatures and demand_signatures[row['mother']]!=demand_signature:raise ValueError('paired beam demand/ID signature differs')
            demand_signatures[row['mother']]=demand_signature
            if metrics['complete']!=1 or metrics['constraint_violations']:raise ValueError('incomplete or illegal terminal')
            terminal_rows[key]=row
            actual[key]=metrics;tables.append(dict(mother=row['mother'],method=method,repeat=row['repeat'],case_id=row['identity']['case_id'],**metrics))
    wanted={(m,name,r) for m in range(40) for name,r in expected}
    if set(actual)!=wanted:raise ValueError('missing complete method/mother/repeat matrix')
    observed_steps={key:set() for key in wanted};decision_count=0
    with gzip.open(d/'logs/steps.jsonl.gz','rt',encoding='utf8') as stream:
        for line in stream:
            step=json.loads(line);method=step['phase'].removeprefix('known_');key=(step['mother'],method,step['repeat'])
            if key not in wanted or step['node']!=config['node']:raise ValueError('unexpected step matrix identity')
            terminal=terminal_rows[key]
            if step['identity']!=terminal['identity'] or step['action_seed']!=terminal['action_seed']:raise ValueError('step/terminal identity or seed differs')
            if action_mismatch(step):raise ValueError('request/actual clipping detected')
            beam=step['beam_id'];wire=terminal['allocation_wire'];ids=wire['beam_ids']
            if beam not in ids or beam in observed_steps[key]:raise ValueError('duplicate/unexpected serviced beam')
            index=ids.index(beam);actual_action=step['actual_action']
            for field,array in [('start','actual_start_slot'),('count','actual_slot_count'),('power_w','actual_whole_beam_power_w')]:
                if actual_action[field]!=decode_array(wire[array])[index]:raise ValueError('actual action differs from terminal allocation')
            observed_steps[key].add(beam);decision_count+=1
    for key in wanted:
        if observed_steps[key]!=set(terminal_rows[key]['allocation_wire']['beam_ids']):raise ValueError('step evidence omits beams or N')
    metrics=('SI_mean','SGM','total_power_w','throughput_bps');summary={};mother_means={};timing={}
    for method in sorted({name for name,r in expected}):
        repeats=[r for name,r in expected if name==method]
        mother=[{metric:sum(actual[(m,method,r)][metric] for r in repeats)/len(repeats) for metric in metrics} for m in range(40)]
        mother_means[method]=mother
        method_rows=[terminal_rows[k] for k in terminal_rows if k[1]==method]
        timing[method]=dict(episodes=len(method_rows),resumed_episodes=sum(bool(row['timing'].get('resumed_from_WAL')) for row in method_rows),
            cumulative_algorithm_s=sum(row['timing'].get('algorithm_with_terminal_s',row['timing']['total_s']) for row in method_rows),
            invocation_wall_to_terminal_s=sum(row['timing'].get('wall_to_terminal_s',0) for row in method_rows),
            previous_algorithm_s=sum(row['timing'].get('previous_algorithm_s',0) for row in method_rows),
            wall_scope='last invocation per episode; not total resumed episode wall',request_execution_mismatches=0,constraint_violations=0)
        summary[method]=dict(mother_count=40,repeats=len(repeats),means={metric:sum(row[metric] for row in mother)/40 for metric in metrics})
    with (out/'known_episodes.csv').open('w',newline='',encoding='utf8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(tables[0]));writer.writeheader();writer.writerows(tables)
    contrasts=[]
    for candidate in sorted(set(summary)-{'B'}):
        for metric in ('SI_mean','SGM'):
            values=[mother_means[candidate][m][metric]-mother_means['B'][m][metric] for m in range(40)]
            contrasts.append(dict(candidate=candidate,reference='B',metric=metric,**bootstrap(values,seed=73027),interval='descriptive paired mother bootstrap only',resamples=10000))
    with (out/'known_mothers.csv').open('w',newline='',encoding='utf8') as stream:
        writer=csv.DictWriter(stream,fieldnames=['method','mother','case_id']+list(metrics));writer.writeheader()
        for method,mothers in mother_means.items():
            for index,row in enumerate(mothers):writer.writerow(dict(method=method,mother=index,case_id=mother_rows[index]['case_id'],**row))
    result=dict(schema='W27KnownResultsV1',node=config['node'],budget_episodes=budget,actual_episodes=len(tables),
                summary=summary,timing=timing,paired_contrasts=contrasts,actual_decisions=decision_count,not_independent_test=True,single_training_seed=51001,training_source_commit=config['training_source_commit'],
                evaluation_source_commit=config['evaluation_source_commit'],source_SHA=config['source_sha'],data_manifest_SHA=config['data_manifest_sha'],
                resume_wall_time_scope='per invocation; cumulative algorithm seconds must not be compared against resumed invocation wall time',
                constraints_passed=True,third_stage_calls=0)
    write_once(out/'known_summary.json',result)
    protected=[d/'known_cursor.json',metadata]+[d/name for name in ('config.json','job.json','status.json') if (d/name).exists()]+list((d/'logs').glob('*.jsonl.gz'))
    write_once(out/'archive_inventory.json',dict(schema='W27KnownArchiveInventoryV1',assets=[dict(path=str(p.resolve()),sha256=sha(p),bytes=p.stat().st_size) for p in protected],policy_refs=config['policies'],known_manifest_ref=config['known_manifest_ref'],copy_of_scientific_assets_created=False))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='action',required=True)
    build=sub.add_parser('candidate');build.add_argument('--inputs',required=True);build.add_argument('--output',required=True)
    catalog=sub.add_parser('catalog');catalog.add_argument('--science-state',required=True);catalog.add_argument('--science-plan-dir',required=True);catalog.add_argument('--w26-science-plan',required=True);catalog.add_argument('--w26-confirmation-plan',required=True);catalog.add_argument('--node',type=int,choices=[50000,150000],required=True);catalog.add_argument('--output',required=True)
    export=sub.add_parser('export');export.add_argument('--config',required=True);export.add_argument('--run-dir',required=True);export.add_argument('--output',required=True);export.add_argument('--known-manifest-metadata',required=True)
    args=p.parse_args()
    if args.action=='candidate':result=build_candidate(**read(args.inputs));write_once(args.output,result)
    elif args.action=='catalog':
        science=read(args.science_state);plans={str(job['port']):read(Path(args.science_plan_dir)/('plan_'+str(job['port'])+'.json')) for job in science['jobs'].values()}
        result=policy_reference_catalog(science,plans,read(args.w26_science_plan),read(args.w26_confirmation_plan),args.node);write_once(args.output,result)
    else:result=export_results(read(args.config),args.run_dir,args.output,args.known_manifest_metadata)
    print(json.dumps(dict(stage={'candidate':'candidate_only','catalog':'metadata_catalog_only','export':'audited_known_results'}[args.action],budget_episodes=result.get('budget_episodes'),models_loaded=0,submitted=False)))

if __name__=='__main__':main()
