"""Metadata-only exact data-phase draft; never stages source or reads airplanes."""
from pathlib import PurePosixPath,Path
from copy import deepcopy
from satellite_rl.workbench.coverage_ai_search import pin
from satellite_rl.workbench.storage import read_json
from satellite_rl.workbench.r7_uniform_data import validate_recipe
from .contract import GROUP,DESIGN_SHA,UNIFORM_RECIPE_SHA,PORTS

def draft(commit,recipe_ref):
    if len(commit)!=40 or recipe_ref['sha256']!=UNIFORM_RECIPE_SHA:raise ValueError('actual new frozen source and root recipe required')
    recipe=read_json(pin(recipe_ref),{});validate_recipe(recipe)
    transports={}
    for port in PORTS:
        lane=f'/home/u2024110224/first_stage_satellite_rl/s1_ai17/{port}/round7'
        metadata=recipe['source_job_ref'];mapping={metadata['sha256']:lane+'/source_inputs/'+metadata['sha256']+'/job.json'}
        for row in recipe['rows']:mapping[row['source_asset_SHA']]=row['source_remote_path']
        transports[str(port)]=dict(schema='R7SourceTransportMapV1',group=GROUP,ssh_port=port,recipe_sha256=UNIFORM_RECIPE_SHA,sha256_to_path=mapping,
                                   source_job_copy_ref=metadata,source_job_destination=mapping[metadata['sha256']],must_hash_every_source_before_build=True)
    return dict(schema='R7DataPhaseMetadataDraftV1',group=GROUP,phase='data_train_dev',scientific_commit=commit,design_sha256=DESIGN_SHA,
                status='draft_pending_root_source_bridge_transport_and_exact_plan_admission',authorized=False,science_POSTs=0,
                recipe_ref=deepcopy(recipe_ref),selected_build_indices=list(range(596)),
                routing={str(i):PORTS[i%2] for i in range(596)},case_parallelism_per_lane=2,case_threads=2,
                minimum_free_bytes=8589934592,lane_parent_children_rss_bytes=4294967296,case_parent_children_rss_bytes=2147483648,
                phase_budget=dict(builds=596,prepare=0,T=0,D5=0,updates=0),transport_drafts=transports,
                prerequisites=dict(source_bridge_schema='R7CompleteSourceInterfaceBridgeV1',qualification_key='complete_source_interface_bridge',
                                   accepted_interface_receipt_sha256='9fbf8fbd1924c977a6a6dd7c066225ce9dcfcd263c7c6bff35fa1df7c2a65e54',
                                   ancestor_interface_source_commit='6496abe61d8bdc5029388ce024508b15a7e306c7',
                                   bridge_gate_phase='shared_interface',root_exact_plan_ref_required=True,root_fresh_runtime_resource_admission_required=True),
                labels_dispatch_allowed=False,all_model_or_heldout_successors_dispatch_allowed=False)
