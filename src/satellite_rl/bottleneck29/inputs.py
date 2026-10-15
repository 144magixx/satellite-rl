"""Read exact transported bytes; never generate a missing trajectory."""
import copy
import json
from pathlib import Path
from .ledger import Halt, digest


def input_loader(config):
    from satellite_rl.t0_width.data import load_published_layout
    from satellite_rl.shared_interface.stages import UniformStage2Input
    from satellite_rl.shared_interface.spherical_gaussian import PHYSICS
    from satellite_rl.shared_interface.content import read_wire
    base = Path(config['input_root']).resolve()
    def load(row):
        row = copy.deepcopy(row)
        for key in ('case_ref','layout_ref','case_authority_ref','publication_authority_ref'):
            ref = row[key]
            path = (base/ref['member']).resolve()
            if base not in path.parents or digest(path) != ref['sha256']:
                raise Halt('transport member identity mismatch')
            row[key] = dict(path=str(path), sha256=ref['sha256'])
        trajectory = row['saved_evaluation_ref']
        path = (base/trajectory['member']).resolve()
        if base not in path.parents or digest(path) != trajectory['sha256']:
            raise Halt('saved trajectory identity mismatch')
        saved=json.loads(path.read_text())
        allocation=read_wire(json.dumps(saved['allocation_wire']).encode())
        if (saved['node']!=150000 or saved['repeat']!=0 or saved['mother']!=row['mother']
                or saved['identity']['case_id']!=row['case_id']
                or saved['identity']['layout_SHA']!=row['layout_content_SHA']
                or allocation['processing_index_to_beam_id']!=row['processing_index_to_beam_id']):
            raise Halt('saved trajectory origin/order mismatch')
        order=row['processing_index_to_beam_id'];actions=row['actions_indices_by_processing_order']
        if len(actions)!=row['N'] or len(order)!=row['N']:raise Halt('trajectory length')
        requested=allocation['requested_actions_separate']
        if len(requested)!=row['N']:raise Halt('requested action completeness')
        for position,(beam_id,action) in enumerate(zip(order,actions)):
            index=allocation['beam_ids'].index(beam_id);f,l,p=action
            old=requested[position]
            if (old['beam_id']!=beam_id or [old['frequency_index'],old['count_index'],old['power_index']]!=action
                    or int(allocation['actual_start_slot'][index])!=f
                    or int(allocation['actual_slot_count'][index])!=l+1
                    or float(allocation['actual_whole_beam_power_w'][index])!=(p+1)*5):
                raise Halt('original requested/actual action mismatch')
        case, layout, authority = load_published_layout(row, role='development')
        stage_input = UniformStage2Input(case,layout,authority,PHYSICS)
        if list(stage_input.processing_index_to_beam_id) != row['processing_index_to_beam_id']:
            raise Halt('original processing order differs')
        return stage_input, row['actions_indices_by_processing_order']
    return load
