"""Actual config dataset SHA == accepted labels gate == qualified source table."""
from .storage import read_json
from .coverage_ai_search import pin

def validate_training_dataset(refs,dataset_ref,model_identity=None):
    if dataset_ref.get('role')!='train' or dataset_ref.get('count')!=576 or not dataset_ref.get('sha256'):raise ValueError('actual new uniform training576 dataset ref')
    dataset=read_json(pin(dataset_ref),{});gate=read_json(pin(refs['labels_gate']),{});qualified=read_json(pin(refs['qualified']),{})
    if dataset.get('schema')!='R7UniformDatasetV1' or dataset.get('role')!='train' or dataset.get('count')!=576 or dataset.get('fixed_request_bps')!=21600000:raise ValueError('actual uniform training dataset object')
    if gate.get('group')!='S1-AI17-R7' or gate.get('phase')!='labels' or gate.get('passed')is not True:raise ValueError('actual accepted R7 labels gate')
    if model_identity is not None and gate.get('model_identity')!=model_identity:raise ValueError('labels gate belongs to another model identity')
    if qualified.get('schema')!='R7QualifiedTrainManifestV1' or qualified.get('role')!='train' or qualified.get('labels_gate_sha256')!=refs['labels_gate']['sha256']:raise ValueError('actual qualified table from this labels gate')
    if gate.get('source_dataset_sha256')!=dataset_ref['sha256'] or qualified.get('source_dataset_sha256')!=dataset_ref['sha256']:raise ValueError('config training dataset differs from labels/qualified actual source SHA')
    if refs.get('authority')!=dataset.get('authority_ref'):raise ValueError('actual dataset and supplied root aircraft authority differ')
    return dict(source_dataset_sha256=dataset_ref['sha256'],labels_gate_ref=refs['labels_gate'],qualified_ref=refs['qualified'],authority_ref=refs['authority'])
