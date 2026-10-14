"""Pure exact denominator/partition and permanent per-stage reservations."""
from pathlib import Path
from .contracts import ARMS,MOTHERS,check_identity,publish,read,typed_asset

def partition(rows,assignment,executor):
 if executor not in (20473,20474):raise ValueError('only admitted73/74')
 if len(rows)!=12 or any(type(v['original_index'])is not int for v in rows) or tuple(v['original_index'] for v in rows)!=MOTHERS:raise ValueError('original12 ordered mother denominator')
 if set(assignment)!={str(i) for i in MOTHERS} or any(v not in (20473,20474) for v in assignment.values()):raise ValueError('exact twelve mother host mapping')
 if any(sum(v==host for v in assignment.values())!=6 for host in (20473,20474)):raise ValueError('fixed6+6 partition')
 for row in rows:check_identity(row['identity'])
 if len({row['identity']['case_id'] for row in rows})!=12 or len({row['identity']['mother_id'] for row in rows})!=12:raise ValueError('mother identities must remain twelve distinct originals')
 return [v for v in rows if assignment[str(v['original_index'])]==executor]

def stage_order(rows):
 """One common followed by five serial methods per mother; never independent replicas."""
 return [(v['original_index'],stage) for v in rows for stage in ('common',*ARMS)]

class Reservations:
 def __init__(self,root,run_id,epoch,identities):self.root=Path(root);self.run_id=run_id;self.epoch=epoch;self.identities=identities
 def reserve(self,index,stage,input_refs,common_ref=None):
  if index not in self.identities or stage not in ('common',*ARMS):raise ValueError('undeclared stage/mother')
  identity=check_identity(self.identities[index]);directory=self.root/'mothers'/str(index)/stage
  if stage!='common':
   if common_ref is None:raise ValueError('same frozen common required')
   common=typed_asset(common_ref) if common_ref.get('compression') else read(common_ref)
   if common.get('original_index')!=index or common.get('identity')!=identity or common.get('run_id')!=self.run_id:raise ValueError('common provenance differs')
  return publish(directory/'stage.wal.json',dict(run_id=self.run_id,epoch=self.epoch,original_index=index,stage=stage,identity=identity,input_refs=input_refs,common_ref=common_ref,reservation='reserved_unknown_until_original_terminal_receipt',actual_algorithm_entry=None,automatic_retry=False),exclusive=True)
 def finish(self,index,stage,value):
  check_identity(value['identity'],self.identities[index])
  directory=self.root/'mothers'/str(index)/stage
  if not (directory/'stage.wal.json').exists():raise ValueError('no permanent reservation')
  if value.get('stage')!=stage or value.get('original_index')!=index or value.get('run_id')!=self.run_id:raise ValueError('terminal stage/run differs')
  return publish(directory/'terminal.json',value,exclusive=True)
