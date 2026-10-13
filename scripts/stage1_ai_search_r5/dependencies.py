"""Metadata-only dependency collector; never reads raw train/val/test arrays."""
import argparse,json
from pathlib import Path
from satellite_rl.training.run_monitor import atomic_json
from .plan_builder import digest
from satellite_rl.workbench.coverage_ai_search_r5 import REGISTRY_SHA

def collect(registry_path,output):
 if digest(registry_path)!=REGISTRY_SHA:raise ValueError('exact registry SHA')
 reg=json.loads(Path(registry_path).read_text('utf-8'));refs={}
 def walk(node):
  if isinstance(node,dict):
   if 'path' in node and 'sha256' in node:
    key=(node['path'],node['sha256']);refs[key]={'original_path':node['path'],'sha256':node['sha256'],'local_path':node.get('local_path'),'remote_path':node.get('remote_path',node['path']),'dependency_only_not_duplicate_evidence':True}
   for value in node.values():walk(value)
  elif isinstance(node,list):
   for value in node:walk(value)
 for case in reg['cases']:walk(case)
 walk(reg['source_manifests']);walk(reg['historical_unknowns'])
 result={'schema':'R5P0DependencyInventoryV1','registry_ref':{'path':str(registry_path),'sha256':REGISTRY_SHA},'refs':list(refs.values()),'algorithm_calls':0,'raw_arrays_read':False,'training_validation_test_dependencies':False};atomic_json(output,result);return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--registry',required=True);p.add_argument('--output',required=True);a=p.parse_args();print(json.dumps({'dependencies':len(collect(a.registry,a.output)['refs']),'algorithm_calls':0}))
