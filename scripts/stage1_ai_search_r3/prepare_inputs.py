"""Prepare exact remote aliases for small signed pilot inputs; no SSH or science."""
import argparse,json
from pathlib import Path
from copy import deepcopy
from satellite_rl.training.run_monitor import atomic_json
from . import ops
from .plan_builder import digest

def configure(inventory_path,identities_path,dataset_ref_path,vendor_refs_path,qualification_path,output):
 inventory_path=Path(inventory_path);source=json.loads(inventory_path.read_text('utf-8'));identity=json.loads(Path(identities_path).read_text('utf-8'));ids=identity.get('development_identities',identity.get('development',identity));source=deepcopy(source);rows={}
 if source.get('passed')is not True or source['case_count']!=24 or len(source['cases'])!=24 or source.get('new_geometry_or_solver_calls')!=0:raise ValueError('accepted original pilot24 anchors required')
 for case in source['cases']:
  for key,ref in case['external_refs'].items():
   local=Path(ref['path']);sha=digest(local)
   if sha!=ref['sha256']:raise ValueError('original source bytes changed')
   remote=ops.BASE+'/source/'+('R2_case_audits/' if key=='audit' else 'R2_anchors/')+sha+'.json';rows[remote]={'remote':remote,'local':str(local),'sha256':sha};ref['path']=remote
 folder=Path(output).resolve().parent;folder.mkdir(parents=True,exist_ok=True);identity_local=folder/'identity_inventory.json';atomic_json(identity_local,{'development':ids});identity_ref={'path':ops.BASE+'/source/identities/'+digest(identity_local)+'.json','sha256':digest(identity_local)};rows[identity_ref['path']]={'remote':identity_ref['path'],'local':str(identity_local),'sha256':identity_ref['sha256']}
 qualification=Path(qualification_path);q=json.loads(qualification.read_text('utf-8'));qref={'path':ops.BASE+'/source/qualifications/'+digest(qualification)+'.json','sha256':digest(qualification)}
 if q.get('passed')is not True:raise ValueError('root source-bound core qualification missing')
 rows[qref['path']]={'remote':qref['path'],'local':str(qualification),'sha256':qref['sha256']}
 vendors=json.loads(Path(vendor_refs_path).read_text('utf-8'));inputs={'reference_inventory':source,'original_reference_inventory_ref':{'path':str(inventory_path),'sha256':digest(inventory_path)},'identities':ids,'identity_ref':identity_ref,'dataset_ref':json.loads(Path(dataset_ref_path).read_text('utf-8')),'qualification_ref':qref,'vendor_refs':{key:{k:r[k] for k in ['path','sha256']} for key,r in vendors.items()},'metadata':list(rows.values())}
 if inputs['dataset_ref']['sha256']!=ids['sha256']:raise ValueError('original development identity/dataset mismatch')
 atomic_json(output,inputs);return {'path':str(output),'sha256':digest(output),'metadata_members':len(rows),'science_calls':0}
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__)
 for name in ['inventory','identities','dataset-ref','vendor-refs','qualification','output']:p.add_argument('--'+name,required=True)
 a=p.parse_args();print(json.dumps(configure(a.inventory,a.identities,a.dataset_ref,a.vendor_refs,a.qualification,a.output)))