"""Content-identical W26 train/dev reuse across the two authorized accounts."""
import hashlib,json,tarfile
from pathlib import Path
from . import remote_ops as r

PARENT=Path('D:/paper/outputs/experiments/s2_width26_20261014')
CORE='9f187e4cd8122a6fce415b590d0ab0a385724b7d840cc88dd874e34ea936aea5'

def main():
 dest=r.LOCAL/'data/training_reuse_01';dest.mkdir(parents=True,exist_ok=True)
 objects=json.loads((PARENT/'collected/original_to_local.json').read_text())
 original={role:json.loads((PARENT/'collected/inputs'/(role+'_published.json')).read_text())for role in ('train','development')}
 pairs={role:[[row['case_content_SHA'],row['layout_content_SHA']]for row in value['rows']]for role,value in original.items()}
 identity=hashlib.sha256(json.dumps(dict(source_SHA=CORE,ordered_content_SHAs=pairs),sort_keys=True,separators=(',',':')).encode()).hexdigest()
 unique={}
 for role,n in [('train',576),('development',20)]:
  value=original[role];assert value['root_approved']and value['role']==role and len(value['rows'])==n
  for row in value['rows']:
   for key in ('case_ref','layout_ref','case_authority_ref','publication_authority_ref'):
    old=row[key];local=objects[old['path']];assert local['sha256']==old['sha256']
    path=Path(local['path']);assert r.sha(path)==old['sha256'];unique[old['sha256']]=path
 transport=dest/'training_objects.tar.gz'
 with tarfile.open(transport,'x:gz',compresslevel=1)as tar:
  for sha,path in sorted(unique.items()):tar.add(path,arcname=sha,recursive=False)
 copied_ref=r.publish(20627,transport,r.NODES[20627]['base']+'/data/training_reuse_01/training_objects.tar.gz')
 root=r.NODES[20627]['base']+'/shared_data/objects'
 expected={sha:path.stat().st_size for sha,path in unique.items()}
 code='''
import pathlib,json,hashlib,tarfile,os
root=pathlib.Path(OBJECT_ROOT);root.mkdir(parents=True,exist_ok=True)
archive=pathlib.Path(ARCHIVE_FILE);expected=EXPECTED
assert hashlib.sha256(archive.read_bytes()).hexdigest()==ARCHIVE_HASH
seen=set()
with tarfile.open(archive,'r:gz')as tar:
 for member in tar:
  if not member.isfile()or member.name not in expected or member.name in seen or member.size!=expected[member.name]:raise ValueError('exact object archive')
  raw=tar.extractfile(member).read()
  if hashlib.sha256(raw).hexdigest()!=member.name:raise ValueError('object byte SHA')
  path=root/member.name
  if path.exists():
   if hashlib.sha256(path.read_bytes()).hexdigest()!=member.name:raise ValueError('existing object differs')
  else:
   with path.open('xb')as stream:stream.write(raw)
  seen.add(member.name)
assert seen==set(expected)
print(json.dumps(dict(passed=True,objects=len(seen),bytes=sum(expected.values()),user_uid=os.getuid())))
'''.replace('OBJECT_ROOT',repr(root)).replace('ARCHIVE_FILE',repr(copied_ref['path'])).replace('ARCHIVE_HASH',repr(copied_ref['sha256'])).replace('EXPECTED',repr(expected))
 readback=r.remote_python(20627,code,interpreter='/usr/bin/python3',timeout=180)
 r.write_once(dest/'new_account_object_readback.json',readback)
 if readback['exit_code']:raise RuntimeError(readback['stderr'])
 gates={};manifests={}
 for port in r.NODES:
  refs={}
  for role,value in original.items():
   value=json.loads(json.dumps(value))
   if port==20627:
    for row in value['rows']:
     for key in ('case_ref','layout_ref','case_authority_ref','publication_authority_ref'):
      row[key]['path']=root+'/'+row[key]['sha256']
   path=dest/(str(port)+'_'+role+'.json');r.write_once(path,value)
   refs[role]=r.publish(port,path,r.NODES[port]['base']+'/shared_data/training_reuse_01/'+str(port)+'_'+role+'.json')
  gate=dict(schema='W27TrainingDataAdmissionV1',experiment_id='S2-W27',root_approved=True,passed=True,executor_port=port,
   source_SHA=CORE,qualified_counts=dict(train=576,development=20),logical_dataset_identity=identity,
   ordered_content_SHAs=pairs,partition_manifest_SHAs={role:ref['sha256']for role,ref in refs.items()},
   original_W26_dataset_gate_SHA='5d31f0a2961ee3cb1d7e6e17eb6bbfa81b6dce3aca5fed955f5109b1391eec87',
   original_semantics_unchanged='same aircraft/layout typed bytes, beam IDs/centres/width/requests and row order; only explicit transport ref paths rebased for20627',
   new_data_prepare_calls=0,known40_arrays_opened=0)
  path=dest/(str(port)+'_data_admission.json');r.write_once(path,gate)
  gates[str(port)]=r.publish(port,path,r.NODES[port]['base']+'/shared_data/training_reuse_01/'+path.name);manifests[str(port)]=refs
 summary=dict(schema='W27TrainingReuseTransportV1',passed=True,logical_dataset_identity=identity,manifests=manifests,admissions=gates,
  new_account_objects=len(unique),new_account_original_bytes=sum(expected.values()),transport_ref=copied_ref,
  source_SHA=CORE,new_data_prepare_calls=0,known40_arrays_opened=0)
 r.write_once(dest/'transport_receipt.json',summary)
 print(json.dumps(dict(passed=True,logical_dataset_identity=identity,objects=len(unique),bytes=sum(expected.values()))))

if __name__=='__main__':main()
