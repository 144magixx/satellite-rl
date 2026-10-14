import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from satellite_rl.shared_interface.content import wire_bytes, content_sha
from satellite_rl.shared_interface.stages import make_case
from satellite_rl.t0_width.data import prepare_binding, prepare_candidate
from satellite_rl.t0_width.publication import CORE_SHA, audit_one_candidate, validate_audit_inputs


def save(path,value,wire=False):
    raw=wire_bytes(value)if wire else json.dumps(value).encode()
    path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
    return {'path':str(path),'sha256':hashlib.sha256(raw).hexdigest()}


class PublicationTests(unittest.TestCase):
    def test_independent_exact_candidate_geometry(self):
        with tempfile.TemporaryDirectory()as folder:
            root=Path(folder)
            case=make_case({'case_id':'audit-fixture','actual_role':'train','source_role':'train',
                'original_mother_id':'audit-fixture','UTC':'2024-01-08T00:02:00Z','source_asset_SHA':'1'*64,'source':{}},
                ['a','b'],[[45.,10.],[45.01,10.01]])
            authority={'schema':'R7PublishedParentRegistryV1','case_sha_by_id':{case['case_id']:case['content_sha256']},
                       'layout_shas':[],'allocation_shas':[]}
            row={'role':'train','case_id':case['case_id'],'case_content_SHA':case['content_sha256'],
                 'case_ref':save(root/'case.json',case,True),'case_authority_ref':save(root/'authority.json',authority),
                 'case_authority_content_SHA':content_sha(authority)}
            row.update(prepare_binding(row,'2'*64,'3'*64))
            request={'row':row,'source_SHA':CORE_SHA,'identity':row['prepare_identity'],'phase_manifest':row['prepare_phase_manifest']}
            request_ref=save(root/'request.json',request)
            prepare_candidate(request,root/'prepared')
            stage_ref={'path':str(root/'prepared/stage_output.json'),
                       'sha256':hashlib.sha256((root/'prepared/stage_output.json').read_bytes()).hexdigest()}
            stage=json.loads((root/'prepared/stage_output.json').read_bytes())
            indexed={'case_index':0,'candidate_ref':stage['candidate_ref'],'stage_output_ref':stage_ref,
                     'accepted_candidate_ref':save(root/'accepted.json',stage),'request_ref':request_ref,
                     'guard_ref':save(root/'guard.json',{'status':'completed','exit_code':0,'output_ref':stage_ref})}
            c,layout=audit_one_candidate(row,indexed)
            self.assertEqual(c['content_sha256'],case['content_sha256'])
            self.assertEqual(int(layout['request_bps_by_beam_int64'].sum()),43200000)
            changed={**request,'source_SHA':'0'*64}
            with self.assertRaisesRegex(ValueError,'request/manifest'):
                audit_one_candidate(row,{**indexed,'request_ref':save(root/'changed_request.json',changed)})

    def test_overlap_shards_rejected_without_layout_reads(self):
        with tempfile.TemporaryDirectory()as folder:
            root=Path(folder)
            rows=[]
            for i in range(636):
                role='train'if i<576 else'development'if i<596 else'test'
                row={'role':role,'case_id':str(i),'UTC':str(i)}
                if i<594:row['native_wenv_admission_ref']={'path':'unused','sha256':'1'*64}
                if role=='test':row['confirmation_source_admission_ref']={'path':'unused','sha256':'2'*64}
                rows.append(row)
            source=Path(__file__).parents[1]/'src/satellite_rl/coverage/ai_search_r5/core.py'
            core=root/'coverage/ai_search_r5/core.py';core.parent.mkdir(parents=True)
            raw=source.read_bytes().replace(b'\r\n',b'\n');core.write_bytes(raw)
            self.assertEqual(hashlib.sha256(raw).hexdigest(),CORE_SHA)
            manifest={'schema':'W26WenvPreparationManifestV1','method':'Wenv','rows':rows,'source_SHA':CORE_SHA,
                      'source_file_refs':[{'path':str(core),'sha256':CORE_SHA}],
                      'nominal_seconds':10,'hard_seconds':20,'raypairs_per_aircraft':1024}
            mref=save(root/'manifest.json',manifest);shards=[]
            for shard in range(4):
                indices=list(range(shard*159,(shard+1)*159))
                if shard==3:indices[0]=0
                native=sum('native_wenv_admission_ref'in rows[i]for i in indices)
                aggregate={'schema':'W26DataPreparationAggregateV1','manifest_SHA':mref['sha256'],'case_indices':indices,
                    'completed':159,'total':159,'candidate_ready':True,'all_dataset_total':636,'reused_published':0,
                    'T_calls':0,'D5_calls':0,'MILP_calls':0,'native_Wenv_imports':native,'new_prepare_started':159-native}
                aggregate_ref=save(root/('aggregate'+str(shard)+'.json'),aggregate)
                shards.append(save(root/('index'+str(shard)+'.json'),{'schema':'W26PreparationShardIndexV1',
                    'manifest_SHA':mref['sha256'],'case_indices':indices,'aggregate_ref':aggregate_ref,
                    'rows':[{'case_index':i}for i in indices]}))
            admission={'schema':'W26DataAuditExecutionAdmissionV1','root_approved':True,'source_commit':'a'*40,
                'manifest_SHA':mref['sha256'],'core_SHA':CORE_SHA,'expected_counts':{'train':576,'development':20,'test':40},
                'max_layout_checks':636,'remote_port':20418,'shard_index_SHAs':[r['sha256']for r in shards],
                'expected_native_imports':594,'expected_new_prepare':42}
            with self.assertRaisesRegex(ValueError,'overlap'):
                validate_audit_inputs({'prepare_manifest_ref':mref,'shard_index_refs':shards,
                    'audit_admission_ref':save(root/'audit.json',admission),'source_commit':'a'*40,'executor_port':20418})


if __name__=='__main__':unittest.main()
