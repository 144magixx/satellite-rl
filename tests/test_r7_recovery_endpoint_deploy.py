import importlib.util,json,unittest
from pathlib import Path

path=Path(__file__).resolve().parents[1]/'scripts/stage1_ai_search_r7/recovery_deploy_once.py'
spec=importlib.util.spec_from_file_location('deploy',path);deploy=importlib.util.module_from_spec(spec);spec.loader.exec_module(deploy)

class DeploymentTests(unittest.TestCase):
    def test_exact_endpoint_only_admission(self):
        plan=dict(schema='R7Labels74EndpointDeploymentPlanV1',operational_commit=deploy.OPS,scientific_commit=deploy.SCIENCE,run_id=deploy.RUN,ssh_port=20474,api_port=9015,python='/home/u2024110224/first_stage_satellite_rl/envs/DQN/bin/python',contract_ref=dict(path='a',sha256='1'*64),recovery_review_ref=dict(path='b',sha256='2'*64))
        raw=json.dumps(plan).encode();a=dict(schema='R7Labels74EndpointDeploymentAdmissionV1',passed=True,plan_sha256=deploy.digest(raw),deployment_script_sha256='3'*64,contract_ref=plan['contract_ref'],recovery_review_ref=plan['recovery_review_ref'],actual_POST_authorized=False)
        deploy.validate_admission(plan,raw,a,'3'*64)
        for change in (dict(actual_POST_authorized=True),dict(plan_sha256='4'*64),dict(deployment_script_sha256='4'*64),dict(passed=False)):
            with self.assertRaises(ValueError):deploy.validate_admission(plan,raw,dict(a,**change),'3'*64)
        with self.assertRaises(ValueError):deploy.validate_admission(dict(plan,ssh_port=20473),raw,a,'3'*64)

    def test_remote_program_is_bounded_get_only(self):
        compile(deploy.REMOTE,'remote_deployment','exec')
        self.assertIn('signal.pidfd_send_signal',deploy.REMOTE)
        self.assertNotIn('SIGKILL',deploy.REMOTE.replace('no SIGKILL',''))
        self.assertNotIn('ops.resume(',deploy.REMOTE)
        self.assertNotIn("method='POST'",deploy.REMOTE)
        self.assertIn("not (directory/'operations_recovery.json').exists()",deploy.REMOTE)

if __name__=='__main__':unittest.main()
