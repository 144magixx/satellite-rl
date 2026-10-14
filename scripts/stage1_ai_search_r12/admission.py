"""Fail-closed prospective checks, no SSH, service, solver or controller entry."""
from datetime import datetime, timezone

DEADLINE = datetime(2026, 10, 14, 23, tzinfo=timezone.utc)
STOP_NEW = datetime(2026, 10, 14, 22, 58, tzinfo=timezone.utc)
INDICES = (0, 6, 12, 18, 24, 36, 42, 48, 54, 60, 72, 78)
# Mathematical binding only. Actual calls still require the separate new root
# execution review, exact source/fixture/input and fresh runtime qualification.
EXECUTION_DESIGN_VERSION = 'R12ContactSlackTwoStepMechanismDesignV2'
DESIGN_SHA='717707c442fa38a818042ae6db2c9515f923e8490683ca1bb1937687dd8896fa'
DESIGN_ACCEPTANCE_SHA='b02302670e6e86a91efb541315eb2f5560e5ef4e173ddf26be773923782afe19'

def require_execution_design(plan):
    if (plan.get('mathematical_design_version')!=EXECUTION_DESIGN_VERSION
        or plan.get('design_ref',{}).get('sha256')!=DESIGN_SHA
        or plan.get('root_design_acceptance_ref',{}).get('sha256')!=DESIGN_ACCEPTANCE_SHA):
        raise ValueError('sole exact root-accepted V2 mathematical contract; V1 remains blocked')
GLOBAL_PAUSE='/home/u2024110224/first_stage_satellite_rl/s1_ai17/control/user_pause_20261015_0700.json'
PORT_PAUSES=tuple('/home/u2024110224/first_stage_satellite_rl/s1_ai17/'+str(port)+'/control/deadline_20261015_0700/user_pause_20261015_0700_state.json' for port in (20473,20474))

def pause_inventory(plan):
    paths=plan.get('watched_pause_paths',[]);own=plan.get('own_stop_marker')
    required={GLOBAL_PAUSE,*PORT_PAUSES}
    if not own or not required.issubset(paths) or own not in paths or own in required or len(paths)!=len(set(paths)):
        raise ValueError('new15th global+both73/74 persistent markers+own stop required')
    return set(paths)

def check(plan, review, *, now, marker_values):
    require_execution_design(plan)
    if now.tzinfo is None:
        raise ValueError('timezone required')
    if now >= STOP_NEW or (DEADLINE - now).total_seconds() < 2550:
        raise ValueError('hard pause envelope; new human continue required after pause')
    if plan.get('schema')!='R12ProductionPlanV2' or plan.get('group') != 'S1-AI17-R12' or plan.get('executor_port') not in (20473, 20474):
        raise ValueError('R12 candidate resource only')
    if plan.get('pause_deadline_utc') != '2026-10-14T23:00:00Z':
        raise ValueError('new pause contract binding')
    paths = plan.get('watched_pause_paths', [])
    if pause_inventory(plan) != set(marker_values):
        raise ValueError('both persistent markers must be freshly read')
    if any(value is not None for value in marker_values.values()):
        raise ValueError('pause marker present; no exemptions')
    required = ('execution_authorized', 'source_accepted', 'fixture_accepted',
                'input_accepted', 'runtime_accepted', 'disk_accepted', 'pause_accepted')
    if review.get('schema') != 'RootR12ExecutionAdmissionV2' or any(review.get(k) is not True for k in required):
        raise ValueError('independent root admission incomplete')
    if review.get('plan_content_SHA') != plan.get('content_SHA') or not plan.get('content_SHA'):
        raise ValueError('exact plan binding')
    if tuple(plan.get('indices', ())) != INDICES or plan.get('request_bps') != 21600000:
        raise ValueError('full12 fixed data contract')
    route = review.get('fresh_route', {})
    age = now.timestamp() - route.get('observed_unix_seconds', float('-inf'))
    if not -5 <= age <= 120 or route.get('port') != plan['executor_port']:
        raise ValueError('fresh actual route required')
    if any(not route.get(k) for k in ('boot_id', 'python', 'task_id', 'owner', 'source_SHA')):
        raise ValueError('boot/interpreter/task/owner/source evidence required')
    if route.get('free_bytes', 0) < 8589934592 or route.get('conflict_free') is not True:
        raise ValueError('actual space and owner checks')
    return True
