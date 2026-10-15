"""Explicit value-only T0 binding: Authority's mappingproxy is never pickled."""
from types import MappingProxyType

def pack_environment(env):
    if not hasattr(env,'stage_input'):return dict(schema='W28FakeBindingV1',environment=env)
    stage=env.stage_input;authority=stage.authority
    values=dict(env.__dict__);values.pop('stage_input')
    stage_values=dict(stage.__dict__);stage_values.pop('authority')
    return dict(schema='W28EnvironmentBindingV1',environment_values=values,stage_values=stage_values,
        authority_values=dict(case_by_id=dict(authority.case_by_id),layout_shas=sorted(authority.layout_shas),
            allocation_shas=sorted(authority.allocation_shas),manifest_sha=authority.manifest_sha))

def unpack_environment(binding):
    if binding['schema']=='W28FakeBindingV1':return binding['environment']
    if binding['schema']!='W28EnvironmentBindingV1':raise ValueError('unknown environment binding schema')
    from satellite_rl.t0_width.environment import T0WidthEnv
    from satellite_rl.shared_interface.stages import UniformStage2Input
    from satellite_rl.shared_interface.contracts import Authority
    from satellite_rl.shared_interface.spherical_gaussian import PHYSICS_SHA
    from satellite_rl.t0_width.logging import content_sha
    authority=Authority.__new__(Authority);values=binding['authority_values']
    authority.case_by_id=MappingProxyType(dict(values['case_by_id']));authority.layout_shas=frozenset(values['layout_shas'])
    authority.allocation_shas=frozenset(values['allocation_shas']);authority.manifest_sha=values['manifest_sha']
    stage=UniformStage2Input.__new__(UniformStage2Input);stage.__dict__.update(binding['stage_values']);stage.authority=authority
    env=T0WidthEnv.__new__(T0WidthEnv);env.__dict__.update(binding['environment_values']);env.stage_input=stage
    if (env.layout['content_sha256']!=stage.layout['content_sha256'] or env.kernel['layout_SHA']!=env.layout['content_sha256']
            or env.kernel['physics_SHA']!=PHYSICS_SHA or content_sha(env.physics)!=content_sha(stage.physics)
            or len(env.ids)!=env.N):raise ValueError('restored immutable environment binding identity differs')
    for array in env.kernel.values():
        if hasattr(array,'setflags'):array.setflags(write=False)
    return env
