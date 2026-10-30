"""Load only the SHA-pinned real Actor, never instantiate a Critic or optimizer.

No load/forward happens at module import. Real load and inference are allowed
only inside root-admitted engineering/science operations controlled by the caller.
"""
import hashlib,io,re
from pathlib import Path
import numpy as np
from .a1_policy import normalize_probabilities


class ActorOnly:
    def __init__(self,network,device,provenance):
        self.network=network;self.device=device;self.provenance=provenance

    @classmethod
    def load(cls,path,expected_SHA,*,device='cuda',expected_metadata=None,allow_cpu_diagnostic=False):
        if not isinstance(expected_SHA,str) or re.fullmatch('[0-9a-f]{64}',expected_SHA) is None:raise ValueError('full Actor SHA required')
        raw=Path(path).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=expected_SHA:raise ValueError('Actor file SHA mismatch')
        import torch
        from satellite_rl.t0_width.agent import WidthEncoder,build_config,schema_identity
        from satellite_rl.agents.workbench_networks import MultiHeadNetwork
        if not str(device).startswith('cuda') and not allow_cpu_diagnostic:raise ValueError('original Actor CUDA inference required; no silent CPU fallback')
        if str(device).startswith('cuda') and not torch.cuda.is_available():raise ValueError('CUDA unavailable')
        value=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=False)
        if value.get('schema')!=schema_identity('S10') or value.get('env_step')!=150000 or value.get('update_step')!=148001:raise ValueError('original S10/150k schema/node/update identity')
        for key,expected in (expected_metadata or {}).items():
            if value.get(key)!=expected:raise ValueError('Actor metadata mismatch: '+key)
        state=value['actor']
        if len(state)!=42 or any(not torch.is_tensor(t) or not bool(torch.isfinite(t).all()) for t in state.values()):raise ValueError('42 finite original Actor tensors required')
        # Preserve the caller's CPU RNG while creating the architecture. All
        # tensors must then load strictly before this object can be returned.
        with torch.random.fork_rng(devices=[]):
            config=build_config('S10',device='cpu',phase='scientific')
            network=MultiHeadNetwork(config,207,(100,10,10))
            network.encoder=WidthEncoder(network.encoder,'S10',51001+100003)
            network.load_state_dict(state,strict=True)
        if sum(t.numel() for t in network.parameters())!=418040:raise ValueError('Actor parameter count')
        network=network.to(device).eval();network.requires_grad_(False)
        provenance=dict(policy_SHA=expected_SHA,env_step=150000,update_step=148001,tensors=42,parameters=418040,
            schema=value['schema'],config_sha=value.get('config_sha'),source_sha=value.get('source_sha'),data_manifest_sha=value.get('data_manifest_sha'),
            device=str(device),Q_constructed=False,optimizers_constructed=False)
        return cls(network,torch.device(device),provenance)

    def forward_probs(self,observation,mask):
        import torch
        from satellite_rl.agents.action_masking import masked_policy
        if np.asarray(observation).shape!=(208,) or np.asarray(observation).dtype!=np.float32:raise ValueError('208 float32 observation required')
        self.network.eval()
        with torch.inference_mode():
            x=torch.as_tensor(observation,dtype=torch.float32,device=self.device).reshape(1,-1)
            fm=torch.as_tensor(mask['frequency_slots'],dtype=torch.bool,device=self.device).unsqueeze(0)
            pm=torch.as_tensor(mask['power'],dtype=torch.bool,device=self.device).unsqueeze(0)
            heads=self.network(x);fl,power,*_=masked_policy(heads,fm,pm)
        raw_fl=fl[0].cpu().numpy().copy();raw_power=power[0].cpu().numpy().copy()
        fl64,p64=normalize_probabilities(raw_fl,raw_power,mask)
        return dict(frequency_slots=fl64,power=p64,forward_calls=1)
