"""Explicit v3 runtime; call in a fresh process before the first CUDA init."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

PROFILE=dict(schema='R6NumericRuntimeV3',torch='2.3.1+cu121',scipy='1.14.1',dtype='FP32',
    autocast=False,GradScaler=False,float32_matmul_precision='highest',cuda_matmul_allow_tf32=False,
    cudnn_allow_tf32=False,cudnn_benchmark=False,cudnn_deterministic=True,
    deterministic_algorithms=True,warn_only=False,CUBLAS_WORKSPACE_CONFIG=':4096:8',
    effective_batch=16,microbatch=4,partitions=[4,4,4,4],torch_compile=False,CUDA_graph_capture=False)


def json_sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def pre_cuda_environment():
    """No import-time torch/CUDA operation; incompatible pre-existing flags fail."""
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG') not in (None,PROFILE['CUBLAS_WORKSPACE_CONFIG']):
        raise RuntimeError('incompatible cuBLAS environment; no runtime fallback')
    torch=sys.modules.get('torch')
    if torch is not None and torch.cuda.is_initialized():raise RuntimeError('v3 must precede any CUDA initialization')
    os.environ['CUBLAS_WORKSPACE_CONFIG']=PROFILE['CUBLAS_WORKSPACE_CONFIG']


def runtime_flags(torch):
    return dict(float32_matmul_precision=torch.get_float32_matmul_precision(),
        cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,cudnn_benchmark=torch.backends.cudnn.benchmark,
        cudnn_deterministic=torch.backends.cudnn.deterministic,
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
        CUBLAS_WORKSPACE_CONFIG=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
        autocast_cuda=torch.is_autocast_enabled(),autocast_cpu=torch.is_autocast_cpu_enabled(),default_dtype=str(torch.get_default_dtype()))


def validate_flags(flags):
    for key in ('float32_matmul_precision','cuda_matmul_allow_tf32','cudnn_allow_tf32','cudnn_benchmark',
                'cudnn_deterministic','deterministic_algorithms','warn_only','CUBLAS_WORKSPACE_CONFIG'):
        if flags.get(key)!=PROFILE[key]:raise RuntimeError('v3 flag drift '+key)
    if flags.get('autocast_cuda') or flags.get('autocast_cpu'):raise RuntimeError('v3 autocast prohibited')
    if flags.get('default_dtype')!='torch.float32':raise RuntimeError('v3 FP32 default required')


def _library_sources(torch):
    directories=[Path(sys.prefix)/'lib',Path(torch.__file__).resolve().parent/'lib']
    site=Path(torch.__file__).resolve().parent.parent
    directories.extend([site/'nvidia/cublas/lib',site/'nvidia/cudnn/lib'])
    files={}
    for directory in directories:
        for pattern in ('libcublas*.so*','libcudnn*.so*'):
            for path in directory.glob(pattern):
                actual=path.resolve()
                if actual.is_file():files[str(actual)]=hashlib.sha256(actual.read_bytes()).hexdigest()
    if not any('cublas' in path for path in files) or not any('cudnn' in path for path in files):
        raise RuntimeError('actual cuBLAS/cuDNN source identity unavailable')
    return files


def configure_v3_runtime(*,expected_identity=None):
    pre_cuda_environment()
    import torch
    import scipy
    if torch.__version__!=PROFILE['torch'] or scipy.__version__!=PROFILE['scipy']:
        raise RuntimeError('v3 exact library versions required')
    torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True,warn_only=False)
    flags=runtime_flags(torch);validate_flags(flags)
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise RuntimeError('one admitted CUDA device required')
    hardware=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name,driver_version,pci.bus_id','--format=csv,noheader'],text=True).strip()
    identity=dict(profile=PROFILE,flags=flags,torch_version=torch.__version__,torch_path=torch.__file__,
        scipy_version=scipy.__version__,scipy_path=scipy.__file__,cuda_build=torch.version.cuda,
        cudnn_version=torch.backends.cudnn.version(),gpu_hardware=hardware,gpu_name=torch.cuda.get_device_name(0),
        library_source_sha256=_library_sources(torch),runtime_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    identity['runtime_profile_SHA']=json_sha(PROFILE)
    if expected_identity is not None and identity!=expected_identity:raise RuntimeError('v3 runtime/library/GPU identity changed')
    return identity


def assert_v3_runtime(identity):
    import torch
    validate_flags(runtime_flags(torch))
    if identity['profile']!=PROFILE or torch.__version__!=identity['torch_version']:raise RuntimeError('v3 profile changed')
    return True
