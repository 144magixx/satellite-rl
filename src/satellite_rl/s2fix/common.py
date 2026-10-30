"""Small shared helpers for the standalone S2-FIX commands."""
from contextlib import contextmanager
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import platform
import random
import subprocess
import sys

import numpy as np
import torch

OUTPUT_ROOT = Path('D:/paper/outputs/experiments/s2_fix_20261018')


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return vars(value)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    default=json_value), encoding='utf-8')


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@contextmanager
def run_manifest(output_dir, config):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[3]
    commit_file = root / 'COMMIT'
    commit = (commit_file.read_text(encoding='utf-8').strip() if commit_file.exists()
              else subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root,
                                           text=True).strip())
    row = dict(config=config if isinstance(config, dict) else vars(config), commit=commit,
               python=sys.version, packages={p: version(p) for p in ('numpy', 'scipy', 'torch')},
               hostname=platform.node(), started=datetime.now(timezone.utc).isoformat())
    try:
        yield row
    finally:
        row['ended'] = datetime.now(timezone.utc).isoformat()
        path = output_dir / 'run_manifest.json'
        history = json.loads(path.read_text(encoding='utf-8')) if path.exists() else []
        history.append(row)
        write_json(path, history)
