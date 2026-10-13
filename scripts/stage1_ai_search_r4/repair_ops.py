"""Absolute bootstrap; PYTHONPATH must retain the original scientific source first."""
import importlib.util
import sys
from pathlib import Path

sys.dont_write_bytecode = True
# Locate the repair module in this separately pinned operations tree, without adding
# that tree to PYTHONPATH or replacing any original scientific package imports.
path = Path(__file__).resolve().parents[2] / 'src/satellite_rl/workbench/r4_monitor_repair.py'
name = 'satellite_rl.workbench.r4_monitor_repair'
spec = importlib.util.spec_from_file_location(name, path)
module = importlib.util.module_from_spec(spec)
sys.modules[name] = module
spec.loader.exec_module(module)
if __name__ == '__main__':
    module.main()
