"""Resolve migrated input/output paths without depending on the working directory.

Legacy names are kept at call sites so experiment logic remains easy to compare.
SATELLITE_RL_OUTPUT_ROOT optionally redirects outputs and newly trained models,
primarily for isolated verification; input data and pretrained weights stay fixed.
"""
from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LEGACY_PATHS = {'cities_china.csv': 'data/cities_china.csv',
 'saved_models/sac-t': 'models/pretrained/sac_transformer',
 'saved_models/sac-cnn': 'models/pretrained/sac_cnn',
 'saved_models/sac-fc': 'models/pretrained/sac_mlp',
 'saved_models/ppo': 'models/pretrained/ppo_direct',
 'saved_models/dqn': 'models/pretrained/dqn',
 'saved_models': 'models/pretrained',
 'saved_sac_transformer_models': 'models/runs/sac_transformer',
 'saved_sac_cnn_models': 'models/runs/sac_cnn',
 'saved_sac_fc_models': 'models/runs/sac_mlp',
 'saved_group_no_freq_models': 'models/runs/ppo',
 'saved_dqn_models': 'models/runs/dqn',
 'sac_transformer_log': 'outputs/logs/sac_transformer',
 'sac_cnn_full_log': 'outputs/logs/sac_cnn',
 'sac_fc_full_log': 'outputs/logs/sac_mlp',
 'ppo_group_no_freq_by_self_log': 'outputs/logs/ppo',
 'dqn_group_log': 'outputs/logs/dqn',
 'sac_full_fig': 'outputs/figures/sac',
 'ppo_group_no_freq_by_self_fig': 'outputs/figures/ppo',
 'dqn_group_fig': 'outputs/figures/dqn',
 'sac_data': 'outputs/metrics/sac',
 'PPO_data': 'outputs/metrics/ppo',
 'DQN_full_data': 'outputs/metrics/dqn',
 'random_data': 'outputs/metrics/random',
 'metrics': 'outputs/metrics',
 'logs': 'outputs/logs',
 'figs': 'outputs/figures',
 'eval_fig': 'outputs/evaluation',
 'sgm_vs_demand_curves.npz': 'outputs/metrics/demand/sgm_vs_demand_curves.npz',
 'si_vs_demand_curves.npz': 'outputs/metrics/demand/si_vs_demand_curves.npz',
 'si_vs_demand_legendstyle.png': 'outputs/figures/demand/si_vs_demand_legendstyle.png',
 'si_vs_demand_legendstyle.eps': 'outputs/figures/demand/si_vs_demand_legendstyle.eps',
 'rate_comparison_bar.png': 'outputs/figures/rate_comparison_bar.png',
 'antenna_pattern.png': 'outputs/figures/antenna_pattern.png'}


def project_path(relative):
    """Return an absolute filename; do not create files or directories."""
    text = str(relative).replace("\\", "/")
    if Path(text).is_absolute():
        return text
    migrated = text
    for old, new in sorted(LEGACY_PATHS.items(), key=lambda pair: len(pair[0]), reverse=True):
        if text == old or text.startswith(old + "/"):
            migrated = new + text[len(old):]
            break
    output_root = os.environ.get("SATELLITE_RL_OUTPUT_ROOT")
    base = PROJECT_ROOT
    if output_root and (migrated.startswith("outputs/") or migrated.startswith("models/runs/")):
        base = Path(output_root).expanduser().resolve()
    return str(base / migrated)
