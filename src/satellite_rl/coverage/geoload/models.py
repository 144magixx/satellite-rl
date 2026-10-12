"""Unchanged pure Torch MLM architecture; no simulation imports."""
from ..nondrl.models import MaskedAssignmentModel
MODEL_FORMAT = 's1_g16_geometry_mlm_v1'
def make_model(network=None):
    return MaskedAssignmentModel(**(network or {}))
