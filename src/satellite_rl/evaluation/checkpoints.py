"""Load required evaluation weights, or stop with an actionable error."""

from pathlib import Path

import torch


class CheckpointLoadError(RuntimeError):
    """A required evaluation checkpoint is absent or incompatible."""


def load_network_checkpoint(
    network,
    checkpoint_path,
    *,
    algorithm,
    map_location="cpu",
    state_dict_key=None,
    require_state_dict_key=False,
    evaluation_mode=False,
):
    """Strictly load a state dict; retain the original failure as its cause."""
    path = Path(checkpoint_path)
    try:
        if not path.is_file():
            raise FileNotFoundError(f"Weight file does not exist: {path}")
        state_dict = torch.load(path, map_location=map_location, weights_only=True)
        if require_state_dict_key and (
            not isinstance(state_dict, dict) or state_dict_key not in state_dict
        ):
            raise ValueError(f"Checkpoint must contain the '{state_dict_key}' state dict")
        if isinstance(state_dict, dict) and state_dict_key in state_dict:
            state_dict = state_dict[state_dict_key]
        network.load_state_dict(state_dict, strict=True)
    except Exception as exc:
        raise CheckpointLoadError(
            f"[{algorithm}] Failed to load required checkpoint '{path}': {exc}"
        ) from exc
    if evaluation_mode:
        network.eval()
    return True


def load_actor_only(
    agent,
    ckpt_dir,
    filenames=("actor.pth", "actor.pt", "sac_actor.pth", "model.pth"),
    *,
    algorithm=None,
):
    """Load the actor using the existing filename search order, or raise."""
    algorithm = algorithm or type(agent).__name__
    if not ckpt_dir:
        raise CheckpointLoadError(f"[{algorithm}] No actor checkpoint directory supplied: {ckpt_dir!r}")
    directory = Path(ckpt_dir)
    if not directory.is_dir():
        raise CheckpointLoadError(f"[{algorithm}] Actor checkpoint directory does not exist: '{directory}'")
    candidates = [(name, False) for name in filenames]
    candidates.extend((name, True) for name in ("checkpoint.pt", "checkpoint.pth"))
    for filename, require_actor_key in candidates:
        path = directory / filename
        if path.is_file():
            load_network_checkpoint(
                agent.actor,
                path,
                algorithm=algorithm,
                state_dict_key="actor",
                require_state_dict_key=require_actor_key,
                evaluation_mode=True,
            )
            print(f"[{algorithm}] Loaded actor: {path}")
            return True
    searched = ", ".join((*filenames, "checkpoint.pt", "checkpoint.pth"))
    raise CheckpointLoadError(
        f"[{algorithm}] No actor weights found in '{directory}'; searched: {searched}"
    )
