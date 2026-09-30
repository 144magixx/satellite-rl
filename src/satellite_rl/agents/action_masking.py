"""Joint frequency/slot and separate power distributions for masked SAC."""

import math

import numpy as np
import torch


def copy_action_mask(mask, action_dims):
    """Validate an environment supplied mask without coercing other values to bool."""
    if not isinstance(mask, dict) or set(mask) != {"frequency_slots", "power"}:
        raise ValueError("action_mask must contain frequency_slots and power")
    frequency_slots = np.asarray(mask["frequency_slots"])
    power = np.asarray(mask["power"])
    if (frequency_slots.dtype != np.bool_ or power.dtype != np.bool_ or
            frequency_slots.shape != tuple(action_dims[:2]) or
            power.shape != (action_dims[2],)):
        raise ValueError("action_mask must have boolean arrays with the configured action shapes")
    if not frequency_slots.any() or not power.any():
        raise ValueError("active action_mask has no feasible action")
    return {"frequency_slots": frequency_slots.copy(), "power": power.copy()}


def mask_batch(masks, action_dims, device):
    """Terminal next masks are absent; use a finite placeholder before zero bootstrap."""
    full = {"frequency_slots": np.ones(action_dims[:2], dtype=np.bool_),
            "power": np.ones(action_dims[2], dtype=np.bool_)}
    masks = [full if mask is None else mask for mask in masks]
    return (torch.as_tensor(np.stack([mask["frequency_slots"] for mask in masks]), device=device),
            torch.as_tensor(np.stack([mask["power"] for mask in masks]), device=device))


def masked_policy(logits, frequency_slots, power):
    """Return exact normalized probabilities and safe logs on feasible actions."""
    f, slots, p = logits
    if (f.ndim != 2 or slots.ndim != 2 or p.ndim != 2 or
            frequency_slots.shape != (f.shape[0], f.shape[1], slots.shape[1]) or
            power.shape != p.shape):
        raise ValueError("batched mask shape does not match policy logits")
    if not torch.isfinite(f).all() or not torch.isfinite(slots).all() or not torch.isfinite(p).all():
        raise FloatingPointError("Nonfinite policy logits")
    if not frequency_slots.any(dim=(1, 2)).all() or not power.any(dim=1).all():
        raise ValueError("active action_mask has no feasible action")
    joint_logits = (f.unsqueeze(2) + slots.unsqueeze(1)).masked_fill(~frequency_slots, -torch.inf)
    power_logits = p.masked_fill(~power, -torch.inf)
    log_fl = joint_logits.flatten(1).log_softmax(-1).reshape_as(joint_logits)
    log_p = power_logits.log_softmax(-1)
    if not torch.isfinite(log_fl[frequency_slots]).all() or not torch.isfinite(log_p[power]).all():
        raise FloatingPointError("Nonfinite masked policy normalization")
    fl = log_fl.exp()
    power_probability = log_p.exp()
    # Zero probability times negative infinity is undefined, including in gradients.
    safe_log_fl = torch.where(frequency_slots, log_fl, torch.zeros_like(log_fl))
    safe_log_p = torch.where(power, log_p, torch.zeros_like(log_p))
    entropy_fl = -(fl * safe_log_fl).sum((1, 2))
    entropy_power = -(power_probability * safe_log_p).sum(1)
    frequency_probability = fl.sum(2)
    safe_log_frequency = torch.where(frequency_probability > 0,
                                     frequency_probability.clamp_min(torch.finfo(fl.dtype).tiny).log(),
                                     torch.zeros_like(frequency_probability))
    entropy_frequency = -(frequency_probability * safe_log_frequency).sum(1)
    entropy_slots_conditional = entropy_fl - entropy_frequency
    return (fl, power_probability, safe_log_fl, safe_log_p,
            entropy_fl, entropy_power, entropy_frequency, entropy_slots_conditional)


def conditional_masked_policy(logits, frequency_slots, power):
    """Exact joint distribution for π(f,l)π(p|f,l) on the feasible product set."""
    fl_logits, conditional_power_logits = logits
    if (fl_logits.ndim != 3 or conditional_power_logits.ndim != 4 or
            frequency_slots.shape != fl_logits.shape or
            conditional_power_logits.shape[:3] != fl_logits.shape or
            power.shape != (fl_logits.shape[0], conditional_power_logits.shape[-1])):
        raise ValueError("batched mask shape does not match conditional policy logits")
    if not torch.isfinite(fl_logits).all() or not torch.isfinite(conditional_power_logits).all():
        raise FloatingPointError("Nonfinite conditional policy logits")
    if not frequency_slots.any(dim=(1, 2)).all() or not power.any(dim=1).all():
        raise ValueError("active action_mask has no feasible action")
    log_fl = fl_logits.masked_fill(~frequency_slots, -torch.inf).flatten(1).log_softmax(-1)
    log_fl = log_fl.reshape_as(fl_logits)
    log_p = conditional_power_logits.masked_fill(~power[:, None, None, :], -torch.inf).log_softmax(-1)
    if not torch.isfinite(log_fl[frequency_slots]).all() or not torch.isfinite(log_p[power[:, None, None, :].expand_as(log_p)]).all():
        raise FloatingPointError("Nonfinite conditional masked policy normalization")
    fl = log_fl.exp()
    conditional_power = log_p.exp()
    safe_log_fl = torch.where(frequency_slots, log_fl, torch.zeros_like(log_fl))
    safe_log_p = torch.where(power[:, None, None, :], log_p, torch.zeros_like(log_p))
    entropy_fl = -(fl * safe_log_fl).sum((1, 2))
    entropy_power_by_fl = -(conditional_power * safe_log_p).sum(-1)
    entropy_power = (fl * entropy_power_by_fl).sum((1, 2))
    frequency_probability = fl.sum(2)
    safe_log_frequency = torch.where(frequency_probability > 0,
                                     frequency_probability.clamp_min(torch.finfo(fl.dtype).tiny).log(),
                                     torch.zeros_like(frequency_probability))
    entropy_frequency = -(frequency_probability * safe_log_frequency).sum(1)
    entropy_slots = entropy_fl - entropy_frequency
    return (fl, conditional_power, safe_log_fl, safe_log_p,
            entropy_fl, entropy_power, entropy_frequency, entropy_slots)


def entropy_ratio(target_entropy, action_dims):
    maximum = math.log(math.prod(action_dims))
    if maximum <= 0 or not math.isfinite(target_entropy) or not 0 <= target_entropy <= maximum:
        raise ValueError("masked SAC target_entropy must be between zero and log(total actions)")
    return float(target_entropy) / maximum


def feasible_target_entropy(frequency_slots, power, ratio):
    legal_fl = frequency_slots.sum((1, 2)).to(torch.float32)
    legal_power = power.sum(1).to(torch.float32)
    return ratio * (legal_fl.log() + legal_power.log())
