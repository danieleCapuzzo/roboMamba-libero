"""
Checkpoint format used for the head-only and LoRA trainers.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from peft import get_peft_model_state_dict

from model.action_head import ActionChunkHead
from model.manip import LinearManip

FORMAT_VERSION = 2


def atomic_save(path: Path, payload: dict) -> None:
    """Writes a checkpoint via temp file + os.replace, so a crash mid-write
    never leaves a corrupt file at `path`."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def build_weights_payload(
    model: nn.Module,
    kind: str,
    epoch: int,
    global_step: int,
    q01: np.ndarray,
    q99: np.ndarray,
    meta_extra: Dict[str, Any],
    config: Dict[str, Any],
    adapter_state: Optional[dict] = None,
    lora_config: Optional[dict] = None,
) -> dict:
    """Builds a format-v2 weights payload.

    Args:
        model: LinearManip (base or PEFT-wrapped); `.action_head` is read
            directly regardless of wrapping, since PEFT swaps Linears in place.
        kind: "head" or "lora+head".
        epoch, global_step: Training progress counters.
        q01, q99: Per-dim action normalization bounds.
        meta_extra: Additional meta fields (suite, trunk_checkpoint, ...).
        config: `dataclasses.asdict(cfg)` of the run's config.
        adapter_state: `get_peft_model_state_dict(model)` output, or None for
            a head-only run.
        lora_config: LoraConfig kwargs dict, or None for a head-only run.
    """
    action_head = model.action_head if hasattr(model, "action_head") else model.base_model.model.action_head
    head_state = action_head.state_dict()
    meta = {
        "format_version": FORMAT_VERSION,
        "kind": kind,
        "epoch": epoch,
        "global_step": global_step,
        "action_q01": q01,
        "action_q99": q99,
        "action_chunk": action_head.chunk,
        **meta_extra,
    }
    assert "trunk_checkpoint" in meta, "meta must carry trunk_checkpoint for eval-side loading"
    return {
        "format_version": FORMAT_VERSION,
        "kind": kind,
        "action_head": head_state,
        "head_state_dict": head_state,  # back-compat alias for vla-benchmark's load_head
        "adapter_state": adapter_state,
        "lora_config": lora_config,
        "meta": meta,
        "config": config,
    }


def load_head_into(model: LinearManip, ckpt: dict) -> None:
    """Loads a foreign or fork-written checkpoint's head weights into
    `model.action_head`, strictly.

    Accepts either the new "action_head" key or the legacy "head_state_dict"
    key, and strips a leading "action_head." prefix if present (e.g. from a
    full `model.state_dict()` dump rather than a head-only one).

    Args:
        model: LinearManip whose `.action_head` will be overwritten in place.
        ckpt: A loaded checkpoint dict.
    """
    state_dict = ckpt.get("action_head") or ckpt["head_state_dict"]
    prefix = "action_head."
    if any(k.startswith(prefix) for k in state_dict):
        state_dict = {k[len(prefix):] if k.startswith(prefix) else k: v
                       for k, v in state_dict.items()}
    model.action_head.load_state_dict(state_dict, strict=True)
