"""Shared runtime helpers for the head-only and LoRA trainers."""

import dataclasses
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch


def print_banner(title: str, components: list, cfg, data_dir: Path, run_dir: Path) -> None:
    """Startup banner shared by train_head.py and train_lora.py."""
    print(f"================ {title} ================")
    print("= components: " + ", ".join(components))
    print("= main parameters:")
    for field in dataclasses.fields(cfg):
        print(f"=   {field.name}={getattr(cfg, field.name)}")
    print(f"=   data_dir={data_dir}")
    print(f"=   run_dir={run_dir}")


def load_resume_files(run_dir: Path) -> Tuple[dict, dict]:
    """Loads last.pt + training_state.pt from `run_dir`, raising if either is missing."""
    last_ckpt_path = run_dir / "last.pt"
    training_state_path = run_dir / "training_state.pt"
    if not last_ckpt_path.exists() or not training_state_path.exists():
        raise RuntimeError(f"--resume given but missing {last_ckpt_path} and/or {training_state_path}")
    resume_weights = torch.load(last_ckpt_path, map_location="cpu", weights_only=False)
    resume_state = torch.load(training_state_path, map_location="cpu", weights_only=False)
    return resume_weights, resume_state


def restore_rng_state(resume_state: Dict[str, Any]) -> None:
    """Restores torch/cuda RNG state saved by `rng_state_dict`."""
    torch.set_rng_state(resume_state["torch_rng_state"])
    if torch.cuda.is_available() and resume_state.get("cuda_rng_state") is not None:
        torch.cuda.set_rng_state_all(resume_state["cuda_rng_state"])


def rng_state_dict() -> Dict[str, Any]:
    """Builds the RNG portion of a training_state.pt payload."""
    return {
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def assert_resume_config_compatible(
    saved_cfg, cfg, relaxed_product_fields: Optional[Tuple[str, str]] = None,
) -> None:
    """Asserts `cfg` is safe to resume `saved_cfg` with.

    All fields must match exactly, except: if `relaxed_product_fields` names
    two fields (e.g. ("batch_size", "grad_accum")), their individual values
    may differ as long as their product is unchanged -- that product is what
    actually determines steps_per_epoch and the optimizer-step boundary. A
    one-line warning is printed when the relaxed fields differ.
    """
    if relaxed_product_fields is None:
        assert saved_cfg == cfg, f"resume config mismatch:\n  saved: {saved_cfg}\n  given: {cfg}"
        return

    field_a, field_b = relaxed_product_fields
    saved_product = getattr(saved_cfg, field_a) * getattr(saved_cfg, field_b)
    cfg_product = getattr(cfg, field_a) * getattr(cfg, field_b)
    saved_rest = dataclasses.replace(saved_cfg, **{field_a: 0, field_b: 0})
    cfg_rest = dataclasses.replace(cfg, **{field_a: 0, field_b: 0})
    assert saved_rest == cfg_rest, f"resume config mismatch:\n  saved: {saved_cfg}\n  given: {cfg}"
    assert saved_product == cfg_product, (
        f"resume {field_a}*{field_b} mismatch: saved {field_a}={getattr(saved_cfg, field_a)} "
        f"{field_b}={getattr(saved_cfg, field_b)} (={saved_product}) vs given "
        f"{field_a}={getattr(cfg, field_a)} {field_b}={getattr(cfg, field_b)} (={cfg_product})"
    )
    if (getattr(cfg, field_a), getattr(cfg, field_b)) != (getattr(saved_cfg, field_a), getattr(saved_cfg, field_b)):
        print(
            f"[resume] {field_a}/{field_b} changed ({getattr(saved_cfg, field_a)}/{getattr(saved_cfg, field_b)} "
            f"-> {getattr(cfg, field_a)}/{getattr(cfg, field_b)}) but their product is unchanged "
            f"({cfg_product}); proceeding."
        )
