"""LoRA attachment and correctness guards for LinearManip fine-tuning.

Two hazards this module exists to prevent:
  1. HF Mamba's fused fast-path forward reads x_proj/out_proj `.weight` raw
     under `self.training`, bypassing PEFT's `lora.Linear.__call__` intercept
     entirely -- `force_mamba_module_path` forces the module-call branch.
  2. `get_peft_model` freezes every non-adapter param, including the folded-in
     action head that pre-exists on LinearManip -- `attach_lora` re-enables it.
"""

from typing import List

import torch.nn as nn
from peft import LoraConfig, PeftModel, get_peft_model
from transformers.models.mamba.modeling_mamba import MambaMixer


def lora_targets(model: nn.Module, lora_vit: bool, lora_projector: bool, lora_mamba: bool) -> List[str]:
    """
    Full module paths of nn.Linear layers to adapt, gated by component flags.

    dt_proj and lm_head are always excluded: dt_proj is read as a raw .weight
    on both the fast and slow Mamba forward paths (an adapter there would be a
    silent no-op), and lm_head does not exist on this model (dropped in
    LinearManip.__init__).
    """
    targets = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if name.startswith("vision.") and lora_vit:
            targets.append(name)
        elif name.startswith("projector.") and lora_projector:
            targets.append(name)
        elif name.startswith("llm.") and lora_mamba and not name.endswith(("dt_proj", "lm_head")):
            targets.append(name)
    return targets


def force_mamba_module_path(model: nn.Module) -> int:
    """
    Flips every MambaMixer's `.training` to False so HF's fused fast-path
    forward takes its module-call branch (self.x_proj(x)) instead of its
    training branch, which reads raw `.weight` and bypasses LoRA.

    Must be re-applied after every `.train()` call, since it recurses and
    would otherwise flip mixers back.

    Returns:
        Number of MambaMixer modules patched, for a startup sanity check.
    """
    n = 0
    for module in model.modules():
        if isinstance(module, MambaMixer):
            module.training = False
            n += 1
    return n


def attach_lora(
    model: nn.Module, lora_rank: int, lora_alpha: int, lora_dropout: float,
    lora_vit: bool, lora_projector: bool, lora_mamba: bool,
) -> PeftModel:
    """
    Wraps `model` in LoRA adapters and re-enables gradients on the folded-in
    action head, which `get_peft_model` would otherwise freeze.

    Args:
        model: LinearManip to wrap.

    Returns:
        A PeftModel; `.action_head` is reachable at
        `peft_model.base_model.model.action_head`, `.tokenizer`/`.transform`
        resolve through PeftModel.__getattr__.
    """
    targets = lora_targets(model, lora_vit, lora_projector, lora_mamba)
    lora_config = LoraConfig(
        r=lora_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
        target_modules=targets, init_lora_weights="gaussian",
    )
    peft_model = get_peft_model(model, lora_config)
    for name, param in peft_model.named_parameters():
        if "lora_" in name:
            param.data = param.data.float()  # adapter master weights stay fp32

    # get_peft_model froze everything not named lora_, including the pre-existing head
    head = peft_model.base_model.model.action_head
    head.requires_grad_(True)
    assert all(p.requires_grad for p in head.parameters()), "action_head was left frozen by PEFT"

    n_patched = force_mamba_module_path(peft_model)
    assert n_patched > 0, "No MambaMixer modules found; the trunk structure may have changed"
    return peft_model


def set_lora_trainable(model: nn.Module, trainable: bool) -> None:
    """
    Toggles requires_grad on every LoRA adapter parameter (head-only warmup switch).
    """
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.requires_grad_(trainable)


def check_gradient_liveness(model: nn.Module, lora_vit: bool, lora_projector: bool, lora_mamba: bool) -> None:
    """
    Asserts backward() produced non-zero gradients for every adapted
    component. Regression check for HF Mamba's fast-path bypass.
    """
    checks = []
    if lora_vit:
        checks.append("vision.model.blocks.0.attn.qkv")
    if lora_projector:
        checks.append("projector.projector.0")
    if lora_mamba:
        checks += [
            "llm.mamba.backbone.layers.0.mixer.in_proj",
            "llm.mamba.backbone.layers.0.mixer.x_proj",
            "llm.mamba.backbone.layers.0.mixer.out_proj",
        ]
    params = dict(model.named_parameters())
    for name in checks:
        key = f"base_model.model.{name}.lora_B.default.weight"
        p = params.get(key)
        assert p is not None, f"expected LoRA param {key} not found -- target_modules mismatch?"
        assert p.grad is not None, f"{key}.grad is None -- LoRA gradient did not flow (fast-path bypass?)"
        assert p.grad.abs().max().item() > 0, f"{key}.grad is all-zero"

    action_head = model.base_model.model.action_head
    for name, p in action_head.named_parameters():
        assert p.requires_grad, f"action_head.{name} is frozen"
    stem_linear_grad = action_head.stem[1].weight.grad
    assert stem_linear_grad is not None and stem_linear_grad.abs().max().item() > 0, (
        "action_head.stem[1].weight.grad is None or all-zero"
    )
