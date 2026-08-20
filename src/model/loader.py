"""Strict, offline loader for the LIBERO LinearManip model."""

from pathlib import Path

import torch
import torch.nn as nn

from model.action_head import ACTION_CHUNK, ACTION_DIM
from model.llm import MambaLLM, mamba_dict
from model.manip import LinearManip
from model.vision import Vision

_ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"

_fp32_scan_installed = False


def enable_fp32_selective_scan() -> bool:
    """
    Routes Mamba's selective scan through fp32 while leaving trunk weights bf16.

    The 16-bit `selective_scan_fwd_kernel` is slower than its fp32 counterpart on
    some GPUs. The scan is ~44% of trunk CUDA time, resulting in a net performance
    gain with no approximation and peak memory cost.

    Idempotent; safe to call more than once.

    Returns:
        True if the patch was installed by this call, False if already installed.
    """
    global _fp32_scan_installed
    if _fp32_scan_installed:
        return False

    import transformers.models.mamba.modeling_mamba as modeling_mamba

    original_scan = modeling_mamba.selective_scan_fn

    def fp32_selective_scan(
        u, delta, A, B, C, D=None, z=None, delta_bias=None,
        delta_softplus=False, return_last_state=False,
    ):
        # A / D / delta_bias are already fp32 upstream; only the activations need casting
        out_dtype = u.dtype
        out = original_scan(
            u.float(), delta.float(), A, B.float(), C.float(), D,
            z=None if z is None else z.float(), delta_bias=delta_bias,
            delta_softplus=delta_softplus, return_last_state=return_last_state,
        )
        if isinstance(out, tuple):
            return (out[0].to(out_dtype),) + out[1:]
        return out.to(out_dtype)

    modeling_mamba.selective_scan_fn = fp32_selective_scan
    _fp32_scan_installed = True
    return True

# state-spaces/mamba-2.8b-hf's own config.json
MAMBA_2_8B_CONFIG = dict(
    vocab_size=50280,
    hidden_size=2560,
    num_hidden_layers=64,
    state_size=16,
    conv_kernel=4,
    expand=2,
    time_step_rank=160,
    use_bias=False,
    use_conv_bias=True,
    hidden_act="silu",
    initializer_range=0.1,
    residual_in_fp32=True,
    time_step_min=0.001,
    time_step_max=0.1,
    time_step_floor=0.0001,
    time_step_scale=1.0,
    time_step_init_scheme="random",
    rescale_prenorm_residual=False,
    use_cache=True,
    rms_norm=True,
    fused_add_norm=True,
    pad_vocab_size_multiple=8,
    pad_token_id=0,
    bos_token_id=0,
    eos_token_id=0,
)


def _build_offline_mamba_llm(mamba_type: str = "mamba-2.8b") -> MambaLLM:
    """
    Builds a MambaLLM shell without any HF Hub/cache access.

    Args:
        mamba_type: Short name from `model.llm.mamba_dict`.

    Returns:
        A MambaLLM with an empty (meta-device) `.mamba` and a local tokenizer.
    """
    from transformers import MambaConfig, MambaForCausalLM, PreTrainedTokenizerFast

    llm = MambaLLM.__new__(MambaLLM)
    nn.Module.__init__(llm)
    llm.mamba_type = mamba_dict[mamba_type]
    llm.mamba = MambaForCausalLM(MambaConfig(**MAMBA_2_8B_CONFIG))
    llm.tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(_ASSETS_DIR / "tokenizer.json"),
        bos_token="<|endoftext|>", eos_token="<|endoftext|>", pad_token="<|endoftext|>",
    )
    llm.hidden_size = llm.mamba.config.hidden_size
    return llm


def build_libero_model(
    trunk_checkpoint: Path,
    device: str,
    dtype: torch.dtype = torch.bfloat16,
    action_chunk: int = ACTION_CHUNK,
    action_dim: int = ACTION_DIM,
) -> LinearManip:
    """
    Builds LinearManip and strictly loads a released trunk checkpoint into it.

    Args:
        trunk_checkpoint: Path to the released trunk .pth (vision/projector/llm keys).
        device: Target device for the loaded model.
        dtype: Compute dtype for the trunk; the action head stays fp32 (see
            `to_trunk_dtype`).
        action_chunk: Actions predicted per forward.
        action_dim: Dimensions per action step.

    Returns:
        A LinearManip in eval mode with the trunk frozen and loaded, and a
        freshly initialized (untrained) action head.
    """
    from accelerate import init_empty_weights

    vision = Vision("CLIP224")
    with init_empty_weights():
        llm = _build_offline_mamba_llm("mamba-2.8b")

    model = LinearManip(vision, llm, action_chunk=action_chunk, action_dim=action_dim)

    state_dict = torch.load(trunk_checkpoint, map_location="cpu", mmap=True, weights_only=True)

    # sanity check on the checkpoint layout
    key_counts = {"vision": 0, "projector": 0, "llm": 0}
    for key in state_dict:
        key_counts[key.split(".", 1)[0]] += 1
    assert key_counts == {"vision": 295, "projector": 6, "llm": 643}, (
        f"unexpected checkpoint key layout: {key_counts}"
    )

    lm_head_key = "llm.mamba.lm_head.weight"
    embed_key = "llm.mamba.backbone.embeddings.weight"
    assert torch.equal(state_dict[lm_head_key], state_dict[embed_key]), (
        "lm_head.weight is not tied to the embedding table -- dropping it would lose weights"
    )
    del state_dict[lm_head_key]

    missing, unexpected = model.load_state_dict(state_dict, strict=False, assign=True)

    # action_head is new and expected to be missing
    missing_non_head = [k for k in missing if not k.startswith("action_head.")]
    assert not missing_non_head, f"checkpoint left params unset: {missing_non_head}"
    assert not unexpected, f"unexpected checkpoint keys: {unexpected}"

    leftover_meta = [n for n, p in model.named_parameters()
                      if p.is_meta and not n.startswith("action_head.")]
    assert not leftover_meta, f"params still on meta device after load: {leftover_meta}"

    model.tokenizer.pad_token = model.tokenizer.eos_token
    model = model.to(device=device)
    to_trunk_dtype(model, dtype)
    model.eval()
    return model


def to_trunk_dtype(model: LinearManip, dtype: torch.dtype) -> LinearManip:
    """
    Casts only the trunk (vision/projector/llm) to `dtype` 
    and pins the head to fp32.
    """
    model.vision.to(dtype=dtype)
    model.projector.to(dtype=dtype)
    model.llm.to(dtype=dtype)
    _pin_head_fp32(model)
    return model


def _pin_head_fp32(model: LinearManip) -> None:
    model.action_head.to(dtype=torch.float32)
