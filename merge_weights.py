#!/usr/bin/env python
"""
Merges a trunk checkpoint with a head-only or LoRA+head checkpoint into a
single self-contained .pt, deployable without peft/LoRA machinery at
inference time. Also writes a dataset_statistics.json sidecar.

Output format:
  - "state_dict":       flat LinearManip weights in the requested dtype.
  - "tokenizer_json":   contents of src/assets/tokenizer.json, verbatim.
  - "tokenizer_kwargs": special tokens for PreTrainedTokenizerFast.
  - "mamba_config":     MAMBA_2_8B_CONFIG, to instantiate MambaForCausalLM.
  - "meta":             architecture scalars, prompt template, the training
                        image preprocessing recipe, action norm bounds, and
                        provenance.
                        
Handles both checkpoints written by train_head.py / train_lora.py
(see src/train/checkpoint.py):
  - "head":      trunk + action_head only, no adapters to merge.
  - "lora+head": trunk + LoRA adapters (merged and folded into the trunk's
                 Linear weights) + action_head.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

_SRC = str(Path(__file__).resolve().parent / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import torch

from data.libero import CLIP_MEAN, CLIP_STD, SUITE_TO_DATASET_DIR, compute_full_action_stats
from model.action_head import (
    ACTION_CHUNK, ACTION_DIM, HEAD_BLOCKS, HEAD_WIDTH, HIDDEN_DIM, IMAGE_SIZE, PROMPT_TEMPLATE,
)
from model.loader import MAMBA_2_8B_CONFIG, build_libero_model
from train.checkpoint import load_head_into

FORMAT_DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16}

_ASSETS_DIR = Path(__file__).resolve().parent / "src" / "assets"

TOKENIZER_KWARGS = {
    "bos_token": "<|endoftext|>",
    "eos_token": "<|endoftext|>",
    "pad_token": "<|endoftext|>",
}

# vision enc shorhand
VISION_ENCODER = "CLIP224"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge a trunk + head/LoRA checkpoint into a single self-contained .pt."
    )
    parser.add_argument("--trunk", type=Path, required=True,
                         help="Released trunk .pth (vision/projector/llm keys).")
    parser.add_argument("--weights", type=Path, required=True,
                         help="Head-only or LoRA+head checkpoint from train_head.py/train_lora.py "
                              "(e.g. checkpoints/lora/<suite>/lora_epoch{N}.pt).")
    parser.add_argument("--output", type=Path, required=True,
                         help="Destination path for the merged checkpoint (.pt).")
    parser.add_argument("--format", choices=list(FORMAT_DTYPES), default="bf16",
                         help="Output weight dtype (default: bf16).")
    parser.add_argument("--data-dir", type=Path, default=None,
                         help="Suite's demo .hdf5 directory, for dataset_statistics.json. "
                              "Defaults to datasets/<dataset_dir_name>_regen, resolved from "
                              "the weights checkpoint's saved suite.")
    parser.add_argument("--no-dataset-stats", dest="dataset_statistics", action="store_false",
                         default=True, help="Skip writing dataset_statistics.json.")
    return parser.parse_args()


def write_dataset_statistics(ckpt: dict, data_dir: Path, output_dir: Path) -> None:
    """Writes dataset_statistics.json (OpenVLA-OFT schema, no proprio -- this
    model has no proprio projector) next to the merged weights."""
    suite = ckpt["meta"]["suite"]
    print(f"[{suite}] computing dataset statistics from {data_dir}...")
    stats = compute_full_action_stats(data_dir, suite)
    mask = [True] * (ACTION_DIM - 1) + [False]  # last dim (gripper) is binary, not normalized
    payload = {
        suite: {
            "action": {
                "mean": stats["mean"].tolist(),
                "std": stats["std"].tolist(),
                "max": stats["max"].tolist(),
                "min": stats["min"].tolist(),
                "q01": stats["q01"].tolist(),
                "q99": stats["q99"].tolist(),
                "mask": mask,
            },
            "num_transitions": stats["num_transitions"],
            "num_trajectories": stats["num_trajectories"],
        }
    }
    stats_path = output_dir / "dataset_statistics.json"
    with open(stats_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"saved dataset statistics: {stats_path}")


def build_meta(ckpt: dict, trunk_checkpoint: Path, format: str) -> dict:
    """
    Collects the architecture, preprocessing and provenance fields.

    The action bounds come from the training checkpoint's own meta rather than
    from the freshly computed dataset statistics, so they always match the
    normalization the head was actually trained under.
    """
    train_meta = ckpt["meta"]
    lora_cfg = ckpt.get("lora_config")
    meta = {
        "suite": train_meta["suite"],
        # architecture
        "vision_encoder": VISION_ENCODER,
        "image_size": IMAGE_SIZE,
        "hidden_dim": HIDDEN_DIM,
        "action_chunk": train_meta.get("action_chunk", ACTION_CHUNK),
        "action_dim": ACTION_DIM,
        "head_width": HEAD_WIDTH,
        "head_blocks": HEAD_BLOCKS,
        "prompt_template": PROMPT_TEMPLATE,
        # preprocessing recipe, as used in training (see data.libero.LiberoFrameDataset)
        "image_mean": list(CLIP_MEAN),
        "image_std": list(CLIP_STD),
        "resize_interpolation": "INTER_AREA",
        "upright": True,
        # action normalization bounds the head was trained with
        "action_q01": _to_list(train_meta["action_q01"]),
        "action_q99": _to_list(train_meta["action_q99"]),
        # provenance
        "dtype": format,
        "trunk_checkpoint": Path(trunk_checkpoint).name,
        "train_kind": ckpt["kind"],
        "train_epoch": train_meta["epoch"],
    }
    if lora_cfg is not None:
        meta["lora"] = {"r": lora_cfg["r"], "alpha": lora_cfg["lora_alpha"]}
    return meta


def _to_list(values) -> list:
    """
    Normalizes a numpy array / tensor / sequence of action bounds to a
    plain list of floats, so the merged file carries no numpy or torch types.
    """
    if hasattr(values, "tolist"):
        values = values.tolist()
    return [float(v) for v in values]


def merge(
    trunk_checkpoint: Path, weights_checkpoint: Path, output: Path, format: str,
    data_dir: Optional[Path], write_stats: bool,
) -> None:
    print(f"loading weights checkpoint: {weights_checkpoint}")

    # load che checkpoint
    ckpt = torch.load(weights_checkpoint, map_location="cpu", weights_only=False)
    kind = ckpt["kind"]
    assert kind in ("head", "lora+head"), f"unrecognized checkpoint kind: {kind!r}"
    print(f"checkpoint kind: {kind}")

    print(f"loading trunk: {trunk_checkpoint}")
    model = build_libero_model(trunk_checkpoint, device="cpu", dtype=torch.float32)

    if kind == "lora+head":
        adapter_state = ckpt["adapter_state"]
        lora_cfg = ckpt["lora_config"]
        target_modules = sorted({
            key.split(".lora_", 1)[0].removeprefix("base_model.model.")
            for key in adapter_state if ".lora_" in key
        })
        assert target_modules, "adapter_state has no lora_ params -- nothing to merge"
        print(f"reattaching LoRA on {len(target_modules)} target modules "
              f"(r={lora_cfg['r']}, alpha={lora_cfg['lora_alpha']})")

        from peft import LoraConfig, get_peft_model, set_peft_model_state_dict

        peft_config = LoraConfig(
            r=lora_cfg["r"], lora_alpha=lora_cfg["lora_alpha"],
            lora_dropout=lora_cfg.get("lora_dropout", 0.0),
            target_modules=target_modules,
            init_lora_weights=lora_cfg.get("init_lora_weights", True),
        )
        peft_model = get_peft_model(model, peft_config)
        missing, unexpected = set_peft_model_state_dict(peft_model, adapter_state)
        missing_lora = [k for k in missing if "lora_" in k]
        assert not missing_lora, f"adapter_state left LoRA params unset: {missing_lora}"
        assert not unexpected, f"adapter_state has unexpected keys: {unexpected}"

        load_head_into(model, ckpt)

        print("merging LoRA adapters into base weights...")
        model = peft_model.merge_and_unload()
    else:
        load_head_into(model, ckpt)

    dtype = FORMAT_DTYPES[format]
    print(f"casting merged model to {format}...")
    model = model.to(dtype=dtype)

    state_dict = model.state_dict()

    tokenizer_path = _ASSETS_DIR / "tokenizer.json"
    assert tokenizer_path.exists(), f"tokenizer asset not found: {tokenizer_path}"
    payload = {
        "state_dict": state_dict,
        "tokenizer_json": tokenizer_path.read_text(),
        "tokenizer_kwargs": dict(TOKENIZER_KWARGS),
        "mamba_config": dict(MAMBA_2_8B_CONFIG),
        "meta": build_meta(ckpt, trunk_checkpoint, format),
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(output)
    n_params = sum(v.numel() for v in state_dict.values())
    size_gb = sum(v.numel() * v.element_size() for v in state_dict.values()) / 1e9
    print(f"saved merged {format} checkpoint: {output} ({n_params:,} params, {size_gb:.2f} GB)")
    print(f"baked in: tokenizer.json ({len(payload['tokenizer_json']):,} chars), "
          f"mamba_config ({len(payload['mamba_config'])} fields), "
          f"meta ({len(payload['meta'])} fields)")

    if write_stats:
        suite = ckpt["meta"]["suite"]
        resolved_data_dir = data_dir or (Path("datasets") / f"{SUITE_TO_DATASET_DIR[suite]}_regen")
        if not resolved_data_dir.exists():
            raise RuntimeError(
                f"Dataset directory not found: {resolved_data_dir} "
                "(pass --data-dir explicitly, or --no-dataset-statistics to skip)"
            )
        write_dataset_statistics(ckpt, resolved_data_dir, output.parent)


if __name__ == "__main__":
    args = parse_args()
    merge(
        args.trunk, args.weights, args.output, args.format,
        args.data_dir, args.dataset_statistics,
    )
