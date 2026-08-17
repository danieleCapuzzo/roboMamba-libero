#!/usr/bin/env python
"""
Online LoRA fine-tuning of LinearManip (CLIP ViT-L/14 + Mamba-2.8B) on one LIBERO suite.

Updates weights via LoRA, every step, online. Hyperparameters live in
LoraTrainConfig below; edit its defaults directly to change them.

Run in the `bench-env` conda env, from the repo root:
    python train_lora.py --suite spatial
    python train_lora.py --suite spatial --resume

Checkpointing under <output-dir>/<suite>/:
  - last.pt: LoRA adapters (unmerged) + fully-trained head. Overwritten every
    --num-save-steps optimizer steps.
  - training_state.pt: optimizer, LR scheduler, epoch/step counters, RNG
    state. This plus last.pt is the --resume target.
  - lora_epoch{N}.pt: same format as last.pt, written every
    --save-every-epochs epochs and never overwritten, also on the final epoch.
"""

import sys
from pathlib import Path

_SRC = str(Path(__file__).resolve().parent / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import argparse
import csv
import dataclasses
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from peft import get_peft_model_state_dict, set_peft_model_state_dict
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.libero import SUITE_TO_DATASET_DIR, LiberoHDF5Dataset, compute_action_bounds, make_collate_fn
from model.loader import build_libero_model
from train.checkpoint import atomic_save, build_weights_payload, load_head_into
from train.lora_utils import attach_lora, check_gradient_liveness, force_mamba_module_path, set_lora_trainable


@dataclass(frozen=True)
class LoraTrainConfig:
    """
    Hyperparameters for one LoRA fine-tuning run. 
    Serialized into every checkpoint, re-asserted equal to the current on --resume.
    """

    suite: str
    data_dir: Optional[Path] = None
    trunk_checkpoint: Path = Path("trained/pre_trained/RoboMamba-224-llava-R300-checkpoint.pth")
    output_dir: Path = Path("checkpoints/lora")

    # LoRA
    lora_rank: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    lora_vit: bool = False
    lora_projector: bool = True
    lora_mamba: bool = True

    # optimization
    epochs: int = 30
    max_steps: Optional[int] = None
    batch_size: int = 8
    grad_accum: int = 1
    grad_checkpointing: bool = True
    lora_lr: float = 5e-4
    head_lr: float = 5e-4
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    warmup_steps: int = 0

    # checkpointing
    num_save_steps: int = 100
    save_every_epochs: int = 5

    # data / io
    augment: bool = True
    num_workers: int = 4
    seed: int = 7


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Online LoRA fine-tuning of LinearManip")
    parser.add_argument("--suite", required=True, choices=list(SUITE_TO_DATASET_DIR))
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--trunk-checkpoint", type=Path, default=LoraTrainConfig.trunk_checkpoint)
    parser.add_argument("--output-dir", type=Path, default=LoraTrainConfig.output_dir)
    parser.add_argument("--lora-rank", type=int, default=LoraTrainConfig.lora_rank)
    parser.add_argument("--lora-alpha", type=int, default=LoraTrainConfig.lora_alpha)
    parser.add_argument("--lora-dropout", type=float, default=LoraTrainConfig.lora_dropout)
    parser.add_argument("--lora-vit", action="store_true", default=LoraTrainConfig.lora_vit)
    parser.add_argument("--no-lora-projector", dest="lora_projector", action="store_false",
                         default=LoraTrainConfig.lora_projector)
    parser.add_argument("--no-lora-mamba", dest="lora_mamba", action="store_false",
                         default=LoraTrainConfig.lora_mamba)
    parser.add_argument("--epochs", type=int, default=LoraTrainConfig.epochs)
    parser.add_argument("--max-steps", type=int, default=LoraTrainConfig.max_steps)
    parser.add_argument("--batch-size", type=int, default=LoraTrainConfig.batch_size)
    parser.add_argument("--grad-accum", type=int, default=LoraTrainConfig.grad_accum)
    parser.add_argument("--grad-checkpointing", action="store_true",
                         default=LoraTrainConfig.grad_checkpointing)
    parser.add_argument("--lora-lr", type=float, default=LoraTrainConfig.lora_lr)
    parser.add_argument("--head-lr", type=float, default=LoraTrainConfig.head_lr)
    parser.add_argument("--weight-decay", type=float, default=LoraTrainConfig.weight_decay)
    parser.add_argument("--grad-clip", type=float, default=LoraTrainConfig.grad_clip)
    parser.add_argument("--warmup-steps", type=int, default=LoraTrainConfig.warmup_steps)
    parser.add_argument("--num-save-steps", type=int, default=LoraTrainConfig.num_save_steps)
    parser.add_argument("--save-every-epochs", type=int, default=LoraTrainConfig.save_every_epochs)
    parser.add_argument("--no-augment", dest="augment", action="store_false",
                         default=LoraTrainConfig.augment)
    parser.add_argument("--num-workers", type=int, default=LoraTrainConfig.num_workers)
    parser.add_argument("--seed", type=int, default=LoraTrainConfig.seed)
    parser.add_argument("--resume", action="store_true",
                         help="Resume from <output-dir>/<suite>/{last,training_state}.pt.")
    return parser.parse_args()


def cfg_from_args(args: argparse.Namespace) -> LoraTrainConfig:
    fields = {f.name for f in dataclasses.fields(LoraTrainConfig)}
    kwargs = {k: v for k, v in vars(args).items() if k in fields}
    return LoraTrainConfig(**kwargs)


def print_banner(cfg: LoraTrainConfig, data_dir: Path, run_dir: Path) -> None:
    """Startup banner."""
    components = []
    if cfg.lora_vit:
        components.append("vision encoder (LoRA)")
    if cfg.lora_projector:
        components.append("projector (LoRA)")
    if cfg.lora_mamba:
        components.append("mamba trunk (LoRA)")
    components.append("action head (full)")

    print("=== LinearManip LoRA Fine-Tuning ===")
    print("= components: " + ", ".join(components))
    print("= main parameters:")
    for field in dataclasses.fields(cfg):
        print(f"=   {field.name}={getattr(cfg, field.name)}")
    print(f"=   data_dir={data_dir}")
    print(f"=   run_dir={run_dir}")


def build_lora_config_dict(cfg: LoraTrainConfig) -> dict:
    """LoraConfig kwargs, saved into checkpoints so eval can rebuild the adapters."""
    return dict(r=cfg.lora_rank, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
                init_lora_weights="gaussian")


def train(cfg: LoraTrainConfig, resume: bool) -> None:

    # set device and seed
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cfg.seed)

    # resolve dataset
    dataset_dir_name = SUITE_TO_DATASET_DIR[cfg.suite]
    data_dir = cfg.data_dir or (Path("datasets") / f"{dataset_dir_name}_regen")
    if not data_dir.exists():
        raise RuntimeError(f"Dataset directory not found: {data_dir}")
    run_dir = cfg.output_dir / cfg.suite
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "train_log.csv"
    last_ckpt_path = run_dir / "last.pt"
    training_state_path = run_dir / "training_state.pt"

    # print startup banner
    print_banner(cfg, data_dir, run_dir)

    # RESUME handling
    resume_weights, resume_state = None, None
    if resume:
        if not last_ckpt_path.exists() or not training_state_path.exists():
            raise RuntimeError(f"--resume given but missing {last_ckpt_path} and/or {training_state_path}")
        resume_weights = torch.load(last_ckpt_path, map_location="cpu", weights_only=False)
        resume_state = torch.load(training_state_path, map_location="cpu", weights_only=False)
        saved_cfg = LoraTrainConfig(**resume_weights["config"])
        assert saved_cfg == cfg, f"resume config mismatch:\n  saved: {saved_cfg}\n  given: {cfg}"
        q01, q99 = np.asarray(resume_state["action_q01"]), np.asarray(resume_state["action_q99"])
        start_epoch = resume_state["epoch"]
        global_step = resume_state["global_step"]
        skip_steps_in_epoch = resume_state.get("step_in_epoch", 0)
        torch.set_rng_state(resume_state["torch_rng_state"])
        if torch.cuda.is_available() and resume_state.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state_all(resume_state["cuda_rng_state"])
    else:
        q01, q99 = compute_action_bounds(data_dir, cfg.suite)
        start_epoch, global_step, skip_steps_in_epoch = 0, 0, 0
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(["step", "epoch", "loss", "lr_lora", "lr_head"])

    print(f"[{cfg.suite}] action bounds q01={q01.round(4)} q99={q99.round(4)}")

    print(f"[{cfg.suite}] loading LinearManip trunk from {cfg.trunk_checkpoint}...")
    t0 = time.perf_counter()
    base_model = build_libero_model(cfg.trunk_checkpoint, device=device, dtype=torch.bfloat16)
    print(f"[{cfg.suite}] trunk loaded in {time.perf_counter() - t0:.1f}s")

    # check padding is consistent
    assert base_model.tokenizer.padding_side == "right", "Mamba is causal-recurrent; right-padding is required"
    assert base_model.tokenizer.pad_token is not None

    # put the model in LoRA mode
    model = attach_lora(
        base_model, cfg.lora_rank, cfg.lora_alpha, 
        cfg.lora_dropout, cfg.lora_vit, cfg.lora_projector, cfg.lora_mamba
    )
    model.print_trainable_parameters()
    model.train()
    force_mamba_module_path(model)  # re-patch: .train() above flipped mixers back

    # set gradient checkpointing if required
    if cfg.grad_checkpointing:
        model.llm.mamba.gradient_checkpointing_enable()

    # create dataset, compute epochs
    dataset = LiberoHDF5Dataset(
        data_dir, cfg.suite, augment=cfg.augment, action_q01=q01, action_q99=q99, seed=cfg.seed,
    )
    steps_per_epoch = len(dataset) // cfg.batch_size // cfg.grad_accum
    print(f"[{cfg.suite}] {len(dataset)} frames  {steps_per_epoch} optimizer steps/epoch")

    # set head and lora params
    head = model.base_model.model.action_head
    lora_params = [p for n, p in model.named_parameters() if "lora_" in n]
    clip_params = lora_params + list(head.parameters())

    # init the optimizer and scheduler
    optimizer = torch.optim.AdamW(
        [{"params": lora_params, "lr": cfg.lora_lr}, {"params": head.parameters(), "lr": cfg.head_lr}],
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps_per_epoch * cfg.epochs)

    if resume_weights is not None:
        set_peft_model_state_dict(model, resume_weights["adapter_state"])
        load_head_into(base_model, resume_weights)
        optimizer.load_state_dict(resume_state["optimizer_state"])
        scheduler.load_state_dict(resume_state["scheduler_state"])
        print(f"[{cfg.suite}] resumed from {run_dir} at epoch {start_epoch}, step {global_step}")

    lora_trainable = global_step >= cfg.warmup_steps
    set_lora_trainable(model, lora_trainable)
    print(f"[{cfg.suite}] LoRA trainable={lora_trainable} (warmup_steps={cfg.warmup_steps})")

    lora_config_dict = build_lora_config_dict(cfg)

    def save_all(epoch: int, global_step: int, step_in_epoch: int) -> dict:
        meta_extra = {
            "suite": cfg.suite,
            "trunk_checkpoint": str(cfg.trunk_checkpoint),
        }
        weights_payload = build_weights_payload(
            model, "lora+head", epoch, global_step, q01, q99, meta_extra,
            dataclasses.asdict(cfg),
            adapter_state=get_peft_model_state_dict(model), lora_config=lora_config_dict,
        )
        state_payload = {
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "step_in_epoch": step_in_epoch,
            "action_q01": q01,
            "action_q99": q99,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "config": dataclasses.asdict(cfg),
        }
        atomic_save(last_ckpt_path, weights_payload)
        atomic_save(training_state_path, state_payload)
        return weights_payload

    checked_liveness = resume_weights is not None  # already proven if resuming

    # setup dataloader
    loader = DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=True,
        num_workers=cfg.num_workers, collate_fn=make_collate_fn(model.tokenizer),
        generator=torch.Generator(),
    )

    # LoRA FINE-TUNE
    for epoch in range(start_epoch, cfg.epochs):
        dataset.set_epoch(epoch)
        loader.generator.manual_seed(cfg.seed + epoch)

        epoch_t0 = time.perf_counter()
        step_in_epoch, last_step_loss = 0, None
        epoch_losses = []
        pbar = tqdm(loader, desc=f"[{cfg.suite}] epoch {epoch + 1}/{cfg.epochs}",
                    unit="frame", unit_scale=cfg.batch_size)
        pbar_iter = iter(pbar)
        stop_early = False

        if epoch == start_epoch and skip_steps_in_epoch:
            for _ in range(skip_steps_in_epoch):
                next(pbar_iter)
            step_in_epoch = skip_steps_in_epoch
            tqdm.write(f"[{cfg.suite}] fast-forwarded {skip_steps_in_epoch} batches into epoch {epoch + 1}")

        for micro_step, batch in enumerate(pbar_iter):
            pixel_values = batch["pixel_values"].to(device=device, dtype=torch.bfloat16)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            actions = batch["actions"].to(device=device, dtype=torch.float32)

            if micro_step % cfg.grad_accum == 0:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                features = model.encode_features(pixel_values, input_ids, attention_mask)
            pred = head(features.float())
            loss = F.l1_loss(pred, actions)
            (loss / cfg.grad_accum).backward()
            step_in_epoch += 1

            if lora_trainable and not checked_liveness:
                check_gradient_liveness(model, cfg.lora_vit, cfg.lora_projector, cfg.lora_mamba)
                checked_liveness = True

            if (micro_step + 1) % cfg.grad_accum != 0:
                continue

            torch.nn.utils.clip_grad_norm_(clip_params, cfg.grad_clip)
            optimizer.step()
            scheduler.step()
            global_step += 1

            if global_step == cfg.warmup_steps and not lora_trainable:
                lora_trainable = True
                set_lora_trainable(model, True)
                tqdm.write(f"[{cfg.suite}] step {global_step}: warmup done, LoRA unfrozen")

            last_step_loss = loss.item()
            epoch_losses.append(last_step_loss)
            pbar.set_postfix(step=global_step, loss=f"{last_step_loss:.4f}")
            with open(log_path, "a", newline="") as f:
                csv.writer(f).writerow([
                    global_step, epoch + 1, last_step_loss,
                    optimizer.param_groups[0]["lr"], optimizer.param_groups[1]["lr"],
                ])

            if global_step % cfg.num_save_steps == 0:
                save_all(epoch, global_step, step_in_epoch)
                tqdm.write(f"[{cfg.suite}] saved checkpoint (step {global_step}): {run_dir}")

            if cfg.max_steps is not None and global_step >= cfg.max_steps:
                tqdm.write(f"[{cfg.suite}] hit max_steps={cfg.max_steps}, stopping (no checkpoint saved)")
                stop_early = True
                break
        if stop_early:
            return

        epoch_time_s = time.perf_counter() - epoch_t0
        epoch_loss = sum(epoch_losses) / len(epoch_losses)
        print(f"[{cfg.suite}] epoch {epoch + 1}/{cfg.epochs} done in {epoch_time_s:.1f}s  "
              f"step {global_step}  train_loss={epoch_loss:.4f}")

        if (epoch + 1) % cfg.save_every_epochs == 0 or epoch + 1 == cfg.epochs:
            meta_extra = {"suite": cfg.suite, "trunk_checkpoint": str(cfg.trunk_checkpoint)}
            epoch_payload = build_weights_payload(
                model, "lora+head", epoch + 1, global_step, q01, q99, meta_extra,
                dataclasses.asdict(cfg),
                adapter_state=get_peft_model_state_dict(model), lora_config=lora_config_dict,
            )
            epoch_ckpt_path = run_dir / f"lora_epoch{epoch + 1}.pt"
            atomic_save(epoch_ckpt_path, epoch_payload)
            tqdm.write(f"[{cfg.suite}] saved epoch checkpoint: {epoch_ckpt_path}")

    save_all(cfg.epochs - 1, global_step, 0)
    print(f"[{cfg.suite}] training done: {cfg.epochs} epochs, {global_step} steps. checkpoints in {run_dir}")


if __name__ == "__main__":
    args = parse_args()
    train(cfg_from_args(args), args.resume)
