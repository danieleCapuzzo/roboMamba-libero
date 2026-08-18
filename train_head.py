#!/usr/bin/env python
"""
Online head-only fine-tuning of LinearManip (frozen trunk) on one LIBERO suite.

Every step re-runs the frozen trunk forward pass (no_grad) on freshly
augmented images. No cached features, so the head sees a different
augmented view of each frame every epoch.

Run in the `bench-env` conda env, from the repo root:
    python train_head.py --suite spatial
    python train_head.py --suite spatial --resume

Checkpointing under <output-dir>/<suite>/:
  - last.pt: head weights + metadata, overwritten every epoch.
  - training_state.pt: optimizer state, epoch counter, RNG state. This plus
    last.pt is the --resume target.
  - epoch{N}.pt: standalone snapshot every --save-every-epochs epochs and on
    the final epoch, never overwritten.
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
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.libero import SUITE_TO_DATASET_DIR, LiberoHDF5Dataset, compute_action_bounds, make_collate_fn
from model.action_head import ACTION_CHUNK, ACTION_DIM, HIDDEN_DIM, IMAGE_SIZE, PROMPT_TEMPLATE
from model.loader import build_libero_model
from train.checkpoint import atomic_save, build_weights_payload, load_head_into
from train.utils import (
    assert_resume_config_compatible, load_resume_files, print_banner, restore_rng_state, rng_state_dict,
)


@dataclass(frozen=True)
class HeadTrainConfig:
    """
    Hyperparameters for one head-only fine-tuning run. 
    Serialized into every checkpoint, asserted equal to the current config on --resume.
    """

    suite: str
    data_dir: Optional[Path] = None
    trunk_checkpoint: Path = Path("pretrained/RoboMamba-224-llava-R300-checkpoint.pth")
    output_dir: Path = Path("checkpoints/head")

    epochs: int = 50
    batch_size: int = 128
    lr: float = 3e-4
    weight_decay: float = 5e-4
    num_workers: int = 8
    seed: int = 7
    save_every_epochs: int = 5
    eager: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Online head-only fine-tuning of LinearManip")
    parser.add_argument("--suite", required=True, choices=list(SUITE_TO_DATASET_DIR))
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--trunk-checkpoint", type=Path,
                         default=HeadTrainConfig.trunk_checkpoint)
    parser.add_argument("--output-dir", type=Path, default=HeadTrainConfig.output_dir)
    parser.add_argument("--epochs", type=int, default=HeadTrainConfig.epochs)
    parser.add_argument("--batch-size", type=int, default=HeadTrainConfig.batch_size)
    parser.add_argument("--lr", type=float, default=HeadTrainConfig.lr)
    parser.add_argument("--weight-decay", type=float, default=HeadTrainConfig.weight_decay)
    parser.add_argument("--num-workers", type=int, default=HeadTrainConfig.num_workers)
    parser.add_argument("--seed", type=int, default=HeadTrainConfig.seed)
    parser.add_argument("--save-every-epochs", type=int, default=HeadTrainConfig.save_every_epochs)
    parser.add_argument("--eager", action="store_true", default=HeadTrainConfig.eager,
                         help="Preload all suite frames into RAM at startup (needs ~20-40GB/suite).")
    parser.add_argument("--resume", action="store_true",
                         help="Resume from <output-dir>/<suite>/{last,training_state}.pt.")
    return parser.parse_args()


def cfg_from_args(args: argparse.Namespace) -> HeadTrainConfig:
    return HeadTrainConfig(
        suite=args.suite, data_dir=args.data_dir, trunk_checkpoint=args.trunk_checkpoint,
        output_dir=args.output_dir, epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, weight_decay=args.weight_decay, num_workers=args.num_workers,
        seed=args.seed, save_every_epochs=args.save_every_epochs, eager=args.eager,
    )


def train(cfg: HeadTrainConfig, resume: bool) -> None:

    # set the seed and device
    torch.manual_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # resolve suite to dataset
    dataset_dir_name = SUITE_TO_DATASET_DIR[cfg.suite]
    data_dir = cfg.data_dir or (Path("datasets") / f"{dataset_dir_name}_regen")
    if not data_dir.exists():
        raise RuntimeError(f"Dataset directory not found: {data_dir}")
    run_dir = cfg.output_dir / cfg.suite
    run_dir.mkdir(parents=True, exist_ok=True)
    last_ckpt_path = run_dir / "last.pt"
    training_state_path = run_dir / "training_state.pt"
    log_path = run_dir / "train_log.csv"

    # show startup banner
    components = ["vision (frozen)", "projector (frozen)", "mamba trunk (frozen)", "action head (full)"]
    print_banner("ROBOMAMBA HEAD TRAINING", components, cfg, data_dir, run_dir)

    # RESUME (--resume) from checkpoint
    resume_weights, resume_state = None, None
    if resume:
        resume_weights, resume_state = load_resume_files(run_dir)
        saved_cfg = HeadTrainConfig(**resume_weights["config"])
        assert_resume_config_compatible(saved_cfg, cfg, ignored_fields=("eager",))

    print(f"\n\n[{cfg.suite}] loading LinearManip trunk from {cfg.trunk_checkpoint}...")
    t0 = time.perf_counter()
    model = build_libero_model(cfg.trunk_checkpoint, device=device)
    print(f"[{cfg.suite}] trunk loaded in {time.perf_counter() - t0:.1f}s")

    # freeze everything but the head
    model.requires_grad_(False)
    model.action_head.requires_grad_(True)

    assert model.tokenizer.padding_side == "right", "Mamba is causal-recurrent; right-padding is required"
    assert model.tokenizer.pad_token is not None

    # action bounds (q01/q99), no train/val split
    if resume_state is not None:
        action_q01 = np.asarray(resume_state["action_q01"])
        action_q99 = np.asarray(resume_state["action_q99"])
    else:
        action_q01, action_q99 = compute_action_bounds(data_dir, cfg.suite)
    print(f"[{cfg.suite}] action bounds q01={action_q01.round(4)} q99={action_q99.round(4)}")

    # create dataset and dataloader
    dataset = LiberoHDF5Dataset(
        data_dir, cfg.suite, augment=True,
        action_q01=action_q01, action_q99=action_q99,
        seed=cfg.seed, eager=cfg.eager,
    )
    loader = DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=True,
        num_workers=cfg.num_workers, 
        collate_fn=make_collate_fn(model.tokenizer),
        generator=torch.Generator(),
    )

    # set optimizer
    optimizer = torch.optim.AdamW(model.action_head.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = torch.nn.L1Loss()

    # FINE-TUNE
    start_epoch = 0
    if resume_weights is not None:
        load_head_into(model, resume_weights)
        optimizer.load_state_dict(resume_state["optimizer_state"])
        start_epoch = resume_state["epoch"]
        restore_rng_state(resume_state)
        print(f"[{cfg.suite}] resumed from {run_dir} at epoch {start_epoch}")
    else:
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "train_loss", "epoch_time_s"])

    if start_epoch >= cfg.epochs:
        print(f"[{cfg.suite}] start_epoch={start_epoch} >= epochs={cfg.epochs}, nothing to do.")
        return

    for epoch in range(start_epoch, cfg.epochs):
        dataset.set_epoch(epoch)
        loader.generator.manual_seed(cfg.seed + epoch)

        epoch_t0 = time.perf_counter()
        model.action_head.train()
        train_losses = []
        pbar = tqdm(loader, desc=f"[{cfg.suite}] epoch {epoch + 1}/{cfg.epochs}",
                    unit="frame", unit_scale=cfg.batch_size)
        for batch in pbar:
            pixel_values = batch["pixel_values"].to(device=device, dtype=model.llm.mamba.dtype)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            actions = batch["actions"].to(device=device, dtype=torch.float32)

            # trunk forward stays no_grad; head forward stays outside it
            with torch.no_grad():
                features = model.encode_features(pixel_values, input_ids, attention_mask)

            optimizer.zero_grad(set_to_none=True)
            pred = model.action_head(features.float())
            loss = loss_fn(pred, actions)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = sum(train_losses) / len(train_losses)
        epoch_time_s = time.perf_counter() - epoch_t0
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch + 1, train_loss, epoch_time_s])

        meta_extra = {
            "suite": cfg.suite,
            "seed": cfg.seed,
            "trunk_checkpoint": str(cfg.trunk_checkpoint),
            "data_dir": str(data_dir),
            "image_size": IMAGE_SIZE,
            "prompt_template": PROMPT_TEMPLATE,
            "action_dim": ACTION_DIM,
            "hidden_dim": HIDDEN_DIM,
            "upright_images": True,
            "readout": "language_pool",
            "train_loss": train_loss,
        }
        weights_payload = build_weights_payload(
            model, "head", epoch + 1, 0, action_q01, action_q99, meta_extra,
            dataclasses.asdict(cfg),
        )
        atomic_save(last_ckpt_path, weights_payload)
        atomic_save(training_state_path, {
            "epoch": epoch + 1,
            "action_q01": action_q01,
            "action_q99": action_q99,
            "optimizer_state": optimizer.state_dict(),
            **rng_state_dict(),
        })
        print(f"[{cfg.suite}] epoch {epoch + 1}/{cfg.epochs} done in {epoch_time_s:.1f}s  train_loss={train_loss:.4f}")

        if (epoch + 1) % cfg.save_every_epochs == 0 or epoch + 1 == cfg.epochs:
            epoch_ckpt_path = run_dir / f"epoch{epoch + 1}.pt"
            atomic_save(epoch_ckpt_path, weights_payload)
            print(f"[{cfg.suite}] saved epoch checkpoint: {epoch_ckpt_path}")

    print(f"[{cfg.suite}] training done. final train_loss={train_loss:.4f}. checkpoints in {run_dir}")


if __name__ == "__main__":
    args = parse_args()
    train(cfg_from_args(args), args.resume)
