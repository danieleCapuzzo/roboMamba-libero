"""LIBERO HDF5 dataset: frame-level samples with action chunking and CLIP preprocessing.

Serves the two fine-tuning regimes in `train_head.py`/`train_lora.py`. Does not
depend on the `libero` simulator package -- task instructions come from the
baked `libero_instructions.json` instead of the benchmark registry, since
regenerated demo files carry no attrs and this fork never touches the simulator.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from model.action_head import ACTION_CHUNK, ACTION_DIM, IMAGE_SIZE, PROMPT_TEMPLATE

# maps short suite name -> dataset directory name (matches vla-benchmark's config/constants.py)
SUITE_TO_DATASET_DIR = {
    "spatial": "libero_spatial",
    "object": "libero_object",
    "goal": "libero_goal",
    "long": "libero_10",
}

_INSTRUCTIONS_PATH = Path(__file__).resolve().parent / "libero_instructions.json"

# CLIP ViT-L/14 (openai) normalization
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

# image augmentation config, matched to OpenVLA-OFT
_CROP_SCALE = 0.9
_BRIGHTNESS_DELTA = 0.2
_CONTRAST_RANGE = (0.8, 1.2)
_SATURATION_RANGE = (0.8, 1.2)
_HUE_DELTA = 0.05


#######################
# image preprocessing #
#######################

def to_upright(image: np.ndarray) -> np.ndarray:
    """Rotates a LIBERO frame 180 degrees (upside-down -> upright)."""
    return np.ascontiguousarray(image[::-1, ::-1])


def resize_frame(image: np.ndarray, size: int) -> np.ndarray:
    """Resizes a uint8 HWC frame to size x size with INTER_AREA."""
    if image.shape[0] == size and image.shape[1] == size:
        return image
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)


def _rgb_to_hsv(img: torch.Tensor) -> torch.Tensor:
    """Converts an (..., 3) RGB tensor in [0,1] to HSV."""
    r, g, b = img.unbind(-1)
    maxc, _ = img.max(-1)
    minc, _ = img.min(-1)
    diff = (maxc - minc).clamp_min(1e-8)
    s = torch.where(maxc > 0, (maxc - minc) / maxc.clamp_min(1e-8), torch.zeros_like(maxc))
    rc, gc, bc = (maxc - r) / diff, (maxc - g) / diff, (maxc - b) / diff
    h = torch.where(r == maxc, bc - gc, torch.where(g == maxc, 2.0 + rc - bc, 4.0 + gc - rc))
    h = torch.where(maxc == minc, torch.zeros_like(h), (h / 6.0) % 1.0)
    return torch.stack([h, s, maxc], dim=-1)


def _hsv_to_rgb(hsv: torch.Tensor) -> torch.Tensor:
    """Converts an (..., 3) HSV tensor back to RGB in [0,1]."""
    h, s, v = hsv.unbind(-1)
    i = torch.floor(h * 6.0)
    f = h * 6.0 - i
    p, q, t = v * (1 - s), v * (1 - s * f), v * (1 - s * (1 - f))
    i = i.long() % 6
    cases = torch.stack(
        [
            torch.stack([v, t, p], -1),
            torch.stack([q, v, p], -1),
            torch.stack([p, v, t], -1),
            torch.stack([p, q, v], -1),
            torch.stack([t, p, v], -1),
            torch.stack([v, p, q], -1),
        ],
        dim=0,
    )
    return torch.gather(cases, 0, i[None, ..., None].expand(1, *i.shape, 3))[0]


def _random_resized_crop(img: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """Torch reimplementation of dlimp `random_resized_crop` (square crop only)."""
    H, W, _ = img.shape
    side = _CROP_SCALE**0.5  # new_height == new_width since ratio == 1
    u = torch.rand((), generator=generator).item()
    off = u * (1.0 - side)
    y_frac = torch.linspace(off, off + side, H)
    x_frac = torch.linspace(off, off + side, W)
    grid_y, grid_x = torch.meshgrid(y_frac * 2 - 1, x_frac * 2 - 1, indexing="ij")
    grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
    chw = img.permute(2, 0, 1).unsqueeze(0)
    out = F.grid_sample(chw, grid, mode="bilinear", align_corners=True, padding_mode="zeros")
    return out.squeeze(0).permute(1, 2, 0)


def _augment(img: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """Applies crop, brightness, contrast, saturation, hue in a fixed order,
    clipping to [0,1] after every op."""
    img = _random_resized_crop(img, generator).clamp(0, 1)

    # brightness
    delta = (torch.rand((), generator=generator).item() * 2 - 1) * _BRIGHTNESS_DELTA
    img = (img + delta).clamp(0, 1)

    # contrast, per-channel mean rather than grayscale
    lo, hi = _CONTRAST_RANGE
    factor = lo + torch.rand((), generator=generator).item() * (hi - lo)
    mean = img.mean(dim=(0, 1), keepdim=True)
    img = ((img - mean) * factor + mean).clamp(0, 1)

    # saturation via HSV
    lo, hi = _SATURATION_RANGE
    factor = lo + torch.rand((), generator=generator).item() * (hi - lo)
    hsv = _rgb_to_hsv(img)
    hsv[..., 1] = (hsv[..., 1] * factor).clamp(0, 1)
    img = _hsv_to_rgb(hsv).clamp(0, 1)

    # hue via HSV
    delta = (torch.rand((), generator=generator).item() * 2 - 1) * _HUE_DELTA
    hsv = _rgb_to_hsv(img)
    hsv[..., 0] = (hsv[..., 0] + delta) % 1.0
    img = _hsv_to_rgb(hsv).clamp(0, 1)

    return img


####################
# action utilities #
####################

def _chunk_actions(demo_actions: np.ndarray, frame_idx: int, chunk: int) -> np.ndarray:
    """Repeats the demo's final action once the chunk window runs past the demo end."""
    n = demo_actions.shape[0]
    idx = np.arange(frame_idx, frame_idx + chunk).clip(max=n - 1)
    return demo_actions[idx]


def normalize_actions(actions: np.ndarray, q01: np.ndarray, q99: np.ndarray) -> np.ndarray:
    """Per-dim q01/q99 bounds normalization into [-1, 1]; values outside the
    bounds saturate at the boundary. Works on np.ndarray or torch.Tensor."""
    normalized = 2 * (actions - q01) / (q99 - q01 + 1e-8) - 1
    return normalized.clip(-1.0, 1.0)


def denormalize_actions(actions: np.ndarray, q01: np.ndarray, q99: np.ndarray) -> np.ndarray:
    """Inverts `normalize_actions`."""
    return 0.5 * (actions + 1) * (q99 - q01 + 1e-8) + q01


##########################
# instructions and files #
##########################

def _demo_files(data_dir: Path) -> List[Path]:
    """Sorted list of *.hdf5 files in data_dir."""
    files = sorted(Path(f) for f in glob.glob(str(data_dir / "*.hdf5")))
    if not files:
        raise FileNotFoundError(f"No .hdf5 files found in {data_dir}")
    return files


def _task_instructions(suite: str) -> Dict[str, str]:
    """Maps task name -> language instruction from the baked JSON.

    Falls back to a demo file's own `problem_info` attr (present on the
    original, non-regenerated LIBERO releases) if the JSON has no entry.
    """
    with open(_INSTRUCTIONS_PATH) as f:
        return json.load(f)[suite]


def _instruction_for(instructions: Dict[str, str], path: Path, h5_file: h5py.File) -> str:
    """Resolves one file's instruction: baked JSON first, then HDF5 attrs, else raises."""
    stem = path.stem.removesuffix("_demo")
    if stem in instructions:
        return instructions[stem]
    if "problem_info" in h5_file["data"].attrs:
        return json.loads(h5_file["data"].attrs["problem_info"])["language_instruction"]
    raise KeyError(f"{path.name}: no instruction in libero_instructions.json or file attrs")


###########
# dataset #
###########

class LiberoHDF5Dataset(Dataset):
    """
    Frame-level dataset over a LIBERO suite's *.hdf5 demo files.

    Args:
        data_dir: Folder of *.hdf5 demo files.
        suite: Suite name from SUITE_TO_DATASET_DIR, used to resolve instructions.
        augment: False applies only resize + CLIP normalize. True additionally
            applies the OFT-matched augmentation pipeline before normalization.
        image_size: Output frame side length.
        action_chunk: Actions per sample; chunks are clamped at demo end.
        action_q01, action_q99: Optional per-dim (action_dim,) bounds. When
            given, actions are bounds-normalized to [-1, 1]; None returns raw
            actions.
        seed: Base seed for the per-sample augmentation RNG.
        eager: False (default) reads frames from HDF5 lazily per __getitem__.
            True preloads every demo's raw agentview_rgb frames into RAM at
            construction time, removing per-step file I/O. Needs enough RAM
            to hold the suite's uncompressed frames (tens of GB).
    """

    def __init__(
        self,
        data_dir: Path,
        suite: str,
        augment: bool = False,
        image_size: int = IMAGE_SIZE,
        action_chunk: int = ACTION_CHUNK,
        action_q01: Optional[np.ndarray] = None,
        action_q99: Optional[np.ndarray] = None,
        seed: int = 0,
        eager: bool = False,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.suite = suite
        self.augment = augment
        self.image_size = image_size
        self.action_chunk = action_chunk
        self.action_q01 = action_q01
        self.action_q99 = action_q99
        self.seed = seed
        self.epoch = 0
        self.eager = eager

        self._files = _demo_files(self.data_dir)
        self._instructions_by_suite = _task_instructions(suite)
        self._mean = torch.tensor(CLIP_MEAN).view(3, 1, 1)
        self._std = torch.tensor(CLIP_STD).view(3, 1, 1)

        # (file_idx, demo_key, frame_idx, demo_id) per frame;
        # actions loaded eagerly per demo (tiny)
        # images stay lazy in HDF5 unless eager=True
        self._index: List[Tuple[int, str, int, int]] = []
        self._demo_actions: List[np.ndarray] = []
        self._demo_image_offset: List[int] = []  # demo_id -> start row in self._images
        _image_chunks: List[np.ndarray] = []
        self._instructions: List[str] = []
        demo_id = 0
        offset = 0
        for file_idx, path in enumerate(self._files):
            with h5py.File(path, "r") as f:
                instruction = _instruction_for(self._instructions_by_suite, path, f)
                for demo_key in f["data"]:
                    actions = f["data"][demo_key]["actions"][:].astype(np.float32)
                    self._demo_actions.append(actions)
                    self._instructions.append(instruction)
                    if self.eager:
                        images = f["data"][demo_key]["obs"]["agentview_rgb"][:]
                        _image_chunks.append(images)
                        self._demo_image_offset.append(offset)
                        offset += images.shape[0]
                    for frame_idx in range(actions.shape[0]):
                        self._index.append((file_idx, demo_key, frame_idx, demo_id))
                    demo_id += 1

        # concatenated into one array (refcounted)
        self._images = np.concatenate(_image_chunks, axis=0) if self.eager else None

        self._handles: Dict[int, h5py.File] = {}

    def set_epoch(self, epoch: int) -> None:
        """Sets the epoch used to seed per-sample augmentation RNG; call before
        each epoch's iteration so augmentation is reproducible across resume."""
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self._index)

    def _file(self, file_idx: int) -> h5py.File:
        # h5py.File handles aren't fork-safe -- opened lazily in the worker that needs them
        if file_idx not in self._handles:
            self._handles[file_idx] = h5py.File(self._files[file_idx], "r")
        return self._handles[file_idx]

    def _seed_for(self, index: int) -> int:
        return (self.seed * 1_000_003 + self.epoch * 97 + index) & 0xFFFFFFFF

    def __getitem__(self, index: int) -> Dict[str, object]:
        file_idx, demo_key, frame_idx, demo_id = self._index[index]
        if self.eager:
            # O(1) due to array random access property
            image = self._images[self._demo_image_offset[demo_id] + frame_idx]
        else:
            f = self._file(file_idx)
            image = f["data"][demo_key]["obs"]["agentview_rgb"][frame_idx]

        frame = to_upright(image)
        frame = resize_frame(frame, self.image_size)
        img = torch.from_numpy(frame).float().div(255.0)  # (H,W,3) in [0,1]

        if self.augment:
            generator = torch.Generator().manual_seed(self._seed_for(index))
            img = _augment(img, generator)

        pixel_values = img.permute(2, 0, 1)  # (3,H,W)
        pixel_values = pixel_values.sub(self._mean).div(self._std)

        actions = _chunk_actions(self._demo_actions[demo_id], frame_idx, self.action_chunk)
        if self.action_q01 is not None:
            actions = normalize_actions(actions, self.action_q01, self.action_q99)

        return {
            "pixel_values": pixel_values,
            "actions": torch.from_numpy(actions.astype(np.float32)),
            "instruction": self._instructions[demo_id],
            "demo_id": demo_id,
        }


def compute_action_bounds(data_dir: Path, suite: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Per-dim q01/q99 raw-action bounds over every frame in data_dir.

    Returns:
        (q01, q99), each (action_dim,) float32.
    """
    dataset = LiberoHDF5Dataset(data_dir, suite, augment=False)
    actions = np.concatenate(dataset._demo_actions, axis=0)
    q01 = np.quantile(actions, 0.01, axis=0).astype(np.float32)
    q99 = np.quantile(actions, 0.99, axis=0).astype(np.float32)
    return q01, q99


def compute_full_action_stats(data_dir: Path, suite: str) -> Dict[str, object]:
    """
    Per-dim mean/std/min/max/q01/q99 raw-action stats over every frame in
    data_dir, plus trajectory/transition counts.

    Returns:
        Dict with "mean", "std", "min", "max", "q01", "q99" (each an
        (action_dim,) float32 ndarray), "num_transitions", "num_trajectories".
    """
    dataset = LiberoHDF5Dataset(data_dir, suite, augment=False)
    demo_actions = dataset._demo_actions
    actions = np.concatenate(demo_actions, axis=0)
    return {
        "mean": actions.mean(axis=0).astype(np.float32),
        "std": actions.std(axis=0).astype(np.float32),
        "min": actions.min(axis=0).astype(np.float32),
        "max": actions.max(axis=0).astype(np.float32),
        "q01": np.quantile(actions, 0.01, axis=0).astype(np.float32),
        "q99": np.quantile(actions, 0.99, axis=0).astype(np.float32),
        "num_transitions": int(actions.shape[0]),
        "num_trajectories": len(demo_actions),
    }


def make_collate_fn(tokenizer):
    """
    Builds a DataLoader collate_fn that formats and tokenizes each sample's
    instruction with PROMPT_TEMPLATE, right-padding to the batch's longest prompt.

    Args:
        tokenizer: The trunk's tokenizer; mutated in place to pad right.

    Returns:
        A callable (list[dict]) -> dict with keys pixel_values, actions,
        demo_id, input_ids, attention_mask.
    """
    tokenizer.padding_side = "right"

    def collate(batch: List[Dict]) -> Dict[str, torch.Tensor]:
        prompts = [PROMPT_TEMPLATE.format(instruction=b["instruction"]) for b in batch]
        tokenized = tokenizer(prompts, return_tensors="pt", padding=True)
        return {
            "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
            "actions": torch.stack([b["actions"] for b in batch]),
            "demo_id": torch.tensor([b["demo_id"] for b in batch], dtype=torch.long),
            "input_ids": tokenized.input_ids,
            "attention_mask": tokenized.attention_mask,
        }

    return collate


def create_libero_dataset(
    data_dir: Path,
    suite: str,
    augment: bool = False,
    action_q01: Optional[np.ndarray] = None,
    action_q99: Optional[np.ndarray] = None,
    seed: int = 0,
    eager: bool = False,
) -> LiberoHDF5Dataset:
    """
    Factory for LiberoHDF5Dataset."""
    return LiberoHDF5Dataset(
        data_dir, suite, augment=augment,
        action_q01=action_q01, action_q99=action_q99, seed=seed,
        eager=eager,
    )
