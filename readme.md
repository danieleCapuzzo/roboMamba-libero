# roboMamba-libero

A fork of [RoboMamba](https://github.com/lmzpai/roboMamba) (`RoboMamba: Multimodal
State Space Model for Efficient Robot Reasoning and Manipulation`), stripped down
to a single purpose: fine-tuning RoboMamba's trunk (CLIP ViT-L/14 + Mamba-2.8B)
with a chunked action head for LIBERO manipulation tasks.

Upstream shipped inference code only, for a single-frame SAPIEN pose-prediction
head (2-D contact point + 6-D rotation). This fork replaces that head with an
action-chunking head (predicts 8 steps of 7-DoF LIBERO actions per forward
pass) and adds the training code upstream withheld: two fine-tuning regimes,
checkpointing, and the LIBERO data pipeline.

Evaluation (simulator rollouts, success-rate metrics) is intentionally **not**
in this repo -- see [vla-benchmark](../vla-benchmark), which loads checkpoints
produced here.

## Layout

```
src/
  model/          trunk (vision/llm/vlm) + LinearManip (trunk + action head)
  data/           LIBERO HDF5 dataset, action chunking, augmentation
  train/          checkpoint format, LoRA attachment, config helpers
  assets/         offline tokenizer files
tests/            no-simulator-required unit + parity tests
train_head.py     frozen-trunk, head-only fine-tuning
train_lora.py     LoRA (trunk) + full (head) fine-tuning
```

`src/` is not a Python package -- it has no `__init__.py` and its modules
mix relative and absolute imports (`model.llm`, `data.libero`, ...). Anything
that imports from this repo must add `src/` to `sys.path` first; every
entrypoint here does that itself.

## Installation

`torch`, `causal-conv1d`, and `mamba-ssm` are CUDA-version specific and must
be installed yourself, before anything else here:

```
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install causal-conv1d==1.4.0 mamba-ssm==2.2.0   # optional, see below
```

Then install this repo:

```
pip install -e .            # from pyproject.toml
# or: pip install -r requirements.txt
```

Both list the same pins this was validated against (`transformers==4.40.1`,
`peft==0.11.1`, `timm==0.9.10`). `causal-conv1d`/`mamba-ssm` are optional --
without them HF's Mamba implementation falls back to a slower pure-torch
path, which is correct but slower.

## Training

```
python train_head.py --suite spatial
python train_lora.py --suite spatial
```

Both expect `datasets/<suite>_regen/` (LIBERO HDF5 demos) and a released
trunk checkpoint at `trained/robomamba/RoboMamba-224-llava-R300-checkpoint.pth`
by default; see `--help` on each script for overrides. Both support
`--resume`.

## Checkpoint format

Checkpoints are a dict with `action_head`/`head_state_dict` (the action
head's own state_dict, unprefixed), `adapter_state`/`lora_config` (LoRA runs
only), and `meta` (including a mandatory `trunk_checkpoint` path). See
`src/train/checkpoint.py`.

## 📚 BibTeX

```bibtex
@inproceedings{liurobomamba,
  title={RoboMamba: Efficient Vision-Language-Action Model for Robotic Reasoning and Manipulation},
  author={Liu, Jiaming and Liu, Mengzhen and Wang, Zhenyu and An, Pengju and Li, Xiaoqi and Zhou, Kaichen and Yang, Senqiao and Zhang, Renrui and Guo, Yandong and Zhang, Shanghang},
  booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems}
}
```
