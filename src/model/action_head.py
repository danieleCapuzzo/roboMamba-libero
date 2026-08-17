"""LIBERO action-chunking head and its shared constants."""

import torch
from torch import nn

# CLIP ViT-L/14@224, fed from 256px renders
IMAGE_SIZE = 224
# Mamba-2.8B hidden size
HIDDEN_DIM = 2560
# LIBERO action space: xyz + axis-angle/rpy + gripper
ACTION_DIM = 7
# actions predicted per trunk forward, executed open-loop by the eval queue
ACTION_CHUNK = 8
# head hidden width and number of residual blocks
HEAD_WIDTH = 1024
HEAD_BLOCKS = 2

PROMPT_TEMPLATE = "<|user|> <image>\n{instruction}\n<|assistant|>"


class _MLPResBlock(nn.Module):
    """Pre-norm residual block: x + Linear(GELU(Linear(LayerNorm(x))))."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width),
            nn.GELU(),
            nn.Linear(width, width),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x)


class ActionChunkHead(nn.Module):
    """
    LIBERO action head, MLP-ResNet:
    LayerNorm(2560) -> Linear(1024) -> HEAD_BLOCKS residual blocks -> LayerNorm -> chunk*7.
    """

    def __init__(self, in_dim: int = HIDDEN_DIM, chunk: int = ACTION_CHUNK,
                 action_dim: int = ACTION_DIM) -> None:
        super().__init__()
        self.chunk = chunk
        self.action_dim = action_dim
        self.stem = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, HEAD_WIDTH),
        )
        self.blocks = nn.Sequential(
            *[_MLPResBlock(HEAD_WIDTH) for _ in range(HEAD_BLOCKS)]
        )
        self.out = nn.Sequential(
            nn.LayerNorm(HEAD_WIDTH), nn.Linear(HEAD_WIDTH, chunk * action_dim)
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """features: (B, in_dim) -> (B, chunk, action_dim)."""
        x = self.blocks(self.stem(features))
        return self.out(x).unflatten(-1, (self.chunk, self.action_dim))
