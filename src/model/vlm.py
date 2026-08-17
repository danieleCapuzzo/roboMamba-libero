"""RoboMamba trunk: CLIP vision encoder -> MLP projector -> Mamba LLM."""

import torch
from torch import nn

from model.llm import MambaLLM
from model.vision import Vision


class Projector(nn.Module):
    """3-layer MLP mapping vision hidden size to LLM hidden size."""

    def __init__(self, input_size, output_size):
        super(Projector, self).__init__()
        self.projector = nn.Sequential(
            nn.Linear(input_size, input_size * 4),
            nn.GELU(),
            nn.Linear(input_size * 4, output_size),
            nn.GELU(),
            nn.Linear(output_size, output_size)
        )

    def forward(self, x):
        return self.projector(x)


def encode_multimodal(vision_module, projector, llm, vision, text, multi_modal=None):
    """
    Prepends projected vision tokens to text embeddings.

    Args:
        vision_module: Vision encoder producing (B, N, vision_hidden).
        projector: Projector mapping vision_hidden -> llm_hidden.
        llm: MambaLLM providing `.embed`.
        vision: Pixel values (B, 3, H, W).
        text: Token ids (B, L).
        multi_modal: Optional (B,) bool mask; True rows get vision tokens prepended.
            Defaults to "any nonzero pixel in the image".

    Returns:
        (B, N + L, llm_hidden) fused embedding sequence, dtype matching `vision`.
    """

    # dtype mismatch silently up/down-casts fused sequence to vision dtype
    assert vision.dtype == next(projector.parameters()).dtype, (
        f"pixel_values dtype {vision.dtype} != trunk compute dtype "
        f"{next(projector.parameters()).dtype}"
    )
    text_encoded = llm.embed(text)
    if multi_modal is None:
        multi_modal = torch.count_nonzero(vision, dim=[-1, -2, -3]) > 0
    vision_encoded = vision_module(vision)
    vision_encoded_result = projector(vision_encoded)[multi_modal]
    shape = list(text_encoded.shape)
    shape[1] += vision_encoded_result.shape[1]

    # dtype pinned to vision
    text_result = torch.zeros(shape, dtype=vision.dtype, device=vision_encoded_result.device)
    text_result[multi_modal] = torch.cat([vision_encoded_result, text_encoded[multi_modal]], dim=1)
    if not multi_modal.all():
        text_result[~multi_modal, :text_encoded.shape[1]] = text_encoded[~multi_modal]
    return text_result


class LinearVLM(nn.Module):
    """
    Frozen RoboMamba trunk: vision + projector + Mamba LLM.
    """

    def __init__(self, vision_encoder, llm):
        super(LinearVLM, self).__init__()
        self.vision: Vision = vision_encoder
        self.llm = llm
        self.projector: Projector = Projector(self.vision.hidden_size, self.llm.hidden_size)

    def encode(self, vision, text, multi_modal=None):
        return encode_multimodal(self.vision, self.projector, self.llm, vision, text, multi_modal)

    def forward(self, vision, text, multi_modal=None, **kwargs):
        inputs_embeds = self.encode(vision, text, multi_modal=multi_modal)
        return self.llm.mamba.backbone(inputs_embeds=inputs_embeds, **kwargs)

    @property
    def tokenizer(self):
        return self.llm.tokenizer

    @property
    def transform(self):
        return self.vision.transform
