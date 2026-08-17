"""LinearManip: RoboMamba trunk + folded-in LIBERO action-chunking head."""

import torch
from torch import nn

from model.action_head import ActionChunkHead, ACTION_CHUNK, ACTION_DIM
from model.llm import MambaLLM
from model.vlm import Projector, encode_multimodal
from model.vision import Vision


class LinearManip(nn.Module):
    """
    RoboMamba trunk (vision + projector + Mamba) with a chunked action head.
    Pools the language-token hidden states (mask-weighted mean) and regresses
    an (action_chunk, action_dim) block in one forward pass.
    """

    def __init__(self, vision_encoder, llm, head_type='action_chunk',
                 action_chunk=ACTION_CHUNK, action_dim=ACTION_DIM):
        super(LinearManip, self).__init__()
        if head_type != 'action_chunk':
            # upstream shipped 'mlp'/'two_mlp'/'ssm+mlp' SAPIEN-pose heads; removed in the
            # strip pass -- this fork only ever trains the LIBERO chunked head
            raise NotImplementedError(
                f"head_type={head_type!r} is not supported; this fork only ships "
                "'action_chunk' (the LIBERO chunked head)."
            )
        self.vision: Vision = vision_encoder
        self.llm = llm
        self.head_type = head_type
        self.projector = Projector(self.vision.hidden_size, self.llm.hidden_size)
        self.action_head = ActionChunkHead(self.llm.hidden_size, action_chunk, action_dim)

        # the LM head projects to vocab size and is never used for regression;
        # dropping it saves ~129M tied params (== embedding table)
        if isinstance(self.llm, MambaLLM):
            self.llm.mamba.lm_head = nn.Identity()

    def encode(self, vision, text, multi_modal=None):
        return encode_multimodal(self.vision, self.projector, self.llm, vision, text, multi_modal)

    def pool_language(self, hidden, num_text_tokens, attention_mask=None):
        """
        Mask-weighted mean over language-token hidden states only.

        Args:
            hidden: (B, num_vision + L, hidden_dim) backbone output.
            num_text_tokens: L, the text sequence length before fusion.
            attention_mask: (B, L), 1 for real tokens / 0 for right-padding.
                None means every row is unpadded.

        Returns:
            (B, hidden_dim) pooled language features.
        """
        num_vision = hidden.shape[1] - num_text_tokens
        lang = hidden[:, num_vision:]
        if attention_mask is None:
            return lang.mean(dim=1)
        mask = attention_mask.unsqueeze(-1).to(lang.dtype)
        return (lang * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

    def encode_features(self, pixel_values, input_ids, attention_mask=None, multi_modal=None):
        """
        Fuses vision + text, runs the trunk, and pools language tokens.
        Returns:
            (B, hidden_dim) pooled features.
        """
        inputs_embeds = self.encode(pixel_values, input_ids, multi_modal)
        # use_cache=False -- no autoregressive decode on this path, skip MambaCache alloc
        hidden = self.llm.mamba.backbone(
            inputs_embeds=inputs_embeds, use_cache=False
        ).last_hidden_state
        return self.pool_language(hidden, input_ids.shape[1], attention_mask)

    def forward(self, pixel_values, input_ids, attention_mask=None, multi_modal=None):
        """Returns (B, action_chunk, action_dim) fp32 action predictions."""
        features = self.encode_features(pixel_values, input_ids, attention_mask, multi_modal)
        return self.action_head(features.float())

    @property
    def tokenizer(self):
        return self.llm.tokenizer

    @property
    def transform(self):
        return self.vision.transform
