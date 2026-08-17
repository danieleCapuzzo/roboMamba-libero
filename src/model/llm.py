"""HF Mamba LLM wrapper used as the RoboMamba trunk's language backbone."""

from transformers import MambaForCausalLM, AutoTokenizer
import torch.nn as nn


# maps short names to their HF Hub repo ids
mamba_dict = {
    'mamba-2.8b': 'state-spaces/mamba-2.8b-hf',
}


class MambaLLM(nn.Module):
    """HF `MambaForCausalLM` + tokenizer, addressed by short name."""

    def __init__(self, mamba_type):
        super(MambaLLM, self).__init__()
        assert mamba_type in mamba_dict, "Unknown mamba type {}".format(mamba_type)
        self.mamba_type = mamba_dict[mamba_type]
        self.mamba = MambaForCausalLM.from_pretrained(self.mamba_type)
        self.tokenizer = AutoTokenizer.from_pretrained(self.mamba_type)
        self.hidden_size = self.mamba.config.hidden_size

    def embed(self, input_ids):
        return self.mamba.backbone.embeddings(input_ids)

    def tokenize(self, text):
        return self.tokenizer.tokenize(text)
