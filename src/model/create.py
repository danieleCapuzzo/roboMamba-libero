from model.vision import Vision
from model.llm import MambaLLM
from model.vlm import LinearVLM
from model.manip import LinearManip


def create_model(vision_encoder, llm_type, types='VLM'):
    """
    Builds either the bare VLM trunk or the LIBERO action-chunking VLA.

    Args:
        vision_encoder: Vision encoder name, e.g. "CLIP224".
        llm_type: Short Mamba LLM name, e.g. "mamba-2.8b".
        types: "VLM" for the trunk alone, "MANIP" for LinearManip (chunked head).

    Returns:
        LinearVLM or LinearManip.
    """
    vision = Vision(vision_encoder)

    if llm_type.startswith('mamba-'):
        llm = MambaLLM(llm_type)
    else:
        assert False, f"{llm_type} is not supported"

    if types == 'VLM':
        return LinearVLM(vision, llm)   # clip+proj+mamba_llm
    if types == 'MANIP':
        return LinearManip(vision, llm) # + action head
    raise NotImplementedError
