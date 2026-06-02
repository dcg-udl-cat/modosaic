from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class HFModelSpec:
    """Hugging Face model loading metadata.

    Attributes:
        model_id: Repository identifier passed to Hugging Face loaders.
        revision: Revision, tag, or commit to load.
        dtype: Torch dtype used for model weights.
        trust_remote_code: Whether to allow custom model code from the repo.
    """

    model_id: str
    revision: str
    dtype: torch.dtype
    trust_remote_code: bool = True
