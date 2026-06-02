import logging
from typing import override

import torch
from transformers import Pipeline, pipeline

from modosaic.core.hf_model_spec import HFModelSpec
from modosaic.core.record import ImageRecord
from modosaic.services.image import ImageService
from modosaic.text.constants import MAX_CAPTIONING_TOKENS, CAPTIONING_PROMPT
from modosaic.text.generators.base.generator import TextGenerator

logger = logging.getLogger(__name__)


class InternVL352B(TextGenerator):
    """InternVL 3.5 2B image-caption generator."""

    pipeline: Pipeline

    def __init__(self) -> None:
        """Load the pinned InternVL 3.5 pipeline."""
        super().__init__()
        self.model_specs = HFModelSpec(
            model_id="OpenGVLab/InternVL3_5-2B-HF",
            revision="3f301ffcf3dcbb47893afae6650ea3e78d96fb6d",
            dtype=self.dtype,
        )
        self.pipeline = self._load_pipeline()

    def _load_pipeline(self) -> Pipeline:
        """Create the Transformers image-to-text pipeline."""
        return pipeline(
            task="image-text-to-text",
            model=self.model_specs.model_id,
            revision=self.model_specs.revision,
            dtype=self.model_specs.dtype,
            device=self.device
        )

    @override
    @torch.inference_mode()
    def generate(self, record: ImageRecord, prompt: str = CAPTIONING_PROMPT, ) -> str:
        """Generate a caption for one image record."""
        logger.debug(f"Generating text for record sample {record.sample_id}")
        with ImageService.temporary_image_file(record.image_bytes) as tmp_image_path:
            messages = InternVL352B._get_messages(tmp_image_path.resolve().as_posix())
            text_caption = self.pipeline(
                text=messages,
                max_new_tokens=MAX_CAPTIONING_TOKENS,
                max_length=MAX_CAPTIONING_TOKENS,
                return_full_text=False,
            )
            caption = text_caption[0]["generated_text"]
        logger.debug(f"Generated text for record sample {record.sample_id} with {len(caption)} characters")
        return caption

    # noinspection PyTypeChecker
    @staticmethod
    def _get_messages(img_path: str) -> list[dict[str, str]]:
        """Build chat messages for an image path."""
        return [{
            "role": "user",
            "content": [
                {"type": "image", "path": img_path},
                {"type": "text", "text": CAPTIONING_PROMPT},
            ],
        }]
