import logging
from typing import override

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer, AutoModel, AutoProcessor

from modosaic.core.record import ImageRecord
from modosaic.services.device import DeviceService
from modosaic.services.image import ImageService
from modosaic.text.constants import VALIDATION_ENRICHMENT
from modosaic.text.validators.base.validator import TextValidator

logger = logging.getLogger(__name__)


class SIGLIP2Validator(TextValidator[float]):
    """Validate caption-image agreement with SIGLIP 2."""

    device: torch.device
    dtype: torch.dtype
    model: PreTrainedModel
    processor: PreTrainedTokenizer

    def __init__(self) -> None:
        """Load the SIGLIP model and processor."""
        self.device, self.dtype = DeviceService.get_device_type()
        self.model = self._load_model()
        self.model.to(self.device)
        self.processor = self._load_processor()

    @override
    @torch.no_grad()
    def validate(self, record: ImageRecord, generated: str) -> float:
        """Score a generated caption against the source image."""
        logger.debug(f"Validating text-image embedding for record sample {record.sample_id}")
        image = ImageService.bytes_to_pil(record.image_bytes)
        text_description = [f"{VALIDATION_ENRICHMENT} {generated}"]
        inputs = self.processor(
            text=text_description,
            images=image,
            padding="max_length",
            max_num_patches=256,
            return_tensors="pt"
        ).to(self.model.device)

        outputs = self.model(**inputs)

        logits_per_image = outputs.logits_per_image
        probs = torch.sigmoid(logits_per_image)
        probability = probs[0][0].item()
        logger.debug(f"Text-image probability for record sample {record.sample_id}: {probability}")
        return probability

    def _load_model(self) -> PreTrainedModel:
        """Load the pinned SIGLIP 2 model."""
        return AutoModel.from_pretrained(
            pretrained_model_name_or_path="google/siglip2-base-patch16-naflex",
            revision="b53b807d3a2d5e2b3911292f2d69e5341cdc064c",
            dtype=self.dtype,
            device_map="auto",
            attn_implementation="sdpa"
        )

    @staticmethod
    def _load_processor() -> PreTrainedTokenizer:
        """Load the pinned SIGLIP 2 processor."""
        return AutoProcessor.from_pretrained(
            pretrained_model_name_or_path="google/siglip2-base-patch16-naflex",
            revision="b53b807d3a2d5e2b3911292f2d69e5341cdc064c",
        )
