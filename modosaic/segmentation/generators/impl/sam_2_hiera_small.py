import logging
from typing import override

import numpy as np
import torch
from transformers import Pipeline, pipeline

from modosaic.core.hf_model_spec import HFModelSpec
from modosaic.core.record import ImageRecord
from modosaic.segmentation.generators.base.generator import SegmentationGenerator
from modosaic.services.image import ImageService

logger = logging.getLogger(__name__)


class SAM2HieraSmall(SegmentationGenerator):
    """SAM 2 Hiera Small mask-generation pipeline."""

    pipeline: Pipeline

    def __init__(self):
        """Load the pinned SAM 2 Transformers pipeline."""
        super().__init__()
        self.model_specs = HFModelSpec(
            model_id="facebook/sam2-hiera-small",
            revision="e080ada8afd19df5e165abe71b006edc7f4c3d4e",
            # The Transformers SAM mask postprocessor runs Torchvision NMS with
            # float32 boxes, so scores must stay float32 as well.
            dtype=torch.float32,
        )
        self.pipeline = self._load_pipeline()

    def _load_pipeline(self):
        """Create the Transformers mask-generation pipeline."""
        return pipeline(
            task="mask-generation",
            model=self.model_specs.model_id,
            revision=self.model_specs.revision,
            dtype=self.model_specs.dtype,
            device=self.device
        )

    @override
    def generate(self, record: ImageRecord) -> list[np.ndarray]:
        """Generate segmentation masks for one image record."""
        logger.debug(f"Generating segmentation masks for record sample {record.sample_id}")
        img = ImageService.bytes_to_pil(record.image_bytes)
        masks = self.pipeline(img, points_per_batch=64)["masks"]
        logger.debug(f"Generated {len(masks)} segmentation masks for record sample {record.sample_id}")
        return [
            mask.detach().cpu().float().numpy()
            for mask in masks
        ]
