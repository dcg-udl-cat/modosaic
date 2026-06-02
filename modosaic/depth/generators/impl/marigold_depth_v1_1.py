import logging
from typing import override

import numpy as np
from diffusers import MarigoldDepthPipeline

from modosaic.core.hf_model_spec import HFModelSpec
from modosaic.core.record import ImageRecord
from modosaic.depth.generators.base.generator import DepthGenerator
from modosaic.services.image import ImageService

logger = logging.getLogger(__name__)


class MarigoldDepthV11(DepthGenerator):
    """Marigold Depth v1.1 generator."""

    pipeline: MarigoldDepthPipeline

    def __init__(self):
        """Load the pinned Marigold diffusion pipeline."""
        super().__init__()
        self.model_specs = HFModelSpec(
            model_id="prs-eth/marigold-depth-v1-1",
            revision="9571e7123e258cf052b4e54241f17971c290e9a8",
            dtype=self.dtype,
        )
        self.pipeline = self._load_pipeline()

    def _load_pipeline(self) -> MarigoldDepthPipeline:
        """Create the Marigold depth pipeline."""
        return MarigoldDepthPipeline.from_pretrained(
            pretrained_model_name_or_path=self.model_specs.model_id,
            revision=self.model_specs.revision,
            dtype=self.model_specs.dtype,
        ).to(self.device)

    @override
    def generate(self, record: ImageRecord) -> np.ndarray:
        """Generate a depth map for one image record."""
        logger.debug(f"Generating depth for record sample {record.sample_id}")
        img = ImageService.bytes_to_pil(record.image_bytes)
        np_depth = self.pipeline(img)["prediction"]

        return np_depth
