import logging
from typing import override

import numpy as np
from transformers import pipeline, Pipeline

from modosaic.core.hf_model_spec import HFModelSpec
from modosaic.core.record import ImageRecord
from modosaic.depth.generators.base.generator import DepthGenerator
from modosaic.services.image import ImageService

logger = logging.getLogger(__name__)


class DepthAnythingSmall(DepthGenerator):
    """Depth Anything Small generator."""

    pipeline: Pipeline

    def __init__(self):
        """Load the pinned Hugging Face depth-estimation pipeline."""
        super().__init__()
        self.model_specs = HFModelSpec(
            model_id="LiheYoung/depth-anything-small-hf",
            revision="25216a913fa218ccb7d58cce818d52b728b6c1f6",
            dtype=self.pipeline_dtype,
        )
        self.pipeline = self._load_pipeline()

    def _load_pipeline(self):
        """Create the Transformers depth-estimation pipeline."""
        return pipeline(
            task="depth-estimation",
            model=self.model_specs.model_id,
            revision=self.model_specs.revision,
            dtype=self.model_specs.dtype,
            device=self.device
        )

    @override
    def generate(self, record: ImageRecord) -> np.ndarray:
        """Generate a depth map for one image record."""
        logger.debug(f"Generating depth anything small image for record sample {record.sample_id}")
        img = ImageService.bytes_to_pil(record.image_bytes)
        depth_tensor = self.pipeline(img)["predicted_depth"]

        np_depth = depth_tensor.detach().cpu().float().numpy()
        return np_depth
