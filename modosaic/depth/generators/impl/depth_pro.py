import logging
from typing import override

import numpy as np
import torch
from torch.fx.tensor_type import TensorType
from transformers import DepthProImageProcessorFast, DepthProForDepthEstimation

from modosaic.core.hf_model_spec import HFModelSpec
from modosaic.core.record import ImageRecord
from modosaic.depth.generators.base.generator import DepthGenerator
from modosaic.services.image import ImageService

logger = logging.getLogger(__name__)


class DepthPro(DepthGenerator):
    """Apple Depth Pro generator."""

    model: DepthProForDepthEstimation
    processor: DepthProImageProcessorFast

    def __init__(self):
        """Load the pinned Depth Pro model and processor."""
        super().__init__()
        self.model_specs = HFModelSpec(
            model_id="apple/DepthPro-hf",
            revision="de816c8ce7168afcb231f96d501d72b869d0beda",
            dtype=self.dtype,
        )
        self.model = self._load_model()
        self.model = self.model.to(self.device)
        self.processor = self._load_processor()

    def _load_model(self) -> DepthProForDepthEstimation:
        """Load the Depth Pro model."""
        return DepthProForDepthEstimation.from_pretrained(
            pretrained_model_name_or_path=self.model_specs.model_id,
            revision=self.model_specs.revision,
            dtype=self.model_specs.dtype,
        )

    def _load_processor(self) -> DepthProImageProcessorFast:
        """Load the Depth Pro image processor."""
        return DepthProImageProcessorFast.from_pretrained(
            pretrained_model_name_or_path=self.model_specs.model_id,
            revision=self.model_specs.revision,
            dtype=self.model_specs.dtype,
        )

    @override
    @torch.no_grad()
    def generate(self, record: ImageRecord) -> np.ndarray:
        """Generate a normalized depth map for one image record."""
        logger.debug(f"Generating depth for record sample {record.sample_id}")
        img = ImageService.bytes_to_pil(record.image_bytes)
        inputs = self.processor(images=img, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)

        post_processed_output = self.processor.post_process_depth_estimation(
            outputs, target_sizes=[(img.height, img.width)],
        )
        depth_tensor = post_processed_output[0]["predicted_depth"]
        return DepthPro._post_process(depth_tensor)

    @staticmethod
    def _post_process(depth: TensorType) -> np.ndarray:
        """Normalize a Depth Pro tensor to a NumPy image-like array."""
        logger.debug(f"Post-processing depth")
        depth = (depth - depth.min()) / (depth.max() - depth.min())
        depth = depth * 255.
        return depth.detach().cpu().float().numpy()
