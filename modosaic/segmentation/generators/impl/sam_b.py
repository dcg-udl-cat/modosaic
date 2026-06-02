import logging
from pathlib import Path
from typing import override

import numpy as np
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

from modosaic.core.record import ImageRecord
from modosaic.segmentation.generators.base.generator import SegmentationGenerator
from modosaic.services.image import ImageService

logger = logging.getLogger(__name__)


class SAM(SegmentationGenerator):
    """Segment Anything ViT-B mask generator using a local checkpoint."""

    generator: SamAutomaticMaskGenerator

    def __init__(self):
        """Load the local SAM checkpoint and automatic mask generator."""
        super().__init__()
        self.generator = self._load_generator()

    def _load_generator(self) -> SamAutomaticMaskGenerator:
        """Create the SAM automatic mask generator.

        Raises:
            FileNotFoundError: If the expected local checkpoint is missing.
        """
        model_type = "vit_b"
        sam_checkpoint = Path("model_weights/sam_vit_b_01ec64.pth")
        if not sam_checkpoint.exists():
            raise FileNotFoundError(
                f"SAM checkpoint not found at {sam_checkpoint}. Please download it and place it in the specified path.")
        logger.debug(f"Loading SAM model from {sam_checkpoint}")
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint).to(self.device)

        generator = SamAutomaticMaskGenerator(sam)
        self._fix_dtype(generator)

        return generator

    def _fix_dtype(self, generator: SamAutomaticMaskGenerator) -> SamAutomaticMaskGenerator:
        """Apply MPS-specific coordinate dtype compatibility fix."""
        if self.device.type == "mps":
            original_apply_coords = generator.predictor.transform.apply_coords

            def apply_coords_float32(coords: np.ndarray, original_size: tuple[int, int]) -> np.ndarray:
                """Return SAM-transformed coordinates as MPS-compatible float32 arrays."""
                # `segment_anything` emits float64 coords by default, which MPS cannot cast.
                return np.asarray(original_apply_coords(coords, original_size), dtype=np.float32)

            generator.predictor.transform.apply_coords = apply_coords_float32
        return generator

    @override
    def generate(self, record: ImageRecord) -> list[np.ndarray]:
        """Generate segmentation masks for one image record."""
        logger.debug(f"Generating segmentation masks for record sample {record.sample_id}")
        img = ImageService.bytes_to_pil(record.image_bytes)
        np_img = np.array(img)
        masks = self.generator.generate(np_img)
        logger.debug(f"Generated {len(masks)} segmentation masks for record sample {record.sample_id}")

        return [
            mask["segmentation"].astype(np.float32)
            for mask in masks
        ]
