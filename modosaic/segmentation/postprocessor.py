import logging
from typing import Any, override

import numpy as np

from modosaic.core.experiment_artifact import ExperimentArtifact
from modosaic.core.modalities import Modalities
from modosaic.core.postprocessor import ModalityPostprocessor
from modosaic.core.record import ImageRecord
from modosaic.services.image import ImageService
from modosaic.segmentation.visualization import SegmentationVisualizationService

logger = logging.getLogger(__name__)


class SegmentationPostprocessor(ModalityPostprocessor[list[np.ndarray]]):
    """Postprocessor that saves segmentation overlays and instance maps.

    Attributes:
        alpha: Overlay alpha used for mask previews.
        seed: Seed used for deterministic mask colors.
    """

    alpha: float
    seed: int

    def __init__(self, alpha: float = 0.55, seed: int = 0) -> None:
        """Initialize the postprocessor.

        Args:
            alpha: Overlay alpha used for mask previews.
            seed: Seed used for deterministic mask colors.
        """
        self.alpha = alpha
        self.seed = seed

    @property
    @override
    def modality(self) -> Modalities:
        """Return the segmentation modality."""
        return Modalities.SEGMENTATION

    @override
    def process(self, record: ImageRecord, generated: list[np.ndarray]) -> list[ExperimentArtifact[Any]]:
        """Create segmentation overlay and instance-map artifacts."""
        logger.debug(
            f"Post-processing {len(generated)} segmentation masks for record sample {record.sample_id}"
        )
        image = ImageService.bytes_to_pil(record.image_bytes)
        H, W = image.size[1], image.size[0]
        overlay = SegmentationVisualizationService.overlay_masks(
            image,
            generated,
            alpha=self.alpha,
            seed=self.seed,
        )
        instance_map = SegmentationVisualizationService.instance_map_from_masks(
            generated,
            H,
            W,
        )

        return [
            ExperimentArtifact.image(self.sample_path(record, ".png"), overlay),
            ExperimentArtifact.array(self.sample_path(record, ".npy"), instance_map),
        ]
