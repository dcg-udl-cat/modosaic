import logging
from typing import override

import numpy as np

from modosaic.core.boundary_alignment_stats import BoundaryAlignmentStats
from modosaic.core.record import ImageRecord
from modosaic.segmentation.validators.base.validator import SegmentationValidator
from modosaic.services.boundary import BoundaryService
from modosaic.services.edge import EdgeService
from modosaic.services.image import ImageService
from modosaic.services.tolerance import ToleranceService

logger = logging.getLogger(__name__)


class SegRgbEdgeOverlapValidator(SegmentationValidator[BoundaryAlignmentStats]):
    """Compare segmentation boundaries with RGB image edges."""

    boundary_thickness: int
    tolerance_radius: int
    rgb_edge_quantile: float

    def __init__(
            self,
            boundary_thickness: int = 1,
            tolerance_radius: int = 2,
            rgb_edge_quantile: float = 0.90,
    ) -> None:
        """Initialize the validator.

        Args:
            boundary_thickness: Mask boundary thickness in pixels.
            tolerance_radius: Pixel tolerance used for precision and recall.
            rgb_edge_quantile: Quantile threshold for RGB edge detection.
        """
        self.boundary_thickness = boundary_thickness
        self.tolerance_radius = tolerance_radius
        self.rgb_edge_quantile = rgb_edge_quantile

    @override
    def validate(self, record: ImageRecord, generated: list[np.ndarray]) -> BoundaryAlignmentStats:
        """Compute RGB-edge alignment for generated segmentation masks."""
        logger.debug(
            f"Computing RGB edge overlap for record sample {record.sample_id} with {len(generated)} masks"
        )
        pil = ImageService.bytes_to_pil(record.image_bytes)
        rgb = np.array(pil, dtype=np.uint8)  # HxWx3 RGB

        boundary = BoundaryService.masks_to_boundary(generated, expected_shape=(rgb.shape[0], rgb.shape[1]),
                                                     thickness=self.boundary_thickness)
        if not np.any(boundary):
            logger.debug(f"No segmentation boundary found for record sample {record.sample_id}")
            return BoundaryAlignmentStats()

        edges = EdgeService.rgb_edge_map(rgb, q=self.rgb_edge_quantile)

        stats = ToleranceService.compute_boundary_alignment_stats(
            edge_map=edges,
            boundary_map=boundary,
            tolerance_radius=self.tolerance_radius,
        )
        logger.debug(f"Computed RGB edge overlap for record sample {record.sample_id}")
        return stats
