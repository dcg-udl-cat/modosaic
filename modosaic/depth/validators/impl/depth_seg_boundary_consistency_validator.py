import logging
from typing import override

import numpy as np

from modosaic.core.boundary_alignment_stats import BoundaryAlignmentStats
from modosaic.core.record import ImageRecord
from modosaic.depth.validators.base.validator import DepthValidator
from modosaic.services.boundary import BoundaryService
from modosaic.services.edge import EdgeService
from modosaic.services.tolerance import ToleranceService

logger = logging.getLogger(__name__)


class DepthSegBoundaryConsistencyValidator(DepthValidator[BoundaryAlignmentStats]):
    """Compare depth-map edges with segmentation-mask boundaries."""

    masks: list[np.ndarray]
    boundary_thickness: int
    tolerance_radius: int
    depth_edge_quantile: float

    def __init__(
            self,
            boundary_thickness: int = 1,
            tolerance_radius: int = 2,
            depth_edge_quantile: float = 0.90,
    ) -> None:
        """Initialize the validator.

        Args:
            boundary_thickness: Mask boundary thickness in pixels.
            tolerance_radius: Pixel tolerance used for precision and recall.
            depth_edge_quantile: Quantile threshold for depth edge detection.
        """
        self.boundary_thickness = boundary_thickness
        self.tolerance_radius = tolerance_radius
        self.depth_edge_quantile = depth_edge_quantile

    @override
    def validate(self, record: ImageRecord, generated: np.ndarray,
                 masks: list[np.ndarray] | None = None) -> BoundaryAlignmentStats:
        """Compute boundary alignment between generated depth and masks."""
        if not masks:
            return BoundaryAlignmentStats()

        depth, valid_mask = BoundaryService.coerce_hw_float_with_mask(generated)
        if not np.any(valid_mask):
            return BoundaryAlignmentStats()

        logger.debug(
            f"Computing boundary alignment stats for record sample {record.sample_id} with depth shape {depth.shape} and "
            f"{len(masks)} segmentation masks")
        boundary = BoundaryService.masks_to_boundary(masks, expected_shape=depth.shape,
                                                     thickness=self.boundary_thickness)

        if not np.any(boundary):
            return BoundaryAlignmentStats()

        depth_edges = EdgeService.sobel_edge_map_float(
            depth,
            blur_ksize=3,
            q=self.depth_edge_quantile,
            valid_mask=valid_mask,
        )

        stats = ToleranceService.compute_boundary_alignment_stats(
            edge_map=depth_edges,
            boundary_map=boundary,
            tolerance_radius=self.tolerance_radius,
        )
        logger.debug(f"Computed boundary alignment stats for record sample {record.sample_id}")
        return stats
