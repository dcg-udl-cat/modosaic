import numpy as np

from modosaic.core.boundary_alignment_stats import BoundaryAlignmentStats
from modosaic.services.boundary import BoundaryService


class ToleranceService:
    """Metrics that compare predicted edges and target boundaries with slack."""

    @staticmethod
    def compute_boundary_alignment_stats(
            edge_map: np.ndarray,  # bool HxW
            boundary_map: np.ndarray,  # bool HxW
            tolerance_radius: int,
    ) -> BoundaryAlignmentStats:
        """Compute precision, recall, F1, and IoU with tolerance dilation.

        Args:
            edge_map: Boolean predicted edge map with shape `HxW`.
            boundary_map: Boolean target boundary map with shape `HxW`.
            tolerance_radius: Pixel radius used to dilate maps for precision
                and recall.

        Returns:
            Boundary-alignment statistics.
        """
        boundary_tol = BoundaryService.dilate_bool(boundary_map, tolerance_radius)
        edge_tol = BoundaryService.dilate_bool(edge_map, tolerance_radius)

        matched_edges = float(np.count_nonzero(edge_map & boundary_tol))
        edge_count = float(np.count_nonzero(edge_map))
        boundary_count = float(np.count_nonzero(boundary_map))

        precision = matched_edges / max(edge_count, 1.0)
        recall = float(np.count_nonzero(boundary_map & edge_tol)) / max(boundary_count, 1.0)
        f1 = (2.0 * precision * recall) / max(precision + recall, 1e-12)

        union = float(np.count_nonzero(edge_map | boundary_map))
        iou = float(np.count_nonzero(edge_map & boundary_map)) / max(union, 1.0)

        return BoundaryAlignmentStats(
            valid=True,
            precision=float(precision),
            recall=float(recall),
            f1=float(f1),
            iou=float(iou),
        )
