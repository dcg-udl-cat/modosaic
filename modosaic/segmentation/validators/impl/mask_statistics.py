import logging
from typing import override

import numpy as np

from modosaic.core.record import ImageRecord
from modosaic.segmentation.mask_validation_stats import MaskValidationStats
from modosaic.segmentation.validators.base.validator import SegmentationValidator
from modosaic.services.boundary import BoundaryService

logger = logging.getLogger(__name__)


class MaskStatsValidator(SegmentationValidator[MaskValidationStats]):
    """Compute coverage, distinctness, and fragmentation scores for masks."""

    @override
    def validate(self, record: ImageRecord, generated: list[np.ndarray]) -> MaskValidationStats:
        """Validate generated masks with aggregate mask statistics."""
        logger.debug(f"Computing mask statistics for record sample {record.sample_id} with {len(generated)} masks")
        stats = self._compute_score(generated)
        logger.debug(f"Computed mask statistics for record sample {record.sample_id}")
        return stats

    @staticmethod
    def _compute_score(masks: list[np.ndarray]) -> MaskValidationStats:
        """Compute mask statistics from a mask list."""
        valid_masks: list[np.ndarray] = []
        expected_shape: tuple[int, int] | None = None

        for mask in masks:
            binary_mask = MaskStatsValidator._coerce_binary_mask(mask, expected_shape)
            if binary_mask is None:
                continue

            if expected_shape is None:
                expected_shape = binary_mask.shape

            if not np.any(binary_mask):
                continue

            valid_masks.append(binary_mask)

        if not valid_masks:
            return MaskValidationStats()

        stack = np.stack(valid_masks, axis=0)
        height, width = stack.shape[1], stack.shape[2]
        image_area = float(height * width)
        if image_area <= 0.0:
            return MaskValidationStats()

        union_pixels = float(np.count_nonzero(np.any(stack, axis=0)))
        union_ratio = union_pixels / image_area

        overlap_pixels = float(np.count_nonzero(np.sum(stack, axis=0) > 1))
        overlap_ratio = overlap_pixels / max(union_pixels, 1.0)

        mask_area_ratios = np.count_nonzero(stack, axis=(1, 2)).astype(np.float64) / image_area
        tiny_ratio = float(np.count_nonzero(mask_area_ratios < 0.01)) / float(stack.shape[0])

        coverage_score = max(0.0, min(union_ratio, 1.0))
        distinctness_score = 1.0 - overlap_ratio
        fragmentation_score = 1.0 - tiny_ratio

        return MaskValidationStats(
            valid=True,
            coverage_score=coverage_score,
            distinctness_score=distinctness_score,
            fragmentation_score=fragmentation_score,
        )

    @staticmethod
    def _coerce_binary_mask(mask: np.ndarray, expected_shape: tuple[int, int] | None) -> np.ndarray | None:
        """Coerce one mask to a boolean array."""
        return BoundaryService.coerce_mask_binary(mask, expected_shape)
