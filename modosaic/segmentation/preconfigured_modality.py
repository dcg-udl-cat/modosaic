import numpy as np

from modosaic.core.configured_modality import ConfiguredModality
from modosaic.core.modalities import Modalities
from modosaic.core.score_functions import boundary_f1_score
from modosaic.core.validation_constraint import ValidationConstraint
from modosaic.core.validator_step import ValidatorStep
from modosaic.segmentation.generators.factory import SegmentationGeneratorFactory
from modosaic.segmentation.mask_validation_stats import MaskValidationStats
from modosaic.segmentation.postprocessor import SegmentationPostprocessor
from modosaic.segmentation.validators.impl.boundary_rgb_edge_overlap import SegRgbEdgeOverlapValidator
from modosaic.segmentation.validators.impl.mask_statistics import MaskStatsValidator

SEGMENTATION_MASK_QUALITY_MINIMUM = 0.75
SEGMENTATION_BOUNDARY_F1_MINIMUM = 0.20


def build_preconfigured_segmentation_modality() -> ConfiguredModality[list[np.ndarray]]:
    """Build the default segmentation modality.

    Returns:
        Configured segmentation modality with mask-statistics and RGB-boundary
        validators.
    """
    return ConfiguredModality(
        modality=Modalities.SEGMENTATION,
        model=SegmentationGeneratorFactory.get(),
        postprocessor=SegmentationPostprocessor(),
        validators=[
            ValidatorStep(
                validator=MaskStatsValidator(),
                constraint=ValidationConstraint(
                    minimum=SEGMENTATION_MASK_QUALITY_MINIMUM,
                    score_name="weighted_mask_quality",
                    score_fn=_mask_quality_score,
                ),
            ),
            ValidatorStep(
                validator=SegRgbEdgeOverlapValidator(),
                constraint=ValidationConstraint(
                    minimum=SEGMENTATION_BOUNDARY_F1_MINIMUM,
                    score_name="rgb_boundary_f1",
                    score_fn=boundary_f1_score,
                ),
            ),
        ],
    )


def _mask_quality_score(stats: MaskValidationStats) -> float | int:
    """Compute the weighted segmentation-mask quality score."""
    if not stats.valid:
        return 0.0

    return (
            0.4 * stats.coverage_score
            + 0.3 * stats.distinctness_score
            + 0.3 * stats.fragmentation_score
    )
