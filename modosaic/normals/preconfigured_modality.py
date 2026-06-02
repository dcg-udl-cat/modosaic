import numpy as np

from modosaic.core.configured_modality import ConfiguredModality
from modosaic.core.modalities import Modalities
from modosaic.core.validation_constraint import ValidationConstraint
from modosaic.core.validator_step import ValidatorStep
from modosaic.normals.generators.factory import NormalsGeneratorFactory
from modosaic.normals.normal_agreement_stats import NormalAgreementStats
from modosaic.normals.normal_field_stats import NormalFieldStats
from modosaic.normals.postprocessor import NormalsPostprocessor
from modosaic.normals.validators.impl.depth_normals_agreement import DepthNormalsAngularAgreementValidator
from modosaic.segmentation.validators.impl.normals_field_quality import NormalsFieldQualityValidator

NORMALS_DEPTH_AGREEMENT_MINIMUM = 0.35
NORMALS_FIELD_QUALITY_MINIMUM = 0.50


def build_preconfigured_normals_modality() -> ConfiguredModality[np.ndarray]:
    """Build the default surface-normal modality.

    Returns:
        Configured normals modality with depth-agreement and field-quality
        validators.
    """
    return ConfiguredModality(
        modality=Modalities.NORMALS,
        model=NormalsGeneratorFactory.get(),
        postprocessor=NormalsPostprocessor(),
        validators=[
            ValidatorStep(
                validator=DepthNormalsAngularAgreementValidator(),
                dependencies={"depth_map": Modalities.DEPTH},
                constraint=ValidationConstraint(
                    minimum=NORMALS_DEPTH_AGREEMENT_MINIMUM,
                    score_name="depth_normals_agreement",
                    score_fn=_depth_normals_agreement_score,
    )
            ),
            ValidatorStep(
                validator=NormalsFieldQualityValidator(),
                constraint=ValidationConstraint(
                    minimum=NORMALS_FIELD_QUALITY_MINIMUM,
                    score_name="normals_field_quality",
                    score_fn=_normals_field_quality_score,
                ),
            ),
        ],
    )


def _depth_normals_agreement_score(stats: NormalAgreementStats) -> float | int:
    """Compute the weighted depth-normal agreement score."""
    if not stats.valid:
        return 0.0

    return (
            0.2 * stats.pct_under_11_25
            + 0.5 * stats.pct_under_22_5
            + 0.3 * stats.pct_under_30
    )


def _normals_field_quality_score(stats: NormalFieldStats) -> float | int:
    """Compute the weighted normal-field quality score."""
    if not stats.valid:
        return 0.0

    smoothness_score = max(0.0, min(1.0, 1.0 - stats.smoothness_median_deg / 45.0))
    integrability_score = 1.0 / (1.0 + max(0.0, stats.integrability_median))
    return 0.7 * smoothness_score + 0.3 * integrability_score
