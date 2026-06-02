import numpy as np

from modosaic.core.configured_modality import ConfiguredModality
from modosaic.core.modalities import Modalities
from modosaic.core.score_functions import boundary_f1_score
from modosaic.core.validation_constraint import ValidationConstraint
from modosaic.core.validator_step import ValidatorStep
from modosaic.depth.generators.factory import DepthGeneratorFactory, DepthGenerationModel
from modosaic.depth.postprocessor import DepthPostprocessor
from modosaic.depth.validators.impl.depth_seg_boundary_consistency_validator import \
    DepthSegBoundaryConsistencyValidator
from modosaic.depth.validators.impl.imagebind import ImageBindValidator

DEPTH_IMAGEBIND_MINIMUM = 0.55
DEPTH_SEGMENTATION_BOUNDARY_F1_MINIMUM = 0.20


def build_preconfigured_depth_modality() -> ConfiguredModality[np.ndarray]:
    """Build the default depth modality.

    Returns:
        Configured depth modality with ImageBind and segmentation-boundary
        validators.
    """
    return ConfiguredModality(
        modality=Modalities.DEPTH,
        model=DepthGeneratorFactory.get(DepthGenerationModel.DEPTH_ANYTHING_V2_SMALL),
        postprocessor=DepthPostprocessor(),
        validators=[
            ValidatorStep(
                validator=ImageBindValidator(DepthGenerationModel.DEPTH_ANYTHING_V2_SMALL),
                constraint=ValidationConstraint(
                    minimum=DEPTH_IMAGEBIND_MINIMUM,
                    score_name="imagebind_similarity",
                ),
            ),
            ValidatorStep(
                validator=DepthSegBoundaryConsistencyValidator(),
                dependencies={"masks": Modalities.SEGMENTATION},
                constraint=ValidationConstraint(
                    minimum=DEPTH_SEGMENTATION_BOUNDARY_F1_MINIMUM,
                    score_name="segmentation_boundary_f1",
                    score_fn=boundary_f1_score,
                ),
            ),
        ],
    )
