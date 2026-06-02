from modosaic.core.configured_modality import ConfiguredModality
from modosaic.core.modalities import Modalities
from modosaic.core.validation_constraint import ValidationConstraint
from modosaic.core.validator_step import ValidatorStep
from modosaic.text.generators.factory import TextGeneratorFactory, TextGenerationModel
from modosaic.text.postprocessor import TextPostprocessor
from modosaic.text.validators.impl.siglip_2 import SIGLIP2Validator

TEXT_SIGLIP_MINIMUM = 0.60


def build_preconfigured_text_modality() -> ConfiguredModality[str]:
    """Build the default text-caption modality.

    Returns:
        Configured text modality with the default caption generator, text
        postprocessor, and SIGLIP quality gate.
    """
    return ConfiguredModality(
        modality=Modalities.TEXT,
        model=TextGeneratorFactory.get(TextGenerationModel.QWEN_2_2B),
        postprocessor=TextPostprocessor(),
        validators=[
            ValidatorStep(
                validator=SIGLIP2Validator(),
                constraint=ValidationConstraint(
                    minimum=TEXT_SIGLIP_MINIMUM,
                    score_name="siglip_probability",
                ),
            ),
        ],
    )
