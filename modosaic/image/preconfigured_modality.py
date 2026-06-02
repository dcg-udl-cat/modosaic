from modosaic.core.configured_modality import ConfiguredModality
from modosaic.core.modalities import Modalities
from modosaic.image.generator import SourceImageGenerator
from modosaic.image.postprocessor import ImagePostprocessor


def build_preconfigured_image_modality() -> ConfiguredModality[bytes]:
    """Build the default source-image passthrough modality.

    Returns:
        Configured image modality that saves the original encoded image bytes.
    """
    return ConfiguredModality(
        modality=Modalities.IMAGE,
        model=SourceImageGenerator(),
        postprocessor=ImagePostprocessor(),
    )
