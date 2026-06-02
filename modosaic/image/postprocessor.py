import logging
from typing import Any, override

from modosaic.core.experiment_artifact import ExperimentArtifact
from modosaic.core.modalities import Modalities
from modosaic.core.postprocessor import ModalityPostprocessor
from modosaic.core.record import ImageRecord
from modosaic.services.image import ImageService

logger = logging.getLogger(__name__)


class ImagePostprocessor(ModalityPostprocessor[bytes]):
    """Postprocessor that saves source images as PNG artifacts."""

    @property
    @override
    def modality(self) -> Modalities:
        """Return the image modality."""
        return Modalities.IMAGE  # type: ignore

    @override
    def process(self, record: ImageRecord, generated: bytes) -> list[ExperimentArtifact[Any]]:
        """Convert encoded image bytes into a PNG artifact."""
        logger.debug(f"Post-processing source image for record sample {record.sample_id}")
        image = ImageService.bytes_to_pil(generated)
        return [
            ExperimentArtifact.image(self.sample_path(record, ".png"), image)
        ]
