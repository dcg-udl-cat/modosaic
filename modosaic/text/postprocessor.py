import logging
from typing import Any, override

from modosaic.core.experiment_artifact import ExperimentArtifact
from modosaic.core.modalities import Modalities
from modosaic.core.postprocessor import ModalityPostprocessor
from modosaic.core.record import ImageRecord

logger = logging.getLogger(__name__)


class TextPostprocessor(ModalityPostprocessor[str]):
    """Postprocessor that saves generated text captions."""

    @property
    @override
    def modality(self) -> Modalities:
        """Return the text modality."""
        return Modalities.TEXT  # type: ignore

    @override
    def process(self, record: ImageRecord, generated: str) -> list[ExperimentArtifact[Any]]:
        """Create a text artifact for a generated caption."""
        logger.debug(f"Post-processing text for record sample {record.sample_id}")
        return [
            ExperimentArtifact.text(self.sample_path(record, ".txt"), generated)
        ]
