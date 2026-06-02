from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from modosaic.core.experiment_artifact import ExperimentArtifact
from modosaic.core.modalities import Modalities
from modosaic.core.record import ImageRecord
from modosaic.services.extension import ExtensionService


class ModalityPostprocessor[T](ABC):
    """Interface for turning generated outputs into experiment artifacts."""

    @property
    @abstractmethod
    def modality(self) -> Modalities:
        """Return the modality type this postprocessor saves."""
        raise NotImplementedError()

    @abstractmethod
    def process(self, record: ImageRecord, generated: T) -> list[ExperimentArtifact[Any]]:
        """Build artifacts for a generated modality output.

        Args:
            record: Input image record.
            generated: Generated modality output.

        Returns:
            Experiment artifacts ready for persistence.
        """
        raise NotImplementedError()

    @staticmethod
    def sample_stem(record: ImageRecord) -> str:
        """Return a filesystem-safe stem for a sample."""
        return ExtensionService.sanitize_fragment(record.sample_id)

    def sample_path(self, record: ImageRecord, suffix: str) -> Path:
        """Build a relative artifact path for the postprocessor modality.

        Args:
            record: Input image record.
            suffix: File suffix to append to the sanitized sample stem.

        Returns:
            Relative path below the modality folder.
        """
        return Path(str(self.modality)) / f"{self.sample_stem(record)}{suffix}"
