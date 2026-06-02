from abc import ABC, abstractmethod
from typing import TypeVar

from modosaic.core.record import ImageRecord

T = TypeVar("T")


class ModalityGenerator[T](ABC):
    """Interface for model-backed modality generators."""

    @abstractmethod
    def generate(self, record: ImageRecord) -> T:
        """Generate a modality output for one image record.

        Args:
            record: Input image record.

        Returns:
            Generated modality output.
        """
        raise NotImplementedError()
