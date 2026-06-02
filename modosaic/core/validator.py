from abc import ABC, abstractmethod
from typing import TypeVar

from modosaic.core.record import ImageRecord

T = TypeVar('T')
U = TypeVar('U')


class ModalityValidator[T, U](ABC):
    """Interface for modality validators."""

    @abstractmethod
    def validate(self, record: ImageRecord, generated: T) -> U:
        """Validate a generated modality output.

        Args:
            record: Input image record.
            generated: Generated modality output.

        Returns:
            Validator-specific score or statistics object.
        """
        raise NotImplementedError()
