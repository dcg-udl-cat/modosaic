from abc import ABC, abstractmethod
from typing import TypeVar, override

from modosaic.core.record import ImageRecord
from modosaic.core.validator import ModalityValidator

U = TypeVar('U')


class TextValidator[U](ModalityValidator[str, U], ABC):
    """Base class for validators that inspect generated text."""

    @override
    @abstractmethod
    def validate(self, record: ImageRecord, generated: str) -> U:
        """Validate generated text."""
        raise NotImplementedError()
