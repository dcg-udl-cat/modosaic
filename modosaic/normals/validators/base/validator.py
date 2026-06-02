from abc import ABC, abstractmethod
from typing import TypeVar, override

import numpy as np

from modosaic.core.record import ImageRecord
from modosaic.core.validator import ModalityValidator

U = TypeVar('U')


class NormalsValidator[U](ModalityValidator[np.ndarray, U], ABC):
    """Base class for validators that inspect generated normal fields."""

    @override
    @abstractmethod
    def validate(self, record: ImageRecord, generated: np.ndarray) -> U:
        """Validate a generated normal field."""
        raise NotImplementedError()
