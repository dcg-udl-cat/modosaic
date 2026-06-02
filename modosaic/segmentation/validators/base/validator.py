from abc import ABC, abstractmethod
from typing import TypeVar, override

import numpy as np

from modosaic.core.record import ImageRecord
from modosaic.core.validator import ModalityValidator

U = TypeVar('U')


class SegmentationValidator[U](ModalityValidator[list[np.ndarray], U], ABC):
    """Base class for validators that inspect generated masks."""

    @override
    @abstractmethod
    def validate(self, record: ImageRecord, generated: list[np.ndarray]) -> U:
        """Validate generated segmentation masks."""
        raise NotImplementedError()
