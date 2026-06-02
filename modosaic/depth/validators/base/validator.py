from abc import ABC, abstractmethod
from typing import TypeVar, override

import numpy as np

from modosaic.core.record import ImageRecord
from modosaic.core.validator import ModalityValidator

U = TypeVar('U')


class DepthValidator[U](ModalityValidator[np.ndarray, U], ABC):
    """Base class for validators that inspect generated depth maps."""

    @override
    @abstractmethod
    def validate(self, record: ImageRecord, generated: np.ndarray) -> U:
        """Validate a generated depth map."""
        raise NotImplementedError()
