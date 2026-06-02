from abc import ABC, abstractmethod
from typing import Iterator

from modosaic.core.record import ImageRecord


class DatasetAdapter(ABC):
    """Interface implemented by dataset backends."""

    @abstractmethod
    def iter_samples(self) -> Iterator[ImageRecord]:
        """Yield image records from the backend.

        Returns:
            Iterator of image records.
        """
        pass
