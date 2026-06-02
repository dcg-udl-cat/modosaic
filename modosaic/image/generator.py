import logging
from typing import override

from modosaic.core.modality_generator import ModalityGenerator
from modosaic.core.record import ImageRecord

logger = logging.getLogger(__name__)


class SourceImageGenerator(ModalityGenerator[bytes]):
    """Generator that forwards the source image bytes unchanged."""

    @override
    def generate(self, record: ImageRecord) -> bytes:
        """Return the encoded image bytes from the input record."""
        logger.debug(f"Using source image for record sample {record.sample_id}")
        return record.image_bytes
