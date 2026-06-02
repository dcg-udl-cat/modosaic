from dataclasses import dataclass, field
from typing import Any, Mapping

from modosaic.services.extension import ExtensionService


@dataclass(slots=True)
class ImageRecord:
    """Image sample exchanged between dataset providers and modalities.

    Attributes:
        sample_id: Stable identifier for the sample inside the dataset.
        image_bytes: Encoded image bytes as read from the source backend.
        extension: Normalized image extension used when saving artifacts.
        metadata: Backend-specific metadata associated with the sample.
    """

    sample_id: str
    image_bytes: bytes
    extension: str = ".bin"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def filename(self) -> str:
        """Return a filesystem-safe filename for the sample.

        Returns:
            The sanitized sample identifier with a normalized extension.
        """
        ext = ExtensionService.normalize_extension(self.extension)
        return f"{ExtensionService.sanitize_fragment(self.sample_id)}{ext}"
