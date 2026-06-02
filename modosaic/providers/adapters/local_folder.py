import logging
from pathlib import Path
from typing import Sequence, Iterator, override

from modosaic.core.constants import DEFAULT_IMAGE_EXTENSIONS
from modosaic.core.record import ImageRecord
from modosaic.providers.adapters.adapter import DatasetAdapter

logger = logging.getLogger(__name__)


class LocalFolderAdapter(DatasetAdapter):
    """Dataset adapter that streams images from a local folder."""

    def __init__(
            self,
            root: str | Path,
            recursive: bool = True,
            extensions: Sequence[str] | None = None,
    ):
        """Initialize the adapter.

        Args:
            root: Folder containing image files.
            recursive: Whether to scan subdirectories.
            extensions: Optional accepted extensions. Values may be passed with
                or without a leading dot.
        """
        self.root = Path(root)
        normalized = extensions or tuple(DEFAULT_IMAGE_EXTENSIONS)
        self.extensions = {ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in normalized}
        self.recursive = recursive

    @override
    def iter_samples(self) -> Iterator[ImageRecord]:
        """Yield image records in stable path order.

        Returns:
            Iterator of local image records.

        Raises:
            FileNotFoundError: If the configured root does not exist.
        """
        if not self.root.exists():
            raise FileNotFoundError(f"Local dataset folder not found: {self.root}")

        logger.debug(f"Scanning local dataset folder {self.root}")
        iterator = self.root.rglob("*") if self.recursive else self.root.glob("*")
        sorted_paths = sorted(iterator, key=lambda path: path.as_posix())
        for image_path in sorted_paths:
            if not image_path.is_file() or image_path.suffix.lower() not in self.extensions:
                continue
            relative = image_path.relative_to(self.root)
            sample_id = "_".join(relative.with_suffix("").parts)
            logger.debug(f"Loading local image sample {sample_id} from {image_path}")
            yield ImageRecord(
                sample_id=sample_id,
                image_bytes=image_path.read_bytes(),
                extension=image_path.suffix.lower(),
                metadata={"source_path": str(image_path)},
            )
