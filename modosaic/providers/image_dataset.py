import logging
from pathlib import Path
from typing import Iterable, Iterator, Callable

from modosaic.core.record import ImageRecord
from modosaic.providers.adapters.adapter import DatasetAdapter
from modosaic.providers.adapters.local_folder import LocalFolderAdapter
from modosaic.providers.adapters.parquet import ParquetAdapter
from modosaic.services.extension import ExtensionService

logger = logging.getLogger(__name__)


class ImageDataset(Iterable[ImageRecord]):
    """Iterable image dataset facade over a concrete backend adapter."""

    def __init__(self, adapter: DatasetAdapter):
        """Initialize the dataset.

        Args:
            adapter: Backend adapter that yields `ImageRecord` objects.
        """
        self.adapter = adapter

    def __iter__(self) -> Iterator[ImageRecord]:
        """Yield image records from the configured adapter."""
        yield from self.adapter.iter_samples()

    def save_to_folder(
            self,
            output_dir: str | Path,
            overwrite: bool = False,
            naming: Callable[[ImageRecord], str] | None = None,
    ) -> list[Path]:
        """Persist every dataset image to a local folder.

        Args:
            output_dir: Destination directory.
            overwrite: Whether to overwrite existing files. When false,
                duplicate filenames receive numeric suffixes.
            naming: Optional callback that returns a filename for each sample.

        Returns:
            Paths written to disk.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        saved_paths: list[Path] = []
        for sample in self:
            filename = naming(sample) if naming else sample.filename
            filename_path = Path(filename)
            stem = ExtensionService.sanitize_fragment(filename_path.stem)
            suffix = ExtensionService.normalize_extension(filename_path.suffix or sample.extension)

            destination = output_path / f"{stem}{suffix}"
            if not overwrite:
                count = 1
                while destination.exists():
                    destination = output_path / f"{stem}_{count}{suffix}"
                    count += 1

            destination.write_bytes(sample.image_bytes)
            saved_paths.append(destination)
            logger.debug(f"Saved image dataset sample {sample.sample_id} to {destination}")
        return saved_paths

    @classmethod
    def from_local_folder(
            cls,
            root: str | Path,
            recursive: bool = True,
            extensions: Iterable[str] | None = None,
    ) -> "ImageDataset":
        """Create a dataset from a local image folder.

        Args:
            root: Folder containing images.
            recursive: Whether to scan subdirectories.
            extensions: Optional accepted image extensions.

        Returns:
            Dataset backed by `LocalFolderAdapter`.
        """
        return cls(LocalFolderAdapter(root, recursive=recursive, extensions=extensions))

    @classmethod
    def from_parquet(
            cls,
            parquet_path: str | Path,
            image_column: str = "image",
            id_column: str | None = None,
            extension_column: str | None = None,
            metadata_columns: Iterable[str] | None = None,
            batch_size: int = 512,
    ) -> "ImageDataset":
        """Create a dataset from a parquet file or directory.

        Args:
            parquet_path: Parquet file or directory containing parquet files.
            image_column: Column or nested path containing image bytes or paths.
            id_column: Optional sample identifier column.
            extension_column: Optional image extension column.
            metadata_columns: Optional columns copied into record metadata.
            batch_size: Number of rows read from parquet at a time.

        Returns:
            Dataset backed by `ParquetAdapter`.
        """
        return cls(
            ParquetAdapter(
                parquet_path,
                image_column=image_column,
                id_column=id_column,
                extension_column=extension_column,
                metadata_columns=metadata_columns,
                batch_size=batch_size,
            )
        )
