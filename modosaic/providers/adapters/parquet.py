import logging
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence, override

import pyarrow.parquet as pq

from modosaic.core.record import ImageRecord
from modosaic.providers.adapters.adapter import DatasetAdapter
from modosaic.services.extension import ExtensionService

logger = logging.getLogger(__name__)


class ParquetAdapter(DatasetAdapter):
    """Dataset adapter that streams images from parquet rows."""

    def __init__(
            self,
            parquet_path: str | Path,
            image_column: str = "image",
            id_column: str | None = None,
            extension_column: str | None = None,
            metadata_columns: Sequence[str] | None = None,
            batch_size: int = 512,
    ):
        """Initialize the adapter.

        Args:
            parquet_path: Parquet file or directory containing parquet files.
            image_column: Column or nested path containing image bytes or image
                paths.
            id_column: Optional column or nested path used as the sample ID.
            extension_column: Optional column or nested path containing file
                extensions.
            metadata_columns: Optional column paths copied into record metadata.
            batch_size: Number of rows read from parquet at a time.
        """
        self.parquet_path = Path(parquet_path)
        self.image_column = image_column
        self.id_column = id_column
        self.extension_column = extension_column
        self.metadata_columns = tuple(metadata_columns or ())
        self.batch_size = batch_size

    @staticmethod
    def _base_column(column_path: str) -> str:
        """Return the top-level parquet column for a nested column path."""
        return column_path.split(".", 1)[0]

    @staticmethod
    def _value_from_column(rows: Mapping[str, Sequence[Any]], column_path: str, index: int) -> Any:
        """Resolve a possibly nested value from a parquet row batch.

        Args:
            rows: Batch data returned by `pyarrow` as Python dictionaries.
            column_path: Top-level or dotted nested column path.
            index: Row index inside the batch.

        Returns:
            Resolved row value.

        Raises:
            KeyError: If a nested path cannot be resolved.
        """
        parts = column_path.split(".")
        value = rows[parts[0]][index]
        for part in parts[1:]:
            if not isinstance(value, Mapping):
                raise KeyError(
                    f"Cannot resolve nested column '{column_path}': segment '{part}' is not a mapping."
                )
            if part not in value:
                raise KeyError(f"Nested column '{column_path}' not found in row value.")
            value = value[part]
        return value

    @staticmethod
    def _coerce_image_value(
            value: Any,
            *,
            parquet_source: Path,
    ) -> tuple[bytes, str | None, str | None]:
        """Convert a parquet image value into encoded bytes.

        Args:
            value: Raw value from the image column.
            parquet_source: Parquet file used to resolve relative image paths.

        Returns:
            Image bytes, an inferred extension if available, and a resolved
            source path when the value referenced an external file.

        Raises:
            ValueError: If the image value is missing.
            TypeError: If the value type cannot be converted to image bytes.
        """
        if value is None:
            raise ValueError("Image value is None")

        if isinstance(value, (bytes, bytearray, memoryview)):
            image_bytes = bytes(value)
            return image_bytes, ExtensionService.infer_extension_from_bytes(image_bytes), None

        if isinstance(value, Mapping):
            raise TypeError(
                "Image column resolved to a mapping/object. "
                "Use a nested path such as 'image.bytes' or 'image.path'."
            )

        if isinstance(value, str):
            candidate = Path(value).expanduser()
            if not candidate.is_absolute():
                candidate = (parquet_source.parent / candidate).resolve()
            image_bytes = candidate.read_bytes()
            return image_bytes, candidate.suffix.lower() or ExtensionService.infer_extension_from_bytes(
                image_bytes), str(candidate)

        if isinstance(value, list) and all(isinstance(x, int) and 0 <= x <= 255 for x in value):
            image_bytes = bytes(value)
            return image_bytes, ExtensionService.infer_extension_from_bytes(image_bytes), None

        raise TypeError(f"Unsupported image value type: {type(value).__name__}")

    def _resolve_parquet_files(self) -> list[Path]:
        """Resolve configured parquet source into concrete files.

        Returns:
            Sorted list of parquet files to read.

        Raises:
            FileNotFoundError: If the configured path does not exist or a
                directory contains no parquet files.
        """
        if self.parquet_path.is_file():
            logger.debug(f"Using parquet dataset file {self.parquet_path}")
            return [self.parquet_path]
        if self.parquet_path.is_dir():
            files = sorted(self.parquet_path.rglob("*.parquet"))
            if not files:
                raise FileNotFoundError(f"No parquet files found under directory: {self.parquet_path}")
            logger.debug(f"Resolved {len(files)} parquet files under {self.parquet_path}")
            return files
        raise FileNotFoundError(f"Parquet dataset path not found: {self.parquet_path}")

    @override
    def iter_samples(self) -> Iterator[ImageRecord]:
        """Yield image records from parquet rows.

        Returns:
            Iterator of image records.
        """
        read_columns = {self._base_column(self.image_column)}
        if self.id_column:
            read_columns.add(self._base_column(self.id_column))
        if self.extension_column:
            read_columns.add(self._base_column(self.extension_column))
        for metadata_column in self.metadata_columns:
            read_columns.add(self._base_column(metadata_column))

        seen_rows = 0
        for parquet_path in self._resolve_parquet_files():
            logger.debug(f"Reading parquet samples from {parquet_path}")
            parquet_file = pq.ParquetFile(parquet_path)
            for batch in parquet_file.iter_batches(batch_size=self.batch_size, columns=sorted(read_columns)):
                rows = batch.to_pydict()
                for idx in range(batch.num_rows):
                    image_value = self._value_from_column(rows, self.image_column, idx)
                    id_value = self._value_from_column(rows, self.id_column, idx) if self.id_column else None
                    extension_value = (
                        self._value_from_column(rows, self.extension_column, idx)
                        if self.extension_column
                        else None
                    )

                    image_bytes, inferred_extension, source_path = self._coerce_image_value(
                        image_value,
                        parquet_source=parquet_path,
                    )

                    sample_id = (
                        str(id_value)
                        if self.id_column and id_value is not None
                        else f"row_{seen_rows}"
                    )

                    extension = (
                        str(extension_value).strip()
                        if self.extension_column and extension_value
                        else inferred_extension
                    )
                    extension = ExtensionService.normalize_extension(extension, fallback=".bin")

                    metadata = {col: self._value_from_column(rows, col, idx) for col in self.metadata_columns}
                    metadata["source_parquet"] = str(parquet_path)
                    if source_path:
                        metadata["resolved_image_path"] = source_path

                    logger.debug(f"Loading parquet image sample {sample_id} from {parquet_path}")
                    yield ImageRecord(
                        sample_id=sample_id,
                        image_bytes=image_bytes,
                        extension=extension,
                        metadata=metadata,
                    )
                    seen_rows += 1
