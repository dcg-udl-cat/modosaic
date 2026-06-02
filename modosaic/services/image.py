import io
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from PIL import Image


class ImageService:
    """Image conversion helpers shared by generators and validators."""

    @staticmethod
    def bytes_to_pil(image_bytes: bytes) -> "Image.Image":
        """Decode encoded image bytes into an RGB PIL image.

        Args:
            image_bytes: Encoded image bytes.

        Returns:
            RGB PIL image.
        """
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")

    @staticmethod
    def save_bytes_to_tmp_file(image_bytes: bytes, suffix: str = ".jpg") -> Path:
        """Save encoded image bytes to a temporary image file.

        Callers own cleanup for the returned path. Prefer
        `temporary_image_file` when the file is only needed within one scoped
        operation.

        Args:
            image_bytes: Encoded image bytes.
            suffix: Filename suffix for the temporary file.

        Returns:
            Path to the created temporary file.
        """
        with io.BytesIO(image_bytes) as img_io:
            with Image.open(img_io) as img:
                tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
                path = Path(tmp.name)
                img.save(path)
                return path

    @staticmethod
    @contextmanager
    def temporary_image_file(image_bytes: bytes, suffix: str = ".jpg") -> Iterator[Path]:
        """Yield a temporary image file path and remove it when the scope exits.

        Args:
            image_bytes: Encoded image bytes.
            suffix: Filename suffix for the temporary file.

        Yields:
            Path to the temporary image file.
        """
        path = ImageService.save_bytes_to_tmp_file(image_bytes, suffix=suffix)
        try:
            yield path
        finally:
            try:
                path.unlink()
            except FileNotFoundError:
                pass
