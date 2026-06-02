from modosaic.core.constants import PNG_BYTES, PNG_EXTENSION, JPG_BYTES, JPG_EXTENSION


class ExtensionService:
    """Helpers for image extensions and filesystem-safe path fragments."""

    @staticmethod
    def infer_extension_from_bytes(data: bytes) -> str | None:
        """Infer a known image extension from encoded bytes.

        Args:
            data: Encoded image bytes.

        Returns:
            `.png`, `.jpg`, or `None` when the signature is unknown.
        """
        if data.startswith(PNG_BYTES):
            return PNG_EXTENSION
        if data.startswith(JPG_BYTES):
            return JPG_EXTENSION
        return None

    @staticmethod
    def sanitize_fragment(value: str) -> str:
        """Convert arbitrary text into a safe filename fragment.

        Args:
            value: Text to sanitize.

        Returns:
            A non-empty fragment containing alphanumeric characters, dots,
            dashes, and underscores only.
        """
        safe = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value)
        return safe.strip("._") or "sample"

    @staticmethod
    def normalize_extension(extension: str | None, fallback: str = ".bin") -> str:
        """Normalize a file extension.

        Args:
            extension: Optional extension with or without a leading dot.
            fallback: Extension returned when `extension` is empty.

        Returns:
            Extension with a leading dot.
        """
        if not extension:
            return fallback
        return extension if extension.startswith(".") else f".{extension}"
