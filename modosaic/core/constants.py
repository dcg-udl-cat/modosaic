from typing import Final, FrozenSet

DEFAULT_IMAGE_EXTENSIONS: Final[FrozenSet[str]] = frozenset({
    ".jpg",
    ".jpeg",
    ".png",
})

PNG_BYTES: Final[bytes] = b"\x89PNG\r\n\x1a\n"
PNG_EXTENSION: Final[str] = ".png"

JPG_BYTES: Final[bytes] = b"\xff\xd8\xff"
JPG_EXTENSION: Final[str] = ".jpg"
