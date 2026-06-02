from enum import Enum, auto


class Modalities(Enum):
    """Canonical modality names and processing order."""

    IMAGE = auto()
    TEXT = auto()
    SEGMENTATION = auto()
    DEPTH = auto()
    NORMALS = auto()

    def __str__(self):
        """Return the lowercase modality name used in paths and configs."""
        return self.name.lower()
