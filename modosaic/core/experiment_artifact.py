from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np
from PIL.Image import Image


class ExperimentArtifactKind(StrEnum):
    """Supported artifact payload types for experiment persistence."""

    TEXT = "text"
    ARRAY = "array"
    ARRAY_COLLECTION = "array_collection"
    IMAGE = "image"
    JSON = "json"
    BYTES = "bytes"


@dataclass(frozen=True, slots=True)
class ExperimentArtifact[T]:
    """Artifact payload plus the relative path where it should be saved.

    Attributes:
        relative_path: Path below the experiment directory.
        payload: Artifact content.
        kind: Serialization strategy used by `ExperimentService`.
    """

    relative_path: Path
    payload: T
    kind: ExperimentArtifactKind

    @classmethod
    def text(cls, relative_path: str | Path, payload: str) -> "ExperimentArtifact[str]":
        """Create a UTF-8 text artifact."""
        return cls(Path(relative_path), payload, ExperimentArtifactKind.TEXT)

    @classmethod
    def array(cls, relative_path: str | Path, payload: np.ndarray) -> "ExperimentArtifact[np.ndarray]":
        """Create a NumPy `.npy` array artifact."""
        return cls(Path(relative_path), payload, ExperimentArtifactKind.ARRAY)

    @classmethod
    def array_collection(
            cls,
            relative_path: str | Path,
            payload: Sequence[np.ndarray] | Mapping[str, np.ndarray],
    ) -> "ExperimentArtifact[Sequence[np.ndarray] | Mapping[str, np.ndarray]]":
        """Create a NumPy `.npz` array-collection artifact."""
        return cls(Path(relative_path), payload, ExperimentArtifactKind.ARRAY_COLLECTION)

    @classmethod
    def image(cls, relative_path: str | Path, payload: Image) -> "ExperimentArtifact[Image]":
        """Create a PIL image artifact."""
        return cls(Path(relative_path), payload, ExperimentArtifactKind.IMAGE)

    @classmethod
    def json(cls, relative_path: str | Path, payload: Any) -> "ExperimentArtifact[Any]":
        """Create a JSON artifact."""
        return cls(Path(relative_path), payload, ExperimentArtifactKind.JSON)

    @classmethod
    def bytes(cls, relative_path: str | Path, payload: bytes) -> "ExperimentArtifact[bytes]":
        """Create a raw bytes artifact."""
        return cls(Path(relative_path), payload, ExperimentArtifactKind.BYTES)
