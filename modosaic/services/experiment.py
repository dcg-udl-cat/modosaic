import json
import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL.Image import Image

from modosaic.core.experiment_artifact import ExperimentArtifact, ExperimentArtifactKind

logger = logging.getLogger(__name__)


class ExperimentService:
    """Persist generated artifacts below one experiment directory.

    Attributes:
        experiment_path: Directory where this run writes artifacts.
    """

    experiment_path: Path

    def __init__(
            self,
            root: str | Path = "experiments",
            experiment_name: str | None = None,
    ):
        """Initialize an experiment folder.

        Args:
            root: Parent directory for experiments.
            experiment_name: Optional run directory name. A timestamp is used
                when omitted.
        """
        current_date = experiment_name or datetime.now().strftime("%Y%m%d-%H%M%S")
        self.experiment_path = Path(root) / current_date
        self.experiment_path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Experiment artifacts will be saved under {self.experiment_path}")

    def save_artifacts(self, artifacts: Iterable[ExperimentArtifact[Any]]) -> list[Path]:
        """Save many artifacts.

        Args:
            artifacts: Artifact objects to persist.

        Returns:
            Paths written to disk.
        """
        return [
            self.save_artifact(artifact)
            for artifact in artifacts
        ]

    def save_artifact(self, artifact: ExperimentArtifact[Any]) -> Path:
        """Save one artifact using its declared kind.

        Args:
            artifact: Artifact to persist.

        Returns:
            Path written to disk.

        Raises:
            ValueError: If the artifact kind is unsupported.
        """
        logger.debug(f"Saving {artifact.kind} artifact to {artifact.relative_path}")
        match artifact.kind:
            case ExperimentArtifactKind.TEXT:
                return self.save_text_artifact(artifact.relative_path, artifact.payload)
            case ExperimentArtifactKind.ARRAY:
                return self.save_array_artifact(artifact.relative_path, artifact.payload)
            case ExperimentArtifactKind.ARRAY_COLLECTION:
                return self.save_array_collection_artifact(artifact.relative_path, artifact.payload)
            case ExperimentArtifactKind.IMAGE:
                return self.save_image_artifact(artifact.relative_path, artifact.payload)
            case ExperimentArtifactKind.JSON:
                return self.save_json_artifact(artifact.relative_path, artifact.payload)
            case ExperimentArtifactKind.BYTES:
                return self.save_bytes_artifact(artifact.relative_path, artifact.payload)
            case _:
                raise ValueError(f"Unsupported experiment artifact kind: {artifact.kind}")

    def save_text_artifact(self, relative_path: str | Path, text: str) -> Path:
        """Save UTF-8 text below the experiment directory."""
        artifact_path = self._artifact_path(relative_path)
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.write_text(text, encoding="utf-8")
        logger.debug(f"Saved text artifact to {artifact_path}")
        return artifact_path

    def save_array_artifact(self, relative_path: str | Path, array: np.ndarray) -> Path:
        """Save a NumPy array with `numpy.save`."""
        artifact_path = self._artifact_path(relative_path)
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        with artifact_path.open("wb") as file:
            np.save(file, array)
        logger.debug(f"Saved array artifact to {artifact_path}")
        return artifact_path

    def save_array_collection_artifact(
            self,
            relative_path: str | Path,
            arrays: Sequence[np.ndarray] | Mapping[str, np.ndarray],
    ) -> Path:
        """Save an array collection with `numpy.savez`."""
        artifact_path = self._artifact_path(relative_path)
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        with artifact_path.open("wb") as file:
            if isinstance(arrays, Mapping):
                np.savez(file, **arrays)
            else:
                np.savez(file, *arrays)
        logger.debug(f"Saved array collection artifact to {artifact_path}")
        return artifact_path

    def save_image_artifact(self, relative_path: str | Path, image: Image) -> Path:
        """Save a PIL image below the experiment directory."""
        artifact_path = self._artifact_path(relative_path)
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(artifact_path)
        logger.debug(f"Saved image artifact to {artifact_path}")
        return artifact_path

    def save_json_artifact(self, relative_path: str | Path, payload: Any) -> Path:
        """Save a JSON-serializable payload below the experiment directory."""
        artifact_path = self._artifact_path(relative_path)
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.write_text(
            json.dumps(payload, default=self._json_default, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        logger.debug(f"Saved JSON artifact to {artifact_path}")
        return artifact_path

    def save_bytes_artifact(self, relative_path: str | Path, payload: bytes) -> Path:
        """Save raw bytes below the experiment directory."""
        artifact_path = self._artifact_path(relative_path)
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.write_bytes(payload)
        logger.debug(f"Saved bytes artifact to {artifact_path}")
        return artifact_path

    def save_text(self, text_caption: str) -> Path:
        """Save a legacy caption artifact under `text/result.txt`."""
        return self.save_text_artifact("text/result.txt", text_caption)

    def save_depth(self, depth: np.ndarray, depth_img: Image) -> list[Path]:
        """Save legacy depth array and preview image artifacts."""
        depth_path = Path("depth")

        return [
            self.save_array_artifact(depth_path / "depth.npy", depth),
            self.save_image_artifact(depth_path / "depth.png", depth_img),
        ]

    def save_normals(self, normals: np.ndarray, normal_img: Image) -> list[Path]:
        """Save legacy normals array and preview image artifacts."""
        normals_path = Path("normals")

        return [
            self.save_array_artifact(normals_path / "normals.npy", normals),
            self.save_image_artifact(normals_path / "normals.png", normal_img),
        ]

    def save_segmentation_masks(
            self,
            segmentation_masks: list[np.ndarray],
            segmentation_masks_img: Image,
    ) -> list[Path]:
        """Save legacy segmentation mask arrays and preview image artifacts."""
        segmentation_masks_path = Path("segmentation_masks")

        return [
            self.save_array_collection_artifact(
                segmentation_masks_path / "segmentation_masks.npz",
                segmentation_masks,
            ),
            self.save_image_artifact(
                segmentation_masks_path / "segmentation_masks.png",
                segmentation_masks_img,
            ),
        ]

    def _artifact_path(self, relative_path: str | Path) -> Path:
        """Resolve and validate an artifact path.

        Args:
            relative_path: Relative path below the experiment directory.

        Returns:
            Absolute path for the artifact.

        Raises:
            ValueError: If the path is absolute, empty, or escapes the
                experiment directory.
        """
        path = Path(relative_path)
        if path.is_absolute() or ".." in path.parts or path == Path("."):
            raise ValueError(
                f"Artifact path must stay inside the experiment folder: {relative_path}"
            )
        return self.experiment_path / path

    @staticmethod
    def _json_default(value: Any) -> Any:
        """Convert supported project objects for JSON serialization."""
        if is_dataclass(value):
            return asdict(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, Path):
            return str(value)

        raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")
