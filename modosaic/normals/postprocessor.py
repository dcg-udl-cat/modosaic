import logging
from typing import Any, override

import numpy as np

from modosaic.core.experiment_artifact import ExperimentArtifact
from modosaic.core.modalities import Modalities
from modosaic.core.postprocessor import ModalityPostprocessor
from modosaic.core.record import ImageRecord
from modosaic.normals.visualization import NormalsVisualizationService

logger = logging.getLogger(__name__)


class NormalsPostprocessor(ModalityPostprocessor[np.ndarray]):
    """Postprocessor that saves normal fields and RGB previews."""

    @property
    @override
    def modality(self) -> Modalities:
        """Return the normals modality."""
        return Modalities.NORMALS

    @override
    def process(self, record: ImageRecord, generated: np.ndarray) -> list[ExperimentArtifact[Any]]:
        """Create normal-field `.npy` and RGB preview artifacts."""
        logger.debug(f"Post-processing normals for record sample {record.sample_id}")
        normals_image = NormalsVisualizationService.normals_to_rgb(generated)

        return [
            ExperimentArtifact.array(self.sample_path(record, ".npy"), generated),
            ExperimentArtifact.image(self.sample_path(record, ".png"), normals_image),
        ]
