import logging
from typing import Any, override

import numpy as np

from modosaic.core.experiment_artifact import ExperimentArtifact
from modosaic.core.modalities import Modalities
from modosaic.core.postprocessor import ModalityPostprocessor
from modosaic.core.record import ImageRecord
from modosaic.depth.visualization import DepthVisualizationService

logger = logging.getLogger(__name__)


class DepthPostprocessor(ModalityPostprocessor[np.ndarray]):
    """Postprocessor that saves depth arrays and preview images."""

    @property
    @override
    def modality(self) -> Modalities:
        """Return the depth modality."""
        return Modalities.DEPTH

    @override
    def process(self, record: ImageRecord, generated: np.ndarray) -> list[ExperimentArtifact[Any]]:
        """Create depth `.npy` and PNG preview artifacts."""
        logger.debug(f"Post-processing depth for record sample {record.sample_id}")
        depth_image = DepthVisualizationService.depth_to_image(generated)

        return [
            ExperimentArtifact.array(self.sample_path(record, ".npy"), generated),
            ExperimentArtifact.image(self.sample_path(record, ".png"), depth_image),
        ]
