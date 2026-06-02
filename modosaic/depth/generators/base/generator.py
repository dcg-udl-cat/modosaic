from abc import ABC, abstractmethod
from typing import override

import numpy as np
import torch

from modosaic.core.modality_generator import ModalityGenerator
from modosaic.core.record import ImageRecord
from modosaic.services.device import DeviceService


class DepthGenerator(ModalityGenerator[np.ndarray], ABC):
    """Base class for depth-map generators.

    Attributes:
        device: Torch device selected for inference.
        dtype: Torch dtype selected for inference.
    """

    device: torch.device
    dtype: torch.dtype

    def __init__(self) -> None:
        """Initialize device and dtype defaults."""
        self.device, self.dtype = DeviceService.get_device_type()

    @property
    def pipeline_dtype(self) -> torch.dtype:
        """Return a dtype compatible with Transformers depth pipeline postprocessing."""
        if self.dtype == torch.bfloat16:
            return torch.float32
        return self.dtype

    @override
    @abstractmethod
    def generate(self, record: ImageRecord) -> np.ndarray:
        """Generate a depth map for one image record."""
        raise NotImplementedError()
