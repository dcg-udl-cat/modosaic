from abc import ABC, abstractmethod
from typing import override

import numpy as np
import torch

from modosaic.core.modality_generator import ModalityGenerator
from modosaic.core.record import ImageRecord
from modosaic.services.device import DeviceService


class SegmentationGenerator(ModalityGenerator[list[np.ndarray]], ABC):
    """Base class for segmentation-mask generators.

    Attributes:
        device: Torch device selected for inference.
        dtype: Torch dtype selected for inference.
    """

    device: torch.device
    dtype: torch.dtype

    def __init__(self):
        """Initialize device and dtype defaults."""
        self.device, self.dtype = DeviceService.get_device_type()

    @override
    @abstractmethod
    def generate(self, record: ImageRecord) -> list[np.ndarray]:
        """Generate instance masks for one image record."""
        raise NotImplementedError()
