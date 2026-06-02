from abc import ABC, abstractmethod
from typing import override

import torch

from modosaic.core.hf_model_spec import HFModelSpec
from modosaic.core.modality_generator import ModalityGenerator
from modosaic.core.record import ImageRecord
from modosaic.services.device import DeviceService


class TextGenerator(ModalityGenerator[str], ABC):
    """Base class for image-caption generators.

    Attributes:
        device: Torch device selected for inference.
        dtype: Torch dtype selected for inference.
        model_specs: Hugging Face model metadata.
    """

    device: torch.device
    dtype: torch.dtype
    model_specs: HFModelSpec

    def __init__(self):
        """Initialize device and dtype defaults."""
        self.device, self.dtype = DeviceService.get_device_type()

    @override
    @abstractmethod
    def generate(self, record: ImageRecord) -> str:
        """Generate text for one image record."""
        raise NotImplementedError()
