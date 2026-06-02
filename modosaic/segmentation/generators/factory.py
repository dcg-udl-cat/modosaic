import logging
from enum import Enum, auto

from modosaic.segmentation.generators.base.generator import SegmentationGenerator
from modosaic.segmentation.generators.impl.sam_2_hiera_small import SAM2HieraSmall
from modosaic.segmentation.generators.impl.sam_3 import SAM3
from modosaic.segmentation.generators.impl.sam_b import SAM

logger = logging.getLogger(__name__)


class SegmentationGenerationModel(Enum):
    """Segmentation generator implementations supported by the factory."""

    SAM = auto()
    SAM2 = auto()
    SAM3 = auto()


class SegmentationGeneratorFactory:
    """Factory for segmentation generator implementations."""

    @staticmethod
    def get(depth_model: SegmentationGenerationModel = SegmentationGenerationModel.SAM3) -> SegmentationGenerator:
        """Create a segmentation generator.

        Args:
            depth_model: Segmentation model implementation to instantiate.

        Returns:
            Configured segmentation generator.

        Raises:
            ValueError: If the model is unsupported.
        """
        logger.debug(f"Creating segmentation generator for model {depth_model}")
        match depth_model:
            case SegmentationGenerationModel.SAM:
                return SAM()
            case SegmentationGenerationModel.SAM2:
                return SAM2HieraSmall()
            case SegmentationGenerationModel.SAM3:
                return SAM3()
            case _:
                raise ValueError(f"Unsupported depth generation model: {depth_model}")
