import logging
from enum import Enum, auto

from modosaic.depth.generators.base.generator import DepthGenerator
from modosaic.depth.generators.impl.depth_anything_small import DepthAnythingSmall
from modosaic.depth.generators.impl.depth_anything_v2_metric_small import DepthAnythingV2MetricSmall
from modosaic.depth.generators.impl.depth_anything_v2_small import DepthAnythingV2Small
from modosaic.depth.generators.impl.depth_pro import DepthPro
from modosaic.depth.generators.impl.dpt_hybrid_midas import DptHybridMidas
from modosaic.depth.generators.impl.marigold_depth_v1_1 import MarigoldDepthV11

logger = logging.getLogger(__name__)


class DepthGenerationModel(Enum):
    """Depth generator implementations supported by the factory."""

    DEPTH_ANYTHING_SMALL = auto()
    DEPTH_ANYTHING_V2_METRIC_SMALL = auto()
    DEPTH_ANYTHING_V2_SMALL = auto()
    DEPTH_PRO = auto()
    DPT_HYBRID_MIDAS = auto()
    MARIGOLD_DEPTH_V1_1 = auto()


class DepthGeneratorFactory:
    """Factory for depth generator implementations."""

    @staticmethod
    def get(depth_model: DepthGenerationModel = DepthGenerationModel.DEPTH_ANYTHING_V2_SMALL) -> DepthGenerator:
        """Create a depth generator.

        Args:
            depth_model: Model implementation to instantiate.

        Returns:
            Configured depth generator.

        Raises:
            ValueError: If the model is unsupported.
        """
        logger.debug(f"Creating depth generator for model {depth_model}")
        match depth_model:
            case DepthGenerationModel.DEPTH_ANYTHING_SMALL:
                return DepthAnythingSmall()
            case DepthGenerationModel.DEPTH_ANYTHING_V2_METRIC_SMALL:
                return DepthAnythingV2MetricSmall()
            case DepthGenerationModel.DEPTH_ANYTHING_V2_SMALL:
                return DepthAnythingV2Small()
            case DepthGenerationModel.DEPTH_PRO:
                return DepthPro()
            case DepthGenerationModel.DPT_HYBRID_MIDAS:
                return DptHybridMidas()
            case DepthGenerationModel.MARIGOLD_DEPTH_V1_1:
                return MarigoldDepthV11()
            case _:
                raise ValueError(f"Unsupported depth generation model: {depth_model}")
