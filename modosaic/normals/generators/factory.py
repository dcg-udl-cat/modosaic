import logging
from enum import Enum, auto

from modosaic.normals.generators.base.generator import NormalsGenerator
from modosaic.normals.generators.impl.midas_d2n import MidasD2N
from modosaic.normals.generators.impl.omnidata import Omnidata

logger = logging.getLogger(__name__)


class NormalsGenerationModel(Enum):
    """Normals generator implementations supported by the factory."""

    OMNIDATA = auto()
    MIDAS_D2N = auto()


class NormalsGeneratorFactory:
    """Factory for surface-normal generator implementations."""

    @staticmethod
    def get(normals_model: NormalsGenerationModel = NormalsGenerationModel.OMNIDATA) -> NormalsGenerator:
        """Create a normals generator.

        Args:
            normals_model: Model implementation to instantiate.

        Returns:
            Configured normals generator.

        Raises:
            ValueError: If the model is unsupported.
        """
        logger.debug(f"Creating normals generator for model {normals_model}")
        match normals_model:
            case NormalsGenerationModel.OMNIDATA:
                return Omnidata()
            case NormalsGenerationModel.MIDAS_D2N:
                return MidasD2N()
            case _:
                raise ValueError(f"Unsupported normals generation model: {normals_model}")
