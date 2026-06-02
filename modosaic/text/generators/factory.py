import logging
from enum import Enum, auto

from modosaic.text.generators.base.generator import TextGenerator
from modosaic.text.generators.impl.internvl_3_2b import InternVL32B
from modosaic.text.generators.impl.internvl_3_5_2b import InternVL352B
from modosaic.text.generators.impl.qwen_2_2b import QWEN22B
from modosaic.text.generators.impl.qwen_2_5_3b import QWEN253B

logger = logging.getLogger(__name__)


class TextGenerationModel(Enum):
    """Text generator implementations supported by the factory."""

    INTERN_VL_3_5_2B = auto()
    INTERN_VL_3_2B = auto()
    QWEN_2_2B = auto()
    QWEN_2_5_3B = auto()


class TextGeneratorFactory:
    """Factory for text generator implementations."""

    @staticmethod
    def get(text_model: TextGenerationModel = TextGenerationModel.INTERN_VL_3_5_2B) -> TextGenerator:
        """Create a text generator.

        Args:
            text_model: Model implementation to instantiate.

        Returns:
            Configured text generator.

        Raises:
            ValueError: If the model is unsupported.
        """
        logger.debug(f"Creating text generator for model {text_model}")
        match text_model:
            case TextGenerationModel.INTERN_VL_3_5_2B:
                return InternVL352B()
            case TextGenerationModel.INTERN_VL_3_2B:
                return InternVL32B()
            case TextGenerationModel.QWEN_2_5_3B:
                return QWEN253B()
            case TextGenerationModel.QWEN_2_2B:
                return QWEN22B()
            case _:
                raise ValueError(f"Unsupported text generation model: {text_model}")
