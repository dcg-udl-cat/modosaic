import logging
from typing import override

import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, PreTrainedModel, PreTrainedTokenizer, \
    BatchEncoding

from modosaic.core.hf_model_spec import HFModelSpec
from modosaic.core.record import ImageRecord
from modosaic.services.image import ImageService
from modosaic.text.constants import CAPTIONING_PROMPT, MAX_CAPTIONING_TOKENS
from modosaic.text.generators.base.generator import TextGenerator

logger = logging.getLogger(__name__)


class QWEN22B(TextGenerator):
    """Qwen2-VL 2B image-caption generator."""

    model: PreTrainedModel
    tokenizer: PreTrainedTokenizer

    def __init__(self) -> None:
        """Load the pinned Qwen2-VL model and processor."""
        super().__init__()
        self.model_specs = HFModelSpec(
            model_id="Qwen/Qwen2-VL-2B-Instruct",
            revision="895c3a49bc3fa70a340399125c650a463535e71c",
            dtype=self.dtype,
        )
        self.model = QWEN22B._load_model(self.model_specs)
        self.model.to(self.device).eval()
        self.processor = QWEN22B._load_tokenizer(self.model_specs)

    @staticmethod
    def _load_model(specs: HFModelSpec) -> PreTrainedModel:
        """Load the Qwen2-VL model from Hugging Face."""
        return Qwen2VLForConditionalGeneration.from_pretrained(
            pretrained_model_name_or_path=specs.model_id,
            revision=specs.revision,
            torch_dtype=specs.dtype,
            low_cpu_mem_usage=True,
        )

    @staticmethod
    def _load_tokenizer(specs: HFModelSpec) -> PreTrainedTokenizer:
        """Load the Qwen2-VL processor."""
        return AutoProcessor.from_pretrained(
            pretrained_model_name_or_path=specs.model_id,
            revision=specs.revision,
        )

    @override
    @torch.inference_mode
    def generate(self, record: ImageRecord) -> str:
        """Generate a caption for one image record."""
        logger.debug(f"Generating text for record sample {record.sample_id}")
        with ImageService.temporary_image_file(record.image_bytes) as tmp_image_path:
            messages = QWEN22B._get_messages(tmp_image_path.resolve().as_posix())
            inputs = self._get_inference_inputs(messages)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            outputs = self.model.generate(**inputs, max_new_tokens=MAX_CAPTIONING_TOKENS)
            gen = outputs[:, inputs["input_ids"].shape[1]:]
            caption = self.processor.batch_decode(gen, skip_special_tokens=True)[0].strip()
        logger.debug(f"Generated text for record sample {record.sample_id} with {len(caption)} characters")
        return caption

    # noinspection PyTypeChecker
    @staticmethod
    def _get_messages(img_path: str) -> list[dict[str, str]]:
        """Build chat messages for an image path."""
        return [{
            "role": "user",
            "content": [
                {"type": "image", "path": img_path},
                {"type": "text", "text": CAPTIONING_PROMPT},
            ],
        }]

    # noinspection PyTypeChecker
    def _get_inference_inputs(self, messages: list[dict[str, str]]) -> BatchEncoding:
        """Tokenize messages into model inputs."""
        return self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
