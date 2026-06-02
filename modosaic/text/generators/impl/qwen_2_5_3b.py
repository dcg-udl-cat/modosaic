import logging
from pathlib import Path
from typing import override

import torch
from PIL.Image import Image
from qwen_vl_utils import process_vision_info
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, PreTrainedModel, PreTrainedTokenizer, \
    BatchEncoding

from modosaic.core.hf_model_spec import HFModelSpec
from modosaic.core.record import ImageRecord
from modosaic.services.image import ImageService
from modosaic.text.constants import CAPTIONING_PROMPT, MAX_CAPTIONING_TOKENS
from modosaic.text.generators.base.generator import TextGenerator

logger = logging.getLogger(__name__)


class QWEN253B(TextGenerator):
    """Qwen2.5-VL 3B image-caption generator."""

    def __init__(self) -> None:
        """Load the pinned Qwen2.5-VL model and processor."""
        super().__init__()
        self.model_specs = HFModelSpec(
            model_id="Qwen/Qwen2.5-VL-3B-Instruct",
            revision="66285546d2b821cf421d4f5eb2576359d3770cd3",
            dtype=self.dtype,
        )
        self.model = QWEN253B._load_model(self.model_specs)
        self.model.to(self.device).eval()
        self.processor = QWEN253B._load_tokenizer(self.model_specs)

    @staticmethod
    def _load_model(specs: HFModelSpec) -> PreTrainedModel:
        """Load the Qwen2.5-VL model from Hugging Face."""
        return Qwen2_5_VLForConditionalGeneration.from_pretrained(
            pretrained_model_name_or_path=specs.model_id,
            revision=specs.revision,
            torch_dtype=specs.dtype,
            low_cpu_mem_usage=True,
        )

    @staticmethod
    def _load_tokenizer(specs: HFModelSpec) -> PreTrainedTokenizer:
        """Load the Qwen2.5-VL processor."""
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
            file_uri = QWEN253B._get_file_uri(tmp_image_path)
            messages = QWEN253B._get_messages(file_uri)
            inference_text = self._get_inference_text(messages)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self._get_inference_inputs(inference_text, image_inputs, video_inputs)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            outputs = self.model.generate(**inputs, max_new_tokens=MAX_CAPTIONING_TOKENS)
            gen = outputs[:, inputs["input_ids"].shape[1]:]
            caption = self.processor.batch_decode(
                gen,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()
        logger.debug(f"Generated text for record sample {record.sample_id} with {len(caption)} characters")
        return caption

    @staticmethod
    def _get_file_uri(image_path: Path) -> str:
        """Return a file URI for an existing image path."""
        return f"file://{str(image_path.resolve())}"

    # noinspection PyTypeChecker
    @staticmethod
    def _get_messages(file_uri: str) -> list[dict[str, str]]:
        """Build chat messages for an image file URI."""
        return [{
            "role": "user",
            "content": [
                {"type": "image", "image": file_uri},
                {"type": "text", "text": CAPTIONING_PROMPT},
            ],
        }]

    # noinspection PyTypeChecker
    def _get_inference_text(self, messages: list[dict[str, str]]) -> str:
        """Render chat messages to Qwen inference text."""
        return self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def _get_inference_inputs(self, inference_text: str, image_inputs: list[Image], video_inputs: list[torch.Tensor]) -> \
            BatchEncoding:
        """Tokenize text and vision inputs for the Qwen processor."""
        return self.processor(
            text=[inference_text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
