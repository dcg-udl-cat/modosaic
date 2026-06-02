import logging
from typing import override, Any

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from modosaic.core.record import ImageRecord
from modosaic.normals.generators.base.generator import NormalsGenerator
from modosaic.services.image import ImageService

logger = logging.getLogger(__name__)


class Omnidata(NormalsGenerator):
    """Omnidata surface-normal generator."""

    model: Any
    transform: T.Compose

    def __init__(self):
        """Load the Omnidata model and image transform."""
        super().__init__()
        self.model = self._load_model()
        self.transform = Omnidata._get_model_transform()

    def _load_model(self):
        """Load the Omnidata surface-normal model through Torch Hub."""
        return torch.hub.load("alexsax/omnidata_models", "surface_normal_dpt_hybrid_384").to(self.device).eval()

    @staticmethod
    def _get_model_transform(image_size: int = 384) -> T.Compose:
        """Create the image transform used by the Omnidata model."""
        return T.Compose(
            [
                T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR),
                T.CenterCrop(image_size),
                T.ToTensor(),
            ]
        )

    @override
    def generate(self, record: ImageRecord) -> np.ndarray:
        """Generate a normal field for one image record."""
        logger.debug(f"Generating normals for record sample {record.sample_id}")
        pil_img = ImageService.bytes_to_pil(record.image_bytes)
        x = self.transform(pil_img.convert("RGB")).unsqueeze(0).to(self.device)
        y = self.model(x)  # (1,3,H,W)
        y = y[0].detach().float().cpu().permute(1, 2, 0).numpy()
        y = Omnidata._normalize_vec(y).astype(np.float32)

        # Resize back to original size for visual comparison
        width, height = pil_img.size
        y_vis = Omnidata._normals_to_rgb(y).resize((width, height), resample=Image.BILINEAR)
        y2 = (np.array(y_vis).astype(np.float32) / 255.0) * 2.0 - 1.0
        normals = Omnidata._normalize_vec(y2).astype(np.float32)
        return normals

    @staticmethod
    def _normalize_vec(n: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        """Normalize vectors along the last axis."""
        denom = np.linalg.norm(n, axis=-1, keepdims=True)
        return n / (denom + eps)

    @staticmethod
    def _normals_to_rgb(normals: np.ndarray) -> Image.Image:
        """Convert normals to RGB for resizing and visualization."""
        n = np.nan_to_num(normals, nan=0.0, posinf=0.0, neginf=0.0)
        n = np.clip(n, -1.0, 1.0)
        rgb = ((n + 1.0) * 0.5 * 255.0).astype(np.uint8)
        return Image.fromarray(rgb, mode="RGB")
