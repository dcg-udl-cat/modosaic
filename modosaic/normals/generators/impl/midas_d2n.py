import logging
from typing import override, Any

import numpy as np
import torch

from modosaic.core.record import ImageRecord
from modosaic.normals.generators.base.generator import NormalsGenerator
from modosaic.services.image import ImageService

logger = logging.getLogger(__name__)


class MidasD2N(NormalsGenerator):
    """MiDaS-based depth-to-normal generator."""

    model: Any
    transform: Any

    def __init__(self):
        """Load the MiDaS model and transform."""
        super().__init__()
        self.model = self._load_model()
        self.transform = MidasD2N._get_model_transform()

    def _load_model(self):
        """Load the MiDaS small model through Torch Hub."""
        return torch.hub.load("intel-isl/MiDaS", "MiDaS_small").to(self.device).eval()

    @staticmethod
    def _get_model_transform():
        """Load the MiDaS small input transform."""
        transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        return transforms.small_transform

    @override
    def generate(self, record: ImageRecord) -> np.ndarray:
        """Generate a normal field for one image record."""
        logger.debug(f"Generating normals for record sample {record.sample_id}")
        pil_img = ImageService.bytes_to_pil(record.image_bytes)
        img = np.array(pil_img.convert("RGB"))
        inp = self.transform(img).to(self.device)

        pred = self.model(inp)
        pred = torch.nn.functional.interpolate(
            pred.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze(1)

        depth = pred[0].detach().float().cpu().numpy()
        lo, hi = np.percentile(depth, 2), np.percentile(depth, 98)
        d = (depth - lo) / (hi - lo + 1e-6)
        d = np.clip(d, 0.0, 1.0)

        normals = self._depth_to_normals(d)
        return normals

    @staticmethod
    def _depth_to_normals(depth: np.ndarray) -> np.ndarray:
        """Estimate normals from a predicted depth map."""
        dzdx = np.gradient(depth, axis=1)
        dzdy = np.gradient(depth, axis=0)
        n = np.stack((-dzdx, -dzdy, np.ones_like(depth)), axis=-1)
        return MidasD2N._normalize_vec(n).astype(np.float32)

    @staticmethod
    def _normalize_vec(n: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        """Normalize vectors along the last axis."""
        denom = np.linalg.norm(n, axis=-1, keepdims=True)
        return n / (denom + eps)
