import numpy as np
from PIL import Image


class NormalsVisualizationService:
    """Convert normal fields into displayable RGB images."""

    @staticmethod
    def normalize_vec(n: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        """Normalize vectors along the last axis."""
        denom = np.linalg.norm(n, axis=-1, keepdims=True)
        return n / (denom + eps)

    @staticmethod
    def normals_to_rgb(normals: np.ndarray) -> Image.Image:
        """Convert normals in `[-1, 1]` into an RGB PIL image."""
        n = np.nan_to_num(normals, nan=0.0, posinf=0.0, neginf=0.0)
        n = np.clip(n, -1.0, 1.0)
        rgb = ((n + 1.0) * 0.5 * 255.0).astype(np.uint8)
        return Image.fromarray(rgb, mode="RGB")
