import cv2
import numpy as np
from PIL import Image


class DepthVisualizationService:
    """Convert depth arrays into image visualizations."""

    @staticmethod
    def normalize_to_uint8(
            depth: np.ndarray,
            eps: float = 1e-8,
    ) -> tuple[np.ndarray, float, float]:
        """Normalize a depth map to an 8-bit grayscale array.

        Args:
            depth: Depth map.
            eps: Minimum denominator used to avoid division by zero.

        Returns:
            Normalized uint8 array, original minimum, and original maximum.
        """
        depth = depth.astype(np.float32)
        d_min = float(np.nanmin(depth))
        d_max = float(np.nanmax(depth))
        denom = (d_max - d_min) if (d_max - d_min) > eps else eps
        depth_uint8 = ((depth - d_min) / denom * 255.0).clip(0, 255).astype(np.uint8)
        return depth_uint8, d_min, d_max

    @staticmethod
    def depth_to_image(
            depth: np.ndarray,
            size: tuple[int, int] | None = None,
    ) -> Image.Image:
        """Convert a depth map to a grayscale PIL image.

        Args:
            depth: Depth map.
            size: Optional output `(width, height)`.

        Returns:
            PIL image visualization.
        """
        depth_uint8, _, _ = DepthVisualizationService.normalize_to_uint8(depth)
        depth_image = Image.fromarray(depth_uint8)
        if size is not None:
            depth_image = depth_image.resize(size, Image.Resampling.BICUBIC)
        return depth_image

    @staticmethod
    def depth_to_heatmap(
            depth: np.ndarray,
            size: tuple[int, int] | None = None,
            colormap: int = cv2.COLORMAP_TURBO,
    ) -> Image.Image:
        """Convert a depth map to a color heatmap.

        Args:
            depth: Depth map.
            size: Optional output `(width, height)`.
            colormap: OpenCV color-map identifier.

        Returns:
            RGB PIL heatmap image.
        """
        depth_uint8, _, _ = DepthVisualizationService.normalize_to_uint8(depth)
        if size is not None:
            depth_uint8 = cv2.resize(depth_uint8, size, interpolation=cv2.INTER_CUBIC)

        heatmap_bgr = cv2.applyColorMap(depth_uint8, colormap)
        heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(heatmap_rgb)

    @staticmethod
    def overlay_depth(
            image: Image.Image,
            depth: np.ndarray,
            alpha: float = 0.55,
    ) -> Image.Image:
        """Blend a depth preview over an RGB image.

        Args:
            image: Source image.
            depth: Depth map.
            alpha: Blend factor for the depth image.

        Returns:
            Blended RGB image.
        """
        base = image.convert("RGB")
        depth_image = DepthVisualizationService.depth_to_image(
            depth,
            size=base.size,
        ).convert("RGB")
        return Image.blend(base, depth_image, alpha=float(np.clip(alpha, 0.0, 1.0)))
