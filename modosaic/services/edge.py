import cv2
import numpy as np


class EdgeService:
    """Edge-map utilities used by cross-modality validators."""

    @staticmethod
    def _adaptive_threshold(mag: np.ndarray, q: float, valid_mask: np.ndarray | None = None) -> float:
        """Compute a quantile threshold over finite magnitudes."""
        finite = np.isfinite(mag)
        if valid_mask is not None:
            finite &= valid_mask.astype(bool, copy=False)
        m = mag[finite]
        if m.size == 0:
            return float("inf")
        return float(np.quantile(m, q))

    @staticmethod
    def sobel_edge_map_float(
            img: np.ndarray,
            blur_ksize: int = 3,
            q: float = 0.90,
            valid_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute a boolean Sobel edge map from a scalar image.

        Args:
            img: Scalar image or dense map.
            blur_ksize: Gaussian blur kernel size before Sobel filtering.
            q: Quantile threshold applied to gradient magnitudes.
            valid_mask: Optional mask limiting where edges can be detected.

        Returns:
            Boolean edge map.
        """
        x = np.asarray(img).astype(np.float32, copy=False)
        finite = np.isfinite(x)
        if valid_mask is not None:
            finite &= valid_mask.astype(bool, copy=False)
        x = np.where(finite, x, 0.0)

        if blur_ksize and blur_ksize >= 3:
            x = cv2.GaussianBlur(x, (blur_ksize, blur_ksize), 0)

        gx = cv2.Sobel(x, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(x, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(gx * gx + gy * gy)

        grad_valid = finite
        if np.any(finite):
            margin = 1 + max(0, blur_ksize // 2)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * margin + 1, 2 * margin + 1))
            grad_valid = cv2.erode(finite.astype(np.uint8), kernel, iterations=1).astype(bool)

        thr = EdgeService._adaptive_threshold(mag, q, grad_valid)
        return ((mag >= thr) & grad_valid).astype(bool)

    @staticmethod
    def rgb_edge_map(
            rgb_uint8: np.ndarray,
            q: float = 0.90,
    ) -> np.ndarray:
        """Compute a boolean edge map from an RGB uint8 image.

        Args:
            rgb_uint8: RGB image with shape `HxWx3`.
            q: Quantile threshold applied to gradient magnitudes.

        Returns:
            Boolean edge map.
        """
        gray = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2GRAY).astype(np.float32)
        return EdgeService.sobel_edge_map_float(gray, blur_ksize=3, q=q)
