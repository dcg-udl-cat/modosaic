import cv2
import numpy as np


class BoundaryService:
    """Helpers for converting masks and dense maps into boundary maps."""

    @staticmethod
    def coerce_hw_float_with_mask(depth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Coerce a depth-like array to `HxW` float values and validity mask.

        Args:
            depth: Depth-like array with shape `HxW`, `HxWx1`, or `1xHxW`.

        Returns:
            Coerced float array and finite-value mask.

        Raises:
            ValueError: If the input cannot be interpreted as a 2D map.
        """
        d = np.asarray(depth)
        if d.ndim == 3 and d.shape[-1] == 1:
            d = d[..., 0]
        if d.ndim == 3 and d.shape[0] == 1:
            d = d[0]
        if d.ndim != 2:
            raise ValueError(f"Expected HxW depth; got {d.shape}")
        d = d.astype(np.float32, copy=False)
        valid_mask = np.isfinite(d)
        d = np.where(valid_mask, d, np.nan)
        return d, valid_mask

    @staticmethod
    def coerce_hw_float(depth: np.ndarray, convert_np: bool = True) -> np.ndarray:
        """Coerce a depth-like array to `HxW` float values.

        Args:
            depth: Depth-like array.
            convert_np: Whether to replace NaN and infinite values with zeros.

        Returns:
            Coerced `HxW` float array.
        """
        d, _ = BoundaryService.coerce_hw_float_with_mask(depth)
        if convert_np:
            d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
        return d

    @staticmethod
    def coerce_mask_binary(mask: np.ndarray, expected_shape: tuple[int, int] | None = None) -> np.ndarray | None:
        """Coerce a mask-like array to a boolean mask.

        Args:
            mask: Mask-like array.
            expected_shape: Optional expected `(height, width)` shape.

        Returns:
            Boolean mask, or `None` if the mask is empty or shape-incompatible.
        """
        m = np.asarray(mask).squeeze()
        if m.ndim != 2 or m.size == 0:
            return None
        if expected_shape is not None and m.shape != expected_shape:
            return None

        if np.issubdtype(m.dtype, np.bool_):
            return m.astype(bool, copy=False)

        m = np.nan_to_num(m.astype(np.float32, copy=False), nan=0.0, posinf=1.0, neginf=0.0)
        mn, mx = float(np.min(m)), float(np.max(m))

        if 0.0 <= mn and mx <= 1.0:
            return m > 0.5
        if 0.0 <= mn and mx <= 255.0:
            return m > 127.5
        return m > 0.0

    @staticmethod
    def masks_to_boundary(
            masks: list[np.ndarray],
            expected_shape: tuple[int, int] | None = None,
            thickness: int = 1,
    ) -> np.ndarray:
        """Convert instance masks to a combined boundary map.

        Args:
            masks: Instance masks.
            expected_shape: Optional expected output shape.
            thickness: Boundary thickness in pixels.

        Returns:
            Boolean boundary map.
        """
        valid: list[np.ndarray] = []
        for mask in masks:
            bm = BoundaryService.coerce_mask_binary(mask, expected_shape)
            if bm is None or not np.any(bm):
                continue
            valid.append(bm)

        if not valid:
            if expected_shape is None:
                return np.zeros((1, 1), dtype=bool)
            return np.zeros(expected_shape, dtype=bool)

        k = max(1, int(thickness))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
        boundary_maps: list[np.ndarray] = []
        for mask in valid:
            mask_u8 = mask.astype(np.uint8)
            eroded = cv2.erode(mask_u8, kernel, iterations=1)
            boundary_maps.append((mask_u8 ^ eroded).astype(bool))

        return np.any(np.stack(boundary_maps, axis=0), axis=0)

    @staticmethod
    def dilate_bool(x: np.ndarray, radius: int) -> np.ndarray:
        """Dilate a boolean map by a circular kernel.

        Args:
            x: Boolean-like input map.
            radius: Dilation radius in pixels.

        Returns:
            Dilated boolean map.
        """
        if radius <= 0:
            return x.astype(bool, copy=False)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
        y = cv2.dilate(x.astype(np.uint8), kernel, iterations=1)
        return y.astype(bool)
