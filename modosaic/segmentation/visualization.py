import random

import numpy as np
from PIL import Image


class SegmentationVisualizationService:
    """Visualization helpers for generated segmentation masks."""

    @staticmethod
    def overlay_masks(
            pil: Image.Image,
            masks: list[np.ndarray],
            alpha: float = 0.55,
            seed: int = 0,
    ) -> Image.Image:
        """Overlay segmentation masks on an image.

        Args:
            pil: Source image.
            masks: Boolean masks with shape `HxW`.
            alpha: Mask-color blend factor.
            seed: Seed used for deterministic mask colors.

        Returns:
            RGB image with colored masks overlaid.
        """
        rng = random.Random(seed)
        img = np.array(pil.convert("RGB"), dtype=np.uint8)
        H, W = img.shape[:2]

        if not masks:
            return pil

        # Sort masks by area DESC so big regions go first (usually nicer)
        masks_sorted = sorted(masks, key=lambda m: int(m.sum()), reverse=True)

        out = img.astype(np.float32)
        for mask in masks_sorted:
            m = np.asarray(mask).squeeze().astype(bool)
            if m.shape != (H, W):
                continue
            if m.sum() == 0:
                continue
            color = np.array(
                [rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255)],
                dtype=np.float32,
            )
            out[m] = (1 - alpha) * out[m] + alpha * color

        return Image.fromarray(out.clip(0, 255).astype(np.uint8))

    @staticmethod
    def instance_map_from_masks(masks: list[np.ndarray], H: int, W: int) -> np.ndarray:
        """Build an integer instance map from masks.

        Args:
            masks: Boolean masks with shape `HxW`.
            H: Output height.
            W: Output width.

        Returns:
            `uint16` instance map where `0` is background and positive values
            identify masks.
        """
        inst = np.zeros((H, W), dtype=np.uint16)
        masks_sorted = sorted(masks, key=lambda m: int(m.sum()), reverse=True)
        idx = 1
        for mask in masks_sorted:
            m = np.asarray(mask).squeeze().astype(bool)
            if m.shape != (H, W):
                continue
            if m.sum() == 0:
                continue
            inst[m] = idx
            idx += 1
            if idx >= np.iinfo(np.uint16).max:
                break
        return inst
