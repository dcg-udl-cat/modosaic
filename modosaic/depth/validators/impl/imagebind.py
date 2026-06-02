import warnings

with warnings.catch_warnings(record=False):
    warnings.filterwarnings(
        "ignore",
        message=r"pkg_resources is deprecated as an API.*",
        category=UserWarning,
        module=r"imagebind\.data",
    )
    from imagebind import ModalityType
    from imagebind.data import load_and_transform_vision_data
    from imagebind.models.imagebind_model import ImageBindModel, imagebind_huge
   
import logging
from typing import override, Any

import numpy as np
import torch
import torch.nn.functional as F

from modosaic.core.record import ImageRecord
from modosaic.depth.generators.factory import DepthGenerationModel
from modosaic.depth.validators.base.validator import DepthValidator
from modosaic.services.boundary import BoundaryService
from modosaic.services.device import DeviceService
from modosaic.services.image import ImageService

logger = logging.getLogger(__name__)


class ImageBindValidator(DepthValidator[float]):
    """Validate depth-image consistency with ImageBind embeddings."""

    generation_model: DepthGenerationModel
    device: torch.device
    dtype: torch.dtype
    model: ImageBindModel

    def __init__(self, generation_model: DepthGenerationModel) -> None:
        """Initialize the validator.

        Args:
            generation_model: Depth model used to generate the depth map. The
                value informs metric-vs-relative depth normalization.
        """
        self.generation_model = generation_model
        self.device, self.dtype = DeviceService.get_device_type()
        self.model = self._load_model()

    def _load_model(self):
        """Load the pretrained ImageBind model."""
        return imagebind_huge(pretrained=True).to(self.device).eval()

    @override
    def validate(self, record: ImageRecord, generated: np.ndarray) -> float:
        """Score agreement between the source image and generated depth map."""
        logger.debug(f"Validating depth-image embedding for record sample {record.sample_id}")
        inputs = self._build_inputs(record, generated)
        with torch.no_grad():
            embeddings = self.model(inputs)
            vision_embeddings = embeddings[ModalityType.VISION]
            depth_embeddings = embeddings[ModalityType.DEPTH]
            cos_sim = F.cosine_similarity(vision_embeddings, depth_embeddings).item()
            percent_like = max(0.0, min(1.0, (cos_sim + 1) / 2))
            logger.debug(f"Percent like {percent_like}")
            return percent_like

    def _build_inputs(self, record: ImageRecord, generated: np.ndarray) -> dict[str, Any]:
        """Build ImageBind modality inputs for one record."""
        with ImageService.temporary_image_file(record.image_bytes, record.extension) as image_path:
            vision_data = load_and_transform_vision_data(
                image_paths=[image_path],
                device=self.device,
            )

        return {
            ModalityType.VISION: vision_data,
            ModalityType.DEPTH: self.depth_np_to_imagebind_depth(
                depth=generated,
                assume_metric=self._get_assume_metric_from_model(self.generation_model),
            ),
        }

    @staticmethod
    def _resize_center_crop_224(x: torch.Tensor) -> torch.Tensor:
        """Resize and center crop a tensor to ImageBind's `224x224` input size."""
        _, _, height, width = x.shape
        if height == 224 and width == 224:
            return x

        # Scale so that min(H,W) -> 224
        scale = 224.0 / min(height, width)
        new_height, new_width = int(round(height * scale)), int(round(width * scale))
        x = F.interpolate(x, size=(new_height, new_width), mode="bilinear", align_corners=False)

        # Center crop
        top = (new_height - 224) // 2
        left = (new_width - 224) // 2
        return x[:, :, top:top + 224, left:left + 224]

    def depth_np_to_imagebind_depth(
            self,
            depth: np.ndarray,
            assume_metric: bool | None = None,
            prefer_disparity: bool = True,
            use_log_for_metric: bool = True,
            p_lo: float = 1.0,
            p_hi: float = 99.0,
            eps: float = 1e-6,
    ) -> torch.Tensor:
        """Convert a depth map to ImageBind depth input format.

        Args:
            depth: Generated depth map.
            assume_metric: Whether to treat values as metric depth. If omitted,
                a heuristic is used.
            prefer_disparity: Whether metric depth should be converted to
                inverse-depth style values.
            use_log_for_metric: Whether metric depth should be log-compressed
                when `prefer_disparity` is false.
            p_lo: Lower percentile for robust normalization.
            p_hi: Upper percentile for robust normalization.
            eps: Numerical stability constant.

        Returns:
            Torch tensor ready for ImageBind depth input.

        Raises:
            ValueError: If the depth map has no finite values.
        """
        d = BoundaryService.coerce_hw_float(depth, convert_np=False)
        mask = np.isfinite(d)
        if not mask.any():
            raise ValueError("Depth has no finite values.")

        # Infer metric-like scale when caller-provided model metadata is unavailable.
        if assume_metric is None:
            # Metric depth maps are usually positive and bounded to a physical
            # range, while relative maps may use arbitrary or normalized units.
            d99 = np.nanpercentile(d, 99)
            d01 = np.nanpercentile(d, 1)
            # Treat the map as metric-like when the robust range is positive and bounded.
            assume_metric = (d01 >= 0.0 and 0.05 < d99 < 500.0)

        # --- canonicalize representation ---
        dc = d.copy()

        if assume_metric:
            # Metric depth should be positive
            dc = np.where(mask, np.maximum(dc, eps), np.nan)

            if prefer_disparity:
                # disparity-like (inverse depth)
                dc = 1.0 / (dc + eps)
            else:
                # keep depth, maybe log-compress
                if use_log_for_metric:
                    dc = np.log(dc + eps)
        else:
            # Relative depth may encode either distance or inverse-distance
            # ordering. Preserve the original orientation and robust-normalize.
            pass

        # --- robust normalize to roughly standard scale ---
        lo = np.nanpercentile(dc, p_lo)
        hi = np.nanpercentile(dc, p_hi)
        if not np.isfinite(lo) or not np.isfinite(hi) or abs(hi - lo) < 1e-9:
            # fallback: simple mean/std over finite pixels
            mu = np.nanmean(dc)
            sigma = np.nanstd(dc) + eps
            dc = (dc - mu) / sigma
        else:
            dc = np.clip(dc, lo, hi)
            # map to [-1, 1] then (optionally) z-score-like scale
            dc = 2.0 * (dc - lo) / (hi - lo + eps) - 1.0

        # fill nans with 0 (center)
        dc = np.where(np.isfinite(dc), dc, 0.0).astype(np.float32)

        # to torch: 1x1xHxW -> 1x1x224x224
        x = torch.from_numpy(dc).unsqueeze(0).unsqueeze(0)  # 1,1,H,W
        x = ImageBindValidator._resize_center_crop_224(x)
        return x.to(self.device)

    @staticmethod
    def _get_assume_metric_from_model(model: DepthGenerationModel) -> bool:
        """Return whether a depth generator is expected to produce metric depth."""
        match model:
            case DepthGenerationModel.DEPTH_ANYTHING_V2_METRIC_SMALL:
                return True
            case DepthGenerationModel.DEPTH_PRO | \
                 DepthGenerationModel.DPT_HYBRID_MIDAS | \
                 DepthGenerationModel.MARIGOLD_DEPTH_V1_1 | \
                 DepthGenerationModel.DEPTH_ANYTHING_SMALL | \
                 DepthGenerationModel.DEPTH_ANYTHING_V2_SMALL:
                return False
