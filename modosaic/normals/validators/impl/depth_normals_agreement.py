import logging
from typing import override

import numpy as np

from modosaic.core.record import ImageRecord
from modosaic.normals.normal_agreement_stats import NormalAgreementStats
from modosaic.normals.validators.base.validator import NormalsValidator
from modosaic.services.boundary import BoundaryService

logger = logging.getLogger(__name__)


def _normalize_vec(n: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Normalize vectors along the last axis."""
    denom = np.linalg.norm(n, axis=-1, keepdims=True)
    return n / (denom + eps)


def normals_from_depth_ortho(depth: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Estimate orthographic normals from a depth map.

    Args:
        depth: Depth map.
        eps: Numerical stability constant for vector normalization.

    Returns:
        Normal field with shape `HxWx3`.
    """
    d, finite = BoundaryService.coerce_hw_float_with_mask(depth)
    d_filled = np.where(finite, d, 0.0)
    # finite differences (central-ish)
    dzdx = np.zeros_like(d_filled, dtype=np.float32)
    dzdy = np.zeros_like(d_filled, dtype=np.float32)

    dzdx[:, 1:-1] = 0.5 * (d_filled[:, 2:] - d_filled[:, :-2])
    dzdx[:, 0] = d_filled[:, 1] - d_filled[:, 0]
    dzdx[:, -1] = d_filled[:, -1] - d_filled[:, -2]

    dzdy[1:-1, :] = 0.5 * (d_filled[2:, :] - d_filled[:-2, :])
    dzdy[0, :] = d_filled[1, :] - d_filled[0, :]
    dzdy[-1, :] = d_filled[-1, :] - d_filled[-2, :]

    valid_dx = np.zeros_like(finite, dtype=bool)
    valid_dy = np.zeros_like(finite, dtype=bool)

    valid_dx[:, 1:-1] = finite[:, 2:] & finite[:, :-2]
    valid_dx[:, 0] = finite[:, 0] & finite[:, 1]
    valid_dx[:, -1] = finite[:, -1] & finite[:, -2]

    valid_dy[1:-1, :] = finite[2:, :] & finite[:-2, :]
    valid_dy[0, :] = finite[0, :] & finite[1, :]
    valid_dy[-1, :] = finite[-1, :] & finite[-2, :]

    normal_valid = finite & valid_dx & valid_dy

    n = np.stack([-dzdx, -dzdy, np.ones_like(d_filled, dtype=np.float32)], axis=-1)
    n[~normal_valid] = np.nan
    return _normalize_vec(n, eps=eps).astype(np.float32)


class DepthNormalsAngularAgreementValidator(NormalsValidator[NormalAgreementStats]):
    """Compare generated normals with normals estimated from a depth map."""

    @override
    def validate(self, record: ImageRecord, generated: np.ndarray,
                 depth_map: np.ndarray | None = None) -> NormalAgreementStats:
        """Compute angular agreement statistics.

        Args:
            record: Input image record.
            generated: Generated normal field.
            depth_map: Optional generated depth map dependency.

        Returns:
            Angular agreement statistics.
        """
        logger.debug(f"Validating depth-normal angular agreement for record sample {record.sample_id}")
        if depth_map is None:
            logger.debug(f"No depth map available for record sample {record.sample_id}")
            return NormalAgreementStats()

        n_pred = np.asarray(generated).astype(np.float32, copy=False)
        if n_pred.ndim != 3 or n_pred.shape[-1] != 3:
            logger.debug(f"Invalid normals shape for record sample {record.sample_id}: {n_pred.shape}")
            return NormalAgreementStats()

        n_depth = normals_from_depth_ortho(depth_map)
        if n_depth.shape[:2] != n_pred.shape[:2]:
            logger.debug(
                f"Depth-normal shape mismatch for record sample {record.sample_id}: "
                f"depth={n_depth.shape[:2]}, normals={n_pred.shape[:2]}"
            )
            return NormalAgreementStats()

        n_pred = _normalize_vec(np.nan_to_num(n_pred, nan=0.0, posinf=0.0, neginf=0.0))

        cos = np.sum(n_depth * n_pred, axis=-1)
        cos = np.clip(cos, -1.0, 1.0)
        ang = np.degrees(np.arccos(cos)).astype(np.float32)

        finite = np.isfinite(ang)
        if not np.any(finite):
            logger.debug(f"No finite angular agreement values for record sample {record.sample_id}")
            return NormalAgreementStats()

        a = ang[finite]
        mean = float(np.mean(a))
        median = float(np.median(a))

        logger.debug(f"Computed depth-normal angular agreement for record sample {record.sample_id}")
        return NormalAgreementStats(
            valid=True,
            mean_angle_deg=mean,
            median_angle_deg=median,
            pct_under_11_25=float(np.mean(a < 11.25)),
            pct_under_22_5=float(np.mean(a < 22.5)),
            pct_under_30=float(np.mean(a < 30.0)),
        )
