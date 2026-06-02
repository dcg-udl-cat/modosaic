import logging
from typing import override

import numpy as np

from modosaic.core.record import ImageRecord
from modosaic.normals.normal_field_stats import NormalFieldStats
from modosaic.normals.validators.base.validator import NormalsValidator

logger = logging.getLogger(__name__)


class NormalsFieldQualityValidator(NormalsValidator[NormalFieldStats]):
    """Measure smoothness and integrability of a normal field."""

    eps: float
    nz_min: float

    def __init__(self, *, eps: float = 1e-6, nz_min: float = 0.1) -> None:
        """Initialize the validator.

        Args:
            eps: Numerical stability constant for vector normalization.
            nz_min: Minimum absolute z component used for integrability.
        """
        self.eps = eps
        self.nz_min = nz_min

    @override
    def validate(self, record: ImageRecord, generated: np.ndarray) -> NormalFieldStats:
        """Compute normal-field quality statistics."""
        logger.debug(f"Computing normals field quality for record sample {record.sample_id}")
        n = np.asarray(generated).astype(np.float32, copy=False)
        if n.ndim != 3 or n.shape[-1] != 3:
            logger.debug(f"Invalid normals shape for record sample {record.sample_id}: {n.shape}")
            return NormalFieldStats()

        n = np.nan_to_num(n, nan=0.0, posinf=0.0, neginf=0.0)
        n = NormalsFieldQualityValidator._normalize_vec(n, eps=self.eps)

        # --- Smoothness: neighbor angular variation ---
        a = n[:, :-1, :]
        b = n[:, 1:, :]
        ang_x = NormalsFieldQualityValidator._neighbor_angle_deg(a, b)

        a2 = n[:-1, :, :]
        b2 = n[1:, :, :]
        ang_y = NormalsFieldQualityValidator._neighbor_angle_deg(a2, b2)

        ang = np.concatenate([ang_x.reshape(-1), ang_y.reshape(-1)], axis=0)
        ang = ang[np.isfinite(ang)]
        if ang.size == 0:
            logger.debug(f"No finite normals neighbor angles for record sample {record.sample_id}")
            return NormalFieldStats()

        smooth_mean = float(np.mean(ang))
        smooth_median = float(np.median(ang))

        # --- Integrability (orthographic): p=-nx/nz, q=-ny/nz, curl = |dp/dy - dq/dx| ---
        nx, ny, nz = n[..., 0], n[..., 1], n[..., 2]
        valid = np.isfinite(nx) & np.isfinite(ny) & np.isfinite(nz) & (np.abs(nz) >= self.nz_min)

        denom = nz + np.sign(nz) * self.eps
        p = -nx / denom
        q = -ny / denom

        # dp/dy
        dpdy = NormalsFieldQualityValidator._finite_difference_gradient(p, axis=0)
        dqdx = NormalsFieldQualityValidator._finite_difference_gradient(q, axis=1)

        curl = np.abs(dpdy - dqdx)
        curl = curl[valid & np.isfinite(curl)]
        if curl.size == 0:
            logger.debug(f"Computed normals smoothness without integrability for record sample {record.sample_id}")
            return NormalFieldStats(
                valid=True,
                smoothness_mean_deg=smooth_mean,
                smoothness_median_deg=smooth_median,
            )

        logger.debug(f"Computed normals field quality for record sample {record.sample_id}")
        return NormalFieldStats(
            valid=True,
            smoothness_mean_deg=smooth_mean,
            smoothness_median_deg=smooth_median,
            integrability_mean=float(np.mean(curl)),
            integrability_median=float(np.median(curl)),
        )

    @staticmethod
    def _finite_difference_gradient(
            q: np.ndarray,
            axis: int,
            step: float = 1.0,
            dtype=np.float32,
    ) -> np.ndarray:
        """Compute a central finite-difference gradient along one axis."""
        slopefield = np.zeros_like(q, dtype=dtype)
        inv_step = 1.0 / step
        inv_2step = 0.5 * inv_step

        if axis == 0:
            if q.shape[0] < 2:
                return slopefield
            slopefield[1:-1, :] = (q[2:, :] - q[:-2, :]) * inv_2step
            slopefield[0, :] = (q[1, :] - q[0, :]) * inv_step
            slopefield[-1, :] = (q[-1, :] - q[-2, :]) * inv_step
            return slopefield

        if axis == 1:
            if q.shape[1] < 2:
                return slopefield
            slopefield[:, 1:-1] = (q[:, 2:] - q[:, :-2]) * inv_2step
            slopefield[:, 0] = (q[:, 1] - q[:, 0]) * inv_step
            slopefield[:, -1] = (q[:, -1] - q[:, -2]) * inv_step
            return slopefield

        raise ValueError(f"Unsupported axis: {axis}")

    @staticmethod
    def _normalize_vec(n: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        """Normalize vectors along the last axis."""
        denom = np.linalg.norm(n, axis=-1, keepdims=True)
        return n / (denom + eps)

    @staticmethod
    def _neighbor_angle_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Compute angles in degrees between neighboring normal vectors."""
        cos = np.sum(a * b, axis=-1)
        cos = np.clip(cos, -1.0, 1.0)
        return np.degrees(np.arccos(cos)).astype(np.float32)
