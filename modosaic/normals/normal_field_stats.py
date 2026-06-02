from dataclasses import dataclass


@dataclass
class NormalFieldStats:
    """Quality statistics for a generated normal field.

    Attributes:
        valid: Whether the statistic could be computed.
        smoothness_mean_deg: Mean angular change between neighboring normals.
        smoothness_median_deg: Median angular change between neighboring
            normals.
        integrability_mean: Mean integrability residual.
        integrability_median: Median integrability residual.
    """

    valid: bool = False
    smoothness_mean_deg: float = 180.0
    smoothness_median_deg: float = 180.0
    integrability_mean: float = 1e9
    integrability_median: float = 1e9
