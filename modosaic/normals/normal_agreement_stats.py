from dataclasses import dataclass


@dataclass
class NormalAgreementStats:
    """Angular agreement between generated normals and depth-derived normals.

    Attributes:
        valid: Whether the statistic could be computed.
        mean_angle_deg: Mean angular difference in degrees.
        median_angle_deg: Median angular difference in degrees.
        pct_under_11_25: Fraction of pixels below 11.25 degrees.
        pct_under_22_5: Fraction of pixels below 22.5 degrees.
        pct_under_30: Fraction of pixels below 30 degrees.
    """

    valid: bool = False
    mean_angle_deg: float = 180.0
    median_angle_deg: float = 180.0
    pct_under_11_25: float = 0.0
    pct_under_22_5: float = 0.0
    pct_under_30: float = 0.0
