from dataclasses import dataclass


@dataclass
class MaskValidationStats:
    """Quality statistics for generated segmentation masks.

    Attributes:
        valid: Whether the statistic could be computed.
        coverage_score: Image-coverage score for the mask set.
        distinctness_score: Overlap and redundancy score for masks.
        fragmentation_score: Score penalizing excessive fragmentation.
    """

    valid: bool = False
    coverage_score: float = 0.0
    distinctness_score: float = 0.0
    fragmentation_score: float = 0.0
