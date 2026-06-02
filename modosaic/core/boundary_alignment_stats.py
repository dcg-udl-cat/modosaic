from dataclasses import dataclass


@dataclass
class BoundaryAlignmentStats:
    """Boundary overlap statistics returned by edge-alignment validators.

    Attributes:
        valid: Whether the statistic could be computed.
        precision: Fraction of predicted boundary pixels matching a target
            boundary.
        recall: Fraction of target boundary pixels matched by predictions.
        f1: Harmonic mean of precision and recall.
        iou: Intersection-over-union between boundary regions.
    """

    valid: bool = False
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    iou: float = 0.0
