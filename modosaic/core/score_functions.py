from modosaic.core.boundary_alignment_stats import BoundaryAlignmentStats


def boundary_f1_score(stats: BoundaryAlignmentStats) -> float | int:
    """Extract a constraint score from boundary-alignment statistics.

    Args:
        stats: Boundary statistics returned by a validator.

    Returns:
        The F1 score when statistics are valid, otherwise `0.0`.
    """
    if not stats.valid:
        return 0.0

    return stats.f1
