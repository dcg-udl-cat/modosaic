import math
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True, slots=True)
class ValidationConstraint[U]:
    """Quality gate applied to a validator value.

    Attributes:
        minimum: Inclusive minimum score needed for a pass.
        score_fn: Optional converter from a structured validator value to a
            numeric score.
        score_name: Human-readable score name stored in validation output.
    """

    minimum: float
    score_fn: Callable[[U], float] | None = None
    score_name: str = "score"

    def evaluate(self, value: U) -> tuple[float, bool]:
        """Evaluate a validator value against the configured minimum.

        Args:
            value: Raw validator output.

        Returns:
            A tuple containing the numeric score and whether the score passed.

        Raises:
            TypeError: If no `score_fn` is provided and `value` is not numeric.
        """
        raw_score = self.score_fn(value) if self.score_fn else value

        try:
            score = float(raw_score)
            minimum = float(self.minimum)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "ValidationConstraint without score_fn requires a numeric validator value."
            ) from exc

        passed = math.isfinite(score) and math.isfinite(minimum) and score >= minimum
        return score, passed
