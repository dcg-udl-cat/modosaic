from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ValidationResult[T]:
    """Serialized result of one validator execution.

    Attributes:
        validator_name: Name shown in summaries and validation artifacts.
        value: Raw validator output.
        score: Optional numeric score extracted from `value`.
        score_name: Optional name associated with `score`.
        minimum: Optional inclusive threshold used for the score.
        passed: Optional pass/fail result. Unconstrained validators leave this
            field as `None`.
    """

    validator_name: str
    value: T
    score: float | None = None
    score_name: str | None = None
    minimum: float | None = None
    passed: bool | None = None
