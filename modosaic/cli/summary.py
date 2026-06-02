"""Formatting helpers for CLI run summaries."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


def format_summary(results: Sequence[Any], experiment_path: Path, *, as_json: bool = False) -> str:
    """Format pipeline results for CLI output.

    Args:
        results: Pipeline sample results.
        experiment_path: Directory where artifacts were written.
        as_json: Whether to return a JSON summary instead of text.

    Returns:
        Formatted summary string.
    """
    if as_json:
        return json.dumps(
            {
                "experiment_path": str(experiment_path),
                "processed_samples": len(results),
                "results": [_result_to_dict(result) for result in results],
            },
            default=_json_default,
            indent=2,
            sort_keys=True,
        )

    lines = [
        f"Processed {len(results)} sample(s).",
        f"Artifacts written to: {experiment_path}",
    ]
    for result in results:
        generated = ", ".join(str(modality) for modality in result.generated) or "none"
        lines.append(f"{result.record.sample_id}: generated {generated}")
        for modality, validations in result.validations.items():
            for validation in validations:
                lines.append(_format_validation(modality, validation))

    return "\n".join(lines)


def _format_validation(modality, validation) -> str:
    """Format one validation result for text output."""
    if validation.passed is None:
        return f"  {modality} {validation.validator_name}: {validation.value}"

    score = _format_optional_float(validation.score)
    minimum = _format_optional_float(validation.minimum)
    return (
        f"  {modality} {validation.validator_name}: "
        f"{validation.score_name}={score} >= {minimum}, passed={validation.passed}"
    )


def _result_to_dict(result) -> dict[str, Any]:
    """Convert one pipeline result to a JSON-compatible dictionary."""
    return {
        "sample_id": result.record.sample_id,
        "extension": result.record.extension,
        "metadata": result.record.metadata,
        "generated_modalities": [str(modality) for modality in result.generated],
        "validations": {
            str(modality): [asdict(validation) for validation in validations]
            for modality, validations in result.validations.items()
        },
        "artifact_paths": [str(path) for path in result.artifact_paths],
    }


def _format_optional_float(value: float | None) -> str:
    """Format an optional float with three decimal places."""
    if value is None:
        return "n/a"
    return f"{value:.3f}"


def _json_default(value: Any) -> Any:
    """Convert project objects to JSON-compatible values."""
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    return repr(value)
