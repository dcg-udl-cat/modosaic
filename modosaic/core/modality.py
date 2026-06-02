from abc import ABC, abstractmethod
from typing import Any, Iterable, Mapping

from modosaic.core.experiment_artifact import ExperimentArtifact
from modosaic.core.modalities import Modalities
from modosaic.core.modality_generator import ModalityGenerator
from modosaic.core.postprocessor import ModalityPostprocessor
from modosaic.core.record import ImageRecord
from modosaic.core.validation_result import ValidationResult
from modosaic.core.validator import ModalityValidator
from modosaic.core.validator_step import ValidatorStep


class Modality[T](ABC):
    """Base class for an executable modality.

    A modality bundles the three pieces needed by the pipeline: a generator, a
    postprocessor, and optional validators. Subclasses provide the concrete
    `Modalities` value.

    Attributes:
        model: Generator that produces the modality output.
        validators: Validation steps executed after generation.
        postprocessor: Converts accepted generated output into artifacts.
    """

    model: ModalityGenerator[T]
    validators: list[ValidatorStep[T, Any]]
    postprocessor: ModalityPostprocessor[T]

    def __init__(
            self,
            model: ModalityGenerator[T],
            postprocessor: ModalityPostprocessor[T],
            validators: Iterable[
                            ModalityValidator[T, Any] | ValidatorStep[T, Any]
                            ] | None = None,
    ) -> None:
        """Initialize a modality from generator, postprocessor, and validators.

        Args:
            model: Generator used to produce the modality output.
            postprocessor: Artifact builder for accepted outputs.
            validators: Optional validator objects or fully configured
                `ValidatorStep` instances.
        """
        self.model = model
        self.postprocessor = postprocessor
        self.validators = [
            self._build_validator_step(validator)
            for validator in validators or []
        ]

    @property
    @abstractmethod
    def modality(self) -> Modalities:
        """Return the modality type represented by this object."""
        raise NotImplementedError()

    def generate(self, record: ImageRecord) -> T:
        """Generate this modality for a record.

        Args:
            record: Input image record.

        Returns:
            Generated modality output.
        """
        return self.model.generate(record)

    def validate(
            self,
            record: ImageRecord,
            generated: T,
            generated_modalities: Mapping[Modalities, Any] | None = None,
    ) -> list[ValidationResult[Any]]:
        """Run validators for a generated modality output.

        Args:
            record: Input image record.
            generated: Output produced by `generate`.
            generated_modalities: Accepted outputs from earlier modalities,
                used to satisfy validator dependencies.

        Returns:
            Validation results in configured validator order.
        """
        context = generated_modalities or {}
        return [
            validator.validate(record, generated, context)
            for validator in self.validators
        ]

    def postprocess(self, record: ImageRecord, generated: T) -> list[ExperimentArtifact[Any]]:
        """Create experiment artifacts for an accepted modality output.

        Args:
            record: Input image record.
            generated: Accepted generated output.

        Returns:
            Artifacts ready to be saved by `ExperimentService`.
        """
        return self.postprocessor.process(record, generated)

    @staticmethod
    def _build_validator_step(
            validator: ModalityValidator[T, Any] | ValidatorStep[T, Any],
    ) -> ValidatorStep[T, Any]:
        """Normalize a validator object into a `ValidatorStep`.

        Args:
            validator: Plain validator or already configured validation step.

        Returns:
            A validation step.
        """
        if isinstance(validator, ValidatorStep):
            return validator

        return ValidatorStep(validator)
