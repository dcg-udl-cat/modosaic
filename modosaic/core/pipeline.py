import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from modosaic.core.experiment_artifact import ExperimentArtifact
from modosaic.core.modalities import Modalities
from modosaic.core.modality import Modality
from modosaic.core.record import ImageRecord
from modosaic.core.validation_result import ValidationResult
from modosaic.services.experiment import ExperimentService
from modosaic.services.extension import ExtensionService

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class PipelineSampleResult:
    """Outputs collected after processing one image record.

    Attributes:
        record: Original image record processed by the pipeline.
        generated: Accepted generated outputs keyed by modality.
        validations: Validation results keyed by modality.
        artifact_paths: Files written for accepted modality outputs and
            validation reports.
    """

    record: ImageRecord
    generated: Mapping[Modalities, Any]
    validations: Mapping[Modalities, list[ValidationResult[Any]]]
    artifact_paths: list[Path]


class Pipeline:
    """Sequential multimodal generation and validation pipeline.

    The pipeline iterates over input image records, runs each configured
    modality in order, applies validators and optional constraints, and saves
    artifacts only for modalities whose constraints pass.

    Attributes:
        dataset: Iterable source of image records.
        modalities: Ordered modality definitions to run for each sample.
        experiment: Service used to persist accepted artifacts.
    """

    dataset: Iterable[ImageRecord]
    modalities: list[Modality[Any]]
    experiment: ExperimentService

    def __init__(
            self,
            dataset: Iterable[ImageRecord],
            modalities: Iterable[Modality[Any]],
            experiment: ExperimentService | None = None,
    ) -> None:
        """Initialize a pipeline.

        Args:
            dataset: Image records to process.
            modalities: Ordered modality definitions. Later modalities can read
                accepted outputs from earlier modalities through validator
                dependencies.
            experiment: Optional experiment service. A default service writing
                to `experiments/<timestamp>` is created when omitted.
        """
        self.dataset = dataset
        self.modalities = list(modalities)
        self.experiment = experiment or ExperimentService()

    def run(self, limit: int | None = None) -> list[PipelineSampleResult]:
        """Run the pipeline over the configured dataset.

        Args:
            limit: Optional maximum number of samples to process.

        Returns:
            Results for every processed sample.
        """
        results: list[PipelineSampleResult] = []

        for idx, record in enumerate(self.dataset):
            if limit is not None and idx >= limit:
                logger.info(f"Limit reached, stopping pipeline")
                break

            logger.debug(f"Starting pipeline for record sample {record.sample_id}")
            results.append(self.run_sample(record))

        return results

    def run_sample(self, record: ImageRecord) -> PipelineSampleResult:
        """Run all configured modalities for one sample.

        Args:
            record: Image record to process.

        Returns:
            Generated outputs, validation results, and saved artifact paths for
            the sample.
        """
        generated_modalities: dict[Modalities, Any] = {}
        validations: dict[Modalities, list[ValidationResult[Any]]] = {}
        artifact_paths: list[Path] = []

        for modality in self.modalities:
            logger.debug(f"Processing modality: {modality.modality} for record sample {record.sample_id}")

            generated = modality.generate(record)

            modality_validations = modality.validate(
                record=record,
                generated=generated,
                generated_modalities=generated_modalities,
            )
            validations[modality.modality] = modality_validations

            if self._has_failed_constraint(modality_validations):
                logger.info(
                    f"Discarding modality {modality.modality} for record sample {record.sample_id} "
                    f"after failed validation"
                )
                continue

            generated_modalities[modality.modality] = generated

            artifacts = modality.postprocess(record, generated)
            logger.debug(f"Generated {len(artifacts)} artifacts for modality {modality.modality} on record sample {record.sample_id}")
            artifact_paths.extend(self.experiment.save_artifacts(artifacts))
            logger.debug(
                f"Saved {len(artifacts)} artifacts for modality {modality.modality} "
                f"on record sample {record.sample_id}"
            )

            if modality_validations:
                validation_artifact = self._validation_artifact(
                    record=record,
                    modality=modality.modality,
                    validations=modality_validations,
                )
                artifact_paths.append(self.experiment.save_artifact(validation_artifact))

        if self.modalities and not generated_modalities:
            logger.info(f"Discarded record sample {record.sample_id}; no modalities passed validation")
        else:
            logger.info(
                f"Completed record sample {record.sample_id} with {len(generated_modalities)} modalities "
                f"and {len(artifact_paths)} artifacts"
            )

        return PipelineSampleResult(
            record=record,
            generated=generated_modalities,
            validations=validations,
            artifact_paths=artifact_paths,
        )

    @staticmethod
    def _validation_artifact(
            record: ImageRecord,
            modality: Modalities,
            validations: list[ValidationResult[Any]],
    ) -> ExperimentArtifact[Any]:
        """Build the validation JSON artifact for a modality result.

        Args:
            record: Image record being processed.
            modality: Modality whose validations were produced.
            validations: Validation results to serialize.

        Returns:
            JSON experiment artifact containing validation results.
        """
        sample_id = ExtensionService.sanitize_fragment(record.sample_id)
        relative_path = Path("validations") / str(modality) / f"{sample_id}.json"
        return ExperimentArtifact.json(relative_path, validations)

    @staticmethod
    def _has_failed_constraint(validations: list[ValidationResult[Any]]) -> bool:
        """Return whether any validation constraint rejected the modality."""
        return any(validation.passed is False for validation in validations)
