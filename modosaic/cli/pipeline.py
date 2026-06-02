"""Runtime builders used by the Modosaic CLI."""

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from modosaic.cli.config import (
    DatasetConfig,
    DatasetKind,
    ModalityName,
    RunConfig,
)

if TYPE_CHECKING:
    from modosaic.core.modality import Modality
    from modosaic.providers.image_dataset import ImageDataset


@dataclass(frozen=True, slots=True)
class PipelineRun:
    """Result of one CLI-triggered pipeline execution.

    Attributes:
        results: Per-sample pipeline results.
        experiment_path: Directory where artifacts were written.
    """

    results: list[Any]
    experiment_path: Path


def execute_run(config: RunConfig) -> PipelineRun:
    """Execute a complete run from a normalized config.

    Args:
        config: Normalized run configuration.

    Returns:
        Pipeline results and experiment path.

    Raises:
        FileNotFoundError: If the configured dataset path is missing.
        ValueError: If the configuration is invalid.
    """
    validate_run_config(config)

    from modosaic.core.pipeline import Pipeline
    from modosaic.services.experiment import ExperimentService
    from modosaic.services.logging import LoggingService
    from modosaic.services.seeding import SeedingService

    LoggingService.setup_logging(config.log_path)
    SeedingService.set_global_seed(config.seed)

    experiment = ExperimentService(
        root=config.experiment_root,
        experiment_name=config.experiment_name,
    )
    pipeline = Pipeline(
        dataset=build_dataset(config.dataset),
        modalities=build_modalities(config),
        experiment=experiment,
    )
    return PipelineRun(
        results=pipeline.run(limit=config.limit),
        experiment_path=experiment.experiment_path,
    )


def validate_run_config(config: RunConfig) -> None:
    """Validate dataset paths and numeric runtime options.

    Args:
        config: Run configuration to validate.

    Raises:
        FileNotFoundError: If a required dataset path is missing.
        ValueError: If a required option is absent or invalid.
    """
    dataset = config.dataset
    if dataset.kind == DatasetKind.LOCAL:
        if dataset.root is None:
            raise ValueError("--root is required when --dataset local.")
        if not dataset.root.exists():
            raise FileNotFoundError(f"Local dataset folder not found: {dataset.root}")
        if not dataset.root.is_dir():
            raise ValueError(f"Local dataset root must be a directory: {dataset.root}")

    if dataset.kind == DatasetKind.PARQUET:
        if dataset.parquet_path is None:
            raise ValueError("--parquet-path is required when --dataset parquet.")
        if not dataset.parquet_path.exists():
            raise FileNotFoundError(f"Parquet path not found: {dataset.parquet_path}")

    if config.limit is not None and config.limit < 0:
        raise ValueError("--limit must be greater than or equal to 0.")
    if dataset.batch_size <= 0:
        raise ValueError("--batch-size must be greater than 0.")

    validator = config.validators
    non_negative_options = {
        "segmentation_boundary_thickness": validator.segmentation_boundary_thickness,
        "segmentation_tolerance_radius": validator.segmentation_tolerance_radius,
        "depth_boundary_thickness": validator.depth_boundary_thickness,
        "depth_tolerance_radius": validator.depth_tolerance_radius,
    }
    for name, value in non_negative_options.items():
        if value < 0:
            raise ValueError(f"{name} must be greater than or equal to 0.")


def build_dataset(config: DatasetConfig) -> "ImageDataset":
    """Build an `ImageDataset` from dataset configuration.

    Args:
        config: Dataset configuration.

    Returns:
        Configured image dataset.

    Raises:
        ValueError: If the dataset kind is unsupported or required paths are
            absent.
    """
    from modosaic.providers.image_dataset import ImageDataset

    if config.kind == DatasetKind.LOCAL:
        return ImageDataset.from_local_folder(
            root=_required_path(config.root, "root"),
            recursive=config.recursive,
            extensions=config.extensions or None,
        )

    if config.kind == DatasetKind.PARQUET:
        return ImageDataset.from_parquet(
            parquet_path=_required_path(config.parquet_path, "parquet_path"),
            image_column=config.image_column,
            id_column=config.id_column,
            extension_column=config.extension_column,
            metadata_columns=config.metadata_columns or None,
            batch_size=config.batch_size,
        )

    raise ValueError(f"Unsupported dataset kind: {config.kind}")


def build_modalities(config: RunConfig) -> list["Modality[Any]"]:
    """Build configured modalities for a run.

    Args:
        config: Run configuration.

    Returns:
        List of configured modalities in execution order.
    """
    builders = {
        ModalityName.IMAGE: _build_image_modality,
        ModalityName.TEXT: _build_text_modality,
        ModalityName.SEGMENTATION: _build_segmentation_modality,
        ModalityName.DEPTH: _build_depth_modality,
        ModalityName.NORMALS: _build_normals_modality,
    }
    selected = set(config.modalities)
    return [builders[name](config, selected) for name in config.modalities]


def _build_image_modality(config: RunConfig, selected: set[ModalityName]):
    del config, selected
    from modosaic.image.preconfigured_modality import build_preconfigured_image_modality

    return build_preconfigured_image_modality()


def _build_text_modality(config: RunConfig, selected: set[ModalityName]):
    del selected
    from modosaic.core.configured_modality import ConfiguredModality
    from modosaic.core.modalities import Modalities
    from modosaic.core.validator_step import ValidatorStep
    from modosaic.text.generators.factory import TextGenerationModel, TextGeneratorFactory
    from modosaic.text.postprocessor import TextPostprocessor

    validator_steps = []
    if config.validators.enabled:
        from modosaic.text.validators.impl.siglip_2 import SIGLIP2Validator

        validator_steps.append(
            ValidatorStep(
                validator=SIGLIP2Validator(),
                constraint=_constraint(
                    config,
                    minimum=config.validators.constraints.text_siglip_minimum,
                    score_name="siglip_probability",
                ),
            )
        )

    return ConfiguredModality(
        modality=Modalities.TEXT,
        model=TextGeneratorFactory.get(TextGenerationModel[config.models.text.name]),
        postprocessor=TextPostprocessor(),
        validators=validator_steps,
    )


def _build_segmentation_modality(config: RunConfig, selected: set[ModalityName]):
    del selected
    from modosaic.core.configured_modality import ConfiguredModality
    from modosaic.core.modalities import Modalities
    from modosaic.core.score_functions import boundary_f1_score
    from modosaic.core.validator_step import ValidatorStep
    from modosaic.segmentation.generators.factory import (
        SegmentationGenerationModel,
        SegmentationGeneratorFactory,
    )
    from modosaic.segmentation.postprocessor import SegmentationPostprocessor
    from modosaic.segmentation.validators.impl.boundary_rgb_edge_overlap import (
        SegRgbEdgeOverlapValidator,
    )
    from modosaic.segmentation.validators.impl.mask_statistics import MaskStatsValidator

    validator_steps = []
    if config.validators.enabled:
        validator_steps.extend(
            [
                ValidatorStep(
                    validator=MaskStatsValidator(),
                    constraint=_constraint(
                        config,
                        minimum=config.validators.constraints.segmentation_mask_quality_minimum,
                        score_name="weighted_mask_quality",
                        score_fn=_mask_quality_score,
                    ),
                ),
                ValidatorStep(
                    validator=SegRgbEdgeOverlapValidator(
                        boundary_thickness=config.validators.segmentation_boundary_thickness,
                        tolerance_radius=config.validators.segmentation_tolerance_radius,
                        rgb_edge_quantile=config.validators.segmentation_rgb_edge_quantile,
                    ),
                    constraint=_constraint(
                        config,
                        minimum=config.validators.constraints.segmentation_boundary_minimum,
                        score_name="rgb_boundary_f1",
                        score_fn=boundary_f1_score,
                    ),
                ),
            ]
        )

    return ConfiguredModality(
        modality=Modalities.SEGMENTATION,
        model=SegmentationGeneratorFactory.get(
            SegmentationGenerationModel[config.models.segmentation.name]
        ),
        postprocessor=SegmentationPostprocessor(),
        validators=validator_steps,
    )


def _build_depth_modality(config: RunConfig, selected: set[ModalityName]):
    from modosaic.core.configured_modality import ConfiguredModality
    from modosaic.core.modalities import Modalities
    from modosaic.core.score_functions import boundary_f1_score
    from modosaic.core.validator_step import ValidatorStep
    from modosaic.depth.generators.factory import DepthGenerationModel, DepthGeneratorFactory
    from modosaic.depth.postprocessor import DepthPostprocessor

    generation_model = DepthGenerationModel[config.models.depth.name]
    validator_steps = []
    if config.validators.enabled:
        from modosaic.depth.validators.impl.imagebind import ImageBindValidator

        validator_steps.append(
            ValidatorStep(
                validator=ImageBindValidator(generation_model),
                constraint=_constraint(
                    config,
                    minimum=config.validators.constraints.depth_imagebind_minimum,
                    score_name="imagebind_similarity",
                ),
            )
        )

        if ModalityName.SEGMENTATION in selected:
            from modosaic.depth.validators.impl.depth_seg_boundary_consistency_validator import (
                DepthSegBoundaryConsistencyValidator,
            )

            validator_steps.append(
                ValidatorStep(
                    validator=DepthSegBoundaryConsistencyValidator(
                        boundary_thickness=config.validators.depth_boundary_thickness,
                        tolerance_radius=config.validators.depth_tolerance_radius,
                        depth_edge_quantile=config.validators.depth_edge_quantile,
                    ),
                    dependencies={"masks": Modalities.SEGMENTATION},
                    constraint=_constraint(
                        config,
                        minimum=config.validators.constraints.depth_segmentation_boundary_minimum,
                        score_name="segmentation_boundary_f1",
                        score_fn=boundary_f1_score,
                    ),
                )
            )

    return ConfiguredModality(
        modality=Modalities.DEPTH,
        model=DepthGeneratorFactory.get(generation_model),
        postprocessor=DepthPostprocessor(),
        validators=validator_steps,
    )


def _build_normals_modality(config: RunConfig, selected: set[ModalityName]):
    from modosaic.core.configured_modality import ConfiguredModality
    from modosaic.core.modalities import Modalities
    from modosaic.core.validator_step import ValidatorStep
    from modosaic.normals.generators.factory import NormalsGenerationModel, NormalsGeneratorFactory
    from modosaic.normals.postprocessor import NormalsPostprocessor
    from modosaic.segmentation.validators.impl.normals_field_quality import (
        NormalsFieldQualityValidator,
    )

    validator_steps = []
    if config.validators.enabled:
        if ModalityName.DEPTH in selected:
            from modosaic.normals.validators.impl.depth_normals_agreement import (
                DepthNormalsAngularAgreementValidator,
            )

            validator_steps.append(
                ValidatorStep(
                    validator=DepthNormalsAngularAgreementValidator(),
                    dependencies={"depth_map": Modalities.DEPTH},
                    constraint=_constraint(
                        config,
                        minimum=config.validators.constraints.normals_depth_agreement_minimum,
                        score_name="depth_normals_agreement",
                        score_fn=_depth_normals_agreement_score,
                    ),
                )
            )

        validator_steps.append(
            ValidatorStep(
                validator=NormalsFieldQualityValidator(
                    eps=config.validators.normals_eps,
                    nz_min=config.validators.normals_nz_min,
                ),
                constraint=_constraint(
                    config,
                    minimum=config.validators.constraints.normals_field_quality_minimum,
                    score_name="normals_field_quality",
                    score_fn=_normals_field_quality_score,
                ),
            )
        )

    return ConfiguredModality(
        modality=Modalities.NORMALS,
        model=NormalsGeneratorFactory.get(NormalsGenerationModel[config.models.normals.name]),
        postprocessor=NormalsPostprocessor(),
        validators=validator_steps,
    )


def _constraint(
        config: RunConfig,
        *,
        minimum: float,
        score_name: str,
        score_fn=None,
):
    """Create a validation constraint when constraints are enabled."""
    if not config.validators.constraints.enabled:
        return None

    from modosaic.core.validation_constraint import ValidationConstraint

    return ValidationConstraint(
        minimum=minimum,
        score_name=score_name,
        score_fn=score_fn,
    )


def _mask_quality_score(stats) -> float:
    """Compute the weighted segmentation-mask quality score."""
    if not stats.valid:
        return 0.0

    return (
            0.4 * stats.coverage_score
            + 0.3 * stats.distinctness_score
            + 0.3 * stats.fragmentation_score
    )


def _depth_normals_agreement_score(stats) -> float:
    """Compute the weighted depth-normal agreement score."""
    if not stats.valid:
        return 0.0

    return (
            0.2 * stats.pct_under_11_25
            + 0.5 * stats.pct_under_22_5
            + 0.3 * stats.pct_under_30
    )


def _normals_field_quality_score(stats) -> float:
    """Compute the weighted normal-field quality score."""
    if not stats.valid:
        return 0.0

    smoothness_score = max(0.0, min(1.0, 1.0 - stats.smoothness_median_deg / 45.0))
    integrability_score = 1.0 / (1.0 + max(0.0, stats.integrability_median))
    return 0.7 * smoothness_score + 0.3 * integrability_score


def _required_path(path: Path | None, name: str) -> Path:
    """Return a required path or raise a user-facing `ValueError`."""
    if path is None:
        raise ValueError(f"{name} is required.")
    return path
