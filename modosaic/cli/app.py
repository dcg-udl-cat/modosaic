"""Typer command definitions for the Modosaic CLI."""

from collections.abc import Sequence
from dataclasses import replace
from enum import Enum
from pathlib import Path
from typing import Annotated

from typer import Argument, Exit, Option, Typer, echo

from modosaic.cli.config import (
    DEFAULT_LOG_PATH,
    DatasetConfig,
    DatasetKind,
    DepthModelName,
    ModalityName,
    ModelConfig,
    NormalsModelName,
    RunConfig,
    SegmentationModelName,
    TextModelName,
    ConstraintConfig,
    ValidatorConfig,
    load_config_mapping,
    normalize_modalities,
    run_config_from_mapping,
)
from modosaic.cli.pipeline import execute_run
from modosaic.cli.summary import format_summary

app = Typer(no_args_is_help=True, name="modosaic")


@app.command(
    help="Run the default Modosaic pipeline on a local image folder.",
    short_help="Run defaults on local images.",
)
def simple(
        root: Annotated[
            Path,
            Argument(
                exists=True,
                file_okay=False,
                help="Path to a local folder containing source images.",
            ),
        ],
        limit: Annotated[
            int | None,
            Option("--limit", "-n", help="Maximum number of samples to process."),
        ] = None,
        experiment_root: Annotated[
            Path,
            Option(
                "--experiment-root",
                file_okay=False,
                help="Directory where experiment artifacts will be written.",
            ),
        ] = Path("experiments"),
        experiment_name: Annotated[
            str | None,
            Option("--experiment-name", help="Optional experiment folder name."),
        ] = None,
        log_path: Annotated[
            Path | None,
            Option(
                "--log-path",
                file_okay=False,
                help="Directory where logs will be stored.",
            ),
        ] = DEFAULT_LOG_PATH,
        seed: Annotated[
            int,
            Option("--seed", help="Global seed for reproducible model execution."),
        ] = 42,
        json_summary: Annotated[
            bool,
            Option("--json", help="Print the run summary as JSON."),
        ] = False,
) -> None:
    """Run the default pipeline on a local image folder."""
    config = RunConfig(
        dataset=DatasetConfig(kind=DatasetKind.LOCAL, root=root),
        limit=limit,
        experiment_root=experiment_root,
        experiment_name=experiment_name,
        log_path=log_path,
        seed=seed,
        json_summary=json_summary,
    )
    _run_or_exit(config)


@app.command(
    help=(
            "Run the Modosaic pipeline with configurable dataset, modalities, "
            "models, validators, and quality-gate constraints."
    ),
    short_help="Run with CLI configuration.",
)
def run(
        dataset: Annotated[
            DatasetKind,
            Option("--dataset", help="Dataset adapter to use."),
        ] = DatasetKind.LOCAL,
        root: Annotated[
            Path | None,
            Option(
                "--root",
                exists=True,
                file_okay=False,
                help="Local image folder. Required when --dataset local.",
            ),
        ] = None,
        parquet_path: Annotated[
            Path | None,
            Option(
                "--parquet-path",
                exists=True,
                help="Parquet file or directory. Required when --dataset parquet.",
            ),
        ] = None,
        recursive: Annotated[
            bool,
            Option(
                "--recursive/--top-level",
                help="Scan local folders recursively or only at the top level.",
            ),
        ] = True,
        extensions: Annotated[
            list[str] | None,
            Option(
                "--extension",
                "-e",
                help="Local image extension to include. Repeat to include many.",
            ),
        ] = None,
        image_column: Annotated[
            str,
            Option("--image-column", help="Parquet column containing image bytes or paths."),
        ] = "image",
        id_column: Annotated[
            str | None,
            Option("--id-column", help="Optional Parquet sample-id column."),
        ] = None,
        extension_column: Annotated[
            str | None,
            Option("--extension-column", help="Optional Parquet image-extension column."),
        ] = None,
        metadata_columns: Annotated[
            list[str] | None,
            Option(
                "--metadata-column",
                help="Parquet metadata column to keep. Repeat to include many.",
            ),
        ] = None,
        batch_size: Annotated[
            int,
            Option("--batch-size", help="Parquet read batch size."),
        ] = 512,
        modalities: Annotated[
            list[ModalityName] | None,
            Option(
                "--modality",
                "-m",
                help=(
                        "Modality to generate. Repeat to select many; omitted means all "
                        "modalities in dependency-safe order."
                ),
            ),
        ] = None,
        text_model: Annotated[
            TextModelName,
            Option("--text-model", help="Captioning model for the text modality."),
        ] = TextModelName.QWEN_2_2B,
        segmentation_model: Annotated[
            SegmentationModelName,
            Option("--segmentation-model", help="Mask generator for segmentation."),
        ] = SegmentationModelName.SAM3,
        depth_model: Annotated[
            DepthModelName,
            Option("--depth-model", help="Depth generator for depth maps."),
        ] = DepthModelName.DEPTH_ANYTHING_V2_SMALL,
        normals_model: Annotated[
            NormalsModelName,
            Option("--normals-model", help="Normal-field generator for normals."),
        ] = NormalsModelName.OMNIDATA,
        validators: Annotated[
            bool,
            Option("--validators/--no-validators", help="Run modality validators."),
        ] = True,
        constraints: Annotated[
            bool,
            Option(
                "--constraints/--no-constraints",
                help="Use validator constraints as quality gates.",
            ),
        ] = True,
        text_siglip_minimum: Annotated[
            float,
            Option("--text-siglip-min", help="Minimum SIGLIP text-image score."),
        ] = 0.60,
        segmentation_mask_quality_minimum: Annotated[
            float,
            Option("--segmentation-mask-quality-min", help="Minimum weighted mask quality."),
        ] = 0.75,
        segmentation_boundary_minimum: Annotated[
            float,
            Option("--segmentation-boundary-min", help="Minimum RGB boundary F1."),
        ] = 0.20,
        depth_imagebind_minimum: Annotated[
            float,
            Option("--depth-imagebind-min", help="Minimum ImageBind depth-image score."),
        ] = 0.55,
        depth_segmentation_boundary_minimum: Annotated[
            float,
            Option(
                "--depth-segmentation-boundary-min",
                help="Minimum depth-vs-segmentation boundary F1.",
            ),
        ] = 0.20,
        normals_depth_agreement_minimum: Annotated[
            float,
            Option(
                "--normals-depth-agreement-min",
                help="Minimum depth/normal angular agreement score.",
            ),
        ] = 0.35,
        normals_field_quality_minimum: Annotated[
            float,
            Option("--normals-field-quality-min", help="Minimum normal-field quality score."),
        ] = 0.50,
        segmentation_boundary_thickness: Annotated[
            int,
            Option("--segmentation-boundary-thickness", help="Mask boundary thickness in pixels."),
        ] = 1,
        segmentation_tolerance_radius: Annotated[
            int,
            Option("--segmentation-tolerance-radius", help="RGB edge matching tolerance in pixels."),
        ] = 2,
        segmentation_rgb_edge_quantile: Annotated[
            float,
            Option("--segmentation-rgb-edge-quantile", help="RGB edge quantile for boundary checks."),
        ] = 0.90,
        depth_boundary_thickness: Annotated[
            int,
            Option("--depth-boundary-thickness", help="Segmentation boundary thickness for depth checks."),
        ] = 1,
        depth_tolerance_radius: Annotated[
            int,
            Option("--depth-tolerance-radius", help="Depth edge matching tolerance in pixels."),
        ] = 2,
        depth_edge_quantile: Annotated[
            float,
            Option("--depth-edge-quantile", help="Depth edge quantile for boundary checks."),
        ] = 0.90,
        normals_eps: Annotated[
            float,
            Option("--normals-eps", help="Epsilon for normal-vector normalization."),
        ] = 1e-6,
        normals_nz_min: Annotated[
            float,
            Option("--normals-nz-min", help="Minimum |nz| used by normals integrability scoring."),
        ] = 0.1,
        limit: Annotated[
            int | None,
            Option("--limit", "-n", help="Maximum number of samples to process."),
        ] = None,
        experiment_root: Annotated[
            Path,
            Option(
                "--experiment-root",
                file_okay=False,
                help="Directory where experiment artifacts will be written.",
            ),
        ] = Path("experiments"),
        experiment_name: Annotated[
            str | None,
            Option("--experiment-name", help="Optional experiment folder name."),
        ] = None,
        log_path: Annotated[
            Path | None,
            Option("--log-path", file_okay=False, help="Directory where logs will be stored."),
        ] = DEFAULT_LOG_PATH,
        seed: Annotated[
            int,
            Option("--seed", help="Global seed for reproducible model execution."),
        ] = 42,
        json_summary: Annotated[
            bool,
            Option("--json", help="Print the run summary as JSON."),
        ] = False,
) -> None:
    """Run a configurable pipeline from CLI options."""
    config = RunConfig(
        dataset=DatasetConfig(
            kind=dataset,
            root=root,
            parquet_path=parquet_path,
            recursive=recursive,
            extensions=tuple(extensions or ()),
            image_column=image_column,
            id_column=id_column,
            extension_column=extension_column,
            metadata_columns=tuple(metadata_columns or ()),
            batch_size=batch_size,
        ),
        modalities=normalize_modalities(modalities),
        models=ModelConfig(
            text=text_model,
            segmentation=segmentation_model,
            depth=depth_model,
            normals=normals_model,
        ),
        validators=ValidatorConfig(
            enabled=validators,
            constraints=ConstraintConfig(
                enabled=constraints,
                text_siglip_minimum=text_siglip_minimum,
                segmentation_mask_quality_minimum=segmentation_mask_quality_minimum,
                segmentation_boundary_minimum=segmentation_boundary_minimum,
                depth_imagebind_minimum=depth_imagebind_minimum,
                depth_segmentation_boundary_minimum=depth_segmentation_boundary_minimum,
                normals_depth_agreement_minimum=normals_depth_agreement_minimum,
                normals_field_quality_minimum=normals_field_quality_minimum,
            ),
            segmentation_boundary_thickness=segmentation_boundary_thickness,
            segmentation_tolerance_radius=segmentation_tolerance_radius,
            segmentation_rgb_edge_quantile=segmentation_rgb_edge_quantile,
            depth_boundary_thickness=depth_boundary_thickness,
            depth_tolerance_radius=depth_tolerance_radius,
            depth_edge_quantile=depth_edge_quantile,
            normals_eps=normals_eps,
            normals_nz_min=normals_nz_min,
        ),
        limit=limit,
        experiment_root=experiment_root,
        experiment_name=experiment_name,
        log_path=log_path,
        seed=seed,
        json_summary=json_summary,
    )
    _run_or_exit(config)


@app.command(
    help=(
            "Run the Modosaic pipeline from a JSON, TOML, or YAML configuration file. "
            "YAML requires PyYAML to be installed."
    ),
    short_help="Run from a config file.",
)
def pipeline(
        config_path: Annotated[
            Path,
            Argument(
                exists=True,
                dir_okay=False,
                help="Path to a JSON, TOML, YAML, or YML pipeline configuration file.",
            ),
        ],
        log_path: Annotated[
            Path | None,
            Option("--log-path", file_okay=False, help="Override the config log directory."),
        ] = None,
        seed: Annotated[
            int | None,
            Option("--seed", help="Override the config seed."),
        ] = None,
        limit: Annotated[
            int | None,
            Option("--limit", "-n", help="Override the config sample limit."),
        ] = None,
        json_summary: Annotated[
            bool,
            Option("--json", help="Print the run summary as JSON."),
        ] = False,
) -> None:
    """Run a configurable pipeline from a config file."""
    try:
        config = run_config_from_mapping(load_config_mapping(config_path))
    except ValueError as exc:
        _abort(str(exc))

    if log_path is not None:
        config = replace(config, log_path=log_path)
    if seed is not None:
        config = replace(config, seed=seed)
    if limit is not None:
        config = replace(config, limit=limit)
    if json_summary:
        config = replace(config, json_summary=True)

    _run_or_exit(config)


@app.command(
    help="Print valid modality and model names for CLI options and config files.",
    short_help="List choices.",
)
def models() -> None:
    """Print valid modality and model names."""
    groups: dict[str, Sequence[Enum]] = {
        "modalities": tuple(ModalityName),
        "text models": tuple(TextModelName),
        "segmentation models": tuple(SegmentationModelName),
        "depth models": tuple(DepthModelName),
        "normals models": tuple(NormalsModelName),
    }
    for title, values in groups.items():
        echo(f"{title}:")
        for value in values:
            default = _default_label(value)
            echo(f"  {value.value}{default}")


def _run_or_exit(config: RunConfig) -> None:
    """Execute a run and convert user-facing failures into CLI exits."""
    try:
        run_result = execute_run(config)
    except (FileNotFoundError, ValueError) as exc:
        _abort(str(exc))

    echo(
        format_summary(
            run_result.results,
            run_result.experiment_path,
            as_json=config.json_summary,
        )
    )


def _default_label(value: Enum) -> str:
    """Return the default-marker suffix for CLI choice output."""
    defaults = {
        ModalityName.IMAGE,
        ModalityName.TEXT,
        ModalityName.SEGMENTATION,
        ModalityName.DEPTH,
        ModalityName.NORMALS,
        TextModelName.QWEN_2_2B,
        SegmentationModelName.SAM3,
        DepthModelName.DEPTH_ANYTHING_V2_SMALL,
        NormalsModelName.OMNIDATA,
    }
    return " (default)" if value in defaults else ""


def _abort(message: str, code: int = 2) -> None:
    """Print an error message and exit with the provided status code."""
    echo(f"Error: {message}", err=True)
    raise Exit(code)
