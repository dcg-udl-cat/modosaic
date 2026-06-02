"""Configuration models and parsers for the Modosaic CLI."""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, TypeVar

DEFAULT_LOG_PATH = Path.home() / ".modosaic" / "logs"
EnumT = TypeVar("EnumT", bound=Enum)


class DatasetKind(str, Enum):
    """Dataset backend names accepted by CLI and config files."""

    LOCAL = "local"
    PARQUET = "parquet"


class ModalityName(str, Enum):
    """Modality names accepted by CLI and config files."""

    IMAGE = "image"
    TEXT = "text"
    SEGMENTATION = "segmentation"
    DEPTH = "depth"
    NORMALS = "normals"


class TextModelName(str, Enum):
    """Captioning model choices accepted by CLI and config files."""

    INTERN_VL_3_5_2B = "intern-vl-3-5-2b"
    INTERN_VL_3_2B = "intern-vl-3-2b"
    QWEN_2_2B = "qwen-2-2b"
    QWEN_2_5_3B = "qwen-2-5-3b"


class SegmentationModelName(str, Enum):
    """Segmentation model choices accepted by CLI and config files."""

    SAM = "sam"
    SAM2 = "sam2"
    SAM3 = "sam3"


class DepthModelName(str, Enum):
    """Depth model choices accepted by CLI and config files."""

    DEPTH_ANYTHING_SMALL = "depth-anything-small"
    DEPTH_ANYTHING_V2_METRIC_SMALL = "depth-anything-v2-metric-small"
    DEPTH_ANYTHING_V2_SMALL = "depth-anything-v2-small"
    DEPTH_PRO = "depth-pro"
    DPT_HYBRID_MIDAS = "dpt-hybrid-midas"
    MARIGOLD_DEPTH_V1_1 = "marigold-depth-v1-1"


class NormalsModelName(str, Enum):
    """Surface-normal model choices accepted by CLI and config files."""

    OMNIDATA = "omnidata"
    MIDAS_D2N = "midas-d2n"


DEFAULT_MODALITIES = (
    ModalityName.IMAGE,
    ModalityName.TEXT,
    ModalityName.SEGMENTATION,
    ModalityName.DEPTH,
    ModalityName.NORMALS,
)


@dataclass(frozen=True, slots=True)
class DatasetConfig:
    """Dataset configuration for a Modosaic run.

    Attributes:
        kind: Dataset backend to use.
        root: Local image-folder root when `kind` is local.
        parquet_path: Parquet file or directory when `kind` is parquet.
        recursive: Whether local folders are scanned recursively.
        extensions: Optional local-folder extensions to include.
        image_column: Parquet column or nested path containing image data.
        id_column: Optional parquet sample ID column.
        extension_column: Optional parquet extension column.
        metadata_columns: Optional parquet columns preserved as metadata.
        batch_size: Parquet read batch size.
    """

    kind: DatasetKind
    root: Path | None = None
    parquet_path: Path | None = None
    recursive: bool = True
    extensions: tuple[str, ...] = ()
    image_column: str = "image"
    id_column: str | None = None
    extension_column: str | None = None
    metadata_columns: tuple[str, ...] = ()
    batch_size: int = 512


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Model choices for generated modalities."""

    text: TextModelName = TextModelName.QWEN_2_2B
    segmentation: SegmentationModelName = SegmentationModelName.SAM3
    depth: DepthModelName = DepthModelName.DEPTH_ANYTHING_V2_SMALL
    normals: NormalsModelName = NormalsModelName.OMNIDATA


@dataclass(frozen=True, slots=True)
class ConstraintConfig:
    """Validation-threshold configuration used as quality gates."""

    enabled: bool = True
    text_siglip_minimum: float = 0.60
    segmentation_mask_quality_minimum: float = 0.75
    segmentation_boundary_minimum: float = 0.20
    depth_imagebind_minimum: float = 0.55
    depth_segmentation_boundary_minimum: float = 0.20
    normals_depth_agreement_minimum: float = 0.35
    normals_field_quality_minimum: float = 0.50


@dataclass(frozen=True, slots=True)
class ValidatorConfig:
    """Validator and quality-gate options for a Modosaic run."""

    enabled: bool = True
    constraints: ConstraintConfig = ConstraintConfig()
    segmentation_boundary_thickness: int = 1
    segmentation_tolerance_radius: int = 2
    segmentation_rgb_edge_quantile: float = 0.90
    depth_boundary_thickness: int = 1
    depth_tolerance_radius: int = 2
    depth_edge_quantile: float = 0.90
    normals_eps: float = 1e-6
    normals_nz_min: float = 0.1


@dataclass(frozen=True, slots=True)
class RunConfig:
    """Complete CLI/runtime configuration for one Modosaic run."""

    dataset: DatasetConfig
    modalities: tuple[ModalityName, ...] = DEFAULT_MODALITIES
    models: ModelConfig = ModelConfig()
    validators: ValidatorConfig = ValidatorConfig()
    limit: int | None = None
    experiment_root: Path = Path("experiments")
    experiment_name: str | None = None
    log_path: Path | None = DEFAULT_LOG_PATH
    seed: int = 42
    json_summary: bool = False


def load_config_mapping(config_path: Path) -> Mapping[str, Any]:
    """Load a JSON, TOML, YAML, or YML config file.

    Args:
        config_path: Path to the config file.

    Returns:
        Parsed top-level mapping.

    Raises:
        ValueError: If the file extension is unsupported or the parsed payload
            is not a mapping.
    """
    suffix = config_path.suffix.lower()
    text = config_path.read_text(encoding="utf-8")

    if suffix == ".json":
        data = json.loads(text)
    elif suffix == ".toml":
        data = _load_toml(text)
    elif suffix in {".yaml", ".yml"}:
        data = _load_yaml(text)
    else:
        raise ValueError("Config file must end in .json, .toml, .yaml, or .yml.")

    if not isinstance(data, Mapping):
        raise ValueError("Pipeline config must contain a top-level mapping.")
    return data


def run_config_from_mapping(data: Mapping[str, Any]) -> RunConfig:
    """Build a `RunConfig` from a parsed config mapping.

    Args:
        data: Parsed config data.

    Returns:
        Normalized run configuration.

    Raises:
        ValueError: If any section has an invalid shape or enum value.
    """
    dataset_section = _mapping(data.get("dataset"), "dataset")
    modalities_section = data.get("modalities", {})
    validators_section = _mapping(data.get("validators", {}), "validators")
    run_section = _mapping(data.get("run", {}), "run")

    enabled_modalities, model_section = _read_modalities_section(modalities_section)
    constraint_section = _read_constraint_section(validators_section)

    return RunConfig(
        dataset=DatasetConfig(
            kind=_enum_value(
                DatasetKind,
                dataset_section.get("kind", dataset_section.get("type", DatasetKind.LOCAL.value)),
                "dataset.kind",
            ),
            root=_path_or_none(dataset_section.get("root", dataset_section.get("path"))),
            parquet_path=_path_or_none(dataset_section.get("parquet_path", dataset_section.get("path"))),
            recursive=bool(dataset_section.get("recursive", True)),
            extensions=tuple(_string_sequence(dataset_section.get("extensions", ()), "dataset.extensions")),
            image_column=str(dataset_section.get("image_column", "image")),
            id_column=_str_or_none(dataset_section.get("id_column")),
            extension_column=_str_or_none(dataset_section.get("extension_column")),
            metadata_columns=tuple(
                _string_sequence(dataset_section.get("metadata_columns", ()), "dataset.metadata_columns")
            ),
            batch_size=int(dataset_section.get("batch_size", 512)),
        ),
        modalities=normalize_modalities(
            [
                _enum_value(ModalityName, value, "modalities.enabled")
                for value in (enabled_modalities or ())
            ]
            if enabled_modalities is not None
            else None
        ),
        models=ModelConfig(
            text=_enum_value(
                TextModelName,
                model_section.get("text", TextModelName.QWEN_2_2B.value),
                "modalities.models.text",
            ),
            segmentation=_enum_value(
                SegmentationModelName,
                model_section.get("segmentation", SegmentationModelName.SAM3.value),
                "modalities.models.segmentation",
            ),
            depth=_enum_value(
                DepthModelName,
                model_section.get("depth", DepthModelName.DEPTH_ANYTHING_V2_SMALL.value),
                "modalities.models.depth",
            ),
            normals=_enum_value(
                NormalsModelName,
                model_section.get("normals", NormalsModelName.OMNIDATA.value),
                "modalities.models.normals",
            ),
        ),
        validators=ValidatorConfig(
            enabled=bool(validators_section.get("enabled", True)),
            constraints=ConstraintConfig(
                enabled=bool(constraint_section.get("enabled", True)),
                text_siglip_minimum=float(constraint_section.get("text_siglip_minimum", 0.60)),
                segmentation_mask_quality_minimum=float(
                    constraint_section.get("segmentation_mask_quality_minimum", 0.75)
                ),
                segmentation_boundary_minimum=float(
                    constraint_section.get("segmentation_boundary_minimum", 0.20)
                ),
                depth_imagebind_minimum=float(constraint_section.get("depth_imagebind_minimum", 0.55)),
                depth_segmentation_boundary_minimum=float(
                    constraint_section.get("depth_segmentation_boundary_minimum", 0.20)
                ),
                normals_depth_agreement_minimum=float(
                    constraint_section.get("normals_depth_agreement_minimum", 0.35)
                ),
                normals_field_quality_minimum=float(
                    constraint_section.get("normals_field_quality_minimum", 0.50)
                ),
            ),
            segmentation_boundary_thickness=int(
                validators_section.get("segmentation_boundary_thickness", 1)
            ),
            segmentation_tolerance_radius=int(
                validators_section.get("segmentation_tolerance_radius", 2)
            ),
            segmentation_rgb_edge_quantile=float(
                validators_section.get("segmentation_rgb_edge_quantile", 0.90)
            ),
            depth_boundary_thickness=int(validators_section.get("depth_boundary_thickness", 1)),
            depth_tolerance_radius=int(validators_section.get("depth_tolerance_radius", 2)),
            depth_edge_quantile=float(validators_section.get("depth_edge_quantile", 0.90)),
            normals_eps=float(validators_section.get("normals_eps", 1e-6)),
            normals_nz_min=float(validators_section.get("normals_nz_min", 0.1)),
        ),
        limit=_int_or_none(run_section.get("limit")),
        experiment_root=Path(run_section.get("experiment_root", "experiments")),
        experiment_name=_str_or_none(run_section.get("experiment_name")),
        log_path=_path_or_none(run_section.get("log_path", DEFAULT_LOG_PATH)),
        seed=int(run_section.get("seed", 42)),
        json_summary=bool(run_section.get("json_summary", False)),
    )


def normalize_modalities(
        modalities: Sequence[ModalityName] | None,
) -> tuple[ModalityName, ...]:
    """Return selected modalities in dependency-safe default order.

    Args:
        modalities: Requested modality subset. `None` or an empty sequence means
            all default modalities.

    Returns:
        Ordered modality tuple.
    """
    if not modalities:
        return DEFAULT_MODALITIES

    requested = set(modalities)
    return tuple(name for name in DEFAULT_MODALITIES if name in requested)


def _read_modalities_section(value: Any) -> tuple[Sequence[Any] | None, Mapping[str, Any]]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return value, {}

    modalities_section = _mapping(value, "modalities")
    enabled_modalities = _sequence_or_none(
        modalities_section.get("enabled", modalities_section.get("include")),
        "modalities.enabled",
    )
    model_section = _mapping(modalities_section.get("models", {}), "modalities.models")
    return enabled_modalities, model_section


def _read_constraint_section(validators_section: Mapping[str, Any]) -> Mapping[str, Any]:
    constraints_section = validators_section.get("constraints", {})
    if isinstance(constraints_section, bool):
        return {"enabled": constraints_section}

    return _mapping(constraints_section, "validators.constraints")


def _load_toml(text: str) -> Mapping[str, Any]:
    try:
        import tomllib
    except ModuleNotFoundError:
        try:
            import tomli as tomllib
        except ImportError as exc:
            raise ValueError(
                "TOML config files require Python 3.11+ or the tomli package."
            ) from exc

    return tomllib.loads(text)


def _load_yaml(text: str) -> Mapping[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise ValueError("YAML config files require PyYAML. Use JSON or TOML instead.") from exc

    return yaml.safe_load(text)


def _mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a mapping.")
    return value


def _sequence_or_none(value: Any, field_name: str) -> Sequence[Any] | None:
    if value is None:
        return None
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return value
    raise ValueError(f"{field_name} must be a list.")


def _string_sequence(value: Any, field_name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{field_name} must be a list.")
    return tuple(str(item) for item in value)


def _enum_value(enum_type: type[EnumT], value: Any, field_name: str) -> EnumT:
    if isinstance(value, enum_type):
        return value

    normalized = str(value).strip().lower().replace("_", "-")
    for candidate in enum_type:
        if normalized in {candidate.value.lower(), candidate.name.lower().replace("_", "-")}:
            return candidate

    choices = ", ".join(candidate.value for candidate in enum_type)
    raise ValueError(f"{field_name} must be one of: {choices}.")


def _path_or_none(value: Any) -> Path | None:
    if value is None:
        return None
    return value if isinstance(value, Path) else Path(str(value))


def _str_or_none(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)
