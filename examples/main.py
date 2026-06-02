"""Run the full preconfigured Modosaic demo pipeline.

The script demonstrates direct Python usage of the project API. It reads image
records from the bundled parquet-style `data/imagenet-a` dataset, generates all
preconfigured modalities, saves accepted artifacts through `ExperimentService`,
and prints validator summaries for each processed sample.

Example:
    Run the demo from the project root:

    ```bash
    python main.py
    ```
"""

from pathlib import Path

from modosaic import ExperimentService, ImageDataset, Pipeline, LoggingService
from modosaic.depth.preconfigured_modality import build_preconfigured_depth_modality
from modosaic.image import build_preconfigured_image_modality
from modosaic.normals.preconfigured_modality import build_preconfigured_normals_modality
from modosaic.segmentation.preconfigured_modality import build_preconfigured_segmentation_modality
from modosaic.services.seeding import SeedingService
from modosaic.text.preconfigured_modality import build_preconfigured_text_modality


def main() -> None:
    """Execute the demo pipeline and print validation summaries.

    The run uses the default logging configuration, the default global seed, the
    parquet dataset at `data/imagenet-a`, and the preconfigured image, text,
    segmentation, depth, and normals modalities.

    Returns:
        None.
    """
    LoggingService.setup_logging()
    SeedingService.set_global_seed()
    pipeline = Pipeline(
        dataset=ImageDataset.from_parquet(
            parquet_path=Path("data/imagenet-a"),
            image_column="image.bytes",
            metadata_columns=["label"],
        ),
        modalities=[
            build_preconfigured_image_modality(),
            build_preconfigured_text_modality(),
            build_preconfigured_segmentation_modality(),
            build_preconfigured_depth_modality(),
            build_preconfigured_normals_modality(),
        ],
        experiment=ExperimentService(),
    )

    results = pipeline.run(limit=20)

    for result in results:
        print("=" * 80)
        print(result.record.sample_id, result.record.extension, result.record.metadata)
        for modality, validations in result.validations.items():
            for validation in validations:
                if validation.passed is None:
                    print(f"{modality} {validation.validator_name}: {validation.value}")
                    continue

                print(
                    f"{modality} {validation.validator_name}: {validation.value} "
                    f"({validation.score_name}={validation.score:.3f} "
                    f">= {validation.minimum:.3f}, passed={validation.passed})"
                )


if __name__ == "__main__":
    main()
