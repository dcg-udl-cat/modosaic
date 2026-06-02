# Modosaic

Modosaic is a multimodal image-dataset pipeline for generating, validating, and
saving complementary modalities from a shared image source. It provides:

- Dataset loading from local folders and parquet files or directories.
- Preconfigured pipelines for source images, captions, segmentation masks,
  depth maps, and surface normals.
- Validators and quality-gate constraints for filtering generated artifacts.
- Experiment output folders containing artifacts, validation JSON, and logs.
- A CLI for default, configurable, and config-file driven runs.
- A Python API for custom generators, validators, postprocessors, and
  modality compositions.

---

## Installation

```bash
uv sync
```

or:

```bash
pip install -e .
```

or:

```bash
pip install modosaic
```

Python 3.13 is required. CUDA is optional but recommended for model-backed
generation and validation.

---

## Quick Start

List the supported modalities and model names:

```bash
modosaic models
```

Run the default pipeline on a local image folder:

```bash
modosaic simple ./images --limit 10
```

Run from a config file:

```bash
modosaic pipeline examples/config.yaml --limit 5 --seed 123
```

---

## Python API

```python
from modosaic import ExperimentService, ImageDataset, Pipeline
from modosaic.depth.preconfigured_modality import build_preconfigured_depth_modality
from modosaic.image import build_preconfigured_image_modality
from modosaic.segmentation.preconfigured_modality import (
    build_preconfigured_segmentation_modality,
)

dataset = ImageDataset.from_local_folder("images")

pipeline = Pipeline(
    dataset=dataset,
    modalities=[
        build_preconfigured_image_modality(),
        build_preconfigured_segmentation_modality(),
        build_preconfigured_depth_modality(),
    ],
    experiment=ExperimentService(experiment_name="demo"),
)

results = pipeline.run(limit=10)
```

---

## Concepts

- Providers load `ImageRecord` objects from dataset backends.
- Generators produce modality outputs from an input image record.
- Validators score generated outputs, optionally using earlier modalities.
- Constraints turn validator scores into pass/fail quality gates.
- Postprocessors convert accepted outputs into experiment artifacts.
- `ExperimentService` saves artifacts below the configured run folder.

The API reference is generated from Google-style docstrings in the package.
