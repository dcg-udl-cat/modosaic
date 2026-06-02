# Modosaic

Modosaic is a multimodal image-dataset pipeline for generating, validating, and
saving complementary modalities from a shared image source. It gives you:

- A unified dataset layer for local image folders and parquet datasets.
- A configurable generation pipeline for source images, captions, segmentation
  masks, depth maps, and surface-normal fields.
- Validators and quality-gate constraints that decide which generated
  modalities are saved.
- A CLI for default runs, fully configurable runs, and config-file driven runs.
- A clean Python API for composing custom modalities, validators, and
  postprocessors.
- Reproducible experiment folders with generated artifacts, validation JSON, and
  structured logs.

---

## Installation

For an exact source-checkout installation, use the committed lockfile:

```bash
uv sync --locked
```

`uv.lock` pins the Python dependency graph and Git revisions; `--locked` fails
instead of changing that graph. Python 3.13 is required.

For an editable convenience install from an activated environment:

```bash
pip install -e .
```

For the published package:

```bash
pip install modosaic
```

The `pip` routes do not reproduce the repository lock. Runtime dependencies are
intentionally installed as one set because the default pipeline runs every
modality and its cross-modal validators in one process. Splitting model families
into optional groups would make the default workflow depend on several
overlapping extras. Documentation tools are isolated in the `docs` dependency
group.

### Linux with NVIDIA CUDA

The Oxford-IIIT Pet CUDA benchmark in the paper used Google Cloud image
`pytorch-2-9-cu129-ubuntu-2404-nvidia-580-v20260518`: Ubuntu 24.04, Python 3.12,
PyTorch 2.9.0, CUDA 12.9, and NVIDIA driver 580. The exact GPU model and VRAM
were not recorded.

For Modosaic 1.0.0post1, `flake.nix` and `flake.lock` define the current
`x86_64-linux` CUDA development shell, including Python 3.13 and CUDA user-space
libraries; `uv.lock` pins the Python packages. Together they define the locked
software environment. A compatible host NVIDIA driver is still required and is
not supplied by Nix. Check it with `nvidia-smi`.

If your Nix installation does not already allow flakes and the listed binary
caches, add the following generic settings to `/etc/nix/nix.conf`:

```ini
experimental-features = nix-command flakes

extra-substituters = https://cache.nixos.org https://nix-community.cachix.org https://cache.nixos-cuda.org

extra-trusted-public-keys = nix-community.cachix.org-1:mB9ZQ+4kTq9qUqM96H8P6oz+ZWHR+Hh3wlgYx9oSt1A= cache.nixos-cuda.org:74DUi4Ye579gUqzH4ziL9IyiJBlDpMRn9MBN8oNan9M=
```

Restart the Nix daemon after changing its configuration, then enter the shell:

```bash
nix develop
```

The shell runs `uv sync --locked`, activates `.venv`, and downloads the
preconfigured SAM ViT-B checkpoint when it is absent.

### Apple Silicon and CPU

On macOS or a non-Nix setup, run `uv sync --locked` directly. Modosaic selects
CUDA first, then Apple MPS, then CPU. CUDA is strongly recommended for the
heavier text, segmentation, depth, and normals models; CPU execution is
supported but slower. Third-party backend support and memory requirements vary
by model, so no universal VRAM minimum is claimed. The Nix shell is currently
limited to `x86_64-linux`.

### Hugging Face model access

Most model weights are downloaded from their upstream repositories on first
use, so allow network access and sufficient cache space. Local checkpoints can
be placed under `model_weights/`.

If you use the SAM 3 segmentation model (`sam3`, backed by `facebook/sam3`),
run Modosaic with `HF_TOKEN` set to a Hugging Face token from an account that
has access to Meta's SAM 3 model:

```bash
export HF_TOKEN=hf_...
modosaic run --dataset local --root ./images --segmentation-model sam3
```

Do not commit tokens to the repository.

---

## Quick Start

### 1. List supported modalities and models

```bash
modosaic models
```

When running directly from a checkout without installing the console script:

```bash
python -m modosaic.cli.cli models
```

### 2. Run the default pipeline on a local image folder

```bash
modosaic simple ./images --limit 10
```

This runs all default modalities in dependency-safe order:

```text
image -> text -> segmentation -> depth -> normals
```

Artifacts are written under `./experiments/<timestamp>/`.

### 3. Run a selected local-folder experiment

```bash
modosaic run \
  --dataset local \
  --root ./images \
  --modality image \
  --modality segmentation \
  --modality depth \
  --segmentation-model sam3 \
  --depth-model depth-anything-v2-small \
  --segmentation-mask-quality-min 0.70 \
  --depth-segmentation-boundary-min 0.15 \
  --limit 20 \
  --experiment-root ./experiments \
  --experiment-name local-seg-depth
```

Validators can be disabled with `--no-validators`. To run validators without
using them as save/discard gates, pass `--no-constraints`.

### 4. Run from a parquet dataset

```bash
modosaic run \
  --dataset parquet \
  --parquet-path ./data/imagenet-a \
  --image-column image.bytes \
  --metadata-column label \
  --modality image \
  --modality text \
  --text-model qwen-2-2b \
  --text-siglip-min 0.65 \
  --limit 20
```

Parquet image columns can contain bytes, bytearray/memoryview values,
`list[int]`, nested fields such as `image.bytes`, or paths relative to the
parquet file.

---

## Full Pipeline From Config

Pipeline config files can be JSON, TOML, YAML, or YML. YAML requires PyYAML.
The config maps directly into `modosaic.cli.config.RunConfig`.

Example `examples/config.yaml`:

```yaml
dataset:
  type: local
  root: ./images
  recursive: true
  extensions: [.jpg, .png]

modalities:
  enabled: [image, segmentation, depth]
  models:
    segmentation: sam3
    depth: depth-anything-v2-small

validators:
  enabled: true
  constraints:
    enabled: true
    segmentation_mask_quality_minimum: 0.70
    segmentation_boundary_minimum: 0.20
    depth_imagebind_minimum: 0.55
    depth_segmentation_boundary_minimum: 0.15
  segmentation_boundary_thickness: 1
  segmentation_tolerance_radius: 2
  segmentation_rgb_edge_quantile: 0.90
  depth_boundary_thickness: 1
  depth_tolerance_radius: 2
  depth_edge_quantile: 0.90

run:
  limit: 20
  experiment_root: ./experiments
  experiment_name: local-seg-depth
  log_path: ./.logs
  seed: 42
```

Run it with:

```bash
modosaic pipeline examples/config.yaml
```

Override selected config values at execution time:

```bash
modosaic pipeline examples/config.yaml --limit 5 --seed 123 --json
```

---

## Python API

```python
from pathlib import Path

from modosaic import ExperimentService, ImageDataset, LoggingService, Pipeline
from modosaic.depth.preconfigured_modality import build_preconfigured_depth_modality
from modosaic.image import build_preconfigured_image_modality
from modosaic.segmentation.preconfigured_modality import (
    build_preconfigured_segmentation_modality,
)
from modosaic.services.seeding import SeedingService

LoggingService.setup_logging()
SeedingService.set_global_seed(42)

dataset = ImageDataset.from_local_folder(Path("images"))

pipeline = Pipeline(
    dataset=dataset,
    modalities=[
        build_preconfigured_image_modality(),
        build_preconfigured_segmentation_modality(),
        build_preconfigured_depth_modality(),
    ],
    experiment=ExperimentService(
        root="experiments",
        experiment_name="local-seg-depth",
    ),
)

results = pipeline.run(limit=10)

for result in results:
    print(result.record.sample_id, result.artifact_paths)
```

Each `ConfiguredModality` owns a generator, validators, and a postprocessor.
Plain validators receive `(record, generated)`. A `ValidatorStep` can also pass
generated outputs from earlier modalities as keyword dependencies.

```python
from modosaic.core.validation_constraint import ValidationConstraint
from modosaic.core.validator_step import ValidatorStep
from modosaic.segmentation.validators.impl.mask_statistics import MaskStatsValidator

mask_quality_gate = ValidatorStep(
    validator=MaskStatsValidator(),
    constraint=ValidationConstraint(
        minimum=0.75,
        score_name="weighted_mask_quality",
        score_fn=lambda stats: (
            0.4 * stats.coverage_score
            + 0.3 * stats.distinctness_score
            + 0.3 * stats.fragmentation_score
        ),
    ),
)
```

A constrained modality is saved only when every configured constraint passes.
Rejected modalities do not write generated artifacts or validation JSON, and
later validators cannot use them as dependencies.

---

## CLI Reference

### `modosaic simple ROOT [--limit N]`

Run the default Modosaic pipeline on a local image folder.

### `modosaic run [OPTIONS]`

Run with CLI-provided dataset, modality, model, validator, constraint, and
experiment settings.

### `modosaic pipeline CONFIG_PATH [--limit N] [--seed SEED] [--json]`

Run from a JSON, TOML, YAML, or YML config file.

### `modosaic models`

Print valid modality and model names for CLI options and config files.

---

## Concepts & Extensibility

- Dataset adapters -> `modosaic.providers`: local folders and parquet data.
- Generators -> `modosaic.<modality>.generators`: model-backed modality
  generation.
- Validators -> `modosaic.<modality>.validators`: quality checks and
  cross-modality consistency checks.
- Constraints -> `modosaic.core.validation_constraint`: pass/fail gates over
  validator output.
- Postprocessors -> `modosaic.<modality>.postprocessor`: conversion from model
  output to saved experiment artifacts.
- Services -> logging, seeding, image conversion, artifact persistence,
  boundaries, edges, and tolerances.

Add custom components by subclassing:

```text
DatasetAdapter -> providers.adapters.adapter.DatasetAdapter
ModalityGenerator -> core.modality_generator.ModalityGenerator
ModalityValidator -> core.validator.ModalityValidator
ModalityPostprocessor -> core.postprocessor.ModalityPostprocessor
Modality -> core.modality.Modality
```

or by composing existing pieces with `ConfiguredModality`.

---

## Experiment Outputs

`ExperimentService` writes every accepted artifact beneath the configured run
folder. Typical output includes:

```text
experiments/<run>/
  image/
  text/
  segmentation/
  depth/
  normals/
  validations/
```

Validation files include the validator name, raw value, optional score,
threshold, and pass/fail result.

---

## Documentation

MkDocs pages live in `docs/`, and API pages are generated from Google-style
Python docstrings through `mkdocstrings`.

Public classes, functions, and methods should carry useful Google-style
docstrings because they form the API reference. Module docstrings are optional
for simple implementation modules; add them when a file exposes important
package-level behavior, re-exports public symbols, or needs context that is not
clear from the documented objects inside it.

```bash
uv run --group docs mkdocs serve
uv run --group docs mkdocs build --strict
```

---

## Examples

See `examples/main.py` for a complete local demo that loads a parquet dataset, runs the
preconfigured modality stack, and prints validation summaries.

---

## License
This project is licensed under the MIT License. See the LICENSE file for details.
