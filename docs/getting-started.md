# Getting Started

This guide walks through the common ways to run Modosaic and the config fields
that control a run.

---

## Nix Development Shell

The repository includes a locked `x86_64-linux` CUDA development shell. See the
[README installation guide](https://github.com/dcg-udl-cat/modosaic#installation) for the tested
environment, hardware-specific routes, model access, and the unified-dependency
rationale. If needed, configure the Nix daemon to use the flake's binary caches;
otherwise Nix may try to build CUDA and community packages locally.

Add the following to `/etc/nix/nix.conf`:

```ini
experimental-features = nix-command flakes

extra-substituters = https://cache.nixos.org https://nix-community.cachix.org https://cache.nixos-cuda.org

extra-trusted-public-keys = nix-community.cachix.org-1:mB9ZQ+4kTq9qUqM96H8P6oz+ZWHR+Hh3wlgYx9oSt1A= cache.nixos-cuda.org:74DUi4Ye579gUqzH4ziL9IyiJBlDpMRn9MBN8oNan9M=
```

Restart the Nix daemon after changing the file, then enter the shell:

```bash
nix develop
```

The shell installs the Python environment with `uv sync --locked`.

---

## Dataset Sources

### Local folders

```bash
modosaic run --dataset local --root ./images --limit 10
```

Local folders are scanned recursively by default. Use `--top-level` to scan only
the root folder, and repeat `--extension` to restrict accepted image suffixes:

```bash
modosaic run \
  --dataset local \
  --root ./images \
  --top-level \
  --extension .jpg \
  --extension .png
```

### Parquet files

```bash
modosaic run \
  --dataset parquet \
  --parquet-path ./data/imagenet-a \
  --image-column image.bytes \
  --id-column id \
  --metadata-column label
```

The image column may contain encoded bytes, `bytearray`, `memoryview`,
`list[int]`, or image paths. Nested paths such as `image.bytes` are supported.

---

## Modality Selection

When no modalities are provided, Modosaic runs all defaults in this order:

```text
image -> text -> segmentation -> depth -> normals
```

Select a subset by repeating `--modality`:

```bash
modosaic run \
  --dataset local \
  --root ./images \
  --modality image \
  --modality segmentation \
  --modality depth
```

The run config normalizes the selected subset back into dependency-safe order.

---

## Models

Show valid model names:

```bash
modosaic models
```

Configure models from the CLI:

```bash
modosaic run \
  --dataset local \
  --root ./images \
  --text-model qwen-2-2b \
  --segmentation-model sam3 \
  --depth-model depth-anything-v2-small \
  --normals-model omnidata
```

---

## Validators and Constraints

Validators compute quality values for a modality. Constraints convert those
values into quality gates. A constrained modality is saved only if every
constraint passes.

Disable validators entirely:

```bash
modosaic run --dataset local --root ./images --no-validators
```

Run validators but never discard generated modalities:

```bash
modosaic run --dataset local --root ./images --no-constraints
```

Tune thresholds:

```bash
modosaic run \
  --dataset local \
  --root ./images \
  --text-siglip-min 0.65 \
  --segmentation-mask-quality-min 0.70 \
  --depth-segmentation-boundary-min 0.15 \
  --normals-field-quality-min 0.50
```

---

## Config Files

Config files can be JSON, TOML, YAML, or YML. YAML requires PyYAML.

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

run:
  limit: 20
  experiment_root: ./experiments
  experiment_name: local-seg-depth
  log_path: ./.logs
  seed: 42
```

Run and override selected values:

```bash
modosaic pipeline examples/config.yaml --limit 5 --seed 123 --json
```

---

## Experiment Artifacts

Accepted modality outputs are saved under the experiment folder:

```text
experiments/<run>/
  image/
  text/
  segmentation/
  depth/
  normals/
  validations/
```

Validation JSON stores the validator name, raw value, optional score, threshold,
and pass/fail result.

---

## Demo Script

`examples/main.py` is a complete demo script. It loads `data/imagenet-a` as a parquet
dataset, runs all preconfigured modalities, and prints validation summaries:

```bash
python examples/main.py
```

Use it as a starting point when you want direct Python control instead of CLI
configuration.
