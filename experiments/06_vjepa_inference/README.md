# Experiment 06: Official V-JEPA inference

## Objective and prerequisites

Extract video features using an existing official HF-format local checkpoint.

Prerequisites: Video shapes, processor normalization, optional dependency install; chapters 5 and 9.

## Dataset and architecture

User-provided local real video and already acquired checkpoint; no weights/data downloaded by lab.

Official Transformers V-JEPA 2 encoder; skip predictor; mean-pool token features.

## Mathematics and training configuration

z=E(video); half-clip comparison uses cosine similarity, not a validated control metric.

No training. Official preprocessor/config; sample configured frame count. Random tiny model test verifies API only.

## Run and inference

From the repository root:

```sh
uv run wm-vjepa --model /path/to/local/checkpoint --video /path/to/video.mp4
```

tokens and pooled arrays, frame indices and JSON metadata.

See [the adapter](../../src/world_models/inference/vjepa.py). It defaults to local-only loading.

## Baselines, metrics, and reproducibility

Exploratory half-clip similarity; add nearest-neighbor or labeled action probe for a real baseline.

Output shape, synchronized one-clip latency; CUDA peak allocation if CUDA; half cosine similarity.

Random seeds, independent train/test episodes where applicable, library versions, args, and evaluation definitions are stored in every run. Results are small educational measurements. Three seeds do not establish broad statistical significance. Experiments sharing code/data are not independent confirmations.

## Expected and actual outcomes

**Expectation:** Official-weight outcomes unmeasured. Do not infer real checkpoint performance from the passing tiny random test.

**Actual:** consult [verification](../../research/VERIFICATION.md) and linked per-seed metric files; no official-weight or external-benchmark result is implied.

## Failure analysis and improvements

Pending local weights/video. Validate processor, temporal sampling, memory and repeated latency before extending to downstream tasks.

