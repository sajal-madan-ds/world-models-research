# Experiment 01: OpenCV motion tracking

## Objective and prerequisites

Locate a colored moving object and estimate pixel velocity and LK optical flow.

Prerequisites: NumPy arrays, BGR/RGB, moments; chapter 2.

## Dataset and architecture

30 generated 96×96 frames, red circle moving 2 pixels/frame; assumed 30 FPS. No external download.

Color mask → moments centroid; sparse corners → LK flow.

## Mathematics and training configuration

v=(p[t]-p[t-1])*FPS; optical flow estimates frame displacement.

No neural training. Uses deterministic frame generator; repeated seeds do not create independent data.

## Run and inference

From the repository root:

```sh
uv run wm-lab --experiment 01 --seeds 0 1 2 --steps 250
```

tracking.npz contains truth/estimates/velocity/flow; first_frame.png.

See [the shared experiment implementation](../../src/world_models/lab.py). Source components are reusable under data/encoders/dynamics/losses/planning. A saved checkpoint is a weights snapshot; instantiate the same architecture before loading.

## Baselines, metrics, and reproducibility

Truth centroid and known velocity are diagnostic references; threshold tracking compared with LK flow.

Centroid RMSE in pixels, velocity pixels/second, mean LK flow.

Random seeds, independent train/test episodes where applicable, library versions, args, and evaluation definitions are stored in every run. Results are small educational measurements. Three seeds do not establish broad statistical significance. Experiments sharing code/data are not independent confirmations.

## Expected and actual outcomes

**Expectation:** Expect near-exact tracking in this controlled scene; no claim about occlusion/general scenes.

**Actual:** consult [verification](../../research/VERIFICATION.md) and linked per-seed metric files; no official-weight or external-benchmark result is implied.

## Failure analysis and improvements

Change object/background colors, add camera movement and missing detections. Add Kalman tracking and identity association.

