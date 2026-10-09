# Experiment 02: Bouncing-ball comparison

## Objective and prerequisites

Compare constant velocity, exact reflection physics, and a learned transition.

Prerequisites: State/observation distinction; finite differences; chapter 1 and 7.

## Dataset and architecture

96 training and 24 independent test episodes, length 40; states x,y,vx,vy, dt=.08.

64-wide two-hidden-layer residual MLP; point mass reflects elastically in [0,1]^2.

## Mathematics and training configuration

raw=x+dt*v; triangular-wave reflection; supervised MSE for next state.

250 updates, batches up to 128, AdamW LR .002; 24-step free rollout; seeds 0/1/2.

## Run and inference

From the repository root:

```sh
uv run wm-lab --experiment 02 --seeds 0 1 2 --steps 250
```

transition.pt, training_loss.json, trajectories.png, metrics.json.

See [the shared experiment implementation](../../src/world_models/lab.py). Source components are reusable under data/encoders/dynamics/losses/planning. A saved checkpoint is a weights snapshot; instantiate the same architecture before loading.

## Baselines, metrics, and reproducibility

Constant velocity ignores boundaries; analytical dynamics use privileged known physics.

One-step state MSE, horizon MSE, average/final position displacement.

Random seeds, independent train/test episodes where applicable, library versions, args, and evaluation definitions are stored in every run. Results are small educational measurements. Three seeds do not establish broad statistical significance. Experiments sharing code/data are not independent confirmations.

## Expected and actual outcomes

**Expectation:** Expect analytical zero error because truth uses the same laws; neural gains over constant velocity are empirical, not guaranteed.

**Actual:** consult [verification](../../research/VERIFICATION.md) and linked per-seed metric files; no official-weight or external-benchmark result is implied.

## Failure analysis and improvements

MLP may smooth bounce discontinuities; add collision-aware features, multi-step loss, unseen speeds. Experiment 03 reuses this implementation.

