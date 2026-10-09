# Experiment 09: Value-guided educational adaptation

## Objective and prerequisites

Isolate whether learned goal-distance geometry helps routes around a wall.

Prerequisites: Expectile loss, Bellman target, embeddings; chapter6.

## Dataset and architecture

Fixed 7×7 grid, wall at x=3 except door y=5; all state/actions available for sampling; 80 held-out goal pairs.

One-hot position → 64-wide MLP → 8-code; separate latent predictor with one-hot actions.

## Mathematics and training configuration

V(s,g)=-||E(s)-E(g)||₂; asymmetric reaching-cost Bellman loss with absorbing goals.

1000 encoder updates for --steps250, batch256, Adam LR .004; gamma.98/tau.8; predictor250 updates.

## Run and inference

From the repository root:

```sh
uv run wm-lab --experiment 09 --seeds 0 1 2 --steps 250
```

model.pt, value_geometry.png, loss and metrics JSON.

See [the shared experiment implementation](../../src/world_models/lab.py). Source components are reusable under data/encoders/dynamics/losses/planning. A saved checkpoint is a weights snapshot; instantiate the same architecture before loading.

## Baselines, metrics, and reproducibility

Euclidean greedy and shortest-path oracle; known transitions supplied to all planners.

Success within40 steps; mean steps; latent prediction TRAINING error (not generalization).

Random seeds, independent train/test episodes where applicable, library versions, args, and evaluation definitions are stored in every run. Results are small educational measurements. Three seeds do not establish broad statistical significance. Experiments sharing code/data are not independent confirmations.

## Expected and actual outcomes

**Expectation:** Expected geometry improvement is testable; pilot succeeded on this fixed support but does not reproduce the paper.

**Actual:** consult [verification](../../research/VERIFICATION.md) and linked per-seed metric files; no official-weight or external-benchmark result is implied.

## Failure analysis and improvements

Known-transition greedy planning; no image/maze/MPPI/quasimetric/held-out layouts. See change matrix below; next implement each missing official component.

## Difference from 2601.00844v1

| Component | Paper | This adaptation |
|---|---|---|
| Observation | 64×64 image, maze velocity | one-hot discrete cell |
| Layout/data | random wall/maze trajectories | one fixed wall, direct transition samples |
| Encoder | residual CNN, 512-code | MLP, 8-code |
| Value | Euclidean or quasimetric variants | Euclidean only |
| Dynamics | action-conditioned learned prediction | fitted but unused by planning |
| Planner | MPC/MPPI, long horizon | greedy with known one-step transitions |
| Generalization | defined held-out environments | new goal pairs within training state support |

[Primary paper](https://arxiv.org/html/2601.00844v1). Do not publish these results as a reproduction of its table.

