# Experiment 03: One-step versus free-rollout neural dynamics

## Objective and prerequisites

Understand why teacher-forced prediction can be good while imagined trajectories drift.

Prerequisites: PyTorch training, MSE, recursion; chapter 3.

## Dataset and architecture

Same episode-level ball split as experiment 02.

Shared residual MLP transition. This is a reused controlled comparison, not independent evidence.

## Mathematics and training configuration

Local loss E||F(s_t)-s_{t+1}||²; rollout recursively applies F.

Same configuration as 02 so the learning comparison remains reproducible.

## Run and inference

From the repository root:

```sh
uv run wm-lab --experiment 03 --seeds 0 1 2 --steps 250
```

transition.pt supports inference from a 4-vector; plot contrasts three trajectories.

See [the shared experiment implementation](../../src/world_models/lab.py). Source components are reusable under data/encoders/dynamics/losses/planning. A saved checkpoint is a weights snapshot; instantiate the same architecture before loading.

## Baselines, metrics, and reproducibility

Constant velocity and exact physics.

One-step MSE versus MSE by free-rollout horizon; displacement metrics.

Random seeds, independent train/test episodes where applicable, library versions, args, and evaluation definitions are stored in every run. Results are small educational measurements. Three seeds do not establish broad statistical significance. Experiments sharing code/data are not independent confirmations.

## Expected and actual outcomes

**Expectation:** Expect local errors and rollout errors to differ; compare measured values.

**Actual:** consult [verification](../../research/VERIFICATION.md) and linked per-seed metric files; no official-weight or external-benchmark result is implied.

## Failure analysis and improvements

Probe errors near contact, train horizon-specific losses, reserve faster/slower physics regimes.

