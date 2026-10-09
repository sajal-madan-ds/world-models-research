# Experiment 08: Action conditioning ablation

## Objective and prerequisites

Test the effect of including commands on prediction and planning.

Prerequisites: Transition/action semantics; chapter 7 and experiment07.

## Dataset and architecture

Same random-control episode split and planning starts/goals as07.

Residual MLPs with/without action inputs; same hidden width and update count, slightly different parameter counts.

## Mathematics and training configuration

Conditional next-state regression; action-blind model approximates average outcome.

250 updates; shared experimental pipeline with07; independent seeds 0/1/2.

## Run and inference

From the repository root:

```sh
uv run wm-lab --experiment 08 --seeds 0 1 2 --steps 250
```

Saved state_dict models and full metrics support fresh inference.

See [the shared experiment implementation](../../src/world_models/lab.py). Source components are reusable under data/encoders/dynamics/losses/planning. A saved checkpoint is a weights snapshot; instantiate the same architecture before loading.

## Baselines, metrics, and reproducibility

Conditioned versus action-blind; PD/random/zero plans.

24-step MSE, displacement, control success/return with both models.

Random seeds, independent train/test episodes where applicable, library versions, args, and evaluation definitions are stored in every run. Results are small educational measurements. Three seeds do not establish broad statistical significance. Experiments sharing code/data are not independent confirmations.

## Expected and actual outcomes

**Expectation:** In random action data, unconditioned dynamics cannot identify command effects; measured pilot demonstrates this scoped effect.

**Actual:** consult [verification](../../research/VERIFICATION.md) and linked per-seed metric files; no official-weight or external-benchmark result is implied.

## Failure analysis and improvements

Add dummy-action capacity match, shuffled actions, restricted action coverage, time offsets; do not generalize toy success to manipulation.

