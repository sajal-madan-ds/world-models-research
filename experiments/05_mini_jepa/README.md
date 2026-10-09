# Experiment 05: Mini-JEPA mechanics

## Objective and prerequisites

Learn future embeddings with stopped targets and diagnose representation collapse.

Prerequisites: Encoder/predictor, EMA and covariance; chapters 3 and 6.

## Dataset and architecture

Same rendered two-frame ball observations and independent episode split as 04.

Tiny CNN + latent predictor + EMA teacher; explicit variance/covariance penalty. No decoder, ViT, or block masks.

## Mathematics and training configuration

L=prediction MSE+5*variance_penalty+.1*covariance_penalty; teacher=.99*teacher+.01*student.

250 updates, batch128, AdamW LR .002; seeds 0/1/2; teacher target frozen at evaluation.

## Run and inference

From the repository root:

```sh
uv run wm-lab --experiment 05 --seeds 0 1 2 --steps 250
```

model.pt contains student/teacher/predictor; predictor supports iterative code forecasts.

See [the shared experiment implementation](../../src/world_models/lab.py). Source components are reusable under data/encoders/dynamics/losses/planning. A saved checkpoint is a weights snapshot; instantiate the same architecture before loading.

## Baselines, metrics, and reproducibility

Persistence, train-mean state, linear probe; compare 04 separately without treating raw latent losses as scale matched.

Feature spread/effective rank, normalized loss, state probe, free-rollout error.

Random seeds, independent train/test episodes where applicable, library versions, args, and evaluation definitions are stored in every run. Results are small educational measurements. Three seeds do not establish broad statistical significance. Experiments sharing code/data are not independent confirmations.

## Expected and actual outcomes

**Expectation:** Expect spread with regularization; useful physics remains an empirical question.

**Actual:** consult [verification](../../research/VERIFICATION.md) and linked per-seed metric files; no official-weight or external-benchmark result is implied.

## Failure analysis and improvements

Not official I/V-JEPA. Mini-JEPA does not necessarily outperform reconstruction on state probes. Ablate EMA/variance/covariance and maintain a test set.

