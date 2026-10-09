# Experiment 04: Reconstruction-assisted latent dynamics

## Objective and prerequisites

Train a visual encoder and predictor and inspect whether the code preserves state.

Prerequisites: Autoencoder, history and velocity; chapters 2–3.

## Dataset and architecture

64 training /24 test ball episodes, length 28; Gaussian 16×16 images, ordered two-frame observations.

Two strided CNN layers → 16-vector; small MLP decoder; residual latent transition.

## Mathematics and training configuration

L=reconstruction(current)+reconstruction(next)+.1||P(z)-stopgrad(z_next)||².

250 updates, batch128, AdamW LR .002; 12-step latent rollout; final encoders frozen for evaluation.

## Run and inference

From the repository root:

```sh
uv run wm-lab --experiment 04 --seeds 0 1 2 --steps 250
```

model.pt stores encoder/predictor/decoder; latent_rollout.png and loss/metric JSON.

See [the shared experiment implementation](../../src/world_models/lab.py). Source components are reusable under data/encoders/dynamics/losses/planning. A saved checkpoint is a weights snapshot; instantiate the same architecture before loading.

## Baselines, metrics, and reproducibility

Latent persistence and train-mean physical-state baseline; frozen ridge linear probe fitted only on train codes.

Raw/normalized latent MSE, state probe MSE, feature std, effective rank, horizon errors.

Random seeds, independent train/test episodes where applicable, library versions, args, and evaluation definitions are stored in every run. Results are small educational measurements. Three seeds do not establish broad statistical significance. Experiments sharing code/data are not independent confirmations.

## Expected and actual outcomes

**Expectation:** Reconstruction may produce a code that omits velocity or partially collapses; improvement is not guaranteed.

**Actual:** consult [verification](../../research/VERIFICATION.md) and linked per-seed metric files; no official-weight or external-benchmark result is implied.

## Failure analysis and improvements

Seed 0 in the measured pilot has low effective rank and does not beat latent persistence. Try weighted foreground reconstruction, sequence memory, and state probes.

