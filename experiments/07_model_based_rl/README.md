# Experiment 07: Learned dynamics with MPC

## Objective and prerequisites

Use an offline learned model to choose actions in a simple environment.

Prerequisites: MDP, costs, planning; chapter 7.

## Dataset and architecture

96 random training trajectories, 24 held-out prediction trajectories; 12 held-out planning starts/goals per seed.

4-state residual MLP conditioned on 2 acceleration actions; CEM horizon12, candidates128, elites16, iterations3.

## Mathematics and training configuration

v_next=.9*v+.1*a; p_next=p+.1*v_next; CEM minimizes summed position error+.01 action².

250 supervised updates, LR .002; each plan executes one action then observes; 40 control steps.

## Run and inference

From the repository root:

```sh
uv run wm-lab --experiment 07 --seeds 0 1 2 --steps 250
```

conditioned.pt and unconditioned.pt; metrics and loss JSON.

See [the shared experiment implementation](../../src/world_models/lab.py). Source components are reusable under data/encoders/dynamics/losses/planning. A saved checkpoint is a weights snapshot; instantiate the same architecture before loading.

## Baselines, metrics, and reproducibility

Random actions, zero actions, PD controller; action-blind MPC diagnostic.

Success within .10 final distance, mean final distance, cumulative distance reward, rollout error.

Random seeds, independent train/test episodes where applicable, library versions, args, and evaluation definitions are stored in every run. Results are small educational measurements. Three seeds do not establish broad statistical significance. Experiments sharing code/data are not independent confirmations.

## Expected and actual outcomes

**Expectation:** Expect action-conditioned planning to beat random; PD may match or beat learned MPC.

**Actual:** consult [verification](../../research/VERIFICATION.md) and linked per-seed metric files; no official-weight or external-benchmark result is implied.

## Failure analysis and improvements

Offline model-based control, not online Dreamer/actor-critic. Linear fully observed physics; add obstacles, hidden state and model uncertainty.

