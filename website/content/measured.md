# What the small experiments actually did

These are CPU pilot measurements on generated worlds, not real-robot or official model benchmarks. The full record is in the [verification report](research/VERIFICATION.md).

## A good local predictor can still drift

![Known physics follows the true bouncing-ball path. Constant velocity leaves the box, while the neural predictor drifts. The second plot shows error growing with forecast horizon.](assets/plots/ball-trajectories.png)

The analytical baseline matches the generator exactly. The neural MLP beats constant velocity over the evaluated rollouts, but its multi-step error is much larger than its one-step error. This is the distinction between fitting observed transitions and sustaining an imagined trajectory.

[Read the bouncing-ball recipe](experiments/02_bouncing_ball/README.md).

## Low latent loss is not enough

![Reconstruction-trained latent rollout error across twelve prediction steps.](assets/plots/latent-rollout.png)

The reconstruction-assisted model uses a 16-dimensional code but has effective rank around 1.3–1.5 in the pilot. The mini-JEPA maintains more feature diversity, yet does not consistently produce a better physical-state probe.

![Mini-JEPA latent rollout error across twelve prediction steps, evaluated against a frozen teacher.](assets/plots/jepa-rollout.png)

The corrected persistence comparison gives only modest mini-JEPA gains in two seeds and a slight loss in one. A low loss does not prove physical understanding. [Study the JEPA chapter](docs/06_jepa_deep_dive.md).

## Actions change what a model can predict

In the simple controlled point world, conditioned models achieve 24-step state MSE around 0.0001–0.0002, versus 0.0136–0.0182 for action-blind models. Both learned MPC and a simple PD controller succeed on every tested goal. This supports the action-conditioning lesson; it does not establish that a learned planner is necessary.

## Goal geometry can change a route

![Learned goal-distance values over cells of a fixed wall grid with an off-center door.](assets/plots/value-geometry.png)

Learned value distance and a shortest-path oracle reach all 80 tested goals per seed; Euclidean greedy planning reaches 37.5–47.5%. This experiment uses **known transitions in one fixed grid**. It is an educational adaptation, not the paper's image-based JEPA/MPPI reproduction.

[Inspect the exact differences](experiments/09_research_replication/README.md).
