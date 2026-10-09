# Learn by experimenting

The lab moves from visible motion to learned representations and planning. Every recipe includes its objective, data, architecture, baselines, metrics, and limitations.

## Start locally

```sh
git clone https://github.com/sajal-madan-ds/world-models-research.git
cd world-models-research
uv sync --python 3.12 --extra dev
uv run pytest -q
uv run wm-lab --experiment 02 --seeds 0 1 2 --steps 250
```

The small experiments use generated data. Each run creates a fresh results directory with metrics, plots, and checkpoint files. See [local setup](lab.md) for the full instructions.

| Step | Experiment | What you learn |
|---|---|---|
| 1 | [OpenCV motion tracking](experiments/01_opencv_motion/README.md) | Position, velocity, and optical flow |
| 2 | [Bouncing-ball comparison](experiments/02_bouncing_ball/README.md) | Known physics versus constant velocity versus learned dynamics |
| 3 | [Neural dynamics](experiments/03_neural_dynamics/README.md) | One-step error versus free-rollout drift |
| 4 | [Latent world model](experiments/04_latent_world_model/README.md) | Visual encoding, reconstruction, and state probes |
| 5 | [Mini-JEPA](experiments/05_mini_jepa/README.md) | Stopped targets, EMA, and collapse diagnostics |
| 6 | [Pretrained V-JEPA](experiments/06_vjepa_inference/README.md) | Official feature extraction with an existing local checkpoint |
| 7 | [Learned dynamics and MPC](experiments/07_model_based_rl/README.md) | Predictions as a tool for choosing actions |
| 8 | [Action-conditioning ablation](experiments/08_action_conditioned_model/README.md) | Whether commands improve forecasts and plans |
| 9 | [Value-guided adaptation](experiments/09_research_replication/README.md) | Goal geometry and responsible reproduction boundaries |

Eight recipes have local three-seed measurements. Recipes 02/03 and 07/08 reuse shared pipelines, so they are not independent scientific confirmations. Recipe 06 needs real local weights and a video; the completed API test used a tiny random model.

[Explore the measured results](measured.md){ .md-button }
