# World models: from intuition to a research laboratory

Study the [interactive course website](https://sajal-madan-ds.github.io/world-models-research/) with chapter navigation, search, rendered equations and diagrams, and browser-local completion tracking. All original Markdown chapters, experiment code, and measured results are retained here. See [website build and publishing instructions](website/README.md) for maintaining the site.

Research snapshot: **October 8, 2026**, not the whole month. Start with [the guide](docs/01_world_models_eli5.md), follow [the roadmap](docs/12_learning_roadmap.md), and inspect [verification](research/VERIFICATION.md) before treating an example as a reproduced result.

## Read

1. [World-model intuition and prerequisite graph](docs/01_world_models_eli5.md)
2. [Computer vision refresher](docs/02_opencv_foundations.md)
3. [Representation and predictive learning](docs/03_representation_learning.md)
4. [Functional and architectural taxonomy](docs/04_world_model_taxonomy.md)
5. [History and architectures](docs/05_architectures.md)
6. [JEPA and value-guided planning](docs/06_jepa_deep_dive.md)
7. [RL, physics, planning, and control](docs/07_rl_and_planning.md)
8. [Models and datasets](docs/08_models_and_datasets.md)
9. [Training, inference, hardware, and evaluation](docs/09_training_and_inference.md)
10. [Applications](docs/10_use_cases.md)
11. [Research opportunities and proposals](docs/11_research_directions.md)
12. [Twelve-week roadmap](docs/12_learning_roadmap.md)

[Glossary and FAQ](docs/glossary.md) · [Source index](docs/references.md) · [Progress](research/PROGRESS.md)

## Run

```sh
uv sync --python 3.12 --extra dev
uv run pytest
uv run wm-lab --experiment all --seeds 0 1 2
```

Outputs go to a fresh timestamped directory under `results/` and include configurations, JSON metrics, plots, and learned model checkpoints. Default device is CPU; use `--device mps` or `--device cuda` explicitly. Small educational experiments use generated data and require no dataset download. The official DreamerV3 repository uses JAX; this PyTorch lab teaches model-based planning without pretending to reproduce Dreamer.

For local pretrained weights and an existing video:

```sh
uv sync --extra dev --extra pretrained
uv run wm-vjepa --model /absolute/path/to/local/checkpoint --video /absolute/path/to/video.mp4
```

Pretrained loading defaults to local files only. These weights have not been downloaded or executed here. See the [inference guide](docs/09_training_and_inference.md) for shape checks, disk inspection, and the distinction between a representation encoder and a robot policy.

## Evidence conventions

**Published**: reported by an inspected paper, not independently replicated. **Official documentation**: repository/model-card statement. **Measured**: an artifact produced by this lab. **Estimate**: a compute or feasibility judgment. **Hypothesis**: an experiment to test. **Unknown**: information not established from inspected sources. A successful smoke test establishes execution, not scientific validity or generalization.

The toy value-guided study is an explicitly altered adaptation, not an official reproduction. No result is implied to hold on real robot data.
