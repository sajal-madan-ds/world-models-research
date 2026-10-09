# Verification report

Snapshot: **2026-10-08**. This is a first educational/research edition with runnable small experiments. It is not an exhaustive literature review, full-scale pretrained-model reproduction, or independently validated robotics system.

## What was researched

All twelve supplied links were opened with live browsing. The value-guided JEPA paper's full HTML, equations, results and appendix were inspected. Foundational paper records, official repositories, model cards, dataset pages, 2026 discoveries, and supplementary discussion were cross-checked where accessible. [Source index](../docs/references.md) records the level of inspection; [catalog](../docs/08_models_and_datasets.md) retains unknown fields explicitly.

The date cutoff is October 8, not the end of October 2026. Original submission dates and later releases differ. Important examples: 2601.00844v1 is recorded December 28, 2025; V-JEPA's arXiv page records February 15, 2024 despite its 2404 identifier; Dreamer has a later April 2025 Nature publication separate from its 2023 preprint.

LeWorldModel HTML initially failed, but the primary PDF v3 was subsequently accessible and inspected. Cosmos 3 full-paper fetching remained unsuccessful; its search record stays provisional. A supplied Something-Something URL redirects to a provider index that does not establish full dataset metadata. No arXiv-author search was treated as evidence that every paper was read.

## Artifacts created

- Twelve interconnected chapters with progressive intuition, mathematical explanations, diagrams, examples and checks.
- Glossary/FAQ, reference index, resource catalog, fifteen investigated hypotheses, five ranked directions, three proposals and a twelve-week plan.
- PyTorch package with generated data, CNN encoders, transitions, losses, teacher updates, CEM planning, baseline policies and evaluation.
- Nine experiment recipes; eight small recipes executed; optional pretrained encoder adapter.
- Locked dependencies, tests, configuration recipe, saved inference script, plots, per-seed JSON metrics and checkpoints.

Breadth covers the requested outline, but not every term has a separate full thirteen-part teaching treatment. Architecture coverage is deeper for the core JEPA/control families than for newly discovered 2026 systems. Several catalog entries are discovery records rather than fully verified download-ready model/dataset cards. These depth gaps remain explicit.

## Environment and dependency verification

Host reports macOS arm64. Initial filesystem check found about 42 GiB free. RAM query was sandbox-denied; no memory capacity is assumed. Python 3.12.11 was selected rather than default Python 3.14.

Dependency resolution and installation succeeded after network escalation for registry access. The committed uv.lock records the resolved environment. Important executed versions: PyTorch 2.8.0, NumPy 2.5.3, OpenCV 4.14.0.94, Transformers 4.57.6, torchvision 0.23.0. Optional Gymnasium workflows were not executed.

All scientific runs here used CPU, two PyTorch threads, deterministic-algorithm requests with warnings allowed, and explicit seeds. No L40S/H100, MPS benchmark, distributed training, paid API, robot actuation, external publication, or multi-gigabyte checkpoint download occurred.

## Executed checks

Commands:
```sh
uv sync --python 3.12 --extra dev
uv sync --extra dev --extra pretrained
uv run --extra pretrained --extra dev pytest -q
uv run --extra dev ruff check src tests scripts
uv run --extra dev ruff format --check src tests scripts
uv run python scripts/check_docs.py
uv run wm-lab --experiment all --seeds 0 --steps 30
uv run wm-lab --experiment all --seeds 0 1 2 --steps 250
uv run wm-lab --experiment 05 --seeds 0 1 2 --steps 250
uv run python scripts/infer_toy.py --checkpoint results/20261008T074315248466Z/02_seed0/transition.pt --horizon 2
```

Tests cover reflection and speed preservation; seeded data/history; different futures under different actions; expectile weights and stopped Bellman targets; EMA arithmetic/no teacher gradients; collapse detection; supervised action-effect generalization; CEM bounds/rollout shapes; tiny official Transformers V-JEPA processor/encoder output.

The first tiny V-JEPA test exposed an implementation edge case at very small rotary-attention head dimensions. Using a valid larger tiny configuration resolved it; this is not evidence of a bug affecting real checkpoints. Final suite: **9 passed**. Core-only environments skip the optional API test if Transformers is absent.

Documentation checks validate local links, fences, forbidden control characters, and Python snippet syntax. They do not prove every snippet is self-contained or execute it; snippets using variables such as model/inputs/calibration correspondences need the surrounding setup.

## Measured runs and interpretation

[Smoke run](../results/20261008T073917876882Z/summary.json): eight recipes, seed 0, 30 updates. Execution check only.

[Main run](../results/20261008T074315248466Z/summary.json): eight recipes × three seeds, 250 updates (value encoder uses 4× updates). Summed per-experiment wall times about 48.9s. This is measured local suite time, not a GPU estimate or standardized inference benchmark.

[Corrected mini-JEPA run](../results/20261008T083404498291Z/summary.json): persistence baseline now compares current and next features from the same frozen teacher. Use this run for experiment 05 comparisons. The original main-run persistence calculation mixed student current features with teacher future targets, including teacher-lag error; it is superseded.

Experiments 02 and 03 share one dynamics pipeline. Experiments 07 and 08 share one control/ablation pipeline. Do not count duplicated recipes as independent evidence.

| Recipe | Measured result | Scope |
|---|---|---|
| 01 tracking | Centroid RMSE 0; velocity (60,0) pixels/s; LK approximately (2,0) pixels/frame | One generated colored object, stationary camera |
| 02/03 ball, seeds 0/1/2 | Neural one-step MSE .01117/.01245/.01102; 24-step rollout state MSE .06817/.08619/.06484 | Rollout error substantially exceeds local error |
| Ball constant-velocity baseline | Rollout MSE .18437/.29083/.24579; analytical baseline zero | Exact analytical laws match generator by construction |
| 04 reconstruction latents | Effective rank 1.34/1.29/1.48 of 16; state probe MSE .07905/.09866/.08922 | Partial low-rank behavior; low raw latent error is not strong state encoding |
| 05 corrected mini-JEPA | Prediction MSE .004878/.002285/.002425 versus teacher persistence .005418/.002478/.002419 | Modest improvement in two seeds; slightly worse in third |
| 05 physical-state probes | MSE .08734/.09930/.08209 | Does not consistently beat reconstruction probes |
| 07/08 action dynamics | Conditioned 24-step MSE .000194/.000115/.000199 versus blind .01822/.01358/.01546 | Simple randomly controlled linear point world |
| 07/08 planning | Conditioned MPC and PD each succeed on 12/12 goals per seed; blind MPC 0/12 | Small easy problem; learned MPC not necessary to achieve success |
| 09 goal geometry | Learned and shortest-path oracle 80/80 each seed; Euclidean .3875/.475/.375 success | Known transitions, fixed wall, within-support goal pairs |

These are pilot measurements, not externally reproduced benchmark numbers. Goal sets differ by seed; 36 control episodes and 240 grid goal pairs do not establish real-world reliability. The grid predictor training MSE is explicitly training-only; its model is unused by the evaluation planner.

A ball trajectory plot was visually inspected and matched the metric interpretation: constant velocity leaves the box, analytical predictions follow truth, and the neural model drifts/smooths the bounce. Other plots are generated artifacts, not independently adjudicated benchmark evidence.

## Not executed or not established

- Official pretrained V-JEPA weights on real video. Only a tiny **randomly initialized official API** was run; the real inference adapter remains ready for local weights/video.
- Original value-guided paper wall/maze/MPPI/quasimetric experiments. Our grid adaptation changes observations, dynamics use, layouts, architecture, coverage, and planner.
- Dreamer/PlaNet/Genie/Cosmos/Marble training or external benchmark reproduction.
- Full original-paper/code inspection for every architecture and all 2026 additions.
- Complete parameter counts, exact checkpoint revisions, dataset storage, action schemas and license terms for every catalog entry.
- Formal novelty certification for proposals; the search found overlap, and ideas remain hypotheses.
- Tests of real robot safety, sim-to-real transfer, uncertainty calibration, multiple physical regimes or extensive statistical significance.
- Full per-concept thirteen-step expansion, comprehensive standalone implementations of SLAM/stereo/robot dynamics, or every model's equations and original ablations.
- Exact training resume: current checkpoints save weights, not optimizer/sampler/RNG state.

## Next research gates

1. Obtain a suitable checkpoint/video with explicit download authorization if acquisition is needed; run and report real encoder features/latency/memory.
2. Pin official source commits and complete licensing/storage/schema records for the chosen dataset.
3. Match the selected paper's data, architecture and planner before claiming reproduction.
4. Choose one proposal, preregister splits/baselines/budget and run the smallest falsifying experiment.

The durable artifacts support these next steps without implying they have already happened.

## Public course publication — October 9, 2026

- Public source repository: https://github.com/sajal-madan-ds/world-models-research
- Course website: https://sajal-madan-ds.github.io/world-models-research/
- Original Markdown remains authoritative; MkDocs renders it through a build hook.
  The generated site is published separately on `gh-pages`. See `website/README.md`.
- Strict website build passed; documentation link/code checks passed; PyTorch tests:
  **9 passed**. Browser checks visited all **12 chapters**, found **123 rendered
  equations** and **9 rendered diagrams**, and passed search, persistent completion,
  and mobile-homepage overflow checks with **zero JavaScript runtime errors**.
- GitHub reports Pages built and public; the live homepage returned **HTTP 200**.
- This is not a full WCAG audit or exhaustive cross-browser check. No new model
  training, checkpoint download, or original-paper reproduction was performed
  during publication. Completion state is local to each browser.
