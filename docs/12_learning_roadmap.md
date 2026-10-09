# 12. Twelve weeks alongside a full-time job

Previous: [research directions](11_research_directions.md). Companion: [glossary](glossary.md). Budget 6–8 hours/week: two 90-minute weekday sessions and one 3–5-hour weekend block. Expensive training is optional; understanding and measured small experiments come first.

Each week has a deliverable and a gate. If the gate fails, repeat the exercise with a simpler model rather than adding more papers.

| Week | Goal / prerequisites | Read and inspect | Mathematics / exercise | Deliverable / gate / self-check |
|---|---|---|---|---|
| 1 | Intuition and OpenCV; Python arrays only | Chapters 1–2; [World Models](https://arxiv.org/abs/1803.10122); [OpenCV tutorials](https://docs.opencv.org/4.x/d9/df8/tutorial_root.html) | Shapes, normalization, finite-difference velocity; run experiment 01 and alter color/noise | Explain glass loop and write tracking failures. Gate: state versus observation clear. Why is pixel speed not metric speed? 6h |
| 2 | Video motion and basic geometry; week 1 | Chapter 2 camera sections; inspect calibration and optical-flow examples | Projection, disparity, coordinate transforms; manually backproject pixels; run ball analytic baseline | Diagram camera/world axes and plot ball trajectory. Gate: identify hidden velocity/depth. Why does camera motion confuse tracking? 7h |
| 3 | Encoders and neural prediction; PyTorch training | Chapter 3; [VAE](https://arxiv.org/abs/1312.6114); lab encoders/dynamics | MSE, residual prediction, free rollout; run 03/04 and frozen probe | Compare one-step/rollout/persistence errors. Gate: independently recreate a predictor. What causes rollout drift? 7h |
| 4 | Self-supervision and JEPA; week 3 representations | Chapter 6 introduction; [I-JEPA](https://arxiv.org/abs/2301.08243), [VICReg](https://arxiv.org/abs/2105.04906); official I-JEPA repo | Stop-gradient, EMA, covariance; run mini-JEPA with shorter/longer training and disable variance term in a separate branch/copy | Numerical gradient walkthrough and collapse report. Gate: explain why low loss can be bad. Does EMA guarantee non-collapse? 8h |
| 5 | Video feature models; attention basics | [V-JEPA](https://arxiv.org/abs/2404.08471), [V-JEPA 2](https://arxiv.org/abs/2506.09985), [V2.1](https://arxiv.org/abs/2603.14482); official vjepa2 configs | Patch/tubelet token counts, causal versus bidirectional masks; tiny random API test; inspect checkpoint card | Draw architecture with action-free/AC distinction. Gate: predict tensor shapes. Is masked-video training always forecasting? 7h |
| 6 | RL and planning; state/transition knowledge | Chapter 7; Sutton/Barto textbook as an additional reading candidate; inspect lab CEM | MDP/POMDP, return, Bellman backup, finite-horizon cost; run experiment 07 | Write pickup pipeline and compute a return by hand. Gate: policy/value/model distinctions. Why replan? 7h |
| 7 | Dreamer and imagined behavior; week 6 | [Dreamer](https://arxiv.org/abs/1912.01603), [V2](https://arxiv.org/abs/2010.02193), [V3](https://arxiv.org/abs/2301.04104); [official JAX code](https://github.com/danijar/dreamerv3) | Prior/posterior KL, actor/critic returns; trace one replay/imagination cycle in code | Compare PlaNet search to Dreamer behavior learning. Gate: do not label toy MPC “Dreamer.” What is available in imagination? 8h |
| 8 | Real model inference and data selection; preceding video basics | Chapters 8–9; HF V-JEPA docs/card; DROID/BridgeData schemas | Sampling and action alignment; run 06 only with approved/existing checkpoint, otherwise inspect tiny API test | Dataset manifest, license record, sampled clips, feature shapes. Gate: exact action semantics known. Does a semantic label supply motor control? 6h |
| 9 | Compact model training; data audit | Experiment 04/05 recipes; DINO-WM paper/repo | Frozen encoder versus joint adaptation, linear probing, regularization; vary one factor and preserve held-out episodes | Three-seed comparison with plots and saved configs. Gate: compare baselines fairly. Are latent losses scale comparable? 8h |
| 10 | Action conditioning and evaluation; planning implementation | Experiment 08; JEPA-WMs paper and official configs | Equal-condition rollouts, action ablation, bootstrap/binomial uncertainty; inspect offset/jitter effects | Prediction and planning report, including failures. Gate: distinguish observation effects from action effects. Could a predictor ignore actions? 8h |
| 11 | Responsible research adaptation; all prior gates | 2601.00844 full text; chapter 6; experiment 09 | Expectile residual, goal distance, MPPI versus CEM; run grid adaptation and list every change | Reproduction-difference matrix and one additional controlled ablation. Gate: no official replication claim. Why can Euclidean distance mislead? 8h |
| 12 | Original project proposal; reproducibility habits | Chapter 11 plus closest papers/citations for chosen project | Hypothesis, preregistered splits, budget, falsification; perform tiny pilot | 2–4-page proposal, exact configs, baseline results, missing-evidence list. Gate: a failed hypothesis still yields useful evidence. What would falsify your claim? 8h |

## Reading strategy

First read abstract, figures, data, and evaluation. Then trace equations with a symbol list. Inspect code only after you can articulate the input/output contract. Choose one architecture per week; do not attempt full-scale pretraining while learning foundations.

For advanced papers, keep a three-column notebook: author claim, inspected evidence, independent test you would run. Mark preprints and provider demos distinctly.

## Final portfolio

By week 12 you should have a tracking example, dynamics/prediction plots, an encoder/probe comparison, a JEPA toy study, an action/planning ablation, a documented paper adaptation, a real-checkpoint inference record if resources allowed, and a falsifiable research proposal.

A real-checkpoint run is a separate gate; a tiny random API test cannot replace it. If weights are unavailable, finish all other deliverables and retain the exact pending inference recipe.

