# 11. Research opportunities and three feasible proposals

Previous: [applications](10_use_cases.md). Next: [roadmap](12_learning_roadmap.md).

These are investigated **hypotheses**, not declarations of novelty. The live search found significant overlapping 2026 work. In particular, temporal-cost learning already appears in [TD-JEPA](https://arxiv.org/abs/2607.25337); rollout/physics consistency in [Semigroup-JEPA](https://arxiv.org/abs/2609.10464); compact prefixes in [Adaptive Latent Capacity](https://arxiv.org/abs/2609.32921); video-action integration in [V-JEPA Policy](https://arxiv.org/abs/2609.37250) and [OSCAR](https://arxiv.org/abs/2606.04463). Search records for some of these are not full-paper reviews. Read and rerun the closest work before any novelty claim.

The gaps below identify tests not established by this survey, not proof nobody has done them. Compute estimates are proposed budget caps, not measured training requirements. All experiments should begin with three seeds and independently held-out episodes, then increase seeds/evaluations if the pilot justifies it.

## Fifteen candidate studies

### 1. Value-shaped geometry under stochastic dynamics

**Question/motivation:** does goal-distance shaping remain useful when transitions are random? Closest work: [value-guided JEPA](https://arxiv.org/abs/2601.00844), [TD-JEPA](https://arxiv.org/abs/2607.25337). Gap to investigate: controlled observation/action uncertainty rather than only average deterministic success.

**Hypothesis:** value learning improves reachable-goal ranking at low noise but becomes optimistically biased at high noise. **Experiment/model/data:** extend the wall grid to stochastic actions and slippery regions, compare prediction-only encoder, Euclidean VF, separate learned cost, and oracle stochastic planner. **Ablations:** noise type, \(\tau,\gamma\), coverage, mean versus distributional targets. **Metrics:** success, expected cost, Bellman residual, calibration, failure tails. **Compute:** CPU pilot, optional one GPU, proposed ≤8 GPU-hours. **Risk/falsification:** no consistent noise-dependent effect or equal degradation across objectives rejects the hypothesis. **Contribution:** reproducible robustness benchmark or workshop negative result.

### 2. Partial observability and minimal temporal context

**Question:** how much visual history is necessary? Closest work: V-JEPA 2 and RSSM/PlaNet. **Gap:** controlled context-versus-state-sufficiency curves. **Hypothesis:** ordered short history improves velocity identification more than added encoder width.

**Experiment:** balls with occlusion and visually identical opposing velocities; CNN/GRU histories of 1/2/4/8 frames. **Data:** synthetic first, then small Push-T subset after terms review. **Baseline:** single-frame encoder, explicit tracked state, same-capacity shuffled-history control. **Ablations:** order, frame stride, occlusion length, proprioception. **Metrics:** velocity probe, rollout error, planning success, compute. **Budget:** CPU pilot; ≤10 single-GPU hours proposed. **Risk/falsification:** gains disappear with capacity matching or correlate only with image quality. **Contribution:** clear ablation report and dataset generator; broad idea is established, scope is diagnostic.

### 3. Compact latent capacity under task changes

**Question:** can a small code remain useful across new goals? Closest: DINO-WM, Adaptive Latent Capacity, LeWorldModel. **Gap:** task-change failure of aggressively compressed features. **Hypothesis:** prefixes retain easy position tasks but lose hidden dynamics needed under changed goals.

**Experiment/model:** 8/16/32/64 codes with matched decoder/predictor budget; optional frozen ViT patch pooling. **Data:** synthetic physics, selected simulator trajectories. **Baseline:** full code and PCA compression. **Ablations:** width, prefix training, pooling, task-specific versus general representation. **Metrics:** state probes, unseen-goal planning, rank, latency. **Budget:** CPU to ≤20 single-GPU hours. **Risk/falsification:** all small-code gains explained by regularization or goals preserve performance. **Contribution:** capacity benchmark; not a new claim that compact latents are possible.

### 4. Action timestamps as a hidden source of world-model failure

**Question:** how damaging is action–frame misalignment? Closest: JEPA-WMs and robot action-conditioned models. **Gap:** explicit latency sensitivity and cheap correction evaluation.

**Hypothesis:** small offsets can hurt planning more than one-step image metrics reveal. **Experiment:** introduce offsets/jitter to synthetic control, then approved BridgeData subset. **Model:** frozen encoder + small temporal predictor; compare calibrated offset, learned delay, and uncorrected. **Baselines:** true aligned model, action-blind, state model. **Ablations:** delay, sampling frequency, action interpolation, predictor width. **Metrics:** rollout error, success, inferred delay, latency. **Budget:** CPU, then ≤12 GPU-hours. **Risk/falsification:** comparable control despite offset or correction overfits validation. **Contribution:** alignment diagnostic tool and benchmark.

### 5. Ensemble uncertainty that improves decisions

**Question:** does uncertainty predict model exploitation? Closest: [TD-MPC2](https://arxiv.org/abs/2310.16828), model-based ensemble literature. **Gap:** calibration under structured physics shift in compact latent models.

**Hypothesis:** ensemble penalties reduce severe failures on unseen friction, at a measurable nominal-success cost. **Experiment:** 3–5 predictors with fixed total training budget; deterministic versus stochastic heads; held-out dynamics. **Data:** ball/control/ManiSkill trajectories. **Baseline:** no penalty, fixed conservative penalty, oracle uncertainty. **Ablations:** independent seeds versus bootstrap data, coefficient, training support. **Metrics:** error–uncertainty rank correlation, coverage, success, tail cost. **Budget:** ≤20 GPU-hours after CPU pilot. **Risk/falsification:** confident shared failures or no tail improvement. **Contribution:** calibration study, not universal uncertainty guarantees.

### 6. Multi-step training versus short-horizon accuracy

**Question:** does rollout training help planning beyond fitting one step? Closest: Semigroup-JEPA, PlaNet, TD-JEPA. **Gap:** matched-compute, matched-representation comparison.

**Hypothesis:** multi-step losses improve free rollouts at the cost of some local accuracy. **Experiment:** train 1/4/8/16-step losses on fixed episodes; frozen versus jointly trained encoder. **Baseline:** one-step, scheduled input corruption, analytical model. **Ablations:** target stop-gradient, rollout weighting, horizon curriculum. **Metrics:** horizon error, probe, success, stability. **Data/model:** synthetic history encoder/predictor, then selected Push-T. **Budget:** ≤16 GPU-hours proposed. **Risk/falsification:** improvement vanishes under equal updates/FLOPs. **Contribution:** reproducibility and ablation report; overlap is substantial.

### 7. Object-centered versus patch-centered state

**Question:** do tracked objects reduce long-horizon drift? Closest: dense DINO-WM and object-centric modeling. **Gap:** same-data/control-budget comparison with explicit tracking errors.

**Hypothesis:** object states generalize to rearrangements better but fail under occlusion/association mistakes. **Experiment:** multiple balls/blocks, tracked masks versus dense CNN patches. **Baseline:** pooled code and oracle object states. **Ablations:** tracking noise, object count, relational interactions. **Metrics:** identity switches, interaction errors, success, compute. **Budget:** ≤24 GPU-hours. **Risk/falsification:** oracle benefit disappears or tracker destroys gains. **Contribution:** diagnostic object-interaction dataset.

### 8. Physical consistency without overconstraining representation

**Question:** can a small auxiliary state head improve conservation? Closest: Semigroup-JEPA and physics-informed learning. **Gap:** when conservation penalties conflict with contact/energy loss.

**Hypothesis:** appropriate invariants improve elastic motion while wrong invariants harm dissipative motion. **Experiment:** train state-readout conservation penalty on elastic/inelastic mixed data. **Baseline:** no penalty, oracle supervised states, random auxiliary task. **Ablations:** penalty strength, head capacity, known versus inferred parameters. **Metrics:** energy drift, rollout error, control, unseen physics. **Budget:** CPU pilot / ≤16 GPU-hours. **Risk/falsification:** gains only from extra labels or violated laws in contacts. **Contribution:** explicit boundary-of-applicability study.

### 9. Representation drift during fine-tuning

**Question:** does moving the encoder invalidate a predictor? Closest: V-JEPA adaptation and JEPA-WMs. **Gap:** quantified alignment drift versus control deterioration.

**Hypothesis:** a learned alignment adapter recovers predictor usefulness with fewer updates than joint retraining. **Experiment:** freeze/train/fine-tune phases with recorded embeddings; fit linear/MLP alignment. **Data:** synthetic shifts, approved visual dataset. **Baseline:** frozen encoder, full retrain, no alignment. **Ablations:** teacher momentum, LoRA versus full updates if supported, adapter capacity. **Metrics:** drift, latent prediction, probe, planning, update cost. **Budget:** ≤20 GPU-hours. **Risk/falsification:** adapter cannot preserve task state or gains vanish against retrained predictor. **Contribution:** fine-tuning diagnostic package.

### 10. Planner robustness to latent rescaling

**Question:** are planning comparisons confounded by arbitrary code scale? Closest: value-guided JEPA and latent MPC. **Gap:** optimizer-temperature/cost-scale sensitivity.

**Hypothesis:** identical representations scaled differently alter MPPI/CEM behavior despite unchanged state information. **Experiment:** rescale/whiten fixed codes; tune optimizer separately and compare locked defaults. **Data:** wall and point control; later published checkpoint. **Baseline:** normalized distance, known-state planner. **Ablations:** code scale, covariance anisotropy, horizon, temperature. **Metrics:** success, cost ranking, optimizer convergence. **Budget:** CPU / ≤4 GPU-hours. **Risk/falsification:** properly normalized/tuned optimizers eliminate every effect. **Contribution:** evaluation hygiene/negative-result note.

### 11. Distill video representations for domain-local planning

**Question:** can a tiny student preserve task behavior? Closest: V-JEPA dense features and model distillation. **Gap:** preservation of planning utility rather than classification alone.

**Hypothesis:** trajectory-aware distillation outperforms frame-only matching under a fixed student size. **Experiment:** cache approved teacher features; train CNN student plus predictor. **Data:** short robot/synthetic videos. **Baseline:** random student, frame distillation, teacher planner. **Ablations:** temporal loss, pooling, width, action data. **Metrics:** teacher feature match, rollout, success, latency. **Budget:** ≤24 GPU-hours plus teacher extraction after download approval. **Risk/falsification:** success falls despite feature matching. **Contribution:** efficient task-specific encoder; novelty requires focused search.

### 12. Language-selected costs with shared dynamics

**Question:** can language change goals without destabilizing dynamics? Closest: V-JEPA Policy and world-action models. **Gap:** separation of task instruction from transition prediction.

**Hypothesis:** a shared dynamics model plus language-conditioned cost transfers better than entangling instructions into transitions for fixed physics. **Experiment:** same scenes with “move left/right/avoid red” goals. **Baseline:** separate predictors, goal-vector planner, reactive policy. **Ablations:** instruction paraphrases, irrelevant language, seen/unseen goals. **Metrics:** success, dynamics invariance, language generalization. **Budget:** ≤24 GPU-hours with small existing text encoder. **Risk/falsification:** language merely indexes memorized tasks. **Contribution:** factorization study, not a general language-world-model invention.

### 13. Passive video transfer with a small action-calibration set

**Question:** how much action data grounds a passive encoder? Closest: V-JEPA 2-AC, V-JEPA Policy, OSCAR. **Gap:** controlled data-efficiency curves under viewpoint/embodiment shift.

**Hypothesis:** passive pretraining helps perception but action calibration remains necessary. **Experiment:** freeze encoder and train predictors on 1/5/20/100% action data. **Baseline:** random encoder, supervised state, full fine-tune. **Ablations:** view shift, action range, temporal stride. **Metrics:** sample-efficiency curves, held-out action predictions, success. **Data:** simulator paired with passive renderings; later robot subset. **Budget:** ≤40 GPU-hours. **Risk/falsification:** frozen encoder omits relevant state or benefit disappears with matched pretraining data. **Contribution:** careful transfer evidence.

### 14. Synthetic-to-real failure taxonomy

**Question:** which domain differences actually harm control? Closest: Cosmos/OSCAR and domain randomization. **Gap:** factor-wise attribution instead of aggregate transfer score.

**Hypothesis:** camera/action calibration causes greater failures than texture shift for compact latent control. **Experiment:** vary texture, viewpoint, scale, physics, delay independently. **Baseline:** classical state tracker and oracle state predictor. **Ablations:** single versus combined shifts, adaptation labels. **Metrics:** prediction, state probe, success, calibration. **Data:** approved real subset plus simulator counterparts. **Budget:** ≤40 GPU-hours; real collection not assumed. **Risk/falsification:** factors cannot be independently controlled or texture dominates. **Contribution:** evaluation dataset and diagnostics.

### 15. Prediction under a real-time planning budget

**Question:** when is a larger model worse because it misses the control deadline? Closest: JEPA-WMs search and training-only imagination in LeRobot. **Gap:** matched end-to-end time rather than candidate-count comparison.

**Hypothesis:** compact predictors plus frequent feedback outperform slower, more accurate predictors on rapidly changing tasks. **Experiment:** sweep model size, horizon, candidate count, and delayed actuation in a timed simulator. **Baseline:** PD, fixed-budget MPC, reactive distilled policy. **Ablations:** precomputed features, batching, stale observations, uncertainty. **Metrics:** p50/p95 latency, deadline misses, success, memory. **Budget:** CPU timer pilot / ≤16 GPU-hours. **Risk/falsification:** all gains explained by simulator scheduling or faster models lose too much accuracy. **Contribution:** timing-aware benchmark.

## Ranking

Scores are mentor judgments, not evidence of novelty. F: feasibility, N: potential incremental novelty, S: scientific importance, D: data accessibility, C: compute affordability, I: implementation ease, U: usefulness; 5 is favorable, 1 unfavorable. Totals use equal weighting for transparency.

| # | F | N | S | D | C | I | U | Total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 5 | 3 | 5 | 5 | 5 | 4 | 5 | 32 |
| 2 | 5 | 2 | 4 | 5 | 5 | 5 | 5 | 31 |
| 3 | 4 | 2 | 4 | 5 | 4 | 4 | 4 | 27 |
| 4 | 5 | 3 | 5 | 5 | 5 | 5 | 5 | 33 |
| 5 | 4 | 3 | 5 | 5 | 4 | 3 | 5 | 29 |
| 6 | 5 | 2 | 5 | 5 | 4 | 4 | 5 | 30 |
| 7 | 3 | 3 | 4 | 4 | 3 | 2 | 4 | 23 |
| 8 | 4 | 2 | 4 | 5 | 4 | 3 | 4 | 26 |
| 9 | 4 | 3 | 4 | 4 | 4 | 4 | 5 | 28 |
| 10 | 5 | 3 | 4 | 5 | 5 | 5 | 5 | 32 |
| 11 | 4 | 2 | 4 | 4 | 3 | 3 | 5 | 25 |
| 12 | 3 | 2 | 4 | 4 | 3 | 3 | 4 | 23 |
| 13 | 3 | 2 | 5 | 4 | 2 | 3 | 5 | 24 |
| 14 | 3 | 3 | 5 | 3 | 2 | 2 | 5 | 23 |
| 15 | 5 | 3 | 5 | 5 | 4 | 4 | 5 | 31 |

Five strongest under your current background: **4, 1, 10, 2, 15**. The numerical tie between 2 and 15 favors 2 for a first experiment and 15 for engineering relevance. None requires inventing a foundation model.

## Three concrete projects

### Project A: action–observation alignment benchmark

**Problem:** action timing is often hidden inside preprocessing; incorrect alignment can corrupt a learned transition. **Literature:** JEPA-WMs, V-JEPA 2-AC, and robot datasets establish action-conditioned training but do not establish the exact latency robustness curve we need.

**Hypothesis:** correcting alignment improves downstream planning more than expanding a small predictor. **Implementation:** extend experiment 08 with known timestamp shifts and jitter; add parameterized action delay to data generation; train aligned, shifted, action-blind, and learned-delay models. After synthetic validation, use a licensed BridgeData subset without collecting new robot data.

**Evaluation:** hold episodes/scenes fixed; five training seeds; shifts 0/1/2/4 frames; equal update budget; compare rollout MSE and goal success. Use a dummy-action capacity control. **Resources:** local CPU initial study, optional single L40S/H100; proposed ≤12 GPU-hours, storage constrained to a selected subset.

**Success:** a reproducible offset-effect curve and correction benefit across seeds and held-out scenes. **Failure:** no meaningful effect, or correction only helps training scenes. Either outcome can support a useful tool/report. Deliver generator, diagnostic plots, configs, and a documented data-schema adapter.

### Project B: stochastic value-guided JEPA

**Problem:** deterministic goal-distance success may conceal optimistic bias under uncertain transitions. **Literature:** 2601.00844 explicitly motivates value-shaped codes and discusses stochastic bias; TD-JEPA already studies task-aware temporal costs.

**Hypothesis:** expectile value shaping improves deterministic paths but needs uncertainty-aware targets to avoid low-probability shortcuts. **Implementation:** replace fixed grid transitions with stochastic kernels, retain a known-model oracle, then move from one-hot to rendered observations. Compare standard prediction/VCReg, Euclidean VF, joint VF+prediction, and an uncertainty-aware cost. Do not call it a reproduction until official observation, data, and planner settings are matched.

**Evaluation:** five seeds, 200 fixed start–goal pairs per noise level, separate layouts, success/expected steps/tail risk/calibration. Sweep \(\tau,\gamma\) only on validation. **Resources:** CPU first; proposed ≤8 GPU-hours for CNN extension.

**Success:** controlled evidence of the noise-dependent mechanism plus an intervention that improves expected outcomes. **Failure:** bias does not explain failures or uncertainty correction offers no gain. Deliver a transparent benchmark and scoped conclusions, not a universal JEPA claim.

### Project C: planning geometry under scaling and compute limits

**Problem:** latent cost scale and optimizer settings can distort comparisons. **Literature:** DINO-WM/JEPA-WMs plan in learned feature spaces; value-guided work deliberately shapes geometry.

**Hypothesis:** scale changes affect locked-parameter planners but properly normalized, retuned costs reduce those differences; control delay creates an independent model-size trade-off. **Implementation:** retain fixed representations, rescale/whiten them, compare CEM and a documented MPPI implementation at equal wall-clock budget. Add a simulated actuation delay and measure feature extraction separately.

**Evaluation:** rank consistency, planning success, p95 end-to-end latency, deadline misses, candidate count, and physical rollout error; exact same starts/goals and three-to-five seeds. **Resources:** CPU with current toy models; optional one GPU for cached pretrained features; proposed ≤16 GPU-hours.

**Success:** an independently reproducible sensitivity analysis and budget-aware recommendation. **Failure:** effects vanish once every planner is normalized/tuned. That falsifies the broad claim and supports evaluation guidance. Deliver a benchmark harness and tutorial contribution.

Before submitting any of these, expand the literature search to references/citations of the closest paper, confirm code availability and licenses, and narrow the contribution around a tested gap.

