# 9. Training, inference, hardware, and evaluation

Previous: [catalog](08_models_and_datasets.md). Next: [applications](10_use_cases.md).

## A complete small research lifecycle

Choose a falsifiable task: e.g., predict a ball 24 steps ahead or reach a goal using learned dynamics. Define the observation/action interface and held-out distribution before architecture selection. Inspect transitions visually. Then build a nonlearned baseline and only add model complexity that solves an identified failure.

For the default lab: generate independent train and test episodes; preserve time ordering; use small MLP/CNN components; train minibatches with AdamW; save loss curves, weights, exact args, library versions, seeds, metrics, and plots. Defaults are 250 optimizer updates, batch up to 128, LR .002 for supervised dynamics; value adaptation uses 1,000 updates at LR .004. These are educational settings, not a paper recipe.

Run:
```sh
uv sync --python 3.12 --extra dev
uv run pytest -q
uv run wm-lab --experiment all --seeds 0 1 2 --steps 250
uv run wm-lab --experiment 05 --seeds 3 --steps 1000
```

Every run creates a fresh timestamped directory. Experiments 02/03 share the same dynamics implementation with different teaching emphases; 07/08 share the same control implementation, emphasizing planning versus action ablation. Do not count them as independent scientific replications.

Saved-model inference on the measured ball checkpoint:
```sh
uv run python scripts/infer_toy.py --checkpoint results/20261008T074315248466Z/02_seed0/transition.pt --state .5 .5 .2 -.1 --horizon 12
```
For a conditioned control checkpoint, pass `--action .3 -.2`. This script runs the learned transition recursively; its forecast is not ground truth. The JSON suite recipe can be executed with `uv run python scripts/run_suite.py`.

## Real video preprocessing

Video frames must retain sampling timestamps and control alignment. A stride of 8 at 30 FPS is .267 seconds, not one control timestep unless the action pipeline matches it. Sample fixed-duration windows, record effective FPS, and avoid crossing resets. Decode RGB consistently, resize/crop according to the checkpoint processor, and apply the documented normalization.

Use spatial augmentation only if its invariance fits the task. A horizontal flip without changing actions or calibration can create false physics. Temporal reversal is inappropriate for irreversible dynamics. Different augmentation of adjacent frames can fabricate motion.

Do not cache every decoded frame in RAM. Decode batches or short windows; prefetch after measuring the bottleneck. Episode metadata and action tables can be indexed separately from compressed video.

## Optimization and fine-tuning

For small models use full precision first. CUDA AMP can reduce memory; bf16 needs supported hardware and operations. Check numerical stability, especially variances, distance/value losses, and recurrent rollout gradients. Gradient clipping prevents occasional explosions but does not fix incorrect data.

A cosine LR schedule, warmup, weight decay, checkpointing, and early stopping should be selected using validation rather than test performance. Track optimizer, RNG, sampler, scheduler, and AMP-scaler state for exact training resumption. **Current lab checkpoints are inference snapshots, not full resume checkpoints.**

With a pretrained encoder, freeze it and train a small probe/predictor first. Full fine-tuning can move the embedding target distribution and invalidate previously learned dynamics. LoRA can adapt supported attention layers, but no universal JEPA LoRA workflow is claimed here. Establish an official architecture-specific recipe or implement and test it explicitly.

DDP replicates model training across GPUs; FSDP shards parameter/optimizer storage. Neither solves poor video decoding throughput or insufficient data diversity. Use these only after a single-device training step and data pipeline are measured.

## Official V-JEPA inference adapter

The [HF documentation](https://huggingface.co/docs/transformers/main/model_doc/vjepa2) and [model card](https://huggingface.co/facebook/vjepa2-vitl-fpc64-256) show video preprocessing and encoder feature extraction. Prefer those instructions over automatically generated model-page snippets that suggest a tokenizer for a video model.

The local adapter requires an existing Transformers checkpoint directory and video:
```sh
uv sync --extra pretrained --extra dev
uv run wm-vjepa --model /path/to/local/vjepa2 --video /path/to/clip.mp4 --device cuda --output results/my_features.npz
```

It samples the configured frame count, converts BGR to RGB, processes video, requests encoder outputs while skipping the predictor, saves token and pooled features, and compares first/second-half pooled cosine similarity. This is exploratory temporal analysis, not a validated physical-reasoning score.

Documented model input is \((B,T,C,H,W)\); output hidden states are \((B,N,D)\). The adapter uses the processor to enforce normalization. It rejects remote model names and existing output files. Official weights have not been downloaded or run.

The installed optional API must be checked against the locked Transformers version; a tiny randomly initialized V-JEPA test verifies shape/API mechanics separately. It cannot establish that a large real checkpoint fits or performs well.

## Hardware feasibility: facts, estimates, measurements

| Hardware | Verified facts | Practical feasibility estimate | What has been measured here |
|---|---|---|---|
| Local macOS Apple Silicon | arm64; Python 3.12 selected; about 42 GiB free at inspection | Excellent for these tiny CPU labs; MPS useful after op compatibility checks; large pretrained video inference depends on RAM | CPU toy experiments; no MPS training benchmark or real encoder timing |
| NVIDIA L40S | GPU specification page fetch timed out; common 48GB variant must be checked on actual host | Frozen-encoder features and compact predictors plausible; full large-video pretraining unrealistic | No L40S experiment |
| Single H100 | Official [product page](https://www.nvidia.com/en-us/data-center/h100/) inspected; SKU memory varies | Small-batch encoder adaptation and larger predictor studies plausible; video token count controls fit | No H100 experiment |
| Multiple H100 | Aggregate memory requires sharding; interconnect affects throughput | Larger fine-tuning/reproduction with DDP/FSDP possible, but internet-scale pretraining still major work | No distributed experiment |

Parameter storage alone is approximately \(2P\) bytes in bf16, \(4P\) in fp32. Training also needs gradients, optimizer moments, master weights where used, activations, and workspace. A rough Adam full-training budget of 12–20 bytes/parameter **before activations** is a planning estimate, not a measured JEPA requirement. For a 300M encoder, bf16 weights alone are about .6GB decimal; that is not total inference memory.

Attention activation memory depends on \(N\), batch, layers, precision, and attention kernels. Compare native and memory-efficient implementations on the same input rather than asserting GPU fit from parameter count.

CUDA measurement:
```python
torch.cuda.reset_peak_memory_stats()
torch.cuda.synchronize()
start = time.perf_counter()
with torch.inference_mode():
    output = model(**inputs)
torch.cuda.synchronize()
latency = time.perf_counter() - start
peak_bytes = torch.cuda.max_memory_allocated()
```
Imports and `model/inputs` setup are required. Warm up, repeat, report median/p95, and separately measure decoding/end-to-end latency. The lab's wall time includes setup/training/evaluation and is not an inference latency benchmark.

## Evaluation: the metric must match the claim

Let predicted state \(\hat s_{it}\), true state \(s_{it}\), position \(p\), episodes \(i\), and horizon \(T\).

| Metric | Definition / protocol | Useful for | Important caveat |
|---|---|---|---|
| One-step error | Mean squared state error using observed current state | Local transition quality | Teacher forcing hides accumulated drift |
| Rollout error | \(T^{-1}\sum_t\|\hat s_t-s_t\|^2\) from recursively predicted states | Long-horizon dynamics | Report by horizon and state coordinate |
| Average trajectory displacement | \(T^{-1}\sum_t\|\hat p_t-p_t\|_2\) | Path fidelity | Units must be calibrated |
| Final displacement | \(\|\hat p_T-p_T\|_2\) | Endpoint tasks | Can hide intermediate collisions |
| Planning success | Fraction meeting prespecified terminal/task criterion | Goal achievement | Same starts/goals and budget; confidence intervals |
| Cumulative reward | Sum of environment rewards per episode | RL task behavior | Reward definition changes interpretation |
| Sample efficiency | Success/return versus real interactions | Data-efficient learning | Imagined samples are not real data |
| Latent quality | Frozen probes, retrieval, downstream control | Useful representation | Latent MSE depends on scale and geometry |
| Collapse indicators | Feature variance, covariance spectrum, effective rank | Trivial/partial collapse detection | High rank can encode nuisance |
| Generalization | Held-out scenes, objects, physics, actions | Transfer | Random frame split is not enough |
| Physical consistency | Conservation/contact/reachability errors appropriate to environment | Mechanistic validity | Correct-looking frames are insufficient |
| Latency / memory | Synchronized timed trials; allocated/reserved peaks | Deployability | Hardware, precision, clip size, warmup required |

Effective rank uses covariance eigenvalues \(\lambda_j\), normalized \(p_j=\lambda_j/\sum\lambda\), and \(\exp(-\sum p_j\log p_j)\). Constant features need explicit handling because all eigenvalues vanish.

Use episode bootstrap intervals for trajectory averages and binomial intervals for success. Repeat independent training seeds; resampling episodes alone does not estimate training variability. In a three-seed pilot, report all seeds and avoid broad significance claims.

## Ablations and fair comparisons

Required core comparisons: analytical versus neural dynamics; observed-state one-step versus free rollout; persistence versus predicted latents; image reconstruction versus JEPA-style training; action-conditioned versus action-blind dynamics; learned MPC versus random, zero, and PD; Euclidean versus learned goal distance versus shortest-path oracle.

For architecture changes, hold data, updates, precision, optimizer search, evaluation goals, and model capacity as constant as possible. Where action dimension changes parameter count, include a dummy-action capacity-matched control before claiming the effect is purely informational.

Expect real failures: bounce discontinuity, moving target encoders, low-rank reconstructions, action ambiguity, planner exploitation, and unseen layouts. The [measurement report](../research/VERIFICATION.md) describes what occurred.

**Check:** Why does an encoder with a ten-times-smaller latent MSE not necessarily have ten-times-better predictions?
