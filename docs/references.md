# Source index and evidence ledger

Snapshot: **2026-10-08**. Links were accessed through live browsing unless explicitly marked candidate/search-only. This is a curated survey, not an exhaustive literature census. Paper claims are author-reported; no external benchmark was independently reproduced.

## The twelve supplied sources

| Source | Inspection and interpretation |
|---|---|
| [World Labs](https://www.worldlabs.ai/) | Official page accessible; products and research index inspected; not scientific validation of general physics |
| [Value-guided JEPA](https://arxiv.org/abs/2601.00844) | Abstract, full HTML, equations, result table and appendix inspected; v1 dated Dec 28, 2025; CC BY 4.0 |
| [LeCun author search](https://arxiv.org/search/cs?searchtype=author&query=LeCun,+Y) | Accessible discovery index; not evidence that every result was read |
| [Fei-Fei Li LinkedIn discussion](https://www.linkedin.com/posts/fei-fei-li-4541247_world-model-has-become-one-of-the-most-share-7468001018123816961-ahW5/) | Accessible public commentary; use linked essay for taxonomy; commenters' claims are not independently verified |
| [Reddit JEPA/LLM thread](https://www.reddit.com/r/artificial/comments/1tjuats/so_what_is_yann_lecuns_world_models_and_jepa_and/) | Accessible; surfaces pixels-versus-latents confusion, collapse, and replacement claims; not consensus evidence |
| [Functional taxonomy](https://drfeifei.substack.com/p/a-functional-taxonomy-of-world-models) | Full essay inspected; dated Jun 3, 2026; perspective from Li/World Labs |
| [V-JEPA 2 repository](https://github.com/facebookresearch/vjepa2) | README, checkpoint/config interfaces and Hub file targeted; release/version distinctions |
| [JEPA-WMs repository](https://github.com/facebookresearch/jepa-wms) | README, checkpoint/data interfaces and predictor file targeted; no training run |
| [DreamerV3](https://github.com/danijar/dreamerv3) | Official implementation inspected; JAX; our PyTorch MPC lab is different |
| [HF V-JEPA docs](https://huggingface.co/docs/transformers/main/model_doc/vjepa2) | Input/output and processor/model API inspected; main docs mutable; local 4.57.6 test is separate |
| [Genie official family page](https://deepmind.google/models/genie/) | Accessible; cross-checked official announcements |
| [LeRobot v0.6.0](https://huggingface.co/blog/lerobot-release-v060) | Accessible article dated Jul 7, 2026; integration examples not executed |

## Foundational papers and official implementations

| Record / authors | First submission, version inspected | Paper | Official code/project | Scope and limitation |
|---|---|---|---|---|
| World Models; Ha, Schmidhuber | 2018-03-27; v4 record | [1803.10122](https://arxiv.org/abs/1803.10122) | [Project](https://worldmodels.github.io/) | Game world models; historical dependencies |
| PlaNet; Hafner et al. | 2018-11-12; v5 record | [1811.04551](https://arxiv.org/abs/1811.04551) | [Code](https://github.com/google-research/planet) | Pixel control via latent search; code download not tested |
| Dreamer; Hafner et al. | 2019-12-03; v3 record | [1912.01603](https://arxiv.org/abs/1912.01603) | [V3 family code](https://github.com/danijar/dreamerv3) | Imagined behavior; use historical version for exact original reproduction |
| DreamerV2; Hafner et al. | 2020-10-05; v4 record | [2010.02193](https://arxiv.org/abs/2010.02193) | [Family code](https://github.com/danijar/dreamerv3) | Discrete latents, Atari; V3 code not identical to V2 |
| DreamerV3; Hafner, Pasukonis, Ba, Lillicrap | 2023-01-10; v2 April 2024 | [2301.04104](https://arxiv.org/abs/2301.04104) | [Code](https://github.com/danijar/dreamerv3) | Broad-domain control; no local Dreamer run |
| Later Dreamer publication; same authors | 2025-04-02 | [Nature article](https://doi.org/10.1038/s41586-025-08744-2) | Same family repository | Distinguish published article from older preprint title/version |
| I-JEPA; Assran et al., FAIR | 2023-01-19; v3 | [2301.08243](https://arxiv.org/abs/2301.08243) | [Code/checkpoints](https://github.com/facebookresearch/ijepa) | Image masked embeddings, no inherent action control |
| V-JEPA; Bardes et al., FAIR | 2024-02-15 recorded; v1 | [2404.08471](https://arxiv.org/abs/2404.08471) | [Code/checkpoints](https://github.com/facebookresearch/jepa) | Video features; timestamp differs from ID month |
| V-JEPA 2 / AC; Assran et al., FAIR | 2025-06-11; v1 | [2506.09985](https://arxiv.org/abs/2506.09985) | [Code](https://github.com/facebookresearch/vjepa2), [HF](https://huggingface.co/facebook/vjepa2-vitl-fpc64-256) | Encoder and AC stages differ |
| V-JEPA 2.1; Mur-Labadia et al., FAIR | Repo release 2026-03-16 | [2603.14482](https://arxiv.org/abs/2603.14482) | [Code](https://github.com/facebookresearch/vjepa2), [HF issue](https://github.com/facebookresearch/vjepa2/issues/137) | Dense video features; exact assets must be checked |
| DINO-WM; Zhou, Pan, LeCun, Pinto | 2024-11-07; v2 2025-02-01 | [2411.04983](https://arxiv.org/abs/2411.04983) | [Code](https://github.com/gaoyuezhou/dino_wm) | Frozen dense features with action planning |
| JEPA-WMs; Terver, Yang, Ponce, Bardes, LeCun | 2025-12-30 | [2512.24497](https://arxiv.org/abs/2512.24497) | [Code](https://github.com/facebookresearch/jepa-wms), [weights](https://huggingface.co/facebook/jepa-wms) | Component study, environment-specific assets |
| Value-guided JEPA; Destrade et al. | 2025-12-28; v1 | [Full text](https://arxiv.org/html/2601.00844v1) | Separate official release not established; paper references [Sobal et al.](https://arxiv.org/abs/2502.14819) | Small wall/maze; adaptation explicitly altered |
| TD-MPC2; Hansen, Su, Wang | 2023-10-25 | [2310.16828](https://arxiv.org/abs/2310.16828) | [Project](https://tdmpc2.com) from primary abstract | Decoder-free model-based control; not executed |

## Generative, spatial, and 2026 additions

| Record | Source | Status |
|---|---|---|
| Genie; Bruce et al., DeepMind; Feb 2024 | [Paper](https://arxiv.org/abs/2402.15391), [family](https://deepmind.google/models/genie/) | Paper record and official pages inspected |
| Genie 2; DeepMind; Dec 2024 | [Announcement](https://deepmind.google/discover/blog/genie-2-a-large-scale-foundation-world-model/) | Official demonstration |
| Genie 3; DeepMind; Aug 5, 2025 | [Announcement](https://deepmind.google/blog/genie-3-a-new-frontier-for-world-models/) | Official capability/limitation statements |
| Project Genie; Jan 29, 2026 | [Product announcement](https://blog.google/innovation-and-ai/models-and-research/google-deepmind/project-genie/) | Access/product, not open weights |
| Cosmos Predict2.5; NVIDIA; late 2025 | [Paper](https://arxiv.org/abs/2511.00062), [code](https://github.com/nvidia-cosmos/cosmos-predict2.5), [weights](https://huggingface.co/nvidia/Cosmos-Predict2.5-2B) | Code/model card inspected; no GPU run |
| Cosmos 3; June 2026 | [2606.02800](https://arxiv.org/abs/2606.02800) | Search-index discovery; direct fetch failed; provisional |
| Marble; Nov 12, 2025 / API Jan 21, 2026 | [World Labs research index](https://www.worldlabs.ai/blog), [Marble](https://marble.worldlabs.ai/) | Official product evidence; attempted guessed article path not relied upon |
| RTFM; Oct 16, 2025 | [World Labs research index](https://www.worldlabs.ai/blog) | Research preview, separate from Marble |
| LeWorldModel; Maes, Le Lidec, Scieur, LeCun, Balestriero; March 2026 | [PDF v3, June 3](https://arxiv.org/pdf/2603.19312) | Primary PDF inspected after HTML failed; official code/license review pending |
| Semigroup-JEPA; Liu et al.; Sep 9, 2026 | [2609.10464](https://arxiv.org/abs/2609.10464), [project](https://sg-jepa.github.io/) | Primary abstract inspected; project/code not independently executed |
| V-JEPA Policy; Zhang et al.; Sep 29, 2026 | [2609.37250](https://arxiv.org/abs/2609.37250) | Primary abstract inspected; full benchmark/code review pending |
| Adaptive Latent Capacity; Sep 2026 | [2609.32921](https://arxiv.org/abs/2609.32921) | Search-only novelty overlap |
| TD-JEPA; Bai, Xiong; Jul 28, 2026 | [2607.25337](https://arxiv.org/abs/2607.25337) | Primary search-index abstract; full code review pending |
| OSCAR; Wu, Gao; Jun 3, 2026 | [2606.04463](https://arxiv.org/abs/2606.04463) | Primary search-index abstract; replication pending |
| World-Action Models survey; Lu et al.; Sep 13, 2026 | [2609.16074](https://arxiv.org/abs/2609.16074) | Search-index abstract; useful broader reading candidate |
| Waymo World Model; Jiang, Masotto, Sun; Feb 6, 2026 | [Official article](https://waymo.com/blog/2026/02/the-waymo-world-model-a-new-frontier-for-autonomous-driving-simulation/) | Provider deployment/simulation report |

## Mathematical and engineering foundations

- [VAE, Kingma and Welling](https://arxiv.org/abs/1312.6114), original 2013, later v11: distributional compression.
- [BYOL, Grill et al.](https://arxiv.org/abs/2006.07733), 2020: non-contrastive representation learning.
- [VICReg, Bardes, Ponce, LeCun](https://arxiv.org/abs/2105.04906), 2021/v3 2022: explicit variance/covariance regularization.
- [Vision Transformer, Dosovitskiy et al.](https://arxiv.org/abs/2010.11929), 2020: patch-based attention.
- [DDPM, Ho et al.](https://arxiv.org/abs/2006.11239), 2020: primary reading candidate; not fully inspected this run.
- [Flow Matching, Lipman et al.](https://arxiv.org/abs/2210.02747), 2022/v2 2023: generative velocity-field learning.
- [OpenCV tutorials](https://docs.opencv.org/4.x/d9/df8/tutorial_root.html): operations, camera calibration, tracking API.
- [H100 official specifications](https://www.nvidia.com/en-us/data-center/h100/): hardware facts; no lab GPU performance measurements.

## Data index

[DROID](https://droid-dataset.github.io/) · [BridgeData](https://rail-berkeley.github.io/bridgedata/) · [Open X-Embodiment](https://www.cross-embodiment.com/) · [OXE code](https://github.com/google-deepmind/open_x_embodiment) · [Ego4D](https://ego4d-data.org/) · [Something-Something provider](https://www.qualcomm.com/developer/artificial-intelligence/datasets) · [RoboCasa](https://robocasa.ai/) · [ManiSkill](https://www.maniskill.ai/) · [ManiSkill code](https://github.com/haosulab/ManiSkill) · [Meta-World](https://github.com/Farama-Foundation/Metaworld) · [Push-T](https://huggingface.co/datasets/lerobot/pusht) · [BAIR project](https://sites.google.com/view/sna-visual-mpc) · [PHYRE](https://github.com/google-deepmind/phyre).

Each has an evidence/access caveat in [chapter 8](08_models_and_datasets.md). A landing page alone cannot establish download size, license rights, or action schema.

## Supplementary explanations and practitioner concerns

[Medium: JEPA Made Easy, Saif, Jan 5, 2026](https://medium.com/@saif.phy/jepa-made-easy-a-simple-guide-to-learning-world-representations-76fdd6d4ff2c) was found as supplementary reading. It is not used as primary support for architecture or results.

The [supplied Reddit thread](https://www.reddit.com/r/artificial/comments/1tjuats/so_what_is_yann_lecuns_world_models_and_jepa_and/) usefully raises whether JEPA replaces LLMs, whether it predicts pixels, and how collapse is avoided. Corrections in this guide rely on papers and code, not commenter authority. Some comments attribute non-collapse solely to EMA; that is an oversimplification.

[HF papers](https://huggingface.co/papers/2603.14482) and repository issues can help discover assets and implementation changes. A papers page, model card, issue, or Space demonstration does not replace independent evaluation. No additional Space was validated as a reproducible scientific artifact in this run.
