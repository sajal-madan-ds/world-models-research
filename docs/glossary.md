# Glossary and FAQ

Plain meanings and precise meanings are kept separate. Examples are illustrative, not measured capabilities. Chapter numbers link to the guide.

| Term | Plain meaning | Precise meaning | Example | Related concepts / chapter |
|---|---|---|---|---|
| State | Information needed to predict the modeled future | Markov state under chosen dynamics | ball position and velocity | observation, belief; [01](01_world_models_eli5.md) |
| Observation | What sensors measure | Sample from an observation model | camera pixels | state, partial observability; [01](01_world_models_eli5.md) |
| Action | A command that can change the environment | Intervention/control input with units and timing | gripper displacement | transition, policy; [07](07_rl_and_planning.md) |
| Reward | A score for desired behavior | Scalar utility signal used in return | successful lift +1 | cost, return; [07](07_rl_and_planning.md) |
| Cost | A penalty to minimize | Objective term opposite reward convention | one per non-goal step | value, MPC; [07](07_rl_and_planning.md) |
| Transition | Change after a timestep/action | Conditional next-state map/distribution | ball bounce | dynamics, rollout; [01](01_world_models_eli5.md) |
| Dynamics | Rules for how things evolve | State-transition mechanism | acceleration changes velocity | physics, uncertainty; [07](07_rl_and_planning.md) |
| Encoder | Compress measurements into usable numbers | Learned feature map E | CNN maps two frames to 16 codes | representation; [03](03_representation_learning.md) |
| Decoder | Map an internal code to an output | Conditional observation reconstruction/generation map | render a predicted ball | autoencoder, renderer; [03](03_representation_learning.md) |
| Representation | Numerical summary used by a model | Feature vector/grid preserving selected distinctions | position-sensitive patches | embedding, latent; [03](03_representation_learning.md) |
| Embedding | A code assigned to an input | Image of a mapping into a vector space | video feature vector | representation, geometry; [03](03_representation_learning.md) |
| Latent state | An internal state not directly labeled | Learned hidden variable/code for prediction | RSSM latent | belief, dynamics; [03](03_representation_learning.md) |
| Autoencoder | Compress then reconstruct | Encoder/decoder trained with reconstruction objective | rebuild ball frames | VAE; [03](03_representation_learning.md) |
| VAE | Compression with a probabilistic code | Variational latent-variable model | sample Gaussian code | ELBO, KL; [03](03_representation_learning.md) |
| Prior | Prediction before a new measurement | Distribution not conditioned on current observation | imagined next code | posterior, RSSM; [05](05_architectures.md) |
| Posterior | Estimate after using evidence | Conditional latent-state distribution | correct code with new image | prior, belief; [05](05_architectures.md) |
| KL divergence | How one distribution differs from another | Expected log density ratio, asymmetric | posterior-to-prior regularization | VAE; [03](03_representation_learning.md) |
| ELBO | Trainable lower bound on likelihood | Expected log likelihood minus KL | variational world-model loss | VAE, RSSM; [05](05_architectures.md) |
| Self-supervision | Get learning targets from the data | Pretext prediction without external task labels | hidden block target | JEPA, masking; [03](03_representation_learning.md) |
| Contrastive learning | Match related examples and separate others | Representation objective using positive/negative pairs | two views of one object | similarity, temperature; [03](03_representation_learning.md) |
| Non-contrastive learning | Learn related features without explicit negatives | Agreement/prediction with anti-collapse mechanism | EMA target matching | collapse, VICReg; [03](03_representation_learning.md) |
| Predictive learning | Learn by anticipating missing or later information | Optimization of future/target prediction | predict future embedding | masked modeling; [03](03_representation_learning.md) |
| Masked modeling | Hide information to train its prediction | Conditional prediction of selected tokens/regions | occluded video tubelets | causal mask, JEPA; [03](03_representation_learning.md) |
| Representation collapse | Different inputs receive useless identical codes | Degenerate constant or low-rank representation | all-zero encoder | variance, rank; [06](06_jepa_deep_dive.md) |
| EMA | Slowly average changing parameters | Exponential moving average | teacher m=.99 | target encoder; [06](06_jepa_deep_dive.md) |
| Stop-gradient | Prevent one computation branch from learning | Detach tensor from autodiff gradient paths | Bellman target held fixed | teacher, value loss; [06](06_jepa_deep_dive.md) |
| VICReg | Encourage matching, spread, and low redundancy | Variance/invariance/covariance objective | feature spread penalty | collapse, covariance; [03](03_representation_learning.md) |
| SIGReg | Encourage a Gaussian feature distribution | Sketched isotropic Gaussian regularization in relevant methods | LeWorldModel search record | collapse, geometry; [05](05_architectures.md) |
| JEPA | Predict one learned summary from another | Joint Embedding Predictive Architecture framework | context predicts target features | encoder, predictor; [06](06_jepa_deep_dive.md) |
| Context encoder | Encode available information | Student encoder on context subset/history | visible patches | target encoder; [06](06_jepa_deep_dive.md) |
| Target encoder | Compute features to predict | Target branch, often stopped and sometimes EMA updated | full image target | stop-gradient; [06](06_jepa_deep_dive.md) |
| Predictor | Estimate a target or next code | Conditional learned feature-transition function | MLP predicts next latent | dynamics, JEPA; [06](06_jepa_deep_dive.md) |
| Action conditioning | Include the actual command in prediction | Transition depends on intervention variable | left versus right push | causality, coverage; [07](07_rl_and_planning.md) |
| Vision Transformer | Mix image patches with attention | Patch embeddings plus positional info and attention blocks | ViT-L video encoder | tokens, CNN; [02](02_opencv_foundations.md) |
| CNN | Learn spatially shared local filters | Convolutional neural network | small ball encoder | filter, receptive field; [02](02_opencv_foundations.md) |
| Patch token | Vector for an image region | Embedding of a spatial patch | 16×16 patch | ViT, position; [03](03_representation_learning.md) |
| Tubelet | A short space-time patch | Video token over multiple frames | 2-frame tubelet | temporal transformer; [03](03_representation_learning.md) |
| Temporal transformer | Attention over time and visual positions | Sequence model of video/history tokens | past frame context | causal mask; [03](03_representation_learning.md) |
| Autoregression | Predict then reuse earlier predictions | Sequential conditional factorization/rollout | feed next code back | error accumulation; [03](03_representation_learning.md) |
| Diffusion | Generate by reversing noise corruption | Learned iterative denoising model | sample future video | stochastic prediction; [03](03_representation_learning.md) |
| Flow matching | Learn how noise should move into data | Velocity-field regression along probability paths | generative ODE | diffusion, optical flow; [03](03_representation_learning.md) |
| Deterministic model | Same input gives one output | Single-valued transition/prediction | MLP next state | conditional mean; [03](03_representation_learning.md) |
| Stochastic model | Represent several possible outcomes | Conditional predictive distribution | random bounce direction | uncertainty; [03](03_representation_learning.md) |
| Epistemic uncertainty | Unknown model due to limited knowledge | Parameter/model uncertainty reducible with evidence | unseen friction | ensemble; [03](03_representation_learning.md) |
| Aleatoric uncertainty | Irreducible outcome/measurement variability | Data-generating conditional randomness | noisy sensor | calibration; [03](03_representation_learning.md) |
| Calibration | Confidence matches observed frequencies | Reliability of predictive probabilities/intervals | 90% interval covers ~90% | uncertainty, evaluation; [09](09_training_and_inference.md) |
| MDP | State-based decision problem | Markov Decision Process tuple | fully observed game | policy, transition; [07](07_rl_and_planning.md) |
| POMDP | Decide with incomplete measurements | Partially Observable MDP | cube hidden by gripper | belief, memory; [07](07_rl_and_planning.md) |
| Belief state | Distribution over possible current states | Posterior sufficient statistic under POMDP assumptions | possible cube poses | state estimation; [07](07_rl_and_planning.md) |
| Policy | Rule for choosing actions | State/history-conditioned action distribution | PD or learned actor | value, model-free; [07](07_rl_and_planning.md) |
| Value function | Expected longer-term usefulness | Expected discounted return, possibly goal conditioned | negative steps to goal | Bellman, cost; [07](07_rl_and_planning.md) |
| Q function | Score a state-action choice | Expected return after first action | quality of a push | policy, value; [07](07_rl_and_planning.md) |
| Discount factor | How strongly the future counts | Gamma weighting future rewards | gamma=.98 | return, horizon; [07](07_rl_and_planning.md) |
| Bellman equation | Relate value now to value later | Reward plus discounted continuation relation | reaching-cost backup | expectile; [06](06_jepa_deep_dive.md) |
| Expectile regression | Asymmetrically weight squared errors | Squared residual weighted by its sign | tau=.8 positive weight | IQL, value loss; [06](06_jepa_deep_dive.md) |
| IQL | Offline value/policy learning without unrestricted action maximization | Implicit Q-Learning family; paper uses inspired value objective | fit favorable observed continuations | offline RL; [06](06_jepa_deep_dive.md) |
| Quasimetric | A directed distance | Structured distance without required symmetry | one-way reachability | Euclidean geometry; [06](06_jepa_deep_dive.md) |
| Rollout | Repeatedly advance through transitions | Real or modeled trajectory | 24-step forecast | imagination, drift; [07](07_rl_and_planning.md) |
| MPC | Plan briefly, act, observe, replan | Model Predictive Control | execute first of 12 actions | CEM, MPPI; [07](07_rl_and_planning.md) |
| CEM | Refine samples around elite candidates | Cross-Entropy Method search | keep best 16 plans | optimizer; [07](07_rl_and_planning.md) |
| MPPI | Refine plans using cost-weighted perturbations | Model Predictive Path Integral control | weighted noise updates | MPC, temperature; [06](06_jepa_deep_dive.md) |
| Model-based RL | Learn/control using predicted transitions | RL system using a world model | Dreamer imagination | MPC, actor critic; [07](07_rl_and_planning.md) |
| Model-free RL | Learn behavior without explicit transition modeling | Direct policy/value optimization from experience | reactive learned actor | model-based; [07](07_rl_and_planning.md) |
| Offline RL | Learn decisions from existing experience | RL using a fixed dataset | robot demonstration logs | coverage, IQL; [07](07_rl_and_planning.md) |
| Online interaction | Collect new experience while operating | Environment querying during learning/control | new robot trial | feedback; [07](07_rl_and_planning.md) |
| Sim-to-real | Transfer from simulated to physical behavior | Domain transfer across sensing/dynamics gaps | simulation-trained grasp | domain randomization; [07](07_rl_and_planning.md) |
| Model exploitation | Planner benefits from model mistakes | Optimization drives into inaccurate predicted regions | imaginary shortcut | uncertainty, coverage; [07](07_rl_and_planning.md) |
| Proprioception | Robot senses its own configuration | Joint/pose/velocity measurements | gripper pose | state estimation; [07](07_rl_and_planning.md) |
| Renderer | Produce a visible view | Observation-generation component | image from a 3D scene | simulator; [04](04_world_model_taxonomy.md) |
| Simulator | Produce modeled state evolution | Transition-based environment for interventions | rigid-body rollout | renderer, planner; [04](04_world_model_taxonomy.md) |
| Planner | Choose future actions toward a goal | Optimization/search/decision component | pick push sequence | controller; [04](04_world_model_taxonomy.md) |
| Controller | Make an intended motion happen | Feedback command-generation loop | joint torque tracking | planner, policy; [04](04_world_model_taxonomy.md) |
| World-action model | Couple prediction with executable behavior | Family combining future world modeling and action generation | video-action policy | VLA; [10](10_use_cases.md) |
| VLA | Map vision/language into action | Vision-Language-Action model | instruction-following robot | world-action model; [10](10_use_cases.md) |
| Optical flow | Apparent motion in images | Pixel displacement field over time | 2 pixels per frame | tracking, odometry; [02](02_opencv_foundations.md) |
| Tracking | Maintain object identity over frames | Temporal association and state estimation | follow red ball | detection; [02](02_opencv_foundations.md) |
| Depth | Distance along a camera ray | Per-pixel geometric range under convention | 2m point | stereo, projection; [02](02_opencv_foundations.md) |
| Stereo | Infer depth from two viewpoints | Disparity-based triangulation after calibration | Z=fB/d | depth; [02](02_opencv_foundations.md) |
| Intrinsics | How a camera projects onto pixels | Focal lengths/principal point and calibration parameters | matrix K | extrinsics; [02](02_opencv_foundations.md) |
| Extrinsics | Camera location/orientation relative to a frame | Rigid rotation/translation transform | world-to-camera pose | coordinate system; [02](02_opencv_foundations.md) |
| SLAM | Build a map while locating yourself | Simultaneous Localization and Mapping | robot maps room | odometry, loop closure; [02](02_opencv_foundations.md) |
| Visual odometry | Estimate camera movement from images | Relative camera-pose estimation over time | successive poses | SLAM; [02](02_opencv_foundations.md) |
| Point cloud | Collection of 3D samples | Set of spatial points with optional attributes | depth backprojection | mesh; [02](02_opencv_foundations.md) |
| Mesh | Connected surface geometry | Vertices and faces | collision surface | splat, voxel; [02](02_opencv_foundations.md) |
| Voxel | A cell in a 3D grid | Discrete volumetric spatial element | occupancy grid | point cloud; [02](02_opencv_foundations.md) |
| Gaussian splat | A soft 3D rendering primitive | Parameterized Gaussian with appearance/opacity | Marble visual asset | mesh, renderer; [05](05_architectures.md) |
| Generalization | Work beyond training cases | Performance on defined held-out distributions | new layout | distribution shift; [09](09_training_and_inference.md) |
| Ablation | Remove/change one component to identify its effect | Controlled intervention on system design | remove actions | baseline; [09](09_training_and_inference.md) |
| Baseline | A comparison method | Reference under matched protocol | constant velocity | ablation, oracle; [09](09_training_and_inference.md) |
| Oracle | Idealized diagnostic reference | Uses privileged known information | exact physics | baseline, fairness; [09](09_training_and_inference.md) |
| Effective rank | How many feature directions are used | Exponential entropy of covariance eigenvalues | rank ~6 of 16 | collapse; [09](09_training_and_inference.md) |
| Distribution shift | Test cases differ from training | Change in input/transition/task distribution | new camera/friction | generalization; [09](09_training_and_inference.md) |

## FAQ

**Is a world model simply a video generator?** No. Useful predicted latent states need not be rendered. A video generator may offer a visual world-model component, but physical validity and action fidelity need separate tests.

**Can world models replace LLMs?** They address overlapping but different tasks. Language can specify goals and interpret observations; visual/action prediction can support physical decisions. The evidence here does not establish a universal replacement.

**Is JEPA an alternative to autoregression?** JEPA describes prediction in a jointly learned embedding space. A JEPA dynamics predictor can itself roll out autoregressively.

**Why predict embeddings instead of pixels?** It can emphasize predictable task-relevant information and avoid spending capacity on every appearance detail. Which information it retains must be tested.

**Can a model understand physics without a physics engine?** It can learn useful regularities and intervention predictions. Generalization to new forces, contacts, and parameter regimes requires evidence; plausible video alone is insufficient.

**Can passive video teach useful world models?** It can teach features, motion, and some predictive structure. Actual robot action semantics generally require grounding or action data. Inferred latent actions are not automatically calibrated commands.

**Why do actions matter?** The same starting state can have different futures under different commands. An action-blind predictor averages outcomes and cannot rank candidate commands reliably.

**What distinguishes predicting and planning?** Prediction asks what follows a command; planning searches for commands that produce a goal under a cost and constraints.

**Can a pretrained video encoder control a robot?** It can be a component. Dynamics or policy adaptation, action definitions, state estimation, goal/cost, and execution control are additional requirements.

**Why are long-horizon predictions difficult?** Reused prediction errors compound, hidden state persists, interventions leave training support, and several futures may be plausible.

**Why does simulation accuracy matter?** A policy can exploit a simulated shortcut that fails physically. The needed accuracy depends on the decision and its tolerance.

**What makes an experiment scientifically convincing?** Clear hypotheses, independent splits, strong baselines, repeated seeds, matched budgets, ablations, uncertainty reports, saved configurations, and honest negative results.

**What remains unsolved at this snapshot?** Broadly reliable long-horizon prediction and control under partial observations, stochastic contacts, shifting embodiments, limited intervention data, and real-time constraints. This survey is not an exhaustive October 2026 census.

**Is the lab's value experiment a reproduction?** It is a labeled adaptation with one-hot inputs and known-transition greedy planning. The actual paper uses image observations, learned predictors, MPPI and different environments; matching it remains separate work.

**Where should I start?** Read chapter 1, run experiment 01, then run the ball and action-conditioned control experiments before downloading large models.

