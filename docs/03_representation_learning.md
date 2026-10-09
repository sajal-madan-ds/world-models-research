# 3. Representation learning and predictive modeling

Previous: [vision](02_opencv_foundations.md). Next: [taxonomy](04_world_model_taxonomy.md).

## ELI15: keeping information you can use

The problem is that raw observations contain far more detail than a decision needs. A restaurant map suppresses brick texture and retains rooms and paths. An encoder similarly turns (x) into $z=E_\theta(x)$, a learned vector or grid.

**Feature extraction** computes useful measurements. **Representation learning** learns what measurements to compute. An **embedding** is their numerical form. **Latent space** refers to the space of these internal representations; it is not an independently existing physical place.

A decoder $D_\phi(z)$ maps the summary back to an output. An autoencoder trains (D(E(x))) to reconstruct (x), usually minimizing squared error. A small bottleneck constrains capacity, but the network may still preserve texture at the expense of velocity. Predictive usefulness and reconstruction fidelity differ.

In experiment 04, two ball frames enter a CNN. A decoder reconstructs them and a transition network predicts the next code. We evaluate a frozen linear probe for physical state and multi-step latent prediction; a low latent loss alone is insufficient.

**Check:** What information does a single ball image lack that two ordered images might reveal?

## ELI18: deterministic and stochastic codes

A deterministic encoder always returns the same code for the same input. A **variational autoencoder** returns parameters of a distribution, often $\mu(x),\sigma(x)$, samples $z=\mu+\sigma\epsilon$, $\epsilon\sim\mathcal N(0,I)$, and balances reconstruction with distribution regularity:

$$
\mathcal L_{\rm VAE}
=-\mathbb E_{q_\theta(z|x)}\log p_\phi(x|z)
+\beta D_{\rm KL}(q_\theta(z|x)\|p(z)).
$$

(q) approximates the hidden-code posterior; (p(z)) is a prior; KL divergence measures distribution mismatch. $\beta$ trades reconstruction and compression. The reparameterization lets gradients flow through (mu,sigma). It does not make the decoded future physically correct. [VAE](https://arxiv.org/abs/1312.6114).

Minimal reparameterization:
```python
std = (0.5 * logvar).exp()
z = mu + std * torch.randn_like(std)
kl = -.5 * (1 + logvar - mu.square() - logvar.exp()).sum(-1).mean()
```

**Stochastic prediction** models multiple possible futures. A deterministic squared-error predictor learns a conditional mean. If a ball equally likely goes left or right, that mean may describe a nonexistent path through the middle. Distinguish randomness in dynamics from uncertainty about unknown model parameters.

**Check:** Can a deterministic environment still require uncertain prediction?

## Self-supervision: labels from the data itself

The problem is collecting enough useful labels. Self-supervised learning creates targets from observations: hidden patches, other views, future frames, or adjacent states. It is not the absence of a training signal.

**Contrastive learning** attracts related samples and separates negatives. With similarity (s), temperature (T), positive (z^+), and candidates (z_j), a typical objective is
$$
-\log\frac{\exp(s(z,z^+)/T)}{\sum_j\exp(s(z,z_j)/T)}.
$$
A positive pair can be two views of an object. Incorrect positives teach invariance to useful distinctions; false negatives separate physically equivalent scenes.

**Non-contrastive learning** does not require explicit negative pairs. It needs another mechanism to prevent trivial solutions. For prediction loss (|P(E(x))-E(y)|^2), mapping every input to zero gives zero loss. This is **representation collapse**.

**Masked modeling** hides parts of an input and predicts information about them. It can predict pixel values, discrete tokens, or embeddings. Masking within a bidirectional video clip is not necessarily forecasting an unseen future.

**Check:** Why can declining training loss indicate a worse representation?

## Teacher networks and representation geometry

An EMA teacher changes slowly:
$$
\bar\theta\leftarrow m\bar\theta+(1-m)\theta.
$$
$\theta$ is the student parameter vector; $\bar\theta$ is the target vector; (m) is momentum, e.g. .99. Gradients update the student but stop at target outputs. EMA provides stable targets. Architecture, predictor asymmetry, masking, optimization, and data also matter; EMA alone is not a universal guarantee against collapse.

Variance regularization demands feature spread. Covariance regularization reduces redundant dimensions. The lab uses a deliberately altered variance/covariance term alongside EMA. It is not an official V-JEPA recipe. [VICReg](https://arxiv.org/abs/2105.04906), [BYOL](https://arxiv.org/abs/2006.07733).

Useful diagnostics are per-feature standard deviation, covariance eigenspectrum, effective rank, nearest-neighbor structure, and frozen probes. A full-rank nuisance representation can still be useless for control.

**Check:** Would large feature variance prove that a model tracks physical state?

## Video tokens and sequence models

A temporal transformer adds time to image patches. A tubelet covers $p\times p$ pixels over $\tau$ frames. Token count is approximately
$$
N=(T/\tau)(H/p)(W/p).
$$
At $T=64,H=W=256,\tau=2,p=16$, there are 8,192 tokens. Full attention has quadratic pair interactions; long clips can become expensive even with modest parameter counts.

A recurrent model updates hidden state (h_{t+1}=f(h_t,z_t,a_t)). It summarizes history without attending to all past tokens but can lose long-range detail. A temporal transformer attends over a window; causal masks prevent using unavailable future observations.

**Autoregression** predicts the next element conditioned on preceding elements. A latent world model can also be autoregressive: feed predicted (z_{t+1}) back to predict (z_{t+2}). JEPA and autoregression are not mutually exclusive; they describe different design axes.

**Check:** Why is teacher-forced one-step accuracy an optimistic measure of rollout accuracy?

## Generating pixels: diffusion and flow matching

A diffusion model learns to reverse a noise-corruption process. A noise prediction objective can take the form
$$
\mathbb E_{x,\epsilon,t}|\epsilon-\epsilon_\theta(\alpha_tx+\sigma_t\epsilon,t,c)|^2,
$$
where (c) is conditioning, such as a prompt, previous frames, or actions. Repeated denoising produces samples; it can be computationally expensive. [DDPM](https://arxiv.org/abs/2006.11239).

Flow matching learns a velocity field transporting noise into data:
$$
\mathbb E_{t,x_t}|v_\theta(x_t,t,c)-u_t(x_t)|^2.
$$
A numerical ODE solver follows the learned field. This use of “flow” differs from optical flow. Exact normalizing flows additionally emphasize invertible mappings and likelihoods; the names are related but not interchangeable. [Flow matching](https://arxiv.org/abs/2210.02747).

Neither objective alone enforces conservation, reachable actions, camera calibration, or accurate contact dynamics. Evaluate the generated futures against task-specific measurements.

**Check:** Why might several plausible futures be preferable to one sharp prediction?

## ELI21: uncertainty and rollouts

For a learned transition $\hat F$, one-step error can accumulate when predictions are recursively reused. If true dynamics are (L)-Lipschitz and local model error is at most $\epsilon$, a simple bound is
$$
e_{k+1}\le L e_k+\epsilon.
$$
For (L>1), errors can amplify rapidly. The assumptions often fail at contact discontinuities. More rollout training, replanning, ensembles, and better state estimation can help but require experiments.

Ensembles measure disagreement between independently trained models, an imperfect proxy for epistemic uncertainty. Stochastic output heads model outcome variability, an imperfect proxy for aleatoric uncertainty. Validate calibration with held-out coverage and proper scoring rules.

**Check:** When would an ensemble agree and still be wrong?

## Bridges to your LLM experience

| Familiar concept | Useful visual analogy | Where it breaks |
|---|---|---|
| Text tokens | Patches, tubelets, state vectors | Patches are spatial samples, not linguistic units; states can be continuous |
| Token embeddings | Visual feature vectors | Their geometry need not encode controllability or distance |
| Next-token prediction | Future state prediction | Actions intervene; physical hidden state and timing matter |
| Context window | Observation/action history | Sensors may be asynchronous; hidden properties remain missing |
| Decoding | Rendering predicted observations | A dynamics model can be useful without a decoder |
| Fine-tuning | Encoder/predictor adaptation | Frozen feature alignment may change; robot action units must be matched |
| LoRA | Low-rank updates to attention layers | Suitability is architecture/task dependent; support must be checked |

## ELI24–27: representation is a research choice

Ask what is retained, discarded, and stable over time. Compare pixel reconstruction, latent prediction, and task-conditioned objectives under the same data and compute. Freeze a trained encoder and measure state decoding, action prediction, goal-reaching cost, and planning success separately.

A predictive code can encode shortcuts: timestamp, background, or the identity of a trajectory. Splitting adjacent frames randomly leaks those shortcuts across train/test. Split at episode level and reserve layouts, objects, or physical parameters for distribution-shift evaluation.

**Check:** How would you distinguish an encoder learning motion from memorizing backgrounds?
