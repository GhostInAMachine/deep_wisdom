---
title: Normalizing Flows
type: concept
tags: [normalizing-flows, generative-models, vae, probability]
created: 2026-07-04
status: budding
aliases: [Normalizing Flow, Flow-based Models, Coupling Layers, Affine Coupling]
---

# Normalizing Flows

A **normalizing flow** is a way to build a complex probability distribution by pushing a simple one (usually a Gaussian) through an invertible, differentiable map. Because the map is a bijection, the resulting distribution has an **exact, tractable likelihood** — unlike a [[Generative Adversarial Network|GAN]] (no likelihood at all) or a [[Variational Autoencoder|VAE]] (only a lower bound). This makes flows a natural building block wherever you need both expressiveness *and* a computable density: as standalone generative models, and — the focus of the second half of this note — as components inside a VAE.

> [!note] Intuition
> Imagine starting with a plain round blob of probability mass (a standard Gaussian) and running it through a sequence of smooth, reversible deformations — stretching here, compressing there. Each deformation reshapes the blob into something more intricate, and the change-of-variables formula keeps exact track of how the density has to change so it still integrates to 1. "Normalizing" refers to the fact that inverting the flow maps your complicated data back toward that clean, normalized Gaussian.

## Change of variables

A function `f` is just an ordinary deterministic map — it does not, by itself, carry a probability density. What has a density is a random variable `X`, and a *second* random variable `Y := f(X)` obtained by pushing `X` through `f`. If `f` is bijective and differentiable, `Y` inherits a well-defined density of its own, related to `X`'s by the **change-of-variables formula**:

$$
p_X(x) = p_Y(f(x))\,\left|\det \frac{\partial f(x)}{\partial x}\right|
$$

Note the absolute value: a density must stay non-negative regardless of whether the map is orientation-preserving. The Jacobian determinant is the correction factor for how much `f` locally stretches or compresses volume — where `f` compresses space, probability mass piles up and the density rises; where it stretches, the density thins out.

Read the two directions:
- **Forward / normalizing direction** — map complicated data `x` toward a space where a simple density (a Gaussian) applies, and use the formula to read off the exact likelihood of `x`. This is what you differentiate to train by maximum likelihood.
- **Inverse / generating direction** — sample from the simple base density, run `f⁻¹`, and you have a sample from the complex distribution.

Both directions need `f` to be invertible and its Jacobian determinant to be cheap — which is the entire design problem for flows.

### Why bijectivity is what makes the likelihood tractable

The exact-likelihood property flows have — and [[Variational Autoencoder|VAEs]] and [[Generative Adversarial Network|GANs]] lack — comes down entirely to `f` being a bijection. To see why, compare with a general **latent-variable model**, which defines the data density by marginalizing over *every* latent that could have produced `x`:

$$
p_X(x) = \int p(x \mid z)\, p_Z(z)\, dz
$$

This integral is the source of the intractability. Under a stochastic decoder `p(x|z)`, many different `z` can generate the same `x`, so scoring a single datapoint means accumulating their contributions over the whole latent space — an integral with no closed form. That is precisely why a VAE settles for a lower bound (the ELBO) rather than the true likelihood: it cannot evaluate this integral.

A bijection removes the integral outright. If `f` is invertible then, for a given `x`, there is **exactly one** latent that maps to it — `z = f⁻¹(x)`, a single point, not a distribution over candidates. With nothing to marginalize, the density at `x` is just the base density at that one preimage, rescaled by the local volume change — the same change-of-variables formula from the top of this section:

$$
p_X(x) = p_Z\big(f(x)\big)\,\left|\det \frac{\partial f(x)}{\partial x}\right|
$$

Because this is an *identity*, not an approximation, the value is the **exact** likelihood — no bound, no sampling, no marginalization. The bijection is what converts an intractable integral into a single-point evaluation.

That leaves only the terms you actually have to compute, and invertibility keeps each one well-defined:

- **the preimage** `z = f(x)` — a single forward pass, since scoring data runs the normalizing direction;
- **the base density** `p_Z(z)` — essentially free, since the base is a standard Gaussian;
- **the log-determinant** `log|det ∂f/∂x|` — the one genuinely expensive term. Invertibility guarantees the Jacobian is nonsingular, so its determinant is nonzero and the log is always defined — but *cheapness* is not automatic, and driving it down from `O(d³)` is the subject of the next section.

> [!note] The price of a bijection: no bottleneck
> Matching points one-for-one forces the latent `z` to live in the **same dimension** as the data `x` — a flow cannot compress the way a VAE's encoder does. You trade the VAE's cheap low-dimensional bottleneck (and its intractable, only-bounded likelihood) for a full-dimensional invertible stack (and an exact one). Expressiveness then has to come from *depth* — many stacked invertible layers — rather than from a learned code that is smaller than the data.

## Why the determinant is the bottleneck

Computed naively for a `d`-dimensional map, `det(∂f/∂x)` costs `O(d³)` — far too expensive to differentiate through at every training step for a high-dimensional latent. Practical flows are therefore built out of layers whose Jacobian is *triangular*, because the determinant of a triangular matrix is just the product of its diagonal entries. The dominant such construction is the **affine coupling layer** (RealNVP / [[Glow-TTS|Glow]] / WaveGlow):

1. Split the input `x` into two halves along the channel dimension, `x_a` and `x_b`.
2. Leave the first half unchanged: `y_a = x_a`.
3. Transform the second half conditioned on the first — and optionally on external conditioning (e.g. VITS feeds in text features and a speaker embedding): `y_b = x_b ⊙ exp(s(x_a)) + t(x_a)`, where the log-scale `s` and shift `t` come out of an ordinary neural network fed `x_a`.

The Jacobian of this map is block-triangular:

$$
\frac{\partial y}{\partial x} = \begin{bmatrix} I & 0 \\[2pt] \dfrac{\partial y_b}{\partial x_a} & \operatorname{diag}\big(\exp(s(x_a))\big) \end{bmatrix}
$$

so the messy off-diagonal block `∂y_b/∂x_a` never needs to be evaluated at all:

$$
\det\frac{\partial y}{\partial x} = \det(I)\cdot\det\big(\operatorname{diag}(\exp(s(x_a)))\big) = \exp\Big(\sum_i s(x_a)_i\Big)
\quad\Longrightarrow\quad
\log\left|\det\frac{\partial y}{\partial x}\right| = \sum_i s(x_a)_i
$$

The log-determinant of a single coupling layer collapses to a plain sum over its log-scale outputs — an `O(d)` computation that comes almost for free, since `s(x_a)` is already computed as part of the forward pass. The layer is trivially invertible too: given `y`, recover `x_a = y_a`, then `x_b = (y_b - t(x_a)) ⊙ exp(-s(x_a))` — note that `s` and `t` are never themselves inverted, which is why they can be arbitrary neural networks.

### Stacking layers and permutations

A single coupling layer only ever transforms half the channels, so a flow `f` alternates coupling layers with a fixed channel permutation ("flip", or a learned invertible 1×1 convolution as in Glow) so that every channel eventually gets transformed conditioned on every other channel across depth. A pure permutation has Jacobian determinant `±1` (`log|det| = 0`), so it contributes nothing to the running total — its only job is to reshuffle which half is held fixed in the next layer.

Because a flow is a composition of many such layers, `f = f_L ∘ ⋯ ∘ f_1`, the determinant chain rule `det(AB) = det(A)·det(B)` means the log-determinant of the *whole stack* is just the sum of each layer's log-determinant:

$$
\log\left|\det \frac{\partial f(z)}{\partial z}\right| = \sum_{l=1}^{L} \log\left|\det \frac{\partial f_l}{\partial(\text{input to }f_l)}\right| = \sum_{l=1}^{L}\sum_i s_l(\cdot)_i
$$

That single accumulated number is the only Jacobian bookkeeping a coupling flow ever needs, at any depth.

> [!note] Coupling layers vs. autoregressive flows
> Autoregressive flows (IAF/MAF) get the same triangular-Jacobian trick, but pay for it with a mapping that must be computed one dimension at a time in one of the two directions — fast to evaluate but slow to invert, or vice versa. Coupling layers trade away a little expressiveness per layer (only half the channels change at a time) in exchange for **both** directions, `f` and `f⁻¹`, being a single parallel forward pass. That symmetry is exactly why a model like [[VITS]] can afford to run the same flow forward at training time and inverted at inference time without a speed penalty either way.

## Flows inside a VAE

A vanilla [[Variational Autoencoder|VAE]] makes two Gaussian assumptions that both limit it:
- the approximate **posterior** `q_φ(z|x)` is a diagonal Gaussian, which can't capture correlated or multimodal latent structure;
- the **prior** `p(z)` is a fixed `N(0, I)`, which forces the aggregate of encoded data to be squeezed into a simple ball.

A normalizing flow can relax *either* assumption, and **where you insert the flow changes what problem it solves.** The exact-likelihood property is what makes this clean: the extra `log|det|` term just adds into the ELBO wherever the flow sits.

### Placement 1 — flow on the posterior

Start from a simple Gaussian sample `z_0 ~ q_φ(z_0|x)` and warp it through a flow, `z_K = f_φ(z_0)`, so the *effective* posterior `q_φ(z_K|x)` is far richer than a Gaussian (Rezende & Mohamed 2015, *Variational Inference with Normalizing Flows*; Kingma et al. 2016 use an inverse-autoregressive flow, IAF, for this). The flow's log-determinant enters the ELBO through the posterior entropy term:

$$
\log q_\phi(z_K \mid x) = \log q_\phi(z_0 \mid x) - \sum_{k=1}^{K}\log\left|\det\frac{\partial f_k}{\partial z_{k-1}}\right|
$$

- **Goal:** shrink the gap between the approximate posterior and the true posterior, giving a **tighter ELBO** (less slack in the bound) and better-calibrated latents.
- **Direction that must be fast:** the *forward* map (encode → sample → warp), run every training step and at inference for reconstruction. IAF is fast forward, slow inverse — and since a posterior flow only ever runs forward, that's an acceptable trade.

### Placement 2 — flow on the prior

Keep a simple Gaussian posterior, but replace the fixed `N(0, I)` prior with a **learned flow-based prior** `p_θ(z)`: a base Gaussian pushed through a flow so the prior can match the shape the encoder actually wants to use. The `log|det|` term now attaches to the prior side of the KL.

- **Goal:** close the classic **prior hole / aggregate-posterior mismatch** — regions the prior calls likely but the encoder never populates (and vice versa), which is where blurry or off-distribution samples come from.
- **Direction that must be fast:** the *inverse* map, because generation samples the base Gaussian and runs the flow to produce `z`. (A coupling flow is fast both ways, sidestepping the choice entirely.)

### Placement 3 — conditional prior flow (the VITS case)

[[VITS]] is Placement 2 made **conditional**. Its prior over the audio latent is not a fixed Gaussian but a text-conditioned Gaussian `N(μ_θ(c), σ_θ(c))`, and *that* is warped by a flow `f_θ`:

$$
p_\theta(z \mid c) = \mathcal{N}\big(f_\theta(z);\ \mu_\theta(c),\ \sigma_\theta(c)\big)\,\left|\det \frac{\partial f_\theta(z)}{\partial z}\right|
$$

The per-utterance posterior `q_φ(z|x)` is itself only a diagonal Gaussian, so it's tempting to think a Gaussian prior could match it exactly. The catch is that the prior is conditioned on *text* `c`, not on the specific audio `x`: one sentence has many valid renditions (speech is one-to-many), each giving a differently-centered Gaussian posterior. The distribution the prior must actually cover is the **aggregate** of all those posteriors for a given text — a mixture of Gaussians, hence multimodal and decidedly non-Gaussian (for a fixed encoder, the ELBO-optimal prior is exactly this aggregate posterior). A single Gaussian can't represent that mixture; the flow reshapes it into a complex distribution that can, while keeping the exact likelihood that lets the whole model be trained with a KL term at all. VITS trains the flow *forward* (audio latent → prior space, to evaluate the KL) and runs it *inverse* at inference (sample the text prior → decode), which is precisely why it needs the both-directions-cheap coupling construction above. See [[VITS]] for the full derivation, and note it uses the identical change-of-variables machinery a *second* time inside its stochastic duration predictor.

> [!note] The same formula, three jobs
> Posterior flow, prior flow, and conditional prior flow all use the one change-of-variables identity — they differ only in *which* density the flow reshapes and therefore which term of the ELBO the `log|det|` lands in. Getting the placement right is a modeling choice about *where* your simple-Gaussian assumption is hurting you most.

## Key papers

- Rezende, D. J. & Mohamed, S. (2015). *Variational Inference with Normalizing Flows.* ICML. — introduces flows for enriching the VAE posterior.
- Kingma, D. P. et al. (2016). *Improving Variational Inference with Inverse Autoregressive Flow.* NeurIPS. — IAF posteriors.
- Dinh, L., Sohl-Dickstein, J. & Bengio, S. (2017). *Density Estimation using Real NVP.* ICLR. — affine coupling layers.
- Kingma, D. P. & Dhariwal, P. (2018). *Glow: Generative Flow with Invertible 1×1 Convolutions.* NeurIPS. — learned invertible permutations.
- Papamakarios, G. et al. (2021). *Normalizing Flows for Probabilistic Modeling and Inference.* JMLR. — survey.

## Related

- [[VITS]] — end-to-end TTS built on a conditional flow-warped prior (Placement 3) plus a second flow in its duration model
- [[Variational Autoencoder]] — the model flows are inserted into above
- [[Glow-TTS]] — flow-based TTS; origin of the coupling/1×1-conv design VITS reuses
- [[Kullback-Leibler Divergence]] — the term the `log|det|` correction feeds into
- [[Generative Models]]
