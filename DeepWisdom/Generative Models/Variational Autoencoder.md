---
title: Variational Autoencoder
type: architecture
tags: [generative-models, vae, variational-inference, probability]
created: 2026-07-04
status: budding
aliases: [VAE, Variational Autoencoders, Variational Auto-Encoder]
---

# Variational Autoencoder

A **variational autoencoder** (VAE; Kingma & Welling, 2013, *Auto-Encoding Variational Bayes*) is a [[Generative Models|generative model]] that learns a data distribution through a low-dimensional **latent variable** $z$. It pairs a probabilistic **encoder** $q_\phi(z\mid x)$ — which maps data to a distribution over latents — with a **decoder** $p_\theta(x\mid z)$ that maps latents back to data, and trains both jointly by maximizing a lower bound on the data likelihood. Unlike a plain autoencoder, whose latent code is an arbitrary point with no distributional meaning, a VAE's latent space is *regularized toward a known prior*, which is what lets you **sample** new data by drawing $z$ from that prior and decoding it.

> [!note] Intuition
> A plain autoencoder learns to compress each input to a point and reconstruct it — but the space *between* those points is meaningless, so you can't generate from it. A VAE instead encodes each input to a small blob of probability (a Gaussian) and forces all those blobs to collectively fill out a standard Gaussian. That makes the latent space smooth and *navigable* — nearby codes decode to similar things, and you can interpolate between two real examples and get sensible in-betweens. It also lets you sample fresh codes from the prior and decode them, though those samples are only ever as good as the fit between the encoded blobs and the prior — a fit that is never exact (see [[#Failure modes]]).

## The latent-variable model

A VAE defines the data distribution as a **marginal** over a latent variable with a simple fixed prior $p(z) = \mathcal{N}(0, I)$:

$$
p_\theta(x) = \int p_\theta(x \mid z)\, p(z)\, dz
$$

The decoder $p_\theta(x\mid z)$ is a neural network (Gaussian for continuous data, Bernoulli for binary). The trouble is that this integral is **intractable** — you can't marginalize over all $z$ — and so is the exact posterior it induces,

$$
p_\theta(z \mid x) = \frac{p_\theta(x\mid z)\,p(z)}{p_\theta(x)},
$$

because its denominator is that same intractable integral. Training by direct maximum likelihood is therefore off the table. This is precisely the gap [[Variational Inference|variational inference]] is built to close.

## Variational inference and the ELBO

Rather than compute the true posterior $p_\theta(z\mid x)$, the VAE **approximates** it with a tractable **encoder** $q_\phi(z\mid x)$ — typically a diagonal Gaussian whose mean and variance are output by a network. "Amortized" inference: instead of solving a separate optimization per datapoint, one shared network $\phi$ predicts the posterior parameters for *any* $x$ in a single forward pass.

Starting from the marginal log-likelihood and introducing $q_\phi$, the log-likelihood splits exactly into two terms:

$$
\log p_\theta(x) = \underbrace{\mathbb{E}_{q_\phi(z\mid x)}\!\left[\log \frac{p_\theta(x, z)}{q_\phi(z\mid x)}\right]}_{\text{ELBO } \mathcal{L}(\theta,\phi;\,x)} \;+\; \underbrace{D_{\mathrm{KL}}\!\big(q_\phi(z\mid x)\,\|\,p_\theta(z\mid x)\big)}_{\ge\,0}
$$

The second term is a [[Kullback-Leibler Divergence|KL divergence]] and is always non-negative, so the first term is a lower bound on $\log p_\theta(x)$ — the **evidence lower bound (ELBO)**. Because $\log p_\theta(x)$ is fixed with respect to $\phi$, *maximizing the ELBO simultaneously* fits the model and squeezes the approximate posterior toward the true one (shrinking that KL gap). Rearranged into its two working halves:

$$
\mathcal{L}(\theta, \phi;\, x) = \underbrace{\mathbb{E}_{q_\phi(z\mid x)}\big[\log p_\theta(x\mid z)\big]}_{\text{reconstruction}} \;-\; \underbrace{D_{\mathrm{KL}}\!\big(q_\phi(z\mid x)\,\|\,p(z)\big)}_{\text{regularizer}}
$$

- The **reconstruction term** rewards decoding $z$ back to the input $x$ that produced it — this is what makes it an *autoencoder*.
- The **KL regularizer** pulls each per-datapoint posterior toward the prior $\mathcal{N}(0, I)$, keeping the latent space packed against the prior so sampling works. It's in tension with reconstruction: perfect reconstruction wants distinct, spread-out codes; the KL wants everything collapsed onto the prior.

## The cast of distributions, and which network is which

VAE notation trips people up because several distributions over the same two variables ($x$ and $z$) are in play at once, only *two* of them are neural networks, and the two parameter sets $\theta$ and $\phi$ tell you which is which. The whole model is just an **encoder** and a **decoder** plus a fixed prior — everything else is a distribution you reason about but never build.

| Distribution | Name(s) | What in the network | Params | Tractable? |
|---|---|---|---|---|
| $p(z)$ | **prior** | nothing — a fixed choice, $\mathcal{N}(0,I)$ | none | yes, by fiat |
| $q_\phi(z\mid x)$ | **approximate posterior**; recognition / inference model | the **encoder** — outputs $\mu_\phi(x), \log\sigma_\phi^2(x)$ | $\phi$ | yes (diagonal Gaussian) |
| $p_\theta(x\mid z)$ | **likelihood**; observation / generative model | the **decoder** — outputs the params of $x$'s distribution | $\theta$ | yes (Gaussian / Bernoulli) |
| $p_\theta(x)$ | **marginal likelihood** / evidence / model distribution | decoder marginalized over the prior | $\theta$ | **no** — the intractable integral |
| $p_\theta(z\mid x)$ | **true posterior** | nothing — the target $q_\phi$ imitates | $\theta$ | **no** — needs the evidence above |
| $q_\phi(z)$ | **aggregate (marginal) posterior** | encoder averaged over the data, $\mathbb{E}_{x}[q_\phi(z\mid x)]$ | $\phi$ | **no** — no closed form |

The two directions of the model are the only two networks:

- **Encoder = $q_\phi(z\mid x)$** — the *inference* direction, data → latent. It doesn't emit a $z$; it emits the **parameters** of a distribution over $z$, from which you then sample. Its job is to approximate the one distribution you'd love to have but can't compute, the true posterior $p_\theta(z\mid x)$.
- **Decoder = $p_\theta(x\mid z)$** — the *generative* direction, latent → data. Likewise it emits the parameters of a distribution over $x$ (e.g. a per-pixel mean), not the pixels directly.

The three distributions with **no network** are what the objective is defined *in terms of*, not things you evaluate:

- $p(z)$, the **prior**, is a fixed target you regularize toward — it has no parameters to learn, which is exactly why plain prior sampling is limited (see the aggregate-posterior gap under [[#Failure modes]]).
- $p_\theta(x)$, the **evidence**, is what maximum likelihood *would* maximize; because it's intractable, the ELBO stands in for it.
- $p_\theta(z\mid x)$, the **true posterior**, is the intractable thing the encoder approximates; the slack between them is the KL gap in the ELBO decomposition above.

> [!note] Reading the subscripts
> $\theta$ marks everything on the **generative** side (decoder, and the prior/marginal/posterior it induces); $\phi$ marks the **inference** side (the encoder and its aggregate). A distribution with *no* subscript — here $p(z)$ — is fixed and learned by neither. When you see $q_\phi$ think "encoder," when you see $p_\theta(x\mid z)$ think "decoder," and the rest of the notation falls into place.

## The reparameterization trick

To train by gradient descent, the ELBO's reconstruction expectation must be differentiated with respect to $\phi$ — but $\phi$ controls the distribution the expectation is *taken over*, and you can't push a gradient through a random sampling step. The **reparameterization trick** rewrites the sample as a deterministic function of the parameters plus a parameter-free noise source:

$$
z = \mu_\phi(x) + \sigma_\phi(x) \odot \epsilon, \qquad \epsilon \sim \mathcal{N}(0, I)
$$

Now the randomness lives entirely in $\epsilon$, which carries no parameters, so gradients flow cleanly through $\mu_\phi$ and $\sigma_\phi$. This low-variance gradient estimator is the technical key that made VAEs trainable at scale.

> [!note] Why a Gaussian encoder is convenient
> With a diagonal-Gaussian posterior $q_\phi(z\mid x) = \mathcal{N}(\mu_\phi, \operatorname{diag}(\sigma_\phi^2))$ and a standard-normal prior, the KL regularizer has a **closed form** — no sampling needed for that term:
> $$
> D_{\mathrm{KL}}\!\big(q_\phi(z\mid x)\,\|\,\mathcal{N}(0,I)\big) = -\tfrac{1}{2}\sum_{j=1}^{d}\Big(1 + \log \sigma_j^2 - \mu_j^2 - \sigma_j^2\Big)
> $$
> The network conventionally outputs $\log \sigma^2$ rather than $\sigma$, so the variance stays positive without any constraint. Only the reconstruction term is estimated by sampling (usually a single $z$ per datapoint suffices).

## Architecture and training

A VAE is trained end-to-end by maximizing the ELBO (equivalently, minimizing its negative) with ordinary SGD. Each step:

1. Encode $x$ to posterior parameters $\mu_\phi(x)$, $\log\sigma_\phi^2(x)$.
2. Sample $z = \mu_\phi + \sigma_\phi \odot \epsilon$ via reparameterization.
3. Decode $z$ to reconstruct $x$, scoring the reconstruction term.
4. Add the closed-form KL regularizer.
5. Backpropagate through the whole graph.

To **generate** unconditionally, discard the encoder: sample $z \sim \mathcal{N}(0, I)$ and decode. This is the *defined* generative process — being able to ancestral-sample the prior directly is exactly what a fixed prior buys you, and what separates a VAE from a plain autoencoder. In practice, though, it is the weakest thing a vanilla VAE does: the KL only pushes the *aggregate* of encoded codes *toward* the prior, never onto it exactly, so a sampled $z$ can land in a region no training example mapped to and decode to something blurry or off-distribution — the aggregate-posterior mismatch discussed under [[#Failure modes|failure modes]] below.

A separate, more reliable use of the latent space is **editing** rather than de-novo generation: encode a real $x$, then perturb or interpolate its code and decode. This stays on the densely-populated part of the latent space — the region prior samples can miss — which is why latent interpolations and attribute edits look far cleaner than raw prior samples.

## Failure modes

- **Blurry samples.** The single most cited weakness. A Gaussian decoder with an MSE-like reconstruction loss averages over plausible outputs, so VAE samples are smoother and blurrier than [[Generative Adversarial Network|GAN]] samples. This is a large part of why [[Diffusion Models Root|diffusion models]] displaced VAEs for high-fidelity image synthesis.
- **Posterior collapse.** If the decoder is powerful enough to model $x$ without using $z$, the KL term drives $q_\phi(z\mid x)$ all the way to the prior — the latent carries no information and the encoder is ignored. Common with strong autoregressive decoders; mitigated by KL warm-up (annealing the KL weight) or free-bits constraints.
- **Aggregate-posterior / prior mismatch (the "prior hole").** The *marginal* of all encoded posteriors need not match the fixed $\mathcal{N}(0,I)$ prior, leaving prior regions that decode poorly. This is the specific gap a [[Normalizing Flows#Placement 2 — flow on the prior|flow-based prior]] is designed to close.

## Variants and extensions

- **$\beta$-VAE** — scales the KL regularizer by a factor $\beta > 1$ to encourage more disentangled latent factors, trading reconstruction fidelity for a more structured latent space.
- **Richer posteriors and priors via flows** — a [[Normalizing Flows|normalizing flow]] can warp the diagonal-Gaussian posterior into something multimodal (a tighter ELBO), or replace the fixed prior with a learned one; see the three placements worked out in the [[Normalizing Flows#Flows inside a VAE|flows-inside-a-VAE]] section.
- **VQ-VAE** — replaces the continuous Gaussian latent with a *discrete* codebook (vector quantization), sidestepping posterior collapse and underpinning many later latent-space generative models.
- **Conditional VAE (CVAE)** — conditions encoder, decoder, and prior on side information $c$, giving controllable generation. [[VITS]] is a conditional VAE at its core, with a flow-warped text-conditioned prior.

## Comparison to other generative model families

| | VAE | [[Generative Adversarial Network\|GAN]] | [[Normalizing Flows\|Flow]] | [[Diffusion Models Root\|Diffusion]] |
|---|---|---|---|---|
| Likelihood | Lower bound (ELBO) | None (implicit) | Exact | Lower bound (ELBO-derived) |
| Training | Stable | Unstable (adversarial) | Stable | Stable |
| Sampling speed | Fast (1 pass) | Fast (1 pass) | Fast (1 pass) | Slow (many passes) |
| Sample quality | Lower / blurrier | High | Moderate | Highest |
| Latent structure | Smooth, low-dim | Implicit | Same dim as data | Per-step noise |

## Key papers

- Kingma, D. P. & Welling, M. (2013). *Auto-Encoding Variational Bayes.* ICLR 2014. — the original VAE and reparameterization trick.
- Rezende, D. J., Mohamed, S. & Wierstra, D. (2014). *Stochastic Backpropagation and Approximate Inference in Deep Generative Models.* ICML. — concurrent derivation.
- Higgins, I. et al. (2017). *β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework.* ICLR.
- van den Oord, A., Vinyals, O. & Kavukcuoglu, K. (2017). *Neural Discrete Representation Learning (VQ-VAE).* NeurIPS.
- Kingma, D. P. & Welling, M. (2019). *An Introduction to Variational Autoencoders.* Foundations and Trends in ML. — survey.

## Related

- [[Generative Models]] — the family this belongs to
- [[Normalizing Flows]] — inserted into VAEs to enrich the posterior or prior
- [[Diffusion Models Root|Diffusion Models]] — also ELBO-based; can be read as a deep hierarchy of VAEs
- [[Generative Adversarial Network]] — the implicit-likelihood alternative
- [[Kullback-Leibler Divergence]] — the regularizer term of the ELBO
- [[Variational Inference]] — the general framework the ELBO comes from
- [[VITS]] — a conditional VAE for end-to-end text-to-speech
