---
title: Diffusion Models
type: concept
tags: [generative-models, diffusion, deep-learning, computer-vision]
created: 2026-06-29
status: budding
aliases: [Denoising Diffusion Probabilistic Models, DDPM, Diffusion Probabilistic Models]
---

# Diffusion Models

Diffusion models are a class of [[Generative Models|generative model]] that learn to generate data by reversing a gradual noising process. Instead of generating a sample in one shot (as a [[GAN]] or [[Variational Autoencoder|VAE]] does), a diffusion model produces data through many small denoising steps, starting from pure noise and iteratively refining it into a realistic sample.



> [!note] Intuition
> Take a real image and progressively add small amounts of Gaussian noise over many steps until it becomes indistinguishable from pure noise (the **forward process**). A neural network is trained to undo one noising step at a time (the **reverse process**). Run that reverse process starting from random noise, and it produces a new, realistic image.

## The forward (noising) process

The forward process $q$ is a fixed Markov chain that gradually adds Gaussian noise to data $x_0 \sim q(x_0)$ over $T$ timesteps:

$$
q(x_t \mid x_{t-1}) = \mathcal{N}\!\left(x_t;\ \sqrt{1-\beta_t}\, x_{t-1},\ \beta_t I\right)
$$

(see [[Semicolon Notation in Probability Distributions]] if the `;` inside $\mathcal{N}(\cdot)$ is unfamiliar notation, and [[Distribution vs Density Function]] for why a *distribution* $q$ can be written equal to a Gaussian PDF evaluated at $x_t$)

where $\beta_1, \dots, \beta_T$ is a small, increasing **noise schedule**. As $t \to T$, $x_t$ approaches an isotropic Gaussian, $x_T \sim \mathcal{N}(0, I)$.

> [!warning] "Adding noise" is not $x_t = x_{t-1} + \epsilon$
> Sampling from $q(x_t \mid x_{t-1})$ means:
> $$
> x_t = \sqrt{1-\beta_t}\, x_{t-1} + \sqrt{\beta_t}\, \epsilon, \qquad \epsilon \sim \mathcal{N}(0, I)
> $$
> The previous state is **shrunk** by $\sqrt{1-\beta_t}$ at the same time noise scaled by $\sqrt{\beta_t}$ is mixed in — it's a weighted interpolation between signal and noise, not plain addition of noise onto an untouched signal. This is what makes the process **variance-preserving**: if $\text{Var}(x_{t-1}) \approx 1$, then $\text{Var}(x_t) \approx (1-\beta_t) + \beta_t = 1$, so variance stays bounded at every step instead of growing without limit. Without the shrink term, repeated "real addition" of noise would make $x_t$ diverge rather than converge to a clean $\mathcal{N}(0, I)$ by step $T$.

A useful property: you can sample $x_t$ directly from $x_0$ without simulating every intermediate step, using $\bar{\alpha}_t = \prod_{s=1}^t (1-\beta_s)$:

$$
x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1-\bar{\alpha}_t}\, \epsilon, \qquad \epsilon \sim \mathcal{N}(0, I)
$$

Same idea, just compounded over $t$ steps: this is an interpolation between the original signal (weight $\sqrt{\bar{\alpha}_t}$) and pure noise (weight $\sqrt{1-\bar{\alpha}_t}$), with the two weights satisfying $\bar{\alpha}_t + (1-\bar{\alpha}_t) = 1$ — again variance-preserving, not additive. This closed form is what makes training efficient — no need to roll the chain forward step by step.

## The reverse (denoising) process

The generative model learns the reverse transition $p_\theta(x_{t-1} \mid x_t)$, approximated as Gaussian:

$$
p_\theta(x_{t-1} \mid x_t) = \mathcal{N}\!\left(x_{t-1};\ \mu_\theta(x_t, t),\ \Sigma_\theta(x_t, t)\right)
$$

In DDPM, the network doesn't output $\mu_\theta$ directly — it predicts the noise $\epsilon_\theta(x_t, t)$, and $\mu_\theta$ is computed from that prediction in closed form:

$$
\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\, \epsilon_\theta(x_t, t)\right)
$$

This drops straight out of inverting the forward sampling equation $x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon$ and substituting the network's noise estimate for the true $\epsilon$. The variance $\Sigma_\theta(x_t,t)$ is usually **not learned** at all in the original DDPM — it's fixed to either:

$$
\Sigma_\theta(x_t, t) = \beta_t I \qquad \text{or} \qquad \Sigma_\theta(x_t, t) = \tilde\beta_t I = \frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\,\beta_t\, I
$$

(the same $\tilde\beta_t$ from the true posterior variance — see [[Why Reverse Diffusion Adds Noise]]). Later work (e.g. "Improved DDPM") learns an interpolation between these two bounds instead of fixing it.

A network (typically a [[U-Net]] for images, increasingly a [[Vision Transformer|transformer]] in newer work — see [[Diffusion Transformers]]) is trained to predict either:
- the noise $\epsilon$ added at step $t$ (most common, as in DDPM), or
- the original signal $x_0$, or
- the **score** $\nabla_{x_t} \log q(x_t)$ (see [[Score-Based Generative Models]] below).

These parameterizations are mathematically equivalent up to reweighting and reparameterization.

## Training objective

DDPM (Ho et al., 2020) simplifies the [[Diffusion ELBO|variational lower bound]] on the data log-likelihood to a simple denoising loss — predict the noise $\epsilon$ that was added:

$$
\mathcal{L}_{\text{simple}} = \mathbb{E}_{t,\, x_0,\, \epsilon}\left[\ \lVert \epsilon - \epsilon_\theta(x_t, t) \rVert^2\ \right]
$$

This is just a weighted form of denoising score matching — at each training step:
1. Sample a real example $x_0$ and a random timestep $t$.
2. Sample noise $\epsilon$ and form $x_t$ via the closed-form forward equation above.
3. Have the network predict $\epsilon$ from $(x_t, t)$.
4. Minimize MSE between predicted and true noise.

No adversarial loss, no discriminator — training is stable, unlike [[GAN|GANs]].

## Sampling

Generation runs the reverse process starting from $x_T \sim \mathcal{N}(0, I)$ and iteratively denoising down to $x_0$.

- **DDPM sampling**: ancestral sampling through all $T$ steps (often 1000) — slow but high quality. Each step re-injects a small amount of fresh noise on top of the denoised mean; see [[Why Reverse Diffusion Adds Noise]] for why that's correct rather than counterproductive.
- **DDIM sampling** (Denoising Diffusion *Implicit* Models): a deterministic, non-Markovian reformulation that allows skipping steps, cutting sampling to ~20–50 steps with little quality loss. Widely used in practice.
- Faster samplers (DPM-Solver, ancestral samplers with higher-order ODE/SDE solvers) treat sampling as numerically solving a differential equation, framed in [[Score-Based Generative Models|score-based SDE]] terms.

> [!warning] Common pitfall
> Sampling speed vs. quality is the central practical tradeoff of diffusion models. Naively, generating one image can take far more network evaluations than a single GAN forward pass — this motivated DDIM, distillation methods (e.g. consistency models, progressive distillation), and latent-space diffusion.

## Guidance: controlling generation

Plain diffusion models generate unconditionally. To generate *toward* a desired output (e.g. matching a text prompt):

- **Classifier guidance**: use gradients from a separately trained classifier to nudge the sampling trajectory toward a class.
- **Classifier-free guidance (CFG)**: train the model with and without conditioning (e.g. text embeddings) and combine the two predictions at sampling time:

$$
\hat{\epsilon} = \epsilon_\theta(x_t, t, \varnothing) + w \cdot \big(\epsilon_\theta(x_t, t, c) - \epsilon_\theta(x_t, t, \varnothing)\big)
$$

where $c$ is the conditioning (e.g. text embedding from [[CLIP]]) and $w > 1$ strengthens adherence to the condition at some cost to diversity. CFG is the mechanism behind prompt fidelity in text-to-image models.

## Connection to score-based models

Diffusion models and [[Score-Based Generative Models|score-based generative models]] (Song & Ermon) are two views of the same underlying idea: learning $\nabla_x \log p(x)$ at multiple noise levels. The forward noising process can be framed as a continuous-time [[Stochastic Differential Equations|SDE]], with the reverse process given by the corresponding reverse-time SDE — unifying DDPM, score matching, and probability flow ODEs under one framework. See [[Markov Chains and SDEs in Diffusion Models]] for the full derivation of why the same trained network works as both a discrete-chain denoiser and a continuous-time score estimator.

## Latent diffusion

Running diffusion directly in pixel space is expensive. **Latent Diffusion Models** (Rombach et al., 2022 — the basis of Stable Diffusion) instead:
1. Compress images into a lower-dimensional latent space with a pretrained [[Variational Autoencoder|VAE]] (or VQ-VAE) encoder.
2. Run the full forward/reverse diffusion process in that latent space.
3. Decode the final latent back to pixel space.

This dramatically cuts compute while preserving perceptual quality, since most diffusion steps are spent on imperceptible high-frequency pixel detail rather than semantic content.

## Comparison to other generative model families

| | [[GAN]] | [[Variational Autoencoder\|VAE]] | Diffusion |
|---|---|---|---|
| Training stability | Unstable (adversarial) | Stable | Stable |
| Sample quality | High | Lower / blurrier | Highest (SOTA) |
| Sampling speed | Fast (1 pass) | Fast (1 pass) | Slow (many passes), improving |
| Likelihood-based | No | Yes (ELBO) | Yes (ELBO-derived) |
| Mode coverage | Prone to mode collapse | Good | Good |

## Key papers

- Sohl-Dickstein et al., 2015 — *Deep Unsupervised Learning using Nonequilibrium Thermodynamics* (the original diffusion probabilistic model idea)
- Ho, Jain, Abbeel, 2020 — *Denoising Diffusion Probabilistic Models* (DDPM)
- Song & Ermon, 2019/2020 — score-based generative modeling, SDE formulation
- Song et al., 2020 — *Denoising Diffusion Implicit Models* (DDIM)
- Dhariwal & Nichol, 2021 — classifier guidance, *Diffusion Models Beat GANs on Image Synthesis*
- Ho & Salimans, 2022 — *Classifier-Free Diffusion Guidance*
- Rombach et al., 2022 — *High-Resolution Image Synthesis with Latent Diffusion Models* (Stable Diffusion)
- Austin et al., 2021 — *Structured Denoising Diffusion Models in Discrete State-Spaces* ([[D3PM - Discrete Denoising Diffusion|D3PM]] — diffusion for discrete/categorical data)

## Related

- [[U-Net]]
- [[Chess Analogy for Diffusion Models]]
- [[Why Reverse Diffusion Adds Noise]]
- [[Bayes' Theorem in Diffusion Models]]
- [[Diffusion ELBO]]
- [[D3PM - Discrete Denoising Diffusion]]
- [[Markov Chains and SDEs in Diffusion Models]]
- [[Score-Based Generative Models]]
- [[Generative Models]]
- [[GAN]]
- [[Variational Autoencoder]]
- [[Stochastic Differential Equations]]
- [[CLIP]]
- [[Diffusion Transformers]]
