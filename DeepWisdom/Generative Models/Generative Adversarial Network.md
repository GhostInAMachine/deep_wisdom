---
title: Generative Adversarial Network
type: architecture
tags: [generative-models, gan, adversarial, deep-learning]
created: 2026-07-04
status: budding
aliases: [GAN, GANs, Generative Adversarial Networks, Adversarial Network]
---

# Generative Adversarial Network

A **generative adversarial network** (GAN; Goodfellow et al., 2014) is a [[Generative Models|generative model]] trained as a two-player game between a **generator** $G$ and a **discriminator** $D$. The generator turns a noise vector into a fake sample; the discriminator tries to tell real data from the generator's fakes. They improve against each other until the generator's samples are indistinguishable from real data. Crucially, a GAN never computes a density $p(x)$ — it is an **implicit** generative model, trained purely by the adversarial signal. That is the source of both its greatest strength (sharp, high-fidelity samples) and its notorious weaknesses (unstable training, mode collapse).

> [!note] Intuition
> Think of a counterfeiter ($G$) and a detective ($D$). The counterfeiter makes fake banknotes; the detective learns to spot them. Each forces the other to improve: better forgeries push the detective to look harder, and a sharper detective pushes the forger to be more convincing. At the end of this arms race the forgeries are so good the detective can do no better than a coin flip — which means the fakes match the real distribution.

## The minimax game

The generator $G_\theta$ maps a noise sample $z \sim p_z$ (typically $\mathcal{N}(0,I)$) to a fake $G_\theta(z)$, inducing a distribution $p_g$. The discriminator $D_\phi(x)$ outputs the probability that $x$ is real. They optimize a shared **value function** in opposite directions:

$$
\min_{G}\ \max_{D}\ \ V(D, G) = \mathbb{E}_{x \sim p_{\text{data}}}\big[\log D(x)\big] + \mathbb{E}_{z \sim p_z}\big[\log\big(1 - D(G(z))\big)\big]
$$

- $D$ **maximizes**: assign high probability to real $x$ and low probability to fakes $G(z)$.
- $G$ **minimizes**: make $D(G(z))$ high, i.e. fool the discriminator.

There is no likelihood anywhere in this objective — only a classifier's log-loss. This is what makes a GAN implicit: you can *sample* from $p_g$ but never *evaluate* it.

## What the game optimizes

For a **fixed** generator, the optimal discriminator is available in closed form:

$$
D^*(x) = \frac{p_{\text{data}}(x)}{p_{\text{data}}(x) + p_g(x)}
$$

Substituting $D^*$ back into $V$, the generator's objective reduces (up to a constant) to the **Jensen–Shannon divergence** between the data and model distributions:

$$
V(D^*, G) = -\log 4 + 2\cdot D_{\mathrm{JS}}\big(p_{\text{data}} \,\|\, p_g\big)
$$

Since [[Jensen-Shannon Divergence|JS divergence]] — a symmetric, bounded cousin of the [[Kullback-Leibler Divergence|KL divergence]] — is minimized (and equals zero) exactly when $p_g = p_{\text{data}}$, the global optimum of the game is the generator matching the data distribution, with $D^* \equiv \tfrac{1}{2}$ everywhere (the detective reduced to guessing). In practice $G$ and $D$ are updated by alternating gradient steps rather than fully solving each inner problem, and the theoretical optimum is rarely reached exactly.

> [!warning] The saturating-gradient fix
> Early in training, fakes are obviously bad, $D(G(z)) \approx 0$, and the $\log(1 - D(G(z)))$ term **saturates** — its gradient vanishes just when the generator most needs signal. The standard fix is the **non-saturating loss**: instead of minimizing $\log(1 - D(G(z)))$, the generator *maximizes* $\log D(G(z))$. Same fixed point, much stronger gradients when the generator is losing.

## Why GANs are hard to train

The adversarial formulation buys sample quality at the cost of a delicate optimization:

- **Mode collapse.** The generator can win by producing a few highly convincing samples and ignoring the rest of the data distribution — the classic GAN pathology. It maps many $z$ to the same output, so diversity collapses.
- **Non-convergence / instability.** The two networks chase a moving target; the game is a saddle-point problem, not a loss to descend, so training can oscillate or diverge rather than settle.
- **Vanishing gradients.** If $D$ becomes too strong too fast, it classifies perfectly, $V$ saturates, and $G$ receives no usable gradient.
- **No likelihood, hard to evaluate.** With no $p(x)$, there is no held-out likelihood to track. Progress is judged by proxy metrics like **FID** (Fréchet Inception Distance) and **Inception Score**, which are imperfect.

## Key variants

- **DCGAN** — the convolutional architecture recipe (strided convolutions, batch norm) that first made GAN training reliable on images.
- **Conditional GAN (cGAN)** — conditions both $G$ and $D$ on a label or input, enabling class-conditional and image-to-image generation (pix2pix, CycleGAN).
- **Wasserstein GAN (WGAN)** — replaces the JS objective with the **Earth-Mover (Wasserstein) distance**, which gives smoother gradients even when the two distributions barely overlap, easing mode collapse and instability (WGAN-GP adds a gradient penalty).
- **StyleGAN** — a style-based generator that set the standard for high-resolution photorealistic faces and controllable synthesis.
- **HiFi-GAN** — a GAN **vocoder** that turns spectrograms into raw audio waveforms; its adversarial waveform training is reused inside [[VITS]].

## Comparison to other generative model families

| | GAN | [[Variational Autoencoder\|VAE]] | [[Normalizing Flows\|Flow]] | [[Diffusion Models Root\|Diffusion]] |
|---|---|---|---|---|
| Likelihood | **None (implicit)** | Lower bound (ELBO) | Exact | Lower bound (ELBO-derived) |
| Training | Unstable (adversarial) | Stable | Stable | Stable |
| Sampling speed | Fast (1 pass) | Fast (1 pass) | Fast (1 pass) | Slow (many passes) |
| Sample quality | High / sharp | Lower / blurrier | Moderate | Highest |
| Mode coverage | Prone to collapse | Good | Good | Good |

## Key papers

- Goodfellow, I. et al. (2014). *Generative Adversarial Nets.* NeurIPS. — the original GAN.
- Radford, A., Metz, L. & Chintala, S. (2015). *Unsupervised Representation Learning with Deep Convolutional GANs (DCGAN).* ICLR 2016.
- Arjovsky, M., Chintala, S. & Bottou, L. (2017). *Wasserstein GAN.* ICML.
- Gulrajani, I. et al. (2017). *Improved Training of Wasserstein GANs (WGAN-GP).* NeurIPS.
- Karras, T. et al. (2019). *A Style-Based Generator Architecture for GANs (StyleGAN).* CVPR.

## Related

- [[Generative Models]] — the family this belongs to
- [[Jensen-Shannon Divergence]] — what the original GAN objective implicitly minimizes
- [[Kullback-Leibler Divergence]] — the asymmetric divergence JS is built from
- [[Variational Autoencoder]] — the likelihood-based, stable-training counterpart
- [[Diffusion Models Root|Diffusion Models]] — displaced GANs on many image tasks, trading speed for quality and stability
- [[VITS]] — uses a GAN (HiFi-GAN) vocoder for waveform generation
