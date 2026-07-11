---
title: Generative Models
type: moc
tags: [generative-models, deep-learning]
created: 2026-07-04
status: seedling
aliases: [Generative Model, Deep Generative Models]
---

# Generative Models

A **generative model** learns the data distribution $p(x)$ itself — enough to *sample* new data — rather than only a conditional label $p(y \mid x)$ as a discriminative model does. This map-of-content organizes the vault's generative-model notes by the property that most shapes how each family is trained and sampled: **how it handles the likelihood** $p(x)$.

## Organized by likelihood tractability

The central design axis. Whether a model can compute $p(x)$ exactly, only bound it, or not at all, determines its training objective, its sampling procedure, and its failure modes.

- **Exact, tractable likelihood** — [[Normalizing Flows]]. An invertible map with a cheap Jacobian determinant gives an exact density via change-of-variables, so the model trains by direct maximum likelihood. The price is an architectural constraint: every layer must be invertible.
- **Lower-bounded likelihood (ELBO)** — [[Variational Autoencoder]] and [[Diffusion Models Root|Diffusion Models]]. Neither can evaluate $p(x)$ directly, so both optimize a variational lower bound (the ELBO). A VAE does it in one encode/decode; a diffusion model does it across many denoising steps.
- **No explicit likelihood (implicit)** — [[Generative Adversarial Network]]. A generator is trained by an adversarial signal from a discriminator, never by a density. High sample quality, but no likelihood and a training process prone to instability and mode collapse.

## Notes in this vault

- [[Normalizing Flows]] — flow-based models; exact-likelihood building block, also inserted into VAEs and TTS priors
- [[Diffusion Models Root|Diffusion Models]] — the deepest sub-tree here (forward/reverse process, ELBO, score-based view, guidance, latent diffusion)
- [[Variational Autoencoder]] — latent-variable model trained on the ELBO; reparameterization trick, amortized inference
- [[Generative Adversarial Network]] — implicit model trained as a generator-vs-discriminator game; no explicit likelihood

## Applied

- [[VITS Overview|VITS]] — end-to-end TTS that combines a conditional VAE, a normalizing-flow prior, and a GAN vocoder in one model — a concrete case study in mixing three of the families above

## Related

- [[Kullback-Leibler Divergence]] — the objective term that ties the likelihood-based families together
- [[Deep Learning MOC]]
