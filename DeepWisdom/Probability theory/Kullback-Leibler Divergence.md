---
title: Kullback-Leibler Divergence
type: concept
tags: [probability, information-theory, divergence]
created: 2026-07-04
status: budding
aliases: [KL Divergence, KL, Relative Entropy, Kullback–Leibler Divergence]
---

# Kullback-Leibler Divergence

The **Kullback–Leibler (KL) divergence** measures how much one probability distribution $Q$ differs from a reference distribution $P$. Read it as the expected number of *extra* nats (or bits) you pay to encode samples from $P$ when you use a code optimized for $Q$ instead of the true $P$. It is the workhorse objective behind [[Variational Autoencoder|variational autoencoders]], [[Variational Inference|variational inference]], and [[Normalizing Flows|normalizing flows]] — anywhere a model distribution is fit to a target by making them close.

## Definition

For discrete distributions over the same support:

$$
D_{\mathrm{KL}}(P \,\|\, Q) = \sum_{x} P(x)\, \log \frac{P(x)}{Q(x)} = \mathbb{E}_{x \sim P}\!\left[\log \frac{P(x)}{Q(x)}\right]
$$

and for continuous distributions the sum becomes an integral over densities. It is an **expectation under $P$** of the log-ratio $\log(P/Q)$ — the first distribution in the notation is always the one you average over.

## Properties

- **Non-negativity (Gibbs' inequality):** $D_{\mathrm{KL}}(P\|Q) \ge 0$, with equality *if and only if* $P = Q$ almost everywhere. This follows from [[Jensen's Inequality|Jensen's inequality]] applied to the convex function $-\log$. It is what makes KL usable as a "distance to close" in an objective — driving it to zero forces the distributions to match.
- **Asymmetry:** in general $D_{\mathrm{KL}}(P\|Q) \ne D_{\mathrm{KL}}(Q\|P)$. KL is **not** a metric — it violates symmetry and the triangle inequality. (The symmetric, bounded relative is the [[Jensen-Shannon Divergence|Jensen–Shannon divergence]].)
- **Not bounded:** if there is any $x$ where $Q(x) = 0$ but $P(x) > 0$, then $D_{\mathrm{KL}}(P\|Q) = \infty$. $Q$ must put mass everywhere $P$ does (absolute continuity).
- **Units:** nats with $\log_e$, bits with $\log_2$.

## Relation to entropy and cross-entropy

KL decomposes into cross-entropy minus entropy:

$$
D_{\mathrm{KL}}(P \,\|\, Q) = \underbrace{-\sum_x P(x)\log Q(x)}_{H(P,\,Q)\ \text{cross-entropy}} \;-\; \underbrace{\Big(-\sum_x P(x)\log P(x)\Big)}_{H(P)\ \text{entropy}}
$$

This is why minimizing **cross-entropy loss** over a fixed dataset is equivalent to minimizing $D_{\mathrm{KL}}(P_{\text{data}}\|Q_\theta)$: the data entropy $H(P)$ doesn't depend on the model $\theta$, so the two objectives share a gradient. Maximum-likelihood training *is* forward-KL minimization.

## Forward vs reverse KL

Because KL is asymmetric, *which order you write it in changes the behavior of the fit* — a distinction that matters constantly in generative modeling:

- **Forward KL, $D_{\mathrm{KL}}(P\|Q)$** (average under the true $P$): heavily penalizes any $x$ where $P$ has mass but $Q$ doesn't, so the fitted $Q$ is **mode-covering** ("zero-avoiding") — it spreads to cover all of $P$'s support, even smearing mass over gaps. This is the maximum-likelihood direction.
- **Reverse KL, $D_{\mathrm{KL}}(Q\|P)$** (average under the model $Q$): penalizes $Q$ putting mass where $P$ has none, so $Q$ is **mode-seeking** ("zero-forcing") — it locks onto one mode and ignores the rest. This is the direction that appears in the [[Variational Autoencoder#Variational inference and the ELBO|ELBO]] and variational inference.

> [!note] Gaussian closed form
> For two multivariate Gaussians the KL has a closed form; between a diagonal Gaussian and the standard normal $\mathcal{N}(0, I)$ it collapses to the term used in the VAE objective:
> $$
> D_{\mathrm{KL}}\big(\mathcal{N}(\mu, \operatorname{diag}(\sigma^2)) \,\|\, \mathcal{N}(0, I)\big) = -\tfrac{1}{2}\sum_{j=1}^{d}\big(1 + \log \sigma_j^2 - \mu_j^2 - \sigma_j^2\big)
> $$
> Having this in closed form is exactly why the Gaussian encoder/prior pairing is so convenient — the regularizer needs no sampling.

## Where it shows up in this vault

- **[[Variational Autoencoder|VAEs]]** — the KL between the approximate posterior and the prior is the regularizer half of the ELBO; the ELBO gap itself is a KL between the approximate and true posterior.
- **[[Variational Inference]]** — variational inference *is* the minimization of reverse KL to an intractable posterior.
- **[[Normalizing Flows]]** — fitting a flow by maximum likelihood is forward-KL minimization to the data.
- **[[Diffusion Models Root|Diffusion models]]** — each denoising step's loss comes from a KL between the true and learned reverse transition.
- **[[Generative Adversarial Network|GANs]]** — the original GAN objective minimizes [[Jensen-Shannon Divergence|JS divergence]], which is a symmetrized average of two KL terms.

## Related

- [[Jensen-Shannon Divergence]] — the symmetric, bounded divergence built from KL
- [[Cross-Entropy]] — KL plus a constant data entropy; the standard classification loss
- [[Variational Autoencoder]] — KL as the ELBO's regularizer
- [[Variational Inference]] — reverse-KL minimization to a posterior
- [[Jensen's Inequality]] — the convexity result behind KL's non-negativity
