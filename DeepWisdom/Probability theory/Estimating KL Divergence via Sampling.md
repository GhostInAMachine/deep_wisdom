---
title: Estimating KL Divergence via Sampling
type: concept
tags: [probability, information-theory, monte-carlo, variational-inference, deep-learning]
created: 2026-07-03
status: seedling
aliases: [Monte Carlo KL Estimation, Sampling-Based KL Divergence, KL Divergence Estimators]
---

# Estimating KL Divergence via Sampling

When [[Kullback-Leibler Divergence|KL divergence]] between two distributions doesn't have a closed-form integral — because one or both sides are non-conjugate, implicit, or reshaped by something like a [[Normalizing Flows|normalizing flow]] — it can still be estimated from samples, as long as you can evaluate both distributions' log-densities pointwise. This is exactly the situation in [[VITS Overview|VITS]]'s training loss: its audio posterior and its flow-warped, MAS-aligned text prior are not a conjugate pair, so the KL term is computed by sampling rather than by a textbook formula (see [[VITS Overview#How the KL term is computed]] for the full worked derivation).

## Why not just use the closed form?

The familiar analytic KL formulas — e.g. between two diagonal Gaussians — only exist for a handful of conjugate distribution pairs. The moment either side is something more complex (a Gaussian pushed through an invertible flow, an autoregressive model, an implicit distribution defined only by a sampling procedure), there's no closed-form integral left to evaluate. But the definition of KL divergence is still just an expectation:

$$
D_{\text{KL}}(q \,\|\, p) = \mathbb{E}_{x \sim q}\big[\log q(x) - \log p(x)\big]
$$

and any expectation can be estimated by sampling.

## The basic estimator

Draw a single sample `x ~ q` and plug it into the log-ratio directly:

$$
k_1 = \log q(x) - \log p(x)
$$

Since this is exactly the quantity inside the expectation that *defines* the KL divergence, `k_1` is an **unbiased** estimator: $\mathbb{E}_q[k_1] = D_{\text{KL}}(q\|p)$, exactly, for any q and p you can evaluate pointwise. This is the estimator [[VITS Overview|VITS]] uses for its `L_kl` term — reusing the very same latent sample already drawn via the reparameterization trick for the reconstruction loss, so it costs nothing extra to compute.

> [!warning] Unbiased doesn't mean well-behaved
> A single-sample `k_1` estimate can come out **negative**, even though the true KL divergence is always ≥ 0 (via [[Jensen's Inequality|Jensen's/Gibbs' inequality]]). Non-negativity is a property of the *expectation*, not of any individual sample — `log q(x) - log p(x)` can go either way depending on where `x` happened to land. It can also have high variance, especially where `q` and `p` disagree a lot in the tails.

## A better-behaved alternative

Let `r = p(x)/q(x)` be the density ratio, for `x ~ q`. One useful fact: `r` has expectation exactly 1 under `q` (assuming matching support),

$$
\mathbb{E}_{x\sim q}[r] = \int q(x)\,\frac{p(x)}{q(x)}\,dx = \int p(x)\,dx = 1
$$

which means $\mathbb{E}_q[r - 1] = 0$ — a term you can add to any estimator for free without changing its expectation. Adding it to $-\log r = \log q(x) - \log p(x)$ gives a second unbiased estimator:

$$
k_3 = r - 1 - \log r
$$

still unbiased, since $\mathbb{E}_q[k_3] = \mathbb{E}_q[r-1] + \mathbb{E}_q[-\log r] = 0 + D_{\text{KL}}(q\|p)$. What makes `k_3` more useful than `k_1` is that it's **non-negative for every single sample**, not just in expectation — because $r - 1 \ge \log r$ for all $r > 0$ (equality only at $r=1$), which is the same convexity fact ([[Jensen's Inequality]]) behind the proof that KL divergence itself is never negative. In practice `k_3` also tends to have noticeably lower variance than `k_1`. This construction and the `k1`/`k2`/`k3` naming come from John Schulman's widely-cited note *"Approximating KL Divergence"* (2020), written in the context of monitoring/penalizing the KL between reinforcement-learning policies — a setting with exactly the same constraint as VITS's flow-based prior: you can evaluate both log-densities at a sampled point, but have no closed-form integral.

> [!note] A third option: k2
> Schulman's note also describes $k_2 = \tfrac12(\log r)^2$ — always non-negative (it's a square), often lower-variance still, but **biased**: its expectation isn't exactly the KL divergence, only close to it when `q` and `p` are already similar. It's a compromise that `k_3` generally dominates, since `k_3` gets non-negativity *and* exact unbiasedness together.

## Where this shows up

- **[[VITS Overview|VITS]]** — `L_kl` is computed with the `k_1`-style estimator between the posterior encoder's `q_φ(z|x_lin)` and the flow-warped prior `p_θ(z|c)`, reusing the reparameterized sample already drawn for reconstruction.
- **RLHF / policy-gradient fine-tuning** — estimating or penalizing the KL between an updated policy and a reference policy, where only pointwise log-probabilities are available (no analytic form for either policy in general). `k_3` in particular is popular here because it's cheap, unbiased, and never produces a nonsensical negative "divergence."
- Any **[[Variational Autoencoder|VAE]]** whose prior and posterior aren't a simple conjugate pair — flow-based priors, learned/implicit posteriors, hierarchical latents — inherits the same problem and the same fix.

## Practical notes

- Variance shrinks by averaging over more samples (`n` draws from `q` instead of 1), at proportional extra compute — but in a VAE-style training loop, reusing the single sample already drawn for the reconstruction term is usually preferred over paying for more samples just to tighten the KL estimate.
- These estimators only need pointwise log-densities `log q(x)` and `log p(x)` (or their ratio) — they don't require either distribution to have a known form, an accessible CDF, or a tractable normalizing constant beyond what's needed to evaluate the density itself.

## Related

- [[Kullback-Leibler Divergence]]
- [[Jensen's Inequality]]
- [[VITS Overview]]
- [[Variational Autoencoder]]
- [[Normalizing Flows]]
- [[Diffusion ELBO]]
