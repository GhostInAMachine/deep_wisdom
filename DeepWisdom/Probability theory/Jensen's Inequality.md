---
title: Jensen's Inequality
type: concept
tags: [probability, convexity, inequality, foundations, math]
created: 2026-06-30
status: seedling
aliases: [Jensen Inequality, Jensen's Inequality for Expectations]
---

# Jensen's Inequality

Jensen's inequality relates **the function of an average** to **the average of a function**, whenever that function is convex or concave. It is the single fact that turns an intractable $\log\mathbb{E}[\cdot]$ into a tractable $\mathbb{E}[\log\,\cdot]$ — which is exactly the move that produces the [[Diffusion ELBO|ELBO]] (and every other evidence lower bound in variational inference).

## Statement

For a **convex** function $\varphi$ and a random variable $X$:

$$
\varphi\big(\mathbb{E}[X]\big) \;\le\; \mathbb{E}\big[\varphi(X)\big]
$$

For a **concave** function the inequality flips (apply the convex case to $-\varphi$):

$$
\varphi\big(\mathbb{E}[X]\big) \;\ge\; \mathbb{E}\big[\varphi(X)\big]
$$

In words: **push the average through a convex function and it can only grow; through a concave function it can only shrink.** The finite-sum / discrete case is the same statement with weights $\lambda_i \ge 0$, $\sum_i \lambda_i = 1$:

$$
\varphi\!\left(\sum_i \lambda_i x_i\right) \le \sum_i \lambda_i\,\varphi(x_i) \qquad (\varphi \text{ convex})
$$

> [!note] Which way does it point? Use the function's shape
> You never need to memorize the direction — just picture the curve. **Convex** ($\smile$, e.g. $x^2$, $e^x$, $-\log x$): the average lands *above* the curve, so $\mathbb{E}[\varphi(X)] \ge \varphi(\mathbb{E}[X])$. **Concave** ($\frown$, e.g. $\log x$, $\sqrt{x}$): the average lands *below*, so $\mathbb{E}[\varphi(X)] \le \varphi(\mathbb{E}[X])$.

## Geometric intuition: the chord lies above the curve

Convexity *means* that any chord between two points on the graph sits on or above the curve. For two points with weights $\lambda$ and $1-\lambda$, the point $\lambda\varphi(x_1) + (1-\lambda)\varphi(x_2)$ (on the chord) is $\ge \varphi(\lambda x_1 + (1-\lambda)x_2)$ (on the curve). Jensen's inequality is just this fact extended from two points to an arbitrary average — an expectation is a convex combination of values, so its image under a convex curve is pinned below the corresponding combination of images.

> [!note] A cleaner one-line proof (supporting line)
> At the point $\mu = \mathbb{E}[X]$, a convex $\varphi$ has a **supporting line** $\ell(x) = \varphi(\mu) + g\,(x-\mu)$ lying entirely below it: $\varphi(x) \ge \ell(x)$ for all $x$. Take expectations: $\mathbb{E}[\varphi(X)] \ge \mathbb{E}[\ell(X)] = \varphi(\mu) + g\,(\mathbb{E}[X]-\mu) = \varphi(\mu)$. Done.

## When is it an equality?

The gap closes — $\varphi(\mathbb{E}[X]) = \mathbb{E}[\varphi(X)]$ — exactly when **either**:

- $X$ is (almost surely) **constant**, so there's nothing to average over; or
- $\varphi$ is **affine (linear)** on the support of $X$, so the curve *is* its own chord.

Otherwise, for a strictly convex/concave $\varphi$ and a genuinely random $X$, the inequality is **strict**. This "how far from equality" quantity is not noise — it is often the object of interest: in variational inference the Jensen gap *is* a KL divergence (below).

> [!warning] $\mathbb{E}[f(X)] \ne f(\mathbb{E}[X])$ in general
> A constant temptation is to move an expectation inside a nonlinear function. Jensen says you cannot do this for free — you can only do it *as an inequality*, in the direction set by the curvature. Equality requires linearity. This is also why estimators built by plugging a mean into a nonlinear function are biased (e.g. $\mathbb{E}[\log \hat{p}\,] < \log \mathbb{E}[\hat{p}\,]$).

## The case that matters for ML: $\log$ is concave

The logarithm is **concave**, so Jensen gives the most-used form in machine learning:

$$
\log \mathbb{E}[X] \;\ge\; \mathbb{E}[\log X]
$$

This is the lever behind the evidence lower bound. To bound an intractable marginal likelihood $p_\theta(x)$, introduce any distribution $q(z)$ over latents, write the marginal as an expectation, and apply Jensen to pull the $\log$ inside:

$$
\log p_\theta(x) = \log \mathbb{E}_{q(z)}\!\left[\frac{p_\theta(x,z)}{q(z)}\right] \;\ge\; \mathbb{E}_{q(z)}\!\left[\log \frac{p_\theta(x,z)}{q(z)}\right] =: \text{ELBO}
$$

The right-hand side is tractable (no integral inside the log), and it is a genuine *lower* bound precisely because $\log$ is concave. Maximizing it pushes up a floor underneath the true log-likelihood.

> [!note] The Jensen gap here is exactly a KL divergence
> The slack in this inequality is not arbitrary: $\log p_\theta(x) - \text{ELBO} = D_{\text{KL}}\big(q(z)\,\|\,p_\theta(z\mid x)\big) \ge 0$. So the bound is tight exactly when $q$ equals the true posterior — connecting Jensen directly to [[Bayes' Theorem]]. In diffusion models, $z$ is the whole noising trajectory $x_{1:T}$ and $q$ is the fixed forward process; see [[Diffusion ELBO#Deriving the bound|the diffusion ELBO derivation]] for this exact step in context.

## Other appearances

- **AM–GM inequality**: applying Jensen to the concave $\log$ over a uniform average recovers arithmetic-mean $\ge$ geometric-mean.
- **KL divergence $\ge 0$** (Gibbs' inequality): follows from Jensen applied to the convex $-\log$. This non-negativity is what makes every ELBO a valid bound — see [[Kullback-Leibler Divergence]].
- **Entropy bounds** and the concavity of entropy as a function of the distribution.
- **Bias of nonlinear estimators**, as noted above.

## Related

- [[Diffusion ELBO]]
- [[Bayes' Theorem]]
- [[Kullback-Leibler Divergence]]
- [[Variational Autoencoder]]
