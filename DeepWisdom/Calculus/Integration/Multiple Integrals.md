---
title: Multiple Integrals
type: concept
tags: [calculus, integration, foundations, math]
created: 2026-07-01
status: seedling
aliases: [Multivariable Integral, Iterated Integral, n-fold Integral]
---

# Multiple Integrals

A **multiple integral** extends ordinary (single-variable) integration to a function of several variables at once — integrating over a region of $\mathbb{R}^n$ rather than an interval of $\mathbb{R}$. The double, triple, and general $n$-fold integrals

$$
\iint_D f(x,y)\,dA, \qquad \iiint_V f(x,y,z)\,dV, \qquad \int_{\mathbb{R}^n} f(x_1,\dots,x_n)\,dx_1\cdots dx_n
$$

all compute the same kind of thing: the total of $f$ accumulated over a multi-dimensional region, with the volume element ($dA$, $dV$, $dx_1\cdots dx_n$) generalizing the single-variable $dx$.

## Two equivalent ways to read it

- **As one integral over a higher-dimensional space.** The region $D$, $V$, or $\mathbb{R}^n$ is a single object; $f$ assigns a value to each point in it; the integral sums $f$ against volume, exactly like a 1D integral sums a function against length. Nothing here is conceptually different from the single-variable case — only the dimension of the domain changed.
- **As an *iterated* integral.** In practice you compute it one variable at a time, holding the others fixed:
$$
\iint_D f(x,y)\,dx\,dy = \int\!\left(\int f(x,y)\,dx\right)dy
$$
**Fubini's theorem** says that when $f$ is well-behaved (e.g. integrable, non-negative, or absolutely integrable), the order you peel variables off in doesn't matter — every iterated ordering, and the single "all-at-once" integral over the region, agree on the same number.

These two readings are not two different objects — they're two ways of computing the *same* integral. This is exactly the same relationship as [[Chain Rule of Probability]]: a joint object (a probability, or here an integral) can be built up one variable at a time in any order, and all orders give the same result.

## Notation compresses many symbols into one

Just as $x_{1:T}$ bundles $T$ separate variables into one notational object (see [[Diffusion ELBO#^colon-notation|the colon-notation callout]]), a multiple integral's differential bundles many single differentials into one symbol:

$$
dx_{1:T} \;:=\; dx_1\,dx_2\cdots dx_T
$$

$$
\int p_\theta(x_{0:T})\,dx_{1:T} \;=\; \int\!\!\cdots\!\!\int p_\theta(x_0,x_1,\dots,x_T)\,dx_1\,dx_2\cdots dx_T
$$

so this is a genuine $T$-fold multiple integral, one integral sign (and one $dx_t$) per latent variable — the compact notation just hides the repetition. See [[Diffusion ELBO#^colon-notation|the worked-through diffusion example]] for why this specific integral, over $T$ copies of a full-image-sized space, is intractable in closed form.

## Why closed forms are rare in high dimensions

A single-variable integral often has a closed form because there are only so many ways one variable can enter an expression. A multiple integral over $n$ variables has to account for how the integrand couples *all* of them together — coupling that, in general, doesn't factor into a product of easy one-dimensional integrals. Closed forms survive in special cases:

- The integrand **factors** into a product of functions, each depending on only one variable — then the multiple integral factors into a product of ordinary 1D integrals.
- The integrand is a **jointly Gaussian** density (or another distribution family closed under marginalization) — the multivariable integral of a Gaussian over a subset of its variables is again Gaussian, in closed form.

Neither holds for a chain of neural-network conditionals like $p_\theta(x_{0:T})$: each factor $p_\theta(x_{t-1}\mid x_t)$ is Gaussian *given* $x_t$, but the mean $\mu_\theta(x_t,t)$ is a nonlinear function of $x_t$, so the product across all $t$ does not collapse into one clean joint Gaussian over $x_{1:T}$ once $x_t$ is integrated out. That non-factorizing, non-Gaussian coupling across $T$ variables is precisely why $\int p_\theta(x_{0:T})\,dx_{1:T}$ has no closed form — and why [[Diffusion ELBO|the ELBO]] exists as a tractable substitute.

## Related

- [[Diffusion ELBO]]
- [[Chain Rule of Probability]]
- [[Jensen's Inequality]]
