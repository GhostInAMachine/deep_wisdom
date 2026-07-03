---
title: Semicolon Notation in Probability Distributions
type: reference
tags:
  - notation
  - probability
  - math
created: 2026-06-29
status: budding
aliases:
  - Semicolon in Gaussian Notation
---

# Semicolon Notation in Probability Distributions

In an expression like

$$
\mathcal{N}(x;\ \mu,\ \sigma^2)
$$

the semicolon separates **the variable the density is evaluated at** (left of `;`) from **the fixed parameters of the distribution** (right of `;`). It shows up constantly in deep learning papers, including the [[Diffusion Models Root|diffusion forward process]]:

$$
q(x_t \mid x_{t-1}) = \mathcal{N}\!\left(x_t;\ \sqrt{1-\beta_t}\, x_{t-1},\ \beta_t I\right)
$$

## What it does *not* mean

It's easy to misread the semicolon as some kind of conditioning bar (like `|`) or as separating arguments of a joint distribution. It's neither:

- It is **not conditioning**. $\mathcal{N}(x; \mu, \sigma^2)$ is not "$x$ given $\mu,\sigma^2$" in the probabilistic sense of conditioning on a random variable — $\mu$ and $\sigma^2$ aren't being conditioned on, they're just the parameters that define which particular Gaussian curve is being described.
- It is **not a joint distribution** over $x, \mu, \sigma^2$ as if they were all random variables of interest. Only the variable to the left of the semicolon is the one whose probability is in question.

## What it actually means

$\mathcal{N}(x; \mu, \sigma^2)$ is shorthand for "evaluate the Gaussian probability density function, with mean $\mu$ and variance $\sigma^2$ plugged in as fixed parameters, at the point $x$":

$$
\mathcal{N}(x; \mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\!\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)
$$

This is the same convention used throughout statistics for any **parameterized function** $f(x;\theta)$ — the semicolon signals "this is a function of $x$, and $\theta$ happens to be fixed for this expression," as opposed to a function of both $x$ and $\theta$ jointly. It's purely a readability convention, not a different mathematical operator: swapping the semicolon for a comma, $\mathcal{N}(x, \mu, \sigma^2)$, would describe the exact same number — some textbooks do write it that way. The semicolon just makes the asymmetry between "the random variable" and "the knobs describing its distribution" visually explicit.

## Two related but distinct uses of $\mathcal{N}$

It helps to keep these apart, since the same symbol $\mathcal{N}$ is used for both:

| Notation | Meaning |
|---|---|
| $X \sim \mathcal{N}(\mu, \sigma^2)$ | $X$ is a random variable *distributed according to* this Gaussian. No semicolon — this names a distribution, not a number. |
| $\mathcal{N}(x;\ \mu,\ \sigma^2)$ | The *density function* of that same Gaussian, evaluated at a specific point $x$. This is a number (or array), computed by plugging $x$, $\mu$, $\sigma^2$ into the PDF formula above. |

## Reading the diffusion example

$$
q(x_t \mid x_{t-1}) = \mathcal{N}\!\left(x_t;\ \sqrt{1-\beta_t}\, x_{t-1},\ \beta_t I\right)
$$

reads as: "the conditional density $q(x_t \mid x_{t-1})$ — i.e. the density over $x_t$, given a specific value of $x_{t-1}$ — is a Gaussian density evaluated at $x_t$, whose mean is $\sqrt{1-\beta_t}\,x_{t-1}$ and whose covariance is $\beta_t I$." The conditioning on $x_{t-1}$ is carried by the `$\mid$` on the left-hand side *and* shows up consistently on the right as the mean parameter being a function of $x_{t-1}$ — the semicolon notation and the conditioning notation are doing two different, complementary jobs in the same line.

## Related

- [[Diffusion Models Root]]
