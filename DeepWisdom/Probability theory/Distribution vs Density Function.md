---
title: Distribution vs Density Function
type: reference
tags:
  - notation
  - probability
  - math
created: 2026-06-29
status: budding
aliases:
  - PDF vs Distribution
  - Density vs Distribution
---

# Distribution vs Density Function

A common source of confusion in expressions like the [[Diffusion Models Root|diffusion forward process]]:

$$
q(x_t \mid x_{t-1}) = \mathcal{N}\!\left(x_t;\ \sqrt{1-\beta_t}\, x_{t-1},\ \beta_t I\right)
$$

We say "$q$ is a distribution," yet the right-hand side is the **probability density function (PDF)** of a Gaussian *evaluated at the point $x_t$* — which is just a number. How can a distribution equal a number? The resolution is that the symbol $q$ is **overloaded**, and the `=` here is a *functional identity*, not a single-point statement.

## Distribution vs density: two different objects

- A **distribution** is the abstract object that assigns probability to events — formally a probability measure. It answers "how is probability spread over the space?"
- A **density** (PDF) is a *function* $f : \mathbb{R}^d \to [0, \infty)$ that *represents* a continuous distribution: the probability of a region $A$ is recovered by integrating, $P(A) = \int_A f(x)\, dx$.

For a continuous distribution the two determine each other completely: **knowing the density at every point is the same information as knowing the distribution.** So defining the density everywhere *is* defining the distribution — just expressed as a formula instead of as a measure.

## The symbol $q$ is doing double duty

In diffusion-model notation, $q$ is written two ways that look the same but mean different things:

| Expression | What it denotes |
|---|---|
| $q$, or "$x_t \sim q(\cdot \mid x_{t-1})$" | the **distribution** (the abstract object you sample from) |
| $q(x_t \mid x_{t-1})$ with an explicit value $x_t$ | the **density value** — a number: how much probability mass per unit volume sits at that particular $x_t$ |

This reuse of one letter for both the distribution and its density is standard and almost never flagged. It's the same overloading discussed for the Gaussian symbol $\mathcal{N}$ in [[Semicolon Notation in Probability Distributions]].

## Why the equation specifies the whole distribution

The key move: in

$$
q(x_t \mid x_{t-1}) = \mathcal{N}\!\left(x_t;\ \sqrt{1-\beta_t}\, x_{t-1},\ \beta_t I\right)
$$

the $x_t$ on **both** sides is a **free variable**, not one fixed point. The equation is asserted to hold *for every possible value of $x_t \in \mathbb{R}^d$*. Read it as

$$
\forall x_t:\quad q(x_t \mid x_{t-1}) = \mathcal{N}\!\left(x_t;\ \sqrt{1-\beta_t}\, x_{t-1},\ \beta_t I\right).
$$

So it's not "the distribution equals the value at one point." It's "the density function $x_t \mapsto q(x_t \mid x_{t-1})$ is, *as a whole function*, the Gaussian density function." Because density-everywhere $\Leftrightarrow$ distribution, pinning down the function on the left pins down the distribution. The pointwise-looking notation is just the conventional way to *define a function by giving its value at a generic argument* — exactly like writing $f(x) = x^2$ defines the whole function, not a single number.

> [!note] One-line takeaway
> $q(x_t \mid x_{t-1}) = \mathcal{N}(x_t; \dots)$ should be read as "for all $x_t$, the density of $x_t$ equals this Gaussian density." Stating a density at *every* generic point is logically equivalent to specifying the entire distribution — that's why a pointwise-looking formula legitimately defines "the distribution $q$."

## Related

- [[Semicolon Notation in Probability Distributions]]
- [[Diffusion Models Root]]
