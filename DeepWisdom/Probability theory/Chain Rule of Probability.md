---
title: Chain Rule of Probability
type: concept
tags: [probability, foundations, math]
created: 2026-07-01
status: seedling
aliases: [Probability Chain Rule, General Product Rule, Chain Rule for Joint Probability]
---

# Chain Rule of Probability

The chain rule of probability says a **joint distribution over many variables can always be factored into a product of conditionals**, one variable at a time, with no approximation and no independence assumption required:

$$
p(x_1, x_2, \dots, x_n) = p(x_1)\,p(x_2\mid x_1)\,p(x_3\mid x_1, x_2)\,\cdots\,p(x_n \mid x_1, \dots, x_{n-1})
$$

or compactly,

$$
p(x_1, \dots, x_n) = p(x_1)\prod_{i=2}^{n} p(x_i \mid x_1, \dots, x_{i-1})
$$

This is an identity, not a modeling choice — it holds for *any* joint distribution, discrete or continuous.

> [!warning] Not the calculus chain rule
> Same name, unrelated idea. The calculus chain rule differentiates a composition of functions ($\frac{d}{dx}f(g(x))$); this one decomposes a joint probability into conditionals. The shared name is a historical coincidence.

## Where it comes from

It's repeated application of the definition of conditional probability, $p(A\mid B) = \frac{p(A,B)}{p(B)}$, i.e. $p(A,B) = p(A\mid B)\,p(B)$ — see the joint-grid picture in [[Bayes' Theorem#Geometric interpretation: conditioning is slicing a joint|Bayes' Theorem]]. Peel variables off one at a time:

$$
p(x_1,x_2,x_3) = p(x_3 \mid x_1,x_2)\,p(x_1,x_2) = p(x_3\mid x_1,x_2)\,p(x_2\mid x_1)\,p(x_1)
$$

and so on inductively for any number of variables. Nothing about this step requires the variables to be independent, Gaussian, or anything else — it's just bookkeeping on the definition of a conditional.

## The order is arbitrary — but not the terms

The factorization holds for **every** ordering of the variables; $p(x_1,x_2) = p(x_1)p(x_2\mid x_1) = p(x_2)p(x_1\mid x_2)$ are both valid chain-rule expansions of the same joint. Which ordering you pick is a modeling/computational choice — you choose the order that makes the conditionals *tractable to represent or learn*, not because the others are wrong.

> [!note] This is where the freedom in generative modeling lives
> An [[Autoregressive Models|autoregressive model]] commits to one fixed ordering (e.g. left-to-right over pixels or tokens) and learns every conditional $p_\theta(x_i \mid x_{<i})$ directly — no simplification, no Markov assumption, just the chain rule used exactly as written above. This is why autoregressive models can compute an **exact** likelihood: the chain rule already *is* their generative mechanism, so evaluating $p_\theta(x_0)$ is just multiplying together the factors the chain rule guarantees exist.

## How the Markov property cuts it down

The chain rule alone gives conditionals with **growing** conditioning sets ($p(x_i\mid x_1,\dots,x_{i-1})$ — everything before it). That's only tractable if most of that history is irrelevant. The **Markov property** is exactly the assumption that lets you drop it: if $x_i$ depends on the past only through $x_{i-1}$, every conditional collapses to

$$
p(x_i \mid x_1, \dots, x_{i-1}) = p(x_i \mid x_{i-1})
$$

so the general chain-rule product becomes a product of only **pairwise** conditionals:

$$
p(x_1, \dots, x_n) = p(x_1)\prod_{i=2}^n p(x_i \mid x_{i-1})
$$

This is chain rule (always true) **plus** Markov (a simplifying assumption about which conditionals reduce) — two separate ingredients that are easy to conflate because they're usually applied together.

## Where this shows up in diffusion

The [[Diffusion ELBO#Setup: the two trajectory distributions|joint density over an entire diffusion trajectory]] is a direct instance of this combination. Both trajectory distributions in [[Diffusion ELBO]] are chain-rule expansions that collapse to pairwise conditionals *because* the forward and reverse processes are constructed to be Markov chains:

$$
p_\theta(x_{0:T}) = p(x_T)\prod_{t=1}^T p_\theta(x_{t-1}\mid x_t), \qquad q(x_{1:T}\mid x_0) = \prod_{t=1}^T q(x_t\mid x_{t-1})
$$

Without the Markov assumption, the chain rule would still apply, but each factor would have to condition on the *entire* preceding trajectory ($p_\theta(x_{t-1}\mid x_t, x_{t+1}, \dots, x_T)$) instead of just the one adjacent state — intractable to parameterize. Markov structure is what keeps every factor a simple, fixed-size Gaussian conditional.

## Related

- [[Diffusion ELBO]]
- [[Bayes' Theorem]]
- [[Markov Chains and SDEs in Diffusion Models]]
- [[Autoregressive Models]]
- [[Why Generative Models Can Assign a Probability to Data]]
