---
title: Bayes' Theorem with Multiple Conditions
type: concept
tags: [probability, bayesian, foundations, math]
created: 2026-06-30
status: seedling
aliases: [Conditional Bayes, Bayes with Background Information, Bayes Multiple Conditions, Sequential Bayesian Updating]
---

# Bayes' Theorem with Multiple Conditions

The textbook form of [[Bayes' Theorem]] flips a single conditional, $p(A\mid B)$. In practice you almost always have **more than one** variable in play: some you are conditioning *on as fixed background*, and some you are *updating from as evidence*. This note covers the two distinct ways extra variables enter Bayes' rule — and they really are distinct:

1. **A held-fixed background condition $C$** — a context you assume throughout. Every term in Bayes' rule simply gains "$\mid C$".
2. **Extra evidence $B_1, B_2, \dots$** — new observations you fold in, one at a time or all at once, where *yesterday's posterior becomes today's prior*.

Keeping these two roles straight is the whole game.

## 1. A background condition: Bayes "inside a room"

Suppose everything happens under a known context $C$. Then Bayes' theorem holds verbatim, with $C$ carried along in **every** term:

$$
p(A\mid B, C) = \frac{p(B\mid A, C)\,p(A\mid C)}{p(B\mid C)}
$$

> [!note] The intuition: $C$ shrinks the world, then you do ordinary Bayes
> Conditioning on $C$ restricts the full joint distribution to the "sub-world" where $C$ is true. Inside that sub-world, $A$ and $B$ have their own prior, likelihood, and evidence — and Bayes' rule applies exactly as usual. $C$ is not evidence you're weighing; it's the room you've agreed to stand in. Geometrically (see [[Bayes' Theorem#Geometric interpretation: conditioning is slicing a joint|the joint-slicing picture]]), you first slice the joint down to the $C$ layer, then slice *that* along $A$ or $B$.

This is the form that powers the diffusion derivation. In [[Bayes' Theorem in Diffusion Models]] the reverse posterior is taken **given the clean image $x_0$** as background:

$$
q(x_{t-1}\mid x_t, x_0) = \frac{q(x_t\mid x_{t-1}, x_0)\,q(x_{t-1}\mid x_0)}{q(x_t\mid x_0)}
$$

Here $x_0$ plays the role of $C$ — fixed throughout, attached to every term — which is exactly what makes the otherwise-intractable prior collapse into a known Gaussian.

> [!warning] The background condition must appear in *every* term, including the prior
> A common slip is to condition the likelihood on $C$ but forget it in the prior, writing $p(A)$ instead of $p(A\mid C)$. The prior must also live inside the sub-world: it is your belief about $A$ *given $C$*, before seeing $B$. Dropping the $\mid C$ silently mixes two different worlds.

## 2. Multiple pieces of evidence: update one at a time

Now let $B_1$ and $B_2$ be two **observations** you want to learn from. Two equivalent routes give the same posterior.

### All at once

Treat the pair $(B_1, B_2)$ as a single compound observation and apply ordinary Bayes:

$$
p(A\mid B_1, B_2) = \frac{p(B_1, B_2\mid A)\,p(A)}{p(B_1, B_2)}
$$

### Sequentially — yesterday's posterior is today's prior

Fold in $B_1$ first, then treat its posterior as the prior when folding in $B_2$. Using Part 1 (with $B_1$ now serving as background for the second step):

$$
p(A\mid B_1, B_2) = \frac{p(B_2\mid A, B_1)\,\overbrace{p(A\mid B_1)}^{\text{posterior from step 1}}}{p(B_2\mid B_1)}
$$

> [!note] This recursion *is* Bayesian learning
> $$p(A) \;\xrightarrow{\ \text{observe } B_1\ }\; p(A\mid B_1) \;\xrightarrow{\ \text{observe } B_2\ }\; p(A\mid B_1, B_2) \;\xrightarrow{\ \dots\ }$$
> Each observation refines belief, and the order doesn't matter — processing $B_1$ then $B_2$ gives the same final posterior as $B_2$ then $B_1$, because both equal the all-at-once result. This is **sequential (recursive) Bayesian updating**, the backbone of online learning, Kalman filters, and Bayesian inference over streaming data.

## 3. Conditional independence: when the joint likelihood factors

The compound likelihood $p(B_1, B_2\mid A)$ is the term that gets expensive — in general you need the *joint* behavior of all observations given $A$. It simplifies dramatically when the observations are **conditionally independent given $A$**:

$$
B_1 \perp B_2 \mid A \quad\Longrightarrow\quad p(B_1, B_2\mid A) = p(B_1\mid A)\,p(B_2\mid A)
$$

so that

$$
p(A\mid B_1, B_2) \;\propto\; p(A)\,\prod_i p(B_i\mid A)
$$

> [!warning] Conditionally independent ≠ independent
> $B_1$ and $B_2$ may be strongly correlated *marginally* yet independent *once $A$ is known* — knowing $A$ "explains away" the correlation between them. The factorization is licensed only by independence **given $A$**, not by marginal independence. Assuming it when it doesn't hold is exactly the simplification (and the failure mode) of the **naive Bayes classifier**, which treats all features as conditionally independent given the class.

## 4. The Markov special case (used in diffusion)

A particularly clean kind of conditional independence: when conditioning on the *most recent* variable makes earlier ones irrelevant. This is the **Markov property** ([[Markov Chains and SDEs in Diffusion Models|Markov chains]]):

$$
p(x_t\mid x_{t-1}, x_0) = p(x_t\mid x_{t-1})
$$

Once you know $x_{t-1}$, the earlier $x_0$ tells you nothing more about $x_t$. This is precisely the step that collapses the likelihood in the diffusion posterior of Part 1: the term $q(x_t\mid x_{t-1}, x_0)$ — which *looks* like it needs the background $x_0$ — reduces to the known forward kernel $q(x_t\mid x_{t-1})$, leaving every factor in closed form. So in the diffusion derivation **both** multi-condition ideas appear at once: $x_0$ is a held-fixed background ($C$), while the Markov property strips $x_0$ back out of the likelihood specifically.

## Summary

| Role of the extra variable | Form | What it does |
|---|---|---|
| Background context $C$ | $p(A\mid B,C)=\dfrac{p(B\mid A,C)\,p(A\mid C)}{p(B\mid C)}$ | Restricts to a sub-world; attach $\mid C$ to every term |
| Extra evidence $B_2$ | $p(A\mid B_1,B_2)=\dfrac{p(B_2\mid A,B_1)\,p(A\mid B_1)}{p(B_2\mid B_1)}$ | Sequential update; prior $\leftarrow$ previous posterior |
| Conditionally independent evidence | $p(A\mid B_1,B_2)\propto p(A)\prod_i p(B_i\mid A)$ | Likelihood factorizes (naive Bayes) |
| Markov / "explained away" by latest | $p(x_t\mid x_{t-1},x_0)=p(x_t\mid x_{t-1})$ | Drop redundant earlier conditions |

The single rule underneath all of them: **a conditioning bar is just a context; anything to its right is held fixed while ordinary Bayes operates on what's to its left.** Whether a variable is "background" or "evidence" is a choice of viewpoint, not a different theorem.

## Related

- [[Bayes' Theorem]]
- [[Bayes' Theorem in Diffusion Models]]
- [[Medical Test Bayes Worked Example]]
- [[Markov Chains and SDEs in Diffusion Models]]
- [[Diffusion Models Root]]
