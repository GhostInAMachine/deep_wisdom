---
title: Bayes' Theorem
type: concept
tags: [probability, bayesian, foundations, math]
created: 2026-06-29
status: seedling
aliases: [Bayes Theorem, Bayes' Rule, Bayes Rule, Posterior]
---

# Bayes' Theorem

Bayes' theorem tells you how to **flip a conditional** — to turn "the probability of $B$ given $A$" into "the probability of $A$ given $B$":

$$
\underbrace{p(A \mid B)}_{\text{posterior}} \;=\; \frac{\overbrace{p(B \mid A)}^{\text{likelihood}}\;\overbrace{p(A)}^{\text{prior}}}{\underbrace{p(B)}_{\text{evidence}}}, \qquad p(B) = \sum_{A'} p(B \mid A')\,p(A')
$$

The four pieces:

| Term | Name | Reads as |
|---|---|---|
| $p(A\mid B)$ | **posterior** | what you believe about $A$ after seeing $B$ |
| $p(B\mid A)$ | **likelihood** | how well each $A$ explains the observed $B$ |
| $p(A)$ | **prior** | how plausible $A$ was *before* seeing anything |
| $p(B)$ | **evidence** | total probability of $B$, summed over all $A'$ (the normalizer) |

> [!note] The one-line takeaway
> The likelihood alone does **not** tell you the posterior. You must reweight it by the prior $p(A)$ — how probable the cause was in the first place — and renormalize by the evidence. "$A$ explains $B$ well" is worthless if $A$ was never going to happen.

## Geometric interpretation: conditioning is slicing a joint

Put the two variables on perpendicular axes and consider their **joint distribution** $p(A,B)$ as a surface (or, for discrete states, a grid of cells whose areas are the joint probabilities). Every probability in Bayes' theorem is a simple geometric operation on this one object.

> [!note] What $p(A, B)$ means
> The comma reads "**and**": $p(A, B)$ is the probability that $A$ and $B$ occur *together* — one number for each pair of values, e.g. $p(\text{Disease},\,\text{Positive})$. It is symmetric, $p(A,B)=p(B,A)$, since "both happen" doesn't care about order. Do not confuse it with the conditional $p(A\mid B)$ ("$A$ given $B$ has already occurred", which restricts attention to the $B$ slice) or with the [[Semicolon Notation in Probability Distributions|semicolon notation]] $\mathcal{N}(x;\mu,\sigma^2)$ (where the semicolon separates a variable from fixed parameters — not a joint at all). The whole grid of $p(A,B)$ values sums to 1; a single conditional slice sums to 1 only *after* renormalizing. The two link through $p(A,B) = p(A\mid B)\,p(B) = p(B\mid A)\,p(A)$ — a joint cell is a conditional slice scaled back down by the size of the slice it came from.

- **A joint cell** has area $p(A,B)$. It can be decomposed two equivalent ways — row-first or column-first:
$$
p(A,B) = p(B\mid A)\,p(A) = p(A\mid B)\,p(B)
$$
- **A prior** $p(A)$ is the **row sum** (marginalize out $B$): the total height of row $A$.
- **The evidence** $p(B)$ is the **column sum** (marginalize out $A$): the total height of column $B$.
- **A conditional is a normalized slice.** $p(B\mid A)$ is row $A$ rescaled so it sums to 1; $p(A\mid B)$ is column $B$ rescaled so it sums to 1.

Bayes' theorem is then just the statement that **the same rectangle can be sliced along either axis, and the cell areas must agree.** Converting the forward (row) slice into the reverse (column) slice requires knowing how tall each row was — that's exactly where the prior enters.

> [!note] Why the prior is geometrically unavoidable
> A forward conditional $p(B\mid A)$ throws away the *heights* of the rows: every row is renormalized to 1, so a likely cause and a near-impossible one look identical once sliced. To rebuild a column ($p(A\mid B)$) you must restore those heights — multiply each row back by its prior $p(A)$ — before renormalizing down the column. No amount of row information recovers column information without the priors.

## A concrete joint: a worked example

Make the grid literal. A disease affects 1% of people. A test catches 90% of true cases ($p(+\mid D)=0.9$) but also fires on 5% of healthy people ($p(+\mid H)=0.05$). Building the joint grid and normalizing the Positive column gives

$$
p(D \mid +) = \frac{0.009}{0.0585} \approx 0.154,
$$

so a positive result from a 90%-sensitive test still leaves only a **~15%** chance of disease — the prior $p(D)=0.01$ is doing the heavy lifting. This is the [[Chess Analogy for Diffusion Models#Bayes' theorem: forward transitions are only half the story|grandmaster argument]] in numbers: a predecessor can explain the observation perfectly yet still be unlikely, because it was unlikely to occur in the first place.

> [!example] Full computation
> See [[Medical Test Bayes Worked Example]] for the complete joint grid, three independent derivations (grid, formula, and odds form), a frequency picture, what a second test does, and how the posterior scales with prevalence.

## The connection to transitions between states

This geometry becomes especially sharp when $A$ and $B$ are **two states in time** — "before" ($A = x_{t-1}$) and "after" ($B = x_t$) — and the conditional is a **transition**. Collect the forward transitions into a stochastic matrix $T$:

$$
T_{ab} = p(x_t = b \mid x_{t-1} = a) \qquad (\text{each row } a \text{ sums to } 1)
$$

A naive guess for reversing the dynamics would be to **transpose** $T$. That is wrong in general. The correct reverse transition is built by Bayes, and it has a clean matrix-geometric form. Let $\pi_a = p(x_{t-1} = a)$ be the prior over source states. Then:

1. **Build the joint** by scaling each row of $T$ by its prior — this restores the row heights:
$$
J_{ab} = \pi_a\, T_{ab}, \qquad J = \operatorname{diag}(\pi)\, T
$$
2. **Read off the marginal** over the later state as column sums: $p(x_t = b) = \sum_a J_{ab}$.
3. **Normalize each column** of $J$ to get the reverse transition:
$$
p(x_{t-1} = a \mid x_t = b) = \frac{J_{ab}}{\sum_{a'} J_{a'b}} = \frac{\pi_a\,T_{ab}}{\sum_{a'}\pi_{a'}T_{a'b}}
$$

So **reversing a transition = scale rows by the prior, then normalize columns.** The reverse equals the plain transpose $T^\top$ *only* in the special case where the prior is uniform and the marginal is too — otherwise the prior-reweighting bends it away from the transpose.

> [!warning] Knowing the forward transition is not enough to reverse it
> $T$ gives you the row slices. The reverse process needs the column slices. The missing ingredient — the row heights $\pi$, i.e. the **prior probability of being in each predecessor state** — is precisely what a forward transition matrix does not contain. This is the structural reason a [[Diffusion Models Root|diffusion model]] must *learn* something: the forward noising kernel is known for free, but the prior over states (the data distribution) is not. See [[Chess Analogy for Diffusion Models#Bayes' theorem: forward transitions are only half the story|the grandmaster argument]] — a move may be consistent with a wrecked position, but that position has a tiny prior, so its column-normalized reverse probability collapses.

## How this powers reverse diffusion

In a [[Diffusion Models Root|diffusion model]] the states are noise levels of an image and the forward transition is a fixed Gaussian. The reverse step is exactly the Bayes posterior over the predecessor:

$$
q(x_{t-1}\mid x_t) \;\propto\; q(x_t \mid x_{t-1})\,q(x_{t-1})
$$

The likelihood $q(x_t\mid x_{t-1})$ is the known [[Diffusion Models Root#The forward (noising) process|forward kernel]]; the prior $q(x_{t-1})$ is the unknown data distribution at that noise level. Training captures that prior — equivalently, the network [[Markov Chains and SDEs in Diffusion Models#The equivalence: noise prediction *is* score estimation|learns the score]] $\nabla\log q(x_{t-1})$, which is the gradient of the log-prior and so supplies exactly the "row heights" Bayes needs. Conditioning on the clean image $x_0$ pins down that prior into a closed-form Gaussian, which is why $q(x_{t-1}\mid x_t, x_0)$ is the [[Why Reverse Diffusion Adds Noise#The reverse step is "sample the true posterior," not "invert the forward step"|tractable posterior]] used in the DDPM derivation.

For the full step-by-step derivation of the closed-form diffusion posterior $q(x_{t-1}\mid x_t, x_0)$ via Bayes' rule — plus how the same decomposition powers classifier-free guidance — see [[Bayes' Theorem in Diffusion Models]].

## Related

- [[Bayes' Theorem with Multiple Conditions]]
- [[Bayes' Theorem in Diffusion Models]]
- [[Medical Test Bayes Worked Example]]
- [[Chess Analogy for Diffusion Models]]
- [[Diffusion Models Root]]
- [[Why Reverse Diffusion Adds Noise]]
- [[Markov Chains and SDEs in Diffusion Models]]
- [[Semicolon Notation in Probability Distributions]]
