---
title: Medical Test Bayes Worked Example
type: reference
tags:
  - probability
  - bayesian
  - base-rate
  - worked-example
  - math
created: 2026-06-29
status: budding
aliases:
  - Base Rate Fallacy Example
  - Disease Test Bayes Example
  - False Positive Paradox
---

# Medical Test Bayes Worked Example

A fully worked numerical example of [[Bayes' Theorem]], using the canonical disease-screening setup. The goal is to make the **base-rate effect** concrete: why a positive result from a seemingly accurate test can still leave you probably healthy.

## The setup

Three numbers define the problem:

| Quantity | Symbol | Value |
|---|---|---|
| **Prevalence** (prior) | $p(D)$ | $0.01$ — 1% of people have the disease |
| **Sensitivity** (true-positive rate) | $p(+\mid D)$ | $0.90$ — the test catches 90% of true cases |
| **False-positive rate** | $p(+\mid H)$ | $0.05$ — it fires on 5% of healthy people |

The complements follow immediately: $p(H) = 0.99$, the **miss rate** (false-negative) $p(-\mid D)=0.10$, and the **specificity** (true-negative rate) $p(-\mid H)=0.95$.

We want the **posterior** $p(D\mid +)$: given a positive test, how likely is disease?

## Method 1 — build the joint grid

Take **health state** as the rows ($A$) and **test result** as the columns ($B$). Multiply each row's prior by its forward conditional, $J_{ab}=p(A)\,p(B\mid A)$, to fill every joint cell:

|  | Positive ($B_1$) | Negative ($B_2$) | **row sum $= p(A)$ (prior)** |
|---|---|---|---|
| **Disease ($A_1$)** | $0.01\times0.90 = 0.0090$ | $0.01\times0.10 = 0.0010$ | **$0.01$** |
| **Healthy ($A_2$)** | $0.99\times0.05 = 0.0495$ | $0.99\times0.95 = 0.9405$ | **$0.99$** |
| **col sum $= p(B)$ (evidence)** | **$0.0585$** | **$0.9415$** | $1.000$ |

The four interior cells are the four mutually exclusive outcomes for a random person; they sum to $0.0090+0.0010+0.0495+0.9405 = 1.0000$. Now every Bayesian quantity is just reading this one grid a particular way.

### Recovering the forward conditional (normalize a row)

Slice the Disease row and divide by its sum — this should hand back the sensitivity we put in:

$$
p(+\mid D)=\frac{0.0090}{0.01}=0.90, \qquad p(-\mid D)=\frac{0.0010}{0.01}=0.10. \checkmark
$$

### The reverse conditional (normalize a column)

The posterior is the **Positive column rescaled to sum to 1**. The column sum is the evidence:

$$
p(+) = \underbrace{0.0090}_{\text{true pos}} + \underbrace{0.0495}_{\text{false pos}} = 0.0585.
$$

$$
p(D \mid +) = \frac{0.0090}{0.0585} \approx 0.1538, \qquad p(H \mid +) = \frac{0.0495}{0.0585} \approx 0.8462.
$$

These two sum to 1, as a normalized column must. **A positive result from a 90%-sensitive test still leaves only a ~15% chance of disease.**

## Method 2 — the textbook Bayes formula

The same number drops straight out of [[Bayes' Theorem]] with the evidence expanded by the law of total probability:

$$
p(D\mid +) = \frac{p(+\mid D)\,p(D)}{p(+\mid D)\,p(D) + p(+\mid H)\,p(H)}
= \frac{0.90\times 0.01}{0.90\times 0.01 + 0.05\times 0.99}
= \frac{0.0090}{0.0090 + 0.0495} = \frac{0.0090}{0.0585} \approx 0.1538.
$$

The denominator is exactly the column sum from Method 1 — the formula *is* the grid, written on one line.

## Method 3 — odds form (the cleanest intuition)

Bayes in odds form removes the normalizer entirely: **posterior odds = prior odds × likelihood ratio.**

$$
\underbrace{\frac{p(D\mid +)}{p(H\mid +)}}_{\text{posterior odds}}
= \underbrace{\frac{p(D)}{p(H)}}_{\text{prior odds}}\times
\underbrace{\frac{p(+\mid D)}{p(+\mid H)}}_{\text{likelihood ratio (LR}^+)}.
$$

Plug in the numbers:

$$
\frac{p(D)}{p(H)} = \frac{0.01}{0.99} = \frac{1}{99}, \qquad
\text{LR}^+ = \frac{0.90}{0.05} = 18.
$$

$$
\text{posterior odds} = \frac{1}{99}\times 18 = \frac{18}{99} = \frac{2}{11} \approx 0.1818.
$$

Convert odds back to a probability with $p = \dfrac{\text{odds}}{1+\text{odds}}$:

$$
p(D\mid +) = \frac{2/11}{1 + 2/11} = \frac{2}{13} \approx 0.1538. \checkmark
$$

The story in one sentence: the test multiplied the odds of disease by **18**, but the prior odds were so tiny ($1{:}99$) that even an 18× boost only reaches $2{:}11$.

## Why the posterior is so low: the base-rate effect

> [!warning] The prior is doing the heavy lifting
> The forward slice rated Disease a great explanation of a positive test ($0.90$). But the Disease *row was tiny* ($0.01$), so its joint cell ($0.0090$) is dwarfed by the healthy-but-false-positive cell ($0.0495$) sitting in the same column. Among **all** positives, most come from the huge healthy population: $0.0495 / 0.0585 \approx 85\%$ are false alarms. Normalizing the column — which is what reversing the conditional means — is what drags the posterior down to $\approx 0.154$.

A frequency picture makes it visceral. Imagine **10,000 people**:

- $100$ have the disease; $90$ of them test positive (sensitivity $0.90$).
- $9{,}900$ are healthy; $5\%$ — that's $495$ — test positive anyway.
- Total positives: $90 + 495 = 585$. Of those, only $90$ are truly sick.
- $90/585 \approx 15.4\%$. Same answer, no fractions.

This is the **[[Bayes' Theorem|base-rate fallacy]]**: ignoring the prior $p(D)$ and reading the 90% sensitivity *as if* it were the posterior overstates the risk by roughly $6\times$.

## Follow-up: what a second independent test does

Suppose the first test came back positive and we run a **second, independent** test with the same characteristics. Bayesian updating is sequential — yesterday's posterior is today's prior. Working in odds form makes this trivial: just multiply by the likelihood ratio again.

$$
\text{posterior odds after 2 positives} = \frac{1}{99}\times 18 \times 18 = \frac{324}{99} \approx 3.27.
$$

$$
p(D\mid ++) = \frac{3.27}{1+3.27} \approx 0.766.
$$

One positive test: ~15%. **Two** independent positives: ~77%. The base rate that crippled a single test is overwhelmed once the evidence stacks — each LR$^+$ of 18 multiplies the odds, and two of them ($18^2 = 324$) finally beat the $1{:}99$ prior.

> [!note] Conditional independence caveat
> The clean "multiply the likelihood ratios" step assumes the two tests are **conditionally independent given disease status** — their errors are uncorrelated. Real repeat tests of the same biomarker often share failure modes (the same physiological quirk that fooled test 1 fools test 2), which inflates the true posterior estimate. Genuinely independent modalities are what make sequential testing powerful.

## Sensitivity to the prior

Because the posterior hinges on the base rate, it's worth seeing how it moves as prevalence changes (sensitivity $0.90$, false-positive $0.05$ held fixed):

| Prevalence $p(D)$ | Posterior $p(D\mid +)$ |
|---|---|
| $0.001$ (0.1%) | $\approx 0.018$ |
| $0.01$ (1%) | $\approx 0.154$ |
| $0.05$ (5%) | $\approx 0.486$ |
| $0.10$ (10%) | $\approx 0.667$ |
| $0.50$ (50%) | $\approx 0.947$ |

The same test is nearly worthless on a rare condition and quite trustworthy on a common one — **the test's accuracy is fixed, but its meaning is not.** This is why screening programs target high-prevalence subpopulations rather than the general public.

## Related

- [[Bayes' Theorem]] — the theorem this example instantiates
- [[Chess Analogy for Diffusion Models]] — the same "a cause can explain the data yet be unlikely" logic, in diffusion
- [[Distribution vs Density Function]]
- [[Semicolon Notation in Probability Distributions]]
