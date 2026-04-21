# Accuracy Landscape Analysis

## The Random Baseline

Evaluating 10,000 random ensembles (no optimization) at each (nqubit, H) combination
reveals a striking property of the accuracy landscape:

| nqubit | H    | mean  | std    | p90   | goal_sa best |
|--------|------|-------|--------|-------|--------------|
| 4      | 0.0  | 0.268 | 0.070  | 0.350 | 1.000        |
| 4      | 0.5  | 0.268 | 0.070  | 0.350 | 1.000        |
| 4      | 1.0  | 0.269 | 0.069  | 0.350 | 1.000        |
| 6      | 0.0  | 0.292 | 0.024  | 0.324 | 0.997        |
| 6      | 0.5  | 0.292 | 0.024  | 0.324 | 0.945        |
| 6      | 1.0  | 0.292 | 0.024  | 0.321 | 0.896        |
| 8      | 0.0  | 0.290 | 0.0085 | 0.301 | 0.903        |
| 8      | 0.5  | 0.290 | 0.0084 | 0.300 | 0.790        |
| 8      | 1.0  | 0.290 | 0.0082 | 0.300 | 0.625        |

Two things are immediately apparent:

1. **The mean accuracy of random ensembles is ~0.29 across all conditions** — independent
   of nqubit and goal entropy. The distribution does not depend on how hard the problem is.

2. **The standard deviation shrinks dramatically with nqubit** (0.070 → 0.024 → 0.008),
   while the goal_sa ceiling drops toward the random baseline. The gap between the random
   baseline and the optimum narrows with n.

## Concentration of Measure

The shrinking std is a manifestation of **concentration of measure** — a fundamental
property of high-dimensional spaces where random samples cluster tightly around the mean,
with fluctuations that vanish as dimension grows.

Each random ensemble is a set of `nstate` states, each covering a context of `2^(n-1)+1`
rep positions. The accuracy is determined by how well majority votes across all those
contexts align with the goal. With many states contributing to each position, the law of
large numbers stabilizes the majority vote at each rep position independently. So random
ensembles all converge to roughly the same accuracy regardless of goal.

At n=4, a random ensemble might land anywhere from ~0.15 to ~0.45 — there is genuine
variation in quality and a meaningful gradient to follow. At n=8, nearly every random
ensemble lands within a narrow band (std=0.008) around 0.29. The landscape has become
almost flat everywhere except at rare high-accuracy peaks.

## Why the Mean is ~0.29

This is not 0.5 (the naive random-guessing baseline) because many rep positions are NaN —
either no states cover them, or the votes tie exactly. NaN positions do not contribute to
accuracy. The ~0.29 reflects the fraction of goal bits that are covered and decided by a
random 3× ensemble, weighted by the ~0.5 chance of a correct random prediction among
covered positions. This coverage fraction is set by the combinatorics of the context
structure and is largely independent of n once nstate is held at 3× base.

## The Landscape Is Flattening

The gap between the random baseline and the goal_sa ceiling shrinks with n:

| | n=4  | n=6  | n=8  |
|--|------|------|------|
| H=0 gap | 0.73 | 0.71 | 0.61 |
| H=1 gap | 0.73 | 0.60 | 0.34 |

At n=8, H=1, goal_sa achieves only 0.625 — just 0.34 above the random baseline. The good
region is not just sparser in a combinatorial sense; the landscape is physically flattening
as n grows, making it harder to detect a gradient pointing toward the optimum.

---

## Implications for Optimization Strategy

The landscape structure has direct consequences for which optimization strategies are
likely to work at large n.

### What does not work well

**Pure random search**: useless — the random baseline is ~0.29 and is essentially
deterministic. Adding more random ensembles provides no signal.

**Random SA**: the Metropolis framework allows uphill moves and restarts, but the proposals
are random states drawn from the flat plateau. In a nearly flat landscape, most proposals
are neutral (same accuracy) rather than good or bad, so the acceptance criterion provides
almost no signal. This explains why random SA barely outperforms random initialization at
n=8.

**Gradient-based methods**: the landscape is discrete (accuracy changes in steps of
1/ngbits) and nearly flat, so numerical gradients estimated from random perturbations are
extremely noisy. Standard gradient descent would be ineffective.

**Greedy hill-climbing**: works locally but gets stuck quickly on the flat plateau where
most moves are neutral, with no signal pointing toward the rare high-accuracy peaks.

### What does work

**Goal-aware proposals** (`pick_new_alphas`): rather than sampling from the flat
distribution of random states, goal-aware SA constructs proposals that are already
optimally aligned with the current goal and rep. Each proposed state is the best possible
state for its generator — not a random draw from a flat landscape, but a targeted move
toward better accuracy. This is why goal-aware SA's advantage over random SA grows with n:
the goal-aware proposals provide signal precisely where the landscape offers none.

**Metropolis acceptance**: even with goal-aware proposals, the ensemble-level optimization
is non-trivial — a state that is locally good may conflict with others. The ability to
accept temporarily worse moves allows exploration of the ensemble space in ways that pure
greedy methods cannot.

### Broader implications

The combination of a flat random baseline, shrinking std, and dropping ceiling suggests
that as n grows, the optimization problem transitions from a **rugged landscape**
(many local optima of varying quality, explorable by gradient-following) to a
**needle-in-a-haystack landscape** (flat plateau with rare sharp peaks, explorable only
by informed search). In this regime:

- The amount of information provided by a random move approaches zero
- Optimization strategies that incorporate problem structure (like `pick_new_alphas`)
  become not just helpful but essential
- Population-based methods (genetic algorithms, CMA-ES) that maintain diversity while
  sharing structural information across individuals may also perform well, since they
  can collectively cover more of the plateau
- The information-theoretic cost of finding a good solution likely grows faster than
  polynomially with n, suggesting that for large n, problem-specific structure must be
  exploited at every step of the search

This is consistent with the observation that goal_sa's advantage over all other methods
grows with n — it is the only method currently tested that exploits goal structure at the
proposal level.

---

## Structural Analysis: Why H=1 Needs More States Than H=0

### The Observation

The scaling study (scripts/7_scaling/) shows that the optimal ensemble size n* grows
with goal entropy H. At nqubit=6:

| H    | n*  | acc* (at n*) | n*/n*_H0 |
|------|-----|--------------|----------|
| 0.0  | 40  | 0.9245       | 1.00     |
| 0.25 | 52  | 0.9533       | 1.30     |
| 0.5  | 68  | 0.9588       | 1.70     |
| 0.75 | 79  | 0.9560       | 1.98     |
| 0.875| 84  | 0.9451       | 2.10     |
| 1.0  | 104 | 0.9766       | 2.60     |

The question investigated in scripts/11_structural_analysis/ is: **what structural
property of the optimization landscape causes this H-dependent scaling?**

### Hypotheses Tested and Ruled Out

**Script 01 — H=0 oracle picker:** With a perfect picker (full knowledge of which
state is optimal), the H=0 vs H=1 gap persists. The gap is not caused by the picker
making worse choices for H=1.

**Script 02 — H=1 oracle picker:** Confirmed. With an oracle picker for H=1 goals,
H=1 is actually easier than H=0 for the picker alone. The gap is not a picker
deficiency.

**Script 03 — Random ensemble baseline:** Random ensembles score equally across all H
values. The landscape does not inherently favor H=0 goals.

**Scripts 04 & 05 — Both-covered pairs (shuffled pairing test):** In the standard
pairing, two goal bits can share coverage (be covered by the same PseudoGHZ context),
creating "both-covered" pairs that are harder to optimize jointly. Shuffling the pairing
so that no both-covered pairs exist reduces the number of available pairs slightly but
does not change the 2.1× ratio: H=0 n*=84, H=1 n*=176 (ratio 2.09×, same as original
2.10×). Both-covered pairs are not the cause.

**Script 06 — State uniqueness and degeneracy:** Ruled out as a factor.

### The Mechanism: Wrong-Pair Balance (Scripts 07 & 08)

**Script 07 (07_tail_convergence.jl):** Measured the SA step acceptance rate bucketed
by current accuracy. Key result:

| Accuracy bucket | H=0 acceptance | H=1 acceptance |
|----------------|----------------|----------------|
| [0.50, 0.70)   | 88%            | 77%            |
| [0.70, 0.85)   | 78%            | 31%            |
| [0.85, 0.95)   | 40%            | 0.7%           |
| [0.95, 1.00)   | 5.4%           | never reached  |

The acceptance rate collapses far faster for H=1 than H=0 as accuracy increases. This
is the tail behavior: H=1 gets exponentially harder to improve near convergence.

**Script 08 (08_wrong_pair_balance.jl):** The definitive experiment. Measured two
quantities against a **near-converged ensemble** at each accuracy level:

1. **Positive proposal rate p(a):** fraction of freshly proposed states with delta > 0
2. **Wrong-pair balance:** fraction of remaining wrong pairs that want agreement

Results at nqubit=6, shuffled pairing:

| acc  | p_H0  | p_H1   | ratio | H=0 balance | H=1 balance |
|------|-------|--------|-------|-------------|-------------|
| 0.70 | 28.3% | 18.8%  | 1.5×  | 1.000       | ~0.47       |
| 0.80 | 22.5% | 4.1%   | 4.3×  | 1.000       | ~0.47       |
| 0.88 | 13.0% | 0.56%  | 37×   | 1.000       | ~0.48       |
| 0.93 | 4.7%  | ~0%    | ∞     | 1.000       | ~0.47       |

**The mechanism, stated precisely:**

For H=0 (all-zeros goal), all remaining wrong pairs at every accuracy level want the
same correction (balance=1.000). Any new state that covers wrong pairs will have a
positive net delta: agreement among wrong pairs is unanimous, so a theta_z=0 state
always fixes all the wrong covered pairs simultaneously.

For H=1 (balanced goal), remaining wrong pairs are persistently ~50/50 between "want
agreement" and "want disagreement" pairs, at ALL accuracy levels — the balance never
resolves as optimization proceeds. When a new state is proposed:
- theta_z=0 fixes "want agreement" covered wrong pairs but breaks "want disagreement" ones
- theta_z=1 does the opposite
- Net delta ≈ 0 at high accuracy, making positive proposals exponentially rare

This "landscape frustration" is structural and independent of the specific goal instance,
the pairing scheme, or the picker quality. It arises because theta_z uniformly flips all
covered positions — a one-dimensional choice that cannot simultaneously satisfy two
opposing subsets of wrong pairs.

### The Integral SA Model and α Correction

#### Setup

Define the **positive proposal rate** p(a) = P(delta > 0 | accuracy = a), measured
against a near-converged ensemble. This is what script 08 measures.

The SA progress rate is approximated by:
```
d(acc)/d(step) ≈ p_eff(acc) × nstate / ngbits
```

Integrating, the number of states needed to reach accuracy `acc` from baseline `a0=0.5`:
```
nstate(acc) = C_scaled × D(acc)
D(acc) = ∫_{a0}^{acc} da / p_eff(a)     [cumulative difficulty]
C_scaled = ngbits × C_eff / nsteps       [calibration constant]
```

The efficiency metric η = (1 − H_bin(acc)) × ngbits / (N × nstate) peaks at n*, the
optimal ensemble size. This defines acc* as the accuracy at the η peak.

#### The α Problem

Script 08 measures p(a) against a **near-converged** ensemble where every state is
already optimal. But in the actual SA, the replaced state is mediocre — a fresh proposal
can beat it much more easily. This creates a systematic bias:

```
p_eff(a)  >>  p_meas(a)  for H=1 (frustrated landscape)
p_eff(a)  ≈   p_meas(a)  for H=0 (smooth landscape)
```

Define the overestimation factor:
```
α_H = D_H_meas(acc*) / D_H_needed(acc*)
    = D_H_meas(acc*) / (n*_H / C_scaled)
```

At nqubit=6:
- α_H0 = 1.00 (by calibration — smooth landscape, near-converged ≈ typical SA)
- α_H1 = 28.9 (frustrated landscape — near-converged is essentially impossible to improve,
  but a random SA state is easy to beat)

The corrected model is:
```
n*_H = C_scaled × D_H(acc*) / α_H
```

This reproduces both endpoints exactly (by construction) and predicts intermediate H
within the accuracy described below.

#### Interpolating p_H for Intermediate H

Since script 08 only measured p(a) at H=0 and H=1, intermediate H values use a
**geometric interpolation**:
```
p_H(a) = p_H0(a)^(1−H) × p_H1(a)^H
```
This is a log-linear blend. The implied α values for each H (using actual acc* from
the scaling study) are:

| H     | n*  | acc*   | D_H(acc*) | α_implied | α_geo=29^H | α_bal |
|-------|-----|--------|-----------|-----------|------------|-------|
| 0.000 | 40  | 0.9245 | 2.13      | 1.00      | 1.00       | 1.00  |
| 0.250 | 52  | 0.9533 | 5.25      | 1.89      | 2.32       | 1.94  |
| 0.500 | 68  | 0.9588 | 13.42     | 3.70      | 5.37       | 3.53  |
| 0.625 | 56  | 0.9121 | 7.65      | 2.57      | 8.18       | 4.92  |
| 0.750 | 79  | 0.9560 | 34.09     | 8.10      | 12.45      | 7.11  |
| 0.875 | 84  | 0.9451 | 44.80     | 10.01     | 18.96      | 11.20 |
| 1.000 | 104 | 0.9766 | 159.92    | 28.86     | 28.86      | 30.88 |

H=0.625 is anomalous throughout (non-monotone in the scaling study and excluded from
model fitting).

### What Is Empirical vs Theoretical

#### Theoretically derivable

**Wrong-pair balance b(H):**
```
b(H) = 1 − k/ngbits
```
where k is the number of "want-disagreement" pairs in the goal (directly readable from
the goal type). This is exact, not an approximation. For nqubit=6: k=0 (H=0) → b=1.000;
k=182 (H=1) → b=0.500. Confirmed by script 08 at both endpoints.

The balance defines the degree of landscape frustration: b=1 means all wrong pairs point
in the same direction (easy to fix); b=0.5 means perfect frustration (fixing half the
wrong pairs always breaks the other half).

**That α_H=0 = 1:** The H=0 landscape is smooth — near-converged ensembles for H=0 are
as easy to improve as typical SA ensembles — follows from b(H=0) = 1 (no balance conflict).

**The integral SA framework:** The d(acc)/d(step) ≈ p_eff × nstate/ngbits relationship
and the η efficiency metric are derived from the SA dynamics.

#### Empirical (requires measurement)

**p_H0(a) and p_H1(a):** Measured by script 08 at four accuracy levels (0.70, 0.80,
0.88, 0.93), then fit to p(a) = A·exp(−k·a) for extrapolation. The decay rates are:
```
H=0: A=53.7,   k=7.18
H=1: A=1.73×10⁷,  k=25.54
```

**C_scaled:** Calibrated from n*_H0 and D_H0(acc*_H0). Cannot be derived theoretically
without knowing the absolute SA efficiency.

**acc*(H):** The accuracy at which η peaks. This cannot be predicted from p_meas alone —
it would require p_eff (vs the actual SA ensemble). Currently taken from the scaling study.

**The α(b) power law exponent:** The functional form
```
log(α) = A · (1−b)^γ     [A=5.42, γ=0.66, fit from 5 non-anomalous H values]
```
is empirically fit. The exponent γ≈0.66 has no theoretical derivation yet; it should
come from the combinatorics of how wrong-pair coverage distributes across ensemble
states near convergence.

### Prediction Model Summary

Given as inputs: n*_H0 (one scaling study point) and acc*(H) (either from scaling study
or eventually from theory), the corrected model predicts n*_H for all H with:

```
Median |error| = 6%  (balance α model, excluding known-anomalous H=0.625)
Median |error| = 8%  (H-space power law, excluding H=0.625)
Median |error| = 25% (geometric α=29^H)
```

The balance model is preferred because b(H) is theoretically grounded and provides a
natural axis for α.

The model **cannot** currently make a fully standalone prediction: it needs acc*(H) from
the scaling study. To close this gap, p_eff (positive proposal rate vs typical SA
ensemble, not near-converged) would need to be measured. This would shift the acc*
prediction from the near-converged regime to the actual SA regime.

### Regime Interpretation

The key quantity controlling all model behavior is the **cumulative difficulty ratio**
D_H(acc)/D_H0(acc), which grows rapidly with accuracy:

| acc  | D_H1/D_H0 | Regime description |
|------|-----------|--------------------|
| 0.70 | 1.5×      | Low frustration — all H values nearly equivalent |
| 0.80 | 2.1×      | Mild frustration — H=1 needs ~2× more states |
| 0.85 | 3.5×      | Moderate frustration — gap opening |
| 0.90 | 8.4×      | High frustration — gap growing rapidly |
| 0.92 | 16×       | Severe frustration — α correction essential |
| 0.96 | 39×       | Extreme frustration — near-converged H=1 essentially impossible |
| 0.98 | 55×       | Deep tail — exponentially rare positive proposals |

**nqubit=6** has acc* in the range 0.92–0.98 for H=0–1, placing it squarely in the
severe-to-extreme frustration regime. The α correction (factor 1–29×) is essential.

**nqubit=8** (H=0, 0.25, 0.5 available) has acc* in the range 0.81–0.90, placing it in
the mild-to-high frustration regime. The H-gap in n* is much smaller (n*_H0.5/n*_H0 =
1.08 vs 1.70 at N=6), and the implied α≈1 for all measured H values. This means either
(a) the frustration genuinely decreases as N grows, or (b) the N=8 optimizer simply
can't push past acc≈0.90 where frustration is not yet severe. Distinguishing these
requires the N=8 structural study (measuring p_H0 and p_H1 at N=8) and the N=8 H=1
scaling result (still running).

**The critical open question for scaling:** whether acc*(N) decreases faster than the
frustration threshold grows as N increases. If acc* stabilizes at some value as N grows,
the H-gap will compound and become unmanageable. If acc* keeps falling, the system
always operates in the mild-frustration regime and the scaling may remain tractable.
