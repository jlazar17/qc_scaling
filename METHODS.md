# Optimization Methods

## Overview

All methods optimize a fixed-size ensemble of `nstate` PseudoGHZ states to maximize
accuracy — the fraction of a binary goal string correctly predicted by the ensemble's
representation (rep) vector. The rep at position `i` is the majority vote of parity
measurements across all states whose context covers position `i`.

---

## PseudoGHZ States

Each state is parameterized by:
- `theta_s ∈ {0, 1}` — selects even/odd base context
- `theta_z ∈ {0, 1}` — phase parameter
- `alphas ∈ {0,1}^(n-1)` — binary vector of length n-1
- `generator` — a `ParityOperator` drawn from 3^n possibilities

The base context determined by `theta_s` defines a set of `2^(n-1) + 1` parity operators.
The generator shifts these to produce the state's context — the set of rep positions it
contributes to.

### `pick_new_alphas`

Given a fixed generator and `theta_s` (i.e., a fixed context), `pick_new_alphas` finds
the `(theta_z, alphas)` combination that minimizes disagreement between the state's parity
predictions and the current goal/rep. It exhaustively checks all `2 × 2^(n-1)` combinations
and returns the best one. This collapses the alpha search space from 2^(n-1) possibilities
to 1 optimal choice.

---

## Base Greedy Optimizer

Iteratively replaces states in the ensemble one at a time.

**Score function**: each state is scored by how well its context aligns with the goal.
The full rep is recomputed from scratch each iteration (O(3^n)).

**Proposal**: the chosen state's generator is kept fixed; `pick_new_alphas` finds the
best alphas for that generator given the current rep and goal. Only alphas are searched —
the generator never changes.

**Stagnation**: none. The same move type is made every iteration regardless of progress.

---

## Improved Greedy Optimizer

Same iterative structure as the base, with three improvements:

**Incremental rep**: maintains `rep_sum` and `rep_ctr` arrays updated in O(context size)
per swap rather than recomputing the full rep each iteration.

**Generator mutation**: with probability `p_mutate` (default 0.3), instead of keeping the
existing generator, a new one is selected via `get_new_contexts` which looks for
underrepresented regions of the rep space. `pick_new_alphas` then finds the best alphas for
the new generator. This allows the optimizer to escape local optima by exploring new
generators, not just tuning alphas.

**Stagnation handling**: tracks the number of consecutive iterations without accuracy
improvement (`n_same`). After `n_same_tol` stagnant iterations it forces a
generator-mutation move regardless of score-weighted sampling, acting as a manual restart.

---

## Random Simulated Annealing

A Metropolis sampler over the ensemble.

**Proposal**: draw a completely random `PseudoGHZState` — random `theta_s`, `theta_z`,
`alphas`, and generator. No knowledge of the current goal or rep.

**Acceptance**: the proposed swap is always accepted if accuracy improves (Δ ≥ 0);
otherwise accepted with probability `exp(Δ/T)`.

**Cooling**: geometric schedule `T *= alpha` each step. Temperature is calibrated at the
start of each restart by sampling random swaps and setting T₀ so that bad moves of typical
magnitude are accepted with probability 0.8.

**Restarts**: `n_restarts` independent runs from random initializations; best accuracy
across all restarts is returned.

---

## Goal-Aware Simulated Annealing

Identical to random SA in all respects except the proposal step.

**Proposal**:
1. Pick a random generator (3^n possibilities) — same as random SA
2. Pick a random `theta_s ∈ {0, 1}` — same as random SA
3. Call `pick_new_alphas` to find the **optimal** `(theta_z, alphas)` for that generator
   given the current goal and rep

The generator is still chosen randomly, but the alphas are set optimally rather than
randomly. Every proposed state is the best representable state for its generator — no
proposals are wasted on bad alpha choices.

**Why this works**: `pick_new_alphas` collapses the alpha search space from 2^(n-1)
possibilities to 1, so goal-aware SA effectively searches only over generators (~3^n
choices) rather than the full (generator × alpha) space (~3^n × 2^(n-1) choices). The
Metropolis framework then handles the ensemble-level combinatorics that `pick_new_alphas`
alone cannot resolve.

**Incremental rep**: the rep vector is maintained incrementally (updated only at indices
touched by each accepted swap) to keep it current for `pick_new_alphas` at each step
without an O(3^n) recompute.

---

## Comparison

| Property | Base | Improved | Rand SA | Goal SA |
|----------|------|----------|---------|---------|
| Generator search | fixed | occasional mutation | random | random |
| Alpha search | `pick_new_alphas` | `pick_new_alphas` | random | `pick_new_alphas` |
| Accepts worse moves | no | no | yes (Metropolis) | yes (Metropolis) |
| Incremental rep | no | yes | yes | yes |
| Stagnation handling | none | explicit | restarts | restarts |

Goal-aware SA combines the best of both worlds: goal-aligned proposals from the improved
optimizer with the ability to accept worse moves and escape local optima from SA.

---

## Performance (n=8, nstate=156, 3× base)

| Goal entropy H | Base | Improved | Rand SA | **Goal SA** |
|----------------|------|----------|---------|-------------|
| 0.00           | 0.633 | 0.743   | 0.570   | **0.903**   |
| 0.50           | 0.586 | 0.655   | 0.554   | **0.790**   |
| 1.00           | 0.512 | 0.541   | 0.537   | **0.625**   |

Performance decreases monotonically with goal entropy for all methods. Goal-aware SA shows
the strongest entropy dependence because `pick_new_alphas` exploits goal structure most
effectively when the goal is low-entropy (structured).
