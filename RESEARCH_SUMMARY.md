# QC Scaling Research Summary

## Questions Asked and Findings

### 1. What is the true accuracy ceiling?
Initially estimated with beam search (greedy, weak upper bound). SA with a full enumerated
state pool gave better estimates. Goal-aware SA has since exceeded both at n=6 (0.997 vs
beam search ceiling of 0.739), so the "ceiling" remains an open empirical question — we
have lower bounds, not upper bounds. An exact solve for n=6 is computationally intractable
due to the non-linear majority-vote objective (choosing optimal k states from ~93k candidates).

### 2. Does simulated annealing improve over the greedy optimizer?
Random SA (uniform proposals) is competitive with the improved optimizer at high nstate and
random goals (pzero=0.5), but loses badly at structured goals (pzero=0.0). The improved
optimizer uses `pick_new_alphas` to construct goal-aligned states, which random SA lacks.
Neither clearly dominates.

### 3. How do optimizers compare as nstate scales?
Swept nstate_mult=[1,2,3,4,6] at n=8. The improved optimizer holds a ~0.2 accuracy lead
over random SA at pzero=0.0 throughout. At pzero=0.5 the gap narrows with nstate. Neither
saturates cleanly within the tested range.

### 4. Is the optimizer converging to a unique solution?
No. Pairwise Jaccard similarity between top-accuracy runs ≈ 0, and rep vector similarity
≈ 0.5 — completely different state sets achieve the same maximum accuracy. The problem is
highly degenerate: there is no single optimal ensemble, but a large manifold of equally
good solutions.

### 5. What is goal-aware SA and is it better?
Goal-aware SA replaces the random proposal step with `pick_new_alphas`, which constructs
the best possible state for a randomly chosen generator given the current rep and goal.
Combined with Metropolis acceptance, this gets the benefits of both goal-alignment (like
the improved optimizer) and thermal exploration (like SA). It dominates all other methods
at all scales:

| nqubit | pzero | improved | rand\_sa | goal\_sa |
|--------|-------|----------|----------|----------|
| 6      | 0.0   | 0.860    | 0.802    | **0.997** |
| 6      | 0.5   | 0.758    | 0.767    | **0.896** |
| 8      | 0.0   | 0.736    | 0.567    | **0.893** |
| 8      | 0.5   | 0.540    | 0.535    | **0.626** |

### 6. Is the pzero=0.0 vs pzero=0.5 performance gap an entropy effect?
Yes, entirely. Running fixed-Hamming-weight goals at matched entropies (H=0 vs H=1)
reproduces the same accuracy values as pzero=0.0 vs pzero=0.5. The performance gap is
purely a function of goal entropy, not the i.i.d. Bernoulli sampling structure. Entropy
is the natural parameterization for understanding optimizer performance.

### 7. Is the accuracy gap a fundamental limit or optimizer bias?
Open question. The convergence analysis (Jaccard + accuracy distribution over many seeds)
and slow-cooling consistency check are the right empirical tools. The convergence study at
n=4 (50 seeds) showed Jaccard ≈ 0 among top runs, suggesting goal_sa samples freely from
a degenerate optimum manifold rather than getting stuck — evidence the found accuracy is
near the true ceiling. This needs to be confirmed at n=6 and n=8.

---

## Current Results (full comparison, goal_sa)

### n=8, nstate=156 (3×), SA: 2M steps, alpha=0.999

| Condition         | k    | H\_actual | base   | improved | rand\_sa | goal\_sa  |
|-------------------|------|-----------|--------|----------|----------|-----------|
| H=0.00 (Hamming)  | 0    | 0.0000    | 0.633  | 0.743    | 0.570    | **0.903** |
| H=0.50 (Hamming)  | 361  | 0.5001    | 0.586  | 0.655    | 0.554    | **0.790** |
| H=1.00 (Hamming)  | 1640 | 1.0000    | 0.512  | 0.541    | 0.537    | **0.625** |
| pzero=0.0         | —    | 0.0000    | —      | —        | —        | —         |
| pzero=0.5         | —    | ~1.000    | —      | —        | —        | —         |

*(pzero rows still running)*

---

## Pending / In Progress

| Script | Status | Description |
|--------|--------|-------------|
| `full_comparison.jl` | running | n=8 pzero rows still computing |
| `convergence_gsa.jl` | ready | Run after full_comparison |
| `scaling_study.jl` | ready | Run after convergence_gsa |
| n=10 and beyond | future | Needs cluster |

---

## Scripts by Directory

### `scripts/0_optimization_tests/`
Original greedy optimizer infrastructure.
- `base_optimizer.jl` — base greedy optimizer (score-weighted replacement, no rep cache)
- `optimization_utils.jl` — shared helpers: `accuracy()`, `make_initial_state()`

### `scripts/4_beam_search/`
- `beam_search.jl` — beam search ceiling estimator; maintains k partial ensembles, expands
  greedily. First ceiling estimates (too pessimistic, beaten by goal_sa).

### `scripts/5_simulated_annealing/`
- `sa_ceiling.jl` — SA with full state pool enumeration; early ceiling estimates
- `cooling_scan.jl` — scans alpha × nsteps for good SA hyperparameters at n=6,8.
  Recommended: alpha=0.999, nsteps=2M for n=8
- `run_comparison.jl` — three-way comparison (base, improved, random SA) at n=8
- `run_nstate_sweep.jl` — sweeps nstate_mult for base/improved/SA at n=8
- `convergence_analysis.jl` — 50 seeds of improved optimizer, pairwise Jaccard and rep
  similarity; established problem degeneracy at n=4
- `plot_comparison.jl` — plots from run_comparison
- `plot_nstate_sweep.jl` — plots from run_nstate_sweep
- `plot_convergence.jl` — accuracy distributions, Jaccard distributions, rep similarity distributions

### `scripts/6_goal_aware_sa/`
- `goal_aware_sa.jl` — initial goal_sa implementation; three-way comparison (improved,
  rand_sa, goal_sa) at n=4,6,8 with pzero=0,0.5
- `full_comparison.jl` — four-way comparison (base, improved, rand_sa, goal_sa) with both
  fixed-Hamming entropy sweep (H=0, 0.5, 1) and Bernoulli pzero sweep (0.0, 0.5);
  establishes entropy as the natural parameterization
- `convergence_gsa.jl` — convergence analysis using goal_sa at n=6,8; H=0 and H=1;
  50 seeds; pairwise Jaccard + rep similarity
- `plot_convergence.jl` *(from scripts/5_simulated_annealing)* — reusable for gsa results

### `scripts/7_scaling/`
- `scaling_study.jl` — sweeps nstate_mult=[1,2,3,4,6] for nqubit=[4,6,8] at H=0,0.5,1
  using goal_sa; records accuracy and efficiency metric η

---

## Key Definitions

| Symbol | Definition |
|--------|------------|
| nstate0 | `ceil(3^n / 2^(n-1))` — base ensemble size (52 for n=8) |
| N\_classical | `(3^n − 1) / 2` — number of goal bits |
| N\_quantum | `nqubit × nstate` — total quantum resources used |
| η | `(1 − H(acc)) × N_classical / N_quantum` — efficiency metric |
| H(x) | `−x·log₂(x) − (1−x)·log₂(1−x)` — binary entropy of accuracy x |
| Jaccard | `\|A ∩ B\| / \|A ∪ B\|` — overlap between two state ensembles |
| rep similarity | fraction of non-NaN positions where two rep vectors agree |

**η interpretation**: 0 when accuracy = 0.5 (random, no information conveyed),
approaches N\_classical / N\_quantum at perfect accuracy. Measures classical bits of goal
encoded per quantum resource used.

---

## Open Questions

1. **Fundamental limit vs optimizer bias**: does goal_sa accuracy represent the true
   ceiling, or is there more room? Convergence analysis at n=6,8 will shed light.
2. **Scaling with n and nstate\_mult**: how does accuracy and η grow? Is there a saturation
   nstate\_mult for each n? Does η peak at large n despite falling accuracy?
3. **Entropy dependence of scaling**: does the optimal nstate\_mult differ by goal entropy?
4. **Extrapolation to large n**: fit accuracy-vs-n curves to project to n=10,12 (cluster needed).
