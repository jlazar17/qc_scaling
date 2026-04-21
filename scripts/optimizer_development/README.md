# optimizer_development

Chronological development of the optimizer. Each subdirectory represents a distinct algorithmic approach, starting from naive greedy search and progressing to the goal-aware simulated annealing that is the current production optimizer.

## Subdirectories (roughly chronological)

- `naive_beta_swap/` — First optimizer: randomly swaps beta (phase) parameters and keeps improvements. No goal-awareness.
- `improved_optimizer/` — Adds structured search over alpha parameters; benchmarks against naive baseline.
- `smart_alphas/` — Introduces the fingerprint-based alpha selection (pick_new_alphas): chooses alphas that maximize agreement between rep and goal.
- `coverage_first_optimizer/` — Tries prioritizing uncovered positions before optimizing accuracy. Superseded by SA.
- `targeted_escape_optimizer/` — Attempts to escape local minima by explicitly targeting wrong pairs. Superseded by SA.
- `beam_search/` — Beam search over state ensembles. Computationally expensive, did not outperform SA.
- `simulated_annealing/` — First SA implementation: random proposals with Metropolis acceptance. Establishes the SA baseline and cooling schedule.
- `goal_aware_sa/` — Current production optimizer: SA with goal-aware (smart) alpha proposals. Substantially outperforms random-proposal SA, especially at high H.
- `pair_aware_alpha/` — Variant using shuffled-pairing-aware alpha selection (pick_alphas_s). Tested but did not significantly improve over standard goal-aware SA.

## Notes

The final optimizer (goal-aware SA) lives in `goal_aware_sa/goal_aware_sa.jl` and is the version used in all scaling study scripts.
