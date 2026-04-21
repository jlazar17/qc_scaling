# archive

Scripts that were useful during exploration but are no longer part of the active research pipeline. Kept for reference.

## Subdirectories

- `analysis/` — Early accuracy analysis scripts for nqubit=4 and small sweeps. Superseded by the full scaling study.
- `picker_diagnostics/` — Diagnostic scripts that investigated potential explanations for the efficiency gap: both-covered pairs, GF2 capacity, alpha capacity, basin variance, rep convergence, warmstart experiments, and ablation studies. All hypotheses were ruled out; the true cause (wrong-pair balance) was found in `structural_analysis/`.
- `mnist_test/` — Exploratory test using MNIST-derived goal vectors. Not connected to the main scaling results.
