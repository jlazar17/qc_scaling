# scaling_study

Primary data generation and analysis for the QC scaling project. These scripts produce the main results: how optimizer efficiency scales with nqubit, nstate, and goal entropy H.

## Scripts

- `scaling_study.jl` — Original scaling study. Sweeps over nqubit and nstate for a fixed set of goal entropies; records max accuracy across restarts. Outputs `data/scaling_study.h5`.
- `scaling_study_adaptive.jl` — Adaptive version with checkpoint-based early stopping and multi-restart logic. This is the primary data source for all structural analysis. Outputs `data/scaling_study_adaptive.h5`.
- `scaling_study_h_sweep.jl` — Finer H sweep at fixed nqubit to characterize efficiency vs. entropy at higher resolution.
- `scaling_study_n10.jl` — Extends the scaling study to nqubit=10.
- `ablation_smart_vs_random.jl` — Compares goal-aware (smart) alpha selection vs. random selection to quantify the picker's contribution to efficiency.
- `benchmark_hotpath.jl` — Timing benchmark for the core SA inner loop (proposal + acceptance step).
- `time_nqubit.jl` — Measures wall-clock time per SA step as a function of nqubit.
- `random_landscape.jl` — Runs the optimizer on random (non-Hamming-structured) goals to test universality of scaling behavior.
- `ilp_optimal_accuracy.jl` — Computes the ILP upper bound on achievable accuracy for given nstate and goal, for comparison with SA results.
- `download_snp_goals.jl` — Downloads real SNP genotype data from the 1000 Genomes Project to use as biologically-motivated goal vectors.
- `plot_scaling_study.jl` — Plots efficiency η vs. nstate for each (nqubit, H) from the adaptive scaling study data.

## Data

- `data/scaling_study.h5` — Output of `scaling_study.jl`
- `data/scaling_study_adaptive.h5` — Output of `scaling_study_adaptive.jl` (primary dataset)
