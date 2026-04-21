# structural_analysis

Scripts that investigate the structural reasons behind the observed scaling behavior — specifically, why the optimizer's efficiency depends on nstate and goal entropy H.

The main empirical finding: the positive proposal rate p_pos collapses dramatically for H=1 at high accuracy (acc > 0.85), while H=0 maintains useful proposal rates throughout. At acc=0.88 the H=0/H=1 p_pos ratio is ~37x; at acc=0.93 H=1 p_pos is effectively zero. This collapse is why H=1 requires ~2x more states to reach the saturation threshold.

## Active Scripts

- `01_saturation_test.jl` — Measures achievable accuracy as a function of nstate; establishes the saturation threshold n* below which accuracy cannot pass 0.90.
- `02_shuffled_saturation.jl` — Repeats the saturation test using the shuffled pairing (matching positions differing at an odd number of ternary digits) to eliminate both-covered pairs. Confirms the structural result is pairing-independent.
- `03_landscape_frustration.jl` — Directly measures wrong-pair balance b(H) and positive proposal rate p_pos as a function of accuracy during SA runs. Identifies the mechanism behind the H-dependent efficiency gap.
- `04_synthetic_frustration.jl` — Synthetic experiments with controlled b(H) values to confirm the causal link between wrong-pair balance and p_pos collapse.
- `05_ppos_vs_nstate.jl` — Measures p_pos at acc=0.90 as a function of nstate for each H value. Identifies the threshold n*(H) where p_pos first crosses ~0.001.
- `06_ppos_H_sweep.jl` — Full H sweep of the p_pos threshold model: for each H in {0, 1/8, ..., 1}, finds n*(H) and checks consistency with the scaling study.
- `07_plot_landscape_frustration.jl` — Plotting script for the landscape frustration results (wrong-pair balance, p_pos trajectories).

## Data

- `13_ppos_efficiency.h5` — p_pos vs. nstate data from the H sweep (script 06).
- `figures/` — Output plots.

## Detritis

Scripts in `detritis/` were exploratory dead ends or were superseded by the above:
- Early attempts to explain the efficiency gap via both-covered pairs, picker degeneracy, natural parity bias, and GF2 capacity (all ruled out).
- N=8 threshold test and efficiency prediction experiments (inconclusive due to insufficient SA budget).
