# utils

Shared Julia environment and utility code used by all scripts in this repository.

## Files

- `Project.toml` / `Manifest.toml` — Julia package environment. All scripts activate this env via `Pkg.activate(joinpath(@__DIR__, "../utils"))`.
- `optimization_utils.jl` — Common optimizer helpers: state initialization, rep cache update, SA acceptance criterion, checkpoint logic.
- `base_optimizer.jl` — Baseline greedy optimizer (non-SA), used for comparison in early development scripts.
- `io_utils.jl` — HDF5 read/write helpers shared across multiple scripts.
- `0_random_variation/` — Parameter sweep scripts and data for characterizing random variation in optimizer performance.
