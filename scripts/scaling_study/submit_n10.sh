#!/bin/bash
# Submit nqubit=10 scaling study as a SLURM job array — one job per H value.
#
# Each job runs scaling_study_n10.jl with a single H_target argument and
# writes to its own per-H HDF5 file, avoiding any file-locking contention.
# After all jobs finish, run merge_n10.jl to combine into one HDF5.
#
# Usage:
#   cd scripts/scaling_study
#   bash submit_n10.sh
#
# Tune PARTITION, NCORES, TIME_LIMIT, and ACCOUNT for your cluster.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JULIA="${JULIA:-julia}"

PARTITION="${PARTITION:-batch}"
ACCOUNT="${ACCOUNT:-}"
NCORES=20          # must be >= nseeds (20); used for Threads.@threads
TIME_LIMIT="24:00:00"   # worst case ~20h (3 restarts × 5M steps × 20 nstates)

H_VALS=(0.0 0.25 0.5 0.625 0.75 0.875 1.0)
NHVALS=${#H_VALS[@]}

ACCOUNT_FLAG=""
[ -n "$ACCOUNT" ] && ACCOUNT_FLAG="--account=${ACCOUNT}"

echo "Submitting ${NHVALS} jobs (one per H value)..."
echo "  Partition : ${PARTITION}"
echo "  Cores/job : ${NCORES}"
echo "  Time limit: ${TIME_LIMIT}"
echo ""

JOB_IDS=()
for H in "${H_VALS[@]}"; do
    JOB_ID=$(sbatch --parsable \
        --partition="${PARTITION}" \
        ${ACCOUNT_FLAG} \
        --job-name="n10_H${H}" \
        --cpus-per-task="${NCORES}" \
        --time="${TIME_LIMIT}" \
        --output="${SCRIPT_DIR}/data/slurm_n10_H${H}_%j.out" \
        --error="${SCRIPT_DIR}/data/slurm_n10_H${H}_%j.err" \
        --wrap="${JULIA} --threads=${NCORES} ${SCRIPT_DIR}/scaling_study_n10.jl ${H}")
    echo "  H=${H}  → job ${JOB_ID}"
    JOB_IDS+=("${JOB_ID}")
done

# Join job IDs with colons for --dependency
DEPS=$(IFS=:; echo "afterok:${JOB_IDS[*]}")

MERGE_ID=$(sbatch --parsable \
    --partition="${PARTITION}" \
    ${ACCOUNT_FLAG} \
    --job-name="n10_merge" \
    --cpus-per-task=1 \
    --time="00:30:00" \
    --output="${SCRIPT_DIR}/data/slurm_n10_merge_%j.out" \
    --error="${SCRIPT_DIR}/data/slurm_n10_merge_%j.err" \
    --dependency="${DEPS}" \
    --wrap="${JULIA} ${SCRIPT_DIR}/merge_n10.jl")

echo ""
echo "Merge job ${MERGE_ID} will run after all H jobs complete."
echo "Monitor: squeue -u \$USER"
