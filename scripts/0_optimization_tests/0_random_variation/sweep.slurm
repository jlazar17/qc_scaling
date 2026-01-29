#!/bin/bash
#
# Usage:
#   NQUBIT=8 OUTFILE=/path/to/output.h5 bash sweep.slurm
#
# Optional env vars:
#   NITER        (default: 10000)
#   NREPLACE     (default: 1)
#   SAVELEVEL    (default: deltas)
#   PARTITION    (default: batch)
#   PZERO_VALUES (default: "0.0,0.1,0.3,0.5,0.7,0.9,1.0")
#   NSTATE_MULTS (default: "1,2,3,4,5,6,7")
#   NSEEDS       (default: 10)

set -euo pipefail

: "${NQUBIT:?Must set NQUBIT}"
: "${OUTFILE:?Must set OUTFILE}"
NITER="${NITER:-10000}"
NREPLACE="${NREPLACE:-1}"
SAVELEVEL="${SAVELEVEL:-deltas}"
PARTITION="${PARTITION:-batch}"
PZERO_VALUES="${PZERO_VALUES:-0.0,0.1,0.3,0.5,0.7,0.9,1.0}"
NSTATE_MULTS="${NSTATE_MULTS:-1,2,3,4,5,6,7}"
NSEEDS="${NSEEDS:-10}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTDIR="$(dirname "${OUTFILE}")"

mkdir -p "${OUTDIR}"

# Generate random unique seeds
SEEDS=()
declare -A SEEN_SEEDS
while [ ${#SEEDS[@]} -lt "${NSEEDS}" ]; do
    S=$((RANDOM * RANDOM + RANDOM))
    if [ -z "${SEEN_SEEDS[$S]+x}" ]; then
        SEEDS+=("$S")
        SEEN_SEEDS[$S]=1
    fi
done

# Generate params file: each line is "multiplier pzero seed"
PARAMS_FILE="${OUTDIR}/sweep_params.txt"
> "${PARAMS_FILE}"
IFS=',' read -ra MULTS <<< "${NSTATE_MULTS}"
IFS=',' read -ra PZEROS <<< "${PZERO_VALUES}"
for mult in "${MULTS[@]}"; do
    for pzero in "${PZEROS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            echo "${mult} ${pzero} ${seed}" >> "${PARAMS_FILE}"
        done
    done
done

NTASKS=$(wc -l < "${PARAMS_FILE}")
echo "Generated ${NTASKS} parameter combinations in ${PARAMS_FILE}"
echo "Seeds: ${SEEDS[*]}"

# Create empty HDF5 file so all tasks can append to it
julia --project="${SCRIPT_DIR}/.." -e "using HDF5; h5open(\"${OUTFILE}\", \"w\") do _ end"

# Submit array job (each task writes to its own temp file, merge step combines)
TMPDIR_SWEEP="${OUTDIR}/tmp_sweep"
mkdir -p "${TMPDIR_SWEEP}"

ARRAY_JOB_ID=$(sbatch --parsable \
    --partition="${PARTITION}" \
    --job-name="sweep_${NQUBIT}" \
    --output="${OUTDIR}/slurm_%A_%a.out" \
    --error="${OUTDIR}/slurm_%A_%a.err" \
    --array="1-${NTASKS}" \
    --wrap="
PARAMS=\$(sed -n \"\${SLURM_ARRAY_TASK_ID}p\" ${PARAMS_FILE})
MULT=\$(echo \$PARAMS | awk '{print \$1}')
PZERO=\$(echo \$PARAMS | awk '{print \$2}')
SEED=\$(echo \$PARAMS | awk '{print \$3}')
TMPFILE=${TMPDIR_SWEEP}/task_\${SLURM_ARRAY_TASK_ID}.h5
julia --project=${SCRIPT_DIR}/.. ${SCRIPT_DIR}/sweep_single.jl \
    --nqubit ${NQUBIT} \
    --niter ${NITER} \
    --nstate-multiplier \$MULT \
    --pzero \$PZERO \
    --seed \$SEED \
    --outfile \$TMPFILE \
    --savelevel ${SAVELEVEL} \
    --nreplace ${NREPLACE}
")

echo "Submitted array job: ${ARRAY_JOB_ID} (${NTASKS} tasks)"

# Submit merge job
MERGE_JOB_ID=$(sbatch --parsable \
    --partition="${PARTITION}" \
    --job-name="merge_${NQUBIT}" \
    --output="${OUTDIR}/merge_%j.out" \
    --error="${OUTDIR}/merge_%j.err" \
    --dependency="afterok:${ARRAY_JOB_ID}" \
    --wrap="
julia --project=${SCRIPT_DIR}/.. ${SCRIPT_DIR}/merge_sweep.jl \
    --outdir ${TMPDIR_SWEEP} \
    --output ${OUTFILE}
")

echo "Submitted merge job: ${MERGE_JOB_ID} (depends on ${ARRAY_JOB_ID})"
