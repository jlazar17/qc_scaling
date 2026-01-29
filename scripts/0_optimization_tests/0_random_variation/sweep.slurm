#!/bin/bash
#
# Usage:
#   NQUBIT=8 OUTDIR=/path/to/output bash sweep.slurm
#
# Optional env vars:
#   NITER        (default: 10000)
#   NREPLACE     (default: 1)
#   SAVELEVEL    (default: deltas)
#   PARTITION    (default: batch)
#   PZERO_VALUES (default: "0.0,0.1,0.3,0.5,0.7,0.9,1.0")
#   NSTATE_MULTS (default: "1,2,3,4,5,6,7")
#   NSEEDS       (default: 10)
#   MERGED_FILE  (default: ${OUTDIR}/merged.h5)

set -euo pipefail

: "${NQUBIT:?Must set NQUBIT}"
: "${OUTDIR:?Must set OUTDIR}"
NITER="${NITER:-10000}"
NREPLACE="${NREPLACE:-1}"
SAVELEVEL="${SAVELEVEL:-deltas}"
PARTITION="${PARTITION:-batch}"
PZERO_VALUES="${PZERO_VALUES:-0.0,0.1,0.3,0.5,0.7,0.9,1.0}"
NSTATE_MULTS="${NSTATE_MULTS:-1,2,3,4,5,6,7}"
NSEEDS="${NSEEDS:-10}"
MERGED_FILE="${MERGED_FILE:-${OUTDIR}/merged.h5}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "${OUTDIR}"

# Generate params file: each line is "multiplier pzero seed"
PARAMS_FILE="${OUTDIR}/sweep_params.txt"
> "${PARAMS_FILE}"
IFS=',' read -ra MULTS <<< "${NSTATE_MULTS}"
IFS=',' read -ra PZEROS <<< "${PZERO_VALUES}"
for mult in "${MULTS[@]}"; do
    for pzero in "${PZEROS[@]}"; do
        for seed in $(seq 1 "${NSEEDS}"); do
            echo "${mult} ${pzero} ${seed}" >> "${PARAMS_FILE}"
        done
    done
done

NTASKS=$(wc -l < "${PARAMS_FILE}")
echo "Generated ${NTASKS} parameter combinations in ${PARAMS_FILE}"

# Submit array job
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
julia --project=${SCRIPT_DIR}/.. ${SCRIPT_DIR}/sweep_single.jl \
    --nqubit ${NQUBIT} \
    --niter ${NITER} \
    --nstate-multiplier \$MULT \
    --pzero \$PZERO \
    --seed \$SEED \
    --outdir ${OUTDIR} \
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
    --outdir ${OUTDIR} \
    --output ${MERGED_FILE}
")

echo "Submitted merge job: ${MERGE_JOB_ID} (depends on ${ARRAY_JOB_ID})"
