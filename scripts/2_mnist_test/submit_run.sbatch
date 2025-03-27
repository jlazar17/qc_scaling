#!/bin/bash
#SBATCH -J qc_scaling
#SBATCH -c 1 # Number of cores
#SBATCH -p arguelles_delgado,sapphire
#SBATCH --mem 8000
#SBATCH -t 0-10:00:00 # Maximum execution time (D-HH:MM:SS)
#SBATCH -o /n/netscratch/arguelles_delgado_lab/Everyone/jlazar/qc_scaling/%A_%a.out # Standard output
#SBATCH -e /n/netscratch/arguelles_delgado_lab/Everyone/jlazar/qc_scaling/%A_%a.err # Standard output
#SBATCH --array 1-100

if [ -z ${niter+x} ]; then
    niter=10000
fi

if [ -z ${seed+x} ]; then
    seed=${SLURM_ARRAY_TASK_ID}
fi

JULIA_BIN_PATH=$HOME/julias/julia-1.10.4/bin
export PATH=$PATH:$JULIA_BIN_PATH

outfile=/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/jlazar/qc_scaling/MNIST_${SLURM_ARRAY_TASK_ID}.h5

julia /n/home12/jlazar/qc_scaling/scripts/2_mnist_test/run.jl \
    --outfile $outfile \
    --nqubit 8 \
    --seed $seed \
    --goalfile /n/home12/jlazar/qc_scaling/scripts/2_mnist_test/resources/MNIST_strings.jld2:MNIST_${SLURM_ARRAY_TASK_ID} \
    --niter $niter \
    --nstate $nstate
