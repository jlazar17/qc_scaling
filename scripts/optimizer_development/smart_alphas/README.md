# Usage

## Minimal Usage

```bash
nqubit=8
outfile=../../data/naive_beta_swap.jld2
seed=925
niter=1000

julia naive_beta_swap.jl \
    --nqubit $nqubit \
    --outfile $outfile \
    --seed $seed \
    --niter $niter
```

In this case the number of states will be equal to

$$3\times \frac{3^{n}}{2^{n-1}+1}$$

where $n$ is the number of qubits.
This is chosen so that, optimally, each PO will have three states covering it.

## Other options

```bash
nqubit=8
outfile=../../data/naive_beta_swap.jld2
seed=925
niter=1000
nstate=523
goalfile="../../data/goals.jld2" # Binary strings to represent
statesfile="../../data/states/jld2" # Previously used states to continue optimizing
outgroup="my_results" # Name of the group in the JLD2 file

julia naive_beta_swap.jl \
    --nqubit $nqubit \
    --outfile $outfile \
    --seed $seed \
    --niter $niter \
    #--nstate $nstate \
    #--goalfile $goalfile \ # Not yet implemented
    #--statesfile $statesfile \ # Not yet implemented
    #--outgroup $outgroup \
    #--track \ # Use a progress bar
```
