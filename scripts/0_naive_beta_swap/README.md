# Usage

Minimal usage. In this case the number of states will be equal to
$$
3\times \frac{3^{n_{\mathrm{qubit}}}}{2^{n_{\mathrm{qubit}}-1}+1}
$$

```bash
nqubit=8
outfile=../../data/naive_beta_swap.jld2
seed=925
niter=1000

julia naive_beta_swap.jl \
    --nqubit $nqubit \
    --outfile $outfile \
    --seed $seed \
    --niter $niter \
```

You can also 
