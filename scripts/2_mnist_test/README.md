```bash
julia run.jl --outfile test.jld2 --nqubit 8 --seed 3 --goalfile ./resources/MNIST_strings.jld2:MNIST_1 --track --niter 1000
```

```bash
seed=9
for nstate in 153 255 357 459 561 663 765 867 969 1071 1173 1275 1377 1479
do 
    cmd="sbatch -D $PWD --export=seed=$seed,nstate=$nstate submit_run.sbatch"
    $cmd
    seed=$(($seed+34))
done
```
