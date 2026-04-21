includet("./0_random_variation.jl")

args = Dict("n_same_tol" => 1, "nreplace"=>1, "nqubit"=>8, "seed"=>4, "nstate"=> 200, "goalfile"=>"", "statefile"=>"", "niter"=>100, "track"=>false, "outfile"=>"testing.h5", "outgroup"=>"results", "pzero"=>0.0, "savelevel"=>"best_states")

main(args)
