include("./0_random_variation.jl")

function parse_sweep_nqubit_args()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--nqubits"
            arg_type = String
            default = "4,6,8,10"
        "--niter"
            arg_type = Int
            default = 10_000
        "--pzero-values"
            arg_type = String
            default = "0.0,0.1,0.3,0.5,0.7,0.9,1.0"
        "--nstate-multipliers"
            arg_type = String
            default = "1,2,3,4,5,6,7"
        "--nseeds"
            arg_type = Int
            default = 10
        "--outdir"
            arg_type = String
            required = true
        "--savelevel"
            arg_type = String
            default = "deltas"
        "--nreplace"
            arg_type = Int
            default = 1
        "--base-seed"
            arg_type = Int
            default = 12345
    end
    return parse_args(s)
end

function generate_seeds(base_seed, nseeds)
    rng = Random.MersenneTwister(base_seed)
    seeds = rand(rng, UInt32, nseeds)
    while length(unique(seeds)) < nseeds
        seeds = rand(rng, UInt32, nseeds)
    end
    return Int.(seeds)
end

function existing_groups(outfile)
    !isfile(outfile) && return Set{String}()
    h5open(outfile, "r") do h5f
        Set{String}(keys(h5f))
    end
end

function run_sweep_nqubit()
    args = parse_sweep_nqubit_args()
    nqubits = parse.(Int, split(args["nqubits"], ","))
    pzero_values = parse.(Float64, split(args["pzero-values"], ","))
    multipliers = parse.(Int, split(args["nstate-multipliers"], ","))
    nseeds = args["nseeds"]
    outdir = args["outdir"]
    mkpath(outdir)

    seeds = generate_seeds(args["base-seed"], nseeds)

    for nqubit in nqubits
        outfile = joinpath(outdir, "nqubit_$(nqubit).h5")
        base_nstate_exact = 3^nqubit / 2^(nqubit - 1)
        done = existing_groups(outfile)
        println("=== nqubit=$nqubit (outfile=$outfile, $(length(done)) groups already done) ===")

        for mult in multipliers
            nstate = Int(ceil(mult * base_nstate_exact))
            for pzero in pzero_values
                for seed in seeds
                    group = "nstate_$(nstate)_pzero_$(pzero)_seed_$(seed)"
                    if group in done
                        println("Skipping (exists): $group")
                        continue
                    end
                    run_args = Dict(
                        "nqubit" => nqubit,
                        "niter" => args["niter"],
                        "nstate" => nstate,
                        "pzero" => pzero,
                        "seed" => seed,
                        "outfile" => outfile,
                        "outgroup" => group,
                        "savelevel" => args["savelevel"],
                        "nreplace" => args["nreplace"],
                        "n_same_tol" => 10,
                        "track" => false,
                        "goalfile" => "",
                        "statefile" => "",
                    )
                    println("Running: nqubit=$nqubit nstate=$nstate pzero=$pzero seed=$seed")
                    main(run_args)
                end
            end
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_sweep_nqubit()
end
