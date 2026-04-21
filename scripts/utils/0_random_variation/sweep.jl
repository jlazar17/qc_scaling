include("./0_random_variation.jl")

function parse_sweep_args()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--nqubit"
            arg_type = Int
            required = true
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
        "--outfile"
            arg_type = String
            required = true
        "--savelevel"
            arg_type = String
            default = "deltas"
        "--nreplace"
            arg_type = Int
            default = 1
    end
    return parse_args(s)
end

function run_sweep()
    args = parse_sweep_args()
    nqubit = args["nqubit"]
    base_nstate_exact = 3^nqubit / 2^(nqubit - 1)
    pzero_values = parse.(Float64, split(args["pzero-values"], ","))
    multipliers = parse.(Int, split(args["nstate-multipliers"], ","))
    nseeds = args["nseeds"]
    outfile = args["outfile"]
    mkpath(dirname(outfile))

    seeds = rand(UInt32, nseeds)
    while length(unique(seeds)) < nseeds
        seeds = rand(UInt32, nseeds)
    end
    seeds = Int.(seeds)

    for mult in multipliers
        nstate = Int(ceil(mult * base_nstate_exact))
        for pzero in pzero_values
            for seed in seeds
                group = "nstate_$(nstate)_pzero_$(pzero)_seed_$(seed)"
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
                println("Running: nstate=$nstate pzero=$pzero seed=$seed")
                main(run_args)
            end
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_sweep()
end
