include("./0_random_variation.jl")

function parse_sweep_single_args()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--nqubit"
            arg_type = Int
            required = true
        "--niter"
            arg_type = Int
            default = 10_000
        "--pzero"
            arg_type = Float64
            required = true
        "--nstate-multiplier"
            arg_type = Int
            required = true
        "--seed"
            arg_type = Int
            required = true
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

function run_single()
    args = parse_sweep_single_args()
    nqubit = args["nqubit"]
    base_nstate = 3 * Int(ceil(3^nqubit / 2^(nqubit - 1)))
    nstate = args["nstate-multiplier"] * base_nstate
    pzero = args["pzero"]
    seed = args["seed"]
    outfile = args["outfile"]
    mkpath(dirname(outfile))

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
    println("Running: nstate=$nstate pzero=$pzero seed=$seed -> $outfile")
    main(run_args)
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_single()
end
