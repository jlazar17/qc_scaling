# merge_n10.jl — combine per-H HDF5 files produced by scaling_study_n10.jl
# into a single scaling_study_n10.h5 that plot_scaling_study.jl can read.
#
# Usage: julia merge_n10.jl
#
# Reads:  data/scaling_study_n10_H{h}.h5  (one per H value)
# Writes: data/scaling_study_n10.h5

using Pkg
Pkg.activate(joinpath(@__DIR__, "../utils"))

using HDF5, Printf

function main()
    datadir  = joinpath(@__DIR__, "data")
    H_vals   = [0.0, 0.25, 0.5, 0.625, 0.75, 0.875, 1.0]
    outfile  = joinpath(datadir, "scaling_study_n10.h5")

    println("Merging per-H files into $outfile")

    h5open(outfile, "cw") do h5out
        for H in H_vals
            infile = joinpath(datadir, @sprintf("scaling_study_n10_H%.3f.h5", H))
            if !isfile(infile)
                @printf("  H=%.3f: file not found (%s), skipping\n", H, infile)
                continue
            end
            h5open(infile, "r") do h5in
                for nq_key in keys(h5in)
                    nq_in = h5in[nq_key]
                    nq_out = haskey(h5out, nq_key) ? h5out[nq_key] :
                             create_group(h5out, nq_key)

                    # Copy nqubit-level attributes once
                    if !haskey(HDF5.attributes(nq_out), "nqubit")
                        for ak in keys(HDF5.attributes(nq_in))
                            HDF5.attributes(nq_out)[ak] = read(HDF5.attributes(nq_in)[ak])
                        end
                    end

                    for h_key in keys(nq_in)
                        if haskey(nq_out, h_key)
                            @printf("  H=%.3f  %s: already in output, skipping\n", H, h_key)
                            continue
                        end
                        HDF5.copy_object(nq_in[h_key], nq_out, h_key)
                        @printf("  H=%.3f  %s: copied\n", H, h_key)
                    end
                end
            end
        end
    end
    println("Done. Merged file: $outfile")
end

main()
