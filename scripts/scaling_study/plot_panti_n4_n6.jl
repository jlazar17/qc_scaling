using Pkg
Pkg.activate(joinpath(@__DIR__, "../utils"))

using Plots, DelimitedFiles, Printf

data, hdr = readdlm(joinpath(@__DIR__, "data/20_panti_scaling.csv"), ',', header=true)
hdr = vec(hdr)
col(name) = findfirst(==(name), hdr)

H_unique   = sort(unique(Float64.(data[:, col("H")])))
H_colors   = ["#0d0887","#5302a3","#8b0aa5","#b83289","#db5c68","#f48849","#f0f921"]
H_to_color = Dict(H => H_colors[i] for (i, H) in enumerate(H_unique))

plts = []
for nqubit in [4, 6, 8]
    rows = data[Int.(data[:, col("nqubit")]) .== nqubit, :]
    p = plot(title="nqubit = $nqubit", xlabel="nstate",
             ylabel=nqubit == 4 ? "p_anti" : "",
             legend=nqubit == 4 ? :topright : false,
             size=(420, 360), margin=5Plots.mm)
    for H in H_unique
        mask = Float64.(rows[:, col("H")]) .== H
        any(mask) || continue
        sub  = rows[mask, :]
        ns   = Int.(sub[:, col("nstate")])
        pa   = Float64.(sub[:, col("p_anti")])
        idx  = sortperm(ns)
        plot!(p, ns[idx], pa[idx]; label="H=$(H)", color=H_to_color[H],
              lw=2, marker=:circle, ms=5)
    end
    push!(plts, p)
end

fig = plot(plts..., layout=(1, 3), size=(1260, 380))
outfile = joinpath(@__DIR__, "data/20_panti_all_nqubit.png")
savefig(fig, outfile)
println("Saved $outfile")
