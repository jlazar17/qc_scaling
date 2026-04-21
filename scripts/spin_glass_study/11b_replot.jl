# Re-plots 11_picker_quality.csv with improved layout.

using Pkg
Pkg.activate(joinpath(@__DIR__, "../utils"))

using Plots, Statistics, DelimitedFiles

function nanmean(v); fv = filter(!isnan, v); isempty(fv) ? NaN : mean(fv); end
function nanse(v);   fv = filter(!isnan, v); length(fv) < 2 ? NaN : std(fv)/sqrt(length(fv)); end

infile  = joinpath(@__DIR__, "results", "11_picker_quality.csv")
outfile = joinpath(@__DIR__, "results", "11_picker_quality.png")

data, hdr = readdlm(infile, ',', header=true)
hdr = vec(hdr)

col(name) = findfirst(==(name), hdr)
seeds = Int.(data[:, col("seed")])
steps = Int.(data[:, col("step")])

# Replace sentinel -1 with NaN
raw = Float64.(data)
for c in 7:12   # best_pos0..gap1 columns
    raw[raw[:, c] .< 0, c] .= NaN
end

n_seeds = maximum(seeds)
all_steps = sort(unique(steps))
n_cp = length(all_steps)

# Aggregate per step across seeds
function agg_col(colname)
    ci = col(colname)
    means = Float64[]; ses = Float64[]
    for s in all_steps
        vals = raw[steps .== s, ci]
        push!(means, nanmean(vals))
        push!(ses,   nanse(vals))
    end
    return means, ses
end

acc0,  _       = agg_col("acc0")
acc1,  _       = agg_col("acc1")
fpos0, _       = agg_col("frac_pos0")
fpos1, _       = agg_col("frac_pos1")
best0, best0se = agg_col("best_pos0")
best1, best1se = agg_col("best_pos1")
pick0, pick0se = agg_col("pick_pos0")
pick1, pick1se = agg_col("pick_pos1")
gap0,  gap0se  = agg_col("gap0")
gap1,  gap1se  = agg_col("gap1")

# Focus range: steps where both H values still have oracle > 1%
active_mask = (all_steps .>= 500) .& (fpos0 .> 0.01) .& (fpos1 .> 0.01) .&
              .!isnan.(best0) .& .!isnan.(best1) .& (best0 .> 0) .& (best1 .> 0)
xs = all_steps[active_mask]

gap_mask = active_mask .& .!isnan.(gap0) .& .!isnan.(gap1) .&
           (gap0 .> 0) .& (gap1 .> 0)
xg = all_steps[gap_mask]

# ---- Panel 1: Accuracy ----
p_acc = plot(title="Accuracy", xlabel="SA step", ylabel="Accuracy",
             xscale=:log10, legend=:bottomright, xlims=(500, 3.2e5), ylims=(0.3, 1.0))
plot!(p_acc, max.(1, all_steps), acc0, label="H=0", color=:blue, lw=2)
plot!(p_acc, max.(1, all_steps), acc1, label="H=1", color=:red,  lw=2)

# ---- Panel 2: Fraction positive oracle ----
p_frac = plot(title="Fraction of proposals with a positive-delta pattern",
              xlabel="SA step", ylabel="Fraction", xscale=:log10,
              legend=:topright, xlims=(500, 3.2e5))
plot!(p_frac, max.(1, all_steps), fpos0, label="H=0", color=:blue, lw=2)
plot!(p_frac, max.(1, all_steps), fpos1, label="H=1", color=:red,  lw=2)

# ---- Panel 3: best_pos and pick_pos ----
p_delta = plot(title="Mean positive delta — oracle best (solid) vs picker (dashed)",
               xlabel="SA step", ylabel="Mean Δacc | Δacc > 0",
               xscale=:log10, legend=:topright,
               xlims=(xs[1], xs[end]))
plot!(p_delta, xs, best0[active_mask], ribbon=1.96 .* best0se[active_mask],
      fillalpha=0.2, label="H=0 best", color=:blue, lw=2)
plot!(p_delta, xs, best1[active_mask], ribbon=1.96 .* best1se[active_mask],
      fillalpha=0.2, label="H=1 best", color=:red,  lw=2)
plot!(p_delta, xs, pick0[active_mask], ribbon=1.96 .* pick0se[active_mask],
      fillalpha=0.15, label="H=0 picker", color=:blue, lw=2, ls=:dash)
plot!(p_delta, xs, pick1[active_mask], ribbon=1.96 .* pick1se[active_mask],
      fillalpha=0.15, label="H=1 picker", color=:red,  lw=2, ls=:dash)

# ---- Panel 4: absolute gap ----
p_gap = plot(title="Picker gap = oracle_best − picker  (given oracle > 0)",
             xlabel="SA step", ylabel="Mean gap",
             xscale=:log10, legend=:topleft,
             xlims=(xg[1], xg[end]))
plot!(p_gap, xg, gap0[gap_mask], ribbon=1.96 .* gap0se[gap_mask],
      fillalpha=0.2, label="H=0", color=:blue, lw=2)
plot!(p_gap, xg, gap1[gap_mask], ribbon=1.96 .* gap1se[gap_mask],
      fillalpha=0.2, label="H=1", color=:red,  lw=2)

# ---- Panel 5: gap ratio ----
p_ratio = plot(title="Gap ratio  gap₁ / gap₀   (>1 = H=1 picker worse)",
               xlabel="SA step", ylabel="Ratio",
               xscale=:log10, legend=:topleft,
               xlims=(xg[1], xg[end]))
hline!(p_ratio, [1.0], color=:black, ls=:dot, lw=1, label="ratio = 1")
plot!(p_ratio, xg, gap1[gap_mask] ./ gap0[gap_mask],
      color=:purple, lw=2, label="gap₁/gap₀")

combined = plot(p_acc, p_frac, p_delta, p_gap, p_ratio,
                layout=@layout([a b; c d; e _]),
                size=(1400, 1300),
                margin=6Plots.mm,
                plot_title="Picker quality vs SA trajectory  (nqubit=6, nstate=45, n=$(n_seeds) seeds)")

savefig(combined, outfile)
println("Saved to $outfile")
