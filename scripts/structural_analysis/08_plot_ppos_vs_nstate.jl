using Pkg
Pkg.activate(joinpath(@__DIR__, "../../notebooks"))

using CairoMakie
using Printf

outdir = joinpath(@__DIR__, "figures")
mkpath(outdir)

# ---------------------------------------------------------------------------
# Data from 05_ppos_vs_nstate.jl run
# NaN = checkpoint not reached at that (nstate, acc) combination
# ---------------------------------------------------------------------------

const NaN64 = NaN

# nqubit=6
ns6 = [30, 40, 55, 70, 84, 104, 130, 176]

data6 = Dict(
    "H=0.00" => Dict(
        0.70 => [0.0037, 0.0515, 0.2103, 0.2240, 0.2402, 0.2907, 0.3387, 0.3290],
        0.75 => [0.0003, 0.0245, 0.1425, 0.1705, 0.2393, 0.2577, 0.2387, 0.2788],
        0.80 => [0.0000, 0.0027, 0.0427, 0.1080, 0.1862, 0.2575, 0.2643, 0.2380],
        0.85 => [NaN,   0.0007, 0.0258, 0.0750, 0.1008, 0.2202, 0.2045, 0.2097],
        0.90 => [NaN,   NaN,    0.0088, 0.0252, 0.0645, 0.0947, 0.1683, 0.1455],
    ),
    "H=0.25" => Dict(
        0.70 => [0.0008, 0.0268, 0.1728, 0.2258, 0.2423, 0.2603, 0.2430, 0.3558],
        0.75 => [0.0007, 0.0140, 0.0930, 0.1828, 0.1840, 0.2208, 0.2563, 0.2665],
        0.80 => [NaN,    0.0055, 0.0227, 0.1113, 0.1872, 0.2153, 0.2207, 0.2288],
        0.85 => [NaN,    0.0003, 0.0140, 0.0265, 0.0882, 0.1467, 0.2388, 0.2427],
        0.90 => [NaN,    NaN,    0.0012, 0.0088, 0.0308, 0.0768, 0.1085, 0.1398],
    ),
    "H=0.50" => Dict(
        0.70 => [NaN,    0.0053, 0.0727, 0.1707, 0.1830, 0.2407, 0.2337, 0.2703],
        0.75 => [NaN,    0.0002, 0.0453, 0.1192, 0.1637, 0.2275, 0.2083, 0.2648],
        0.80 => [NaN,    NaN,    0.0138, 0.0422, 0.1310, 0.1515, 0.2053, 0.3013],
        0.85 => [NaN,    NaN,    0.0020, 0.0098, 0.0418, 0.0663, 0.1613, 0.1312],
        0.90 => [NaN,    NaN,    0.0000, 0.0015, 0.0105, 0.0202, 0.0503, 0.0730],
    ),
    "H=0.75" => Dict(
        0.70 => [NaN,    0.0010, 0.0218, 0.0663, 0.1257, 0.1987, 0.2520, 0.4002],
        0.75 => [NaN,    0.0000, 0.0043, 0.0200, 0.0875, 0.2040, 0.2210, 0.2575],
        0.80 => [NaN,    NaN,    0.0002, 0.0042, 0.0252, 0.0795, 0.1388, 0.2200],
        0.85 => [NaN,    NaN,    NaN,    0.0002, 0.0013, 0.0410, 0.0718, 0.1295],
        0.90 => [NaN,    NaN,    NaN,    0.0000, 0.0007, 0.0098, 0.0212, 0.0503],
    ),
    "H=1.00" => Dict(
        0.70 => [NaN,    0.0003, 0.0138, 0.0510, 0.1055, 0.1870, 0.2927, 0.3625],
        0.75 => [NaN,    0.0000, 0.0018, 0.0175, 0.0623, 0.1268, 0.2162, 0.2937],
        0.80 => [NaN,    NaN,    0.0000, 0.0038, 0.0185, 0.0368, 0.0887, 0.2393],
        0.85 => [NaN,    NaN,    NaN,    0.0007, 0.0055, 0.0180, 0.0635, 0.1412],
        0.90 => [NaN,    NaN,    NaN,    0.0000, 0.0000, 0.0013, 0.0150, 0.0615],
    ),
)

# nqubit=8
ns8 = [100, 130, 175, 230, 300, 400, 500]

data8 = Dict(
    "H=0.00" => Dict(
        0.70 => [0.0150, 0.1047, 0.3168, 0.3750, 0.3752, 0.3788, 0.3433],
        0.75 => [0.0015, 0.0367, 0.1657, 0.3463, 0.3482, 0.3958, 0.3270],
        0.80 => [NaN,    0.0023, 0.0493, 0.2073, 0.3075, 0.3652, 0.3465],
        0.85 => [NaN,    NaN,    0.0107, 0.0590, 0.1567, 0.3040, 0.2762],
        0.90 => [NaN,    NaN,    0.0010, 0.0133, 0.0613, 0.1575, 0.2290],
    ),
    "H=0.25" => Dict(
        0.70 => [0.0022, 0.0610, 0.2193, 0.3438, 0.3940, 0.3678, 0.3487],
        0.75 => [0.0000, 0.0123, 0.0980, 0.2492, 0.3465, 0.3485, 0.4162],
        0.80 => [NaN,    0.0003, 0.0185, 0.1193, 0.2310, 0.3170, 0.3230],
        0.85 => [NaN,    NaN,    0.0007, 0.0192, 0.1005, 0.2225, 0.2902],
        0.90 => [NaN,    NaN,    NaN,    0.0003, 0.0173, 0.0465, 0.1197],
    ),
    "H=0.50" => Dict(
        0.70 => [0.0000, 0.0140, 0.1202, 0.2680, 0.3525, 0.3312, 0.3033],
        0.75 => [NaN,    0.0002, 0.0258, 0.0917, 0.2490, 0.3032, 0.2997],
        0.80 => [NaN,    NaN,    0.0003, 0.0167, 0.0793, 0.1918, 0.2317],
        0.85 => [NaN,    NaN,    NaN,    0.0007, 0.0105, 0.0377, 0.0708],
        0.90 => [NaN,    NaN,    NaN,    NaN,    NaN,    0.0000, 0.0017],
    ),
    "H=0.75" => Dict(
        0.70 => [NaN,    NaN,    0.0063, 0.0535, 0.1205, 0.2707, 0.2588],
        0.75 => [NaN,    NaN,    NaN,    0.0033, 0.0197, 0.0968, 0.1392],
        0.80 => [NaN,    NaN,    NaN,    NaN,    0.0007, 0.0097, 0.0223],
        0.85 => [NaN,    NaN,    NaN,    NaN,    NaN,    0.0000, 0.0002],
        0.90 => [NaN,    NaN,    NaN,    NaN,    NaN,    NaN,    NaN   ],
    ),
    "H=1.00" => Dict(
        0.70 => [NaN,    NaN,    NaN,    0.0000, 0.0025, 0.0427, 0.1462],
        0.75 => [NaN,    NaN,    NaN,    NaN,    0.0000, 0.0028, 0.0325],
        0.80 => [NaN,    NaN,    NaN,    NaN,    NaN,    0.0000, 0.0022],
        0.85 => [NaN,    NaN,    NaN,    NaN,    NaN,    NaN,    NaN   ],
        0.90 => [NaN,    NaN,    NaN,    NaN,    NaN,    NaN,    NaN   ],
    ),
)

# ---------------------------------------------------------------------------
# Plot helper
# ---------------------------------------------------------------------------

H_labels = ["H=0.00", "H=0.25", "H=0.50", "H=0.75", "H=1.00"]
acc_vals  = [0.70, 0.75, 0.80, 0.85, 0.90]

# Sequential colors for accuracy levels (light → dark)
acc_colors = Makie.cgrad(:viridis, length(acc_vals), categorical=true)

function make_figure(nqubit, ns, data)
    fig = Figure(size=(1400, 320))

    axes = [Axis(fig[1, j],
                 title  = H_labels[j],
                 xlabel = j == 3 ? "nstate" : "",
                 ylabel = j == 1 ? "p_pos" : "",
                 yscale = log10,
                 yticks = ([1e-4, 1e-3, 0.01, 0.1],
                            ["10⁻⁴", "10⁻³", "10⁻²", "10⁻¹"]),
                 yminorticksvisible = true,
                 yminorgridvisible  = true,
                 )
            for j in 1:5]

    for (j, hlabel) in enumerate(H_labels)
        ax = axes[j]
        hdata = data[hlabel]
        for (k, acc) in enumerate(acc_vals)
            ys = hdata[acc]
            # Replace zeros and NaNs: zeros become a floor for log scale display
            ys_plot = [isnan(y) || y == 0.0 ? NaN : y for y in ys]
            valid = .!isnan.(ys_plot)
            any(valid) || continue
            lines!(ax, ns[valid], ys_plot[valid],
                   color=acc_colors[k], linewidth=2,
                   label=@sprintf("acc=%.2f", acc))
            scatter!(ax, ns[valid], ys_plot[valid],
                     color=acc_colors[k], markersize=7)
        end
        ylims!(ax, 5e-5, 0.6)
        xlims!(ax, ns[1] - (ns[end]-ns[1])*0.05,
                   ns[end] + (ns[end]-ns[1])*0.05)
        j > 1 && hideydecorations!(ax, grid=false, minorgrid=false)
    end

    # Shared legend
    Legend(fig[1, 6],
           [LineElement(color=acc_colors[k], linewidth=2) for k in 1:length(acc_vals)],
           [@sprintf("acc=%.2f", a) for a in acc_vals],
           "accuracy",
           framevisible=false)

    Label(fig[0, 1:6],
          "p_pos vs nstate — nqubit=$nqubit (shuffled pairing, 4 trials)",
          fontsize=15, font=:bold)

    return fig
end

fig6 = make_figure(6, ns6, data6)
save(joinpath(outdir, "fig_ppos_vs_nstate_n6.pdf"), fig6)
save(joinpath(outdir, "fig_ppos_vs_nstate_n6.png"), fig6, px_per_unit=2)
println("Saved fig_ppos_vs_nstate_n6")

fig8 = make_figure(8, ns8, data8)
save(joinpath(outdir, "fig_ppos_vs_nstate_n8.pdf"), fig8)
save(joinpath(outdir, "fig_ppos_vs_nstate_n8.png"), fig8, px_per_unit=2)
println("Saved fig_ppos_vs_nstate_n8")
