using Pkg
Pkg.activate(joinpath(@__DIR__, "../utils"))

using HDF5
using Printf
using Statistics

# ---------------------------------------------------------------------------
# Data source: 1000 Genomes Project Phase 3, chromosome 1
# Accessed via bcftools streaming remote tabix-indexed VCF.
# Reference: 1000 Genomes Project Consortium, Nature 526, 68–74 (2015).
# ---------------------------------------------------------------------------

const VCF_URL = "http://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/" *
                "ALL.chr1.phase3_shapeit2_mvncall_integrated_v5b.20130502.genotypes.vcf.gz"

function ngbits(nqubit)
    return (3^nqubit - 1) ÷ 2
end

function hamming_entropy(k, N)
    (k == 0 || k == N) && return 0.0
    p = k / N
    return -p * log2(p) - (1 - p) * log2(1 - p)
end

function parse_gt(gt_str)
    # Extract GT field (before any colon for FORMAT subfields)
    gt = split(gt_str, ':')[1]
    sep = '|' in gt ? '|' : '/'
    parts = split(gt, sep)
    length(parts) != 2 && return missing
    ('.' in parts[1] || '.' in parts[2]) && return missing
    a1 = parse(Int, parts[1])
    a2 = parse(Int, parts[2])
    return Int8((a1 + a2) > 0 ? 1 : 0)
end

function fetch_genotypes(region, min_maf; max_snps, nsamples_want)
    # -m2 -M2: biallelic sites only
    # -v snps: SNPs only (no indels)
    # --min-af: minor allele frequency filter
    cmd = ignorestatus(`bcftools view -r $(region) -m2 -M2 -v snps --min-af $(min_maf):minor $(VCF_URL)`)

    sample_ids = String[]
    rows = Vector{Vector{Int8}}()
    nsamples = 0

    io = open(cmd, "r")
    try
        for line in eachline(io)
            if startswith(line, "##")
                continue
            elseif startswith(line, "#CHROM")
                fields = split(line, '\t')
                all_samples = fields[10:end]
                nsamples = min(length(all_samples), nsamples_want)
                sample_ids = all_samples[1:nsamples]
                continue
            end

            length(rows) >= max_snps && break

            fields = split(line, '\t')
            length(fields) < 10 + nsamples - 1 && continue

            gts = [parse_gt(fields[9 + s]) for s in 1:nsamples]
            any(ismissing, gts) && continue
            push!(rows, Int8[g for g in gts])
        end
    finally
        try; close(io); catch; end
    end

    nsnps = length(rows)
    mat = zeros(Int8, nsnps, nsamples)
    for (i, row) in enumerate(rows)
        mat[i, :] = row
    end
    return sample_ids, mat
end

function main()
    nqubits      = [4, 6, 8, 10]
    nsamples     = 20       # number of individuals to use as goal vectors
    min_maf      = 0.05     # keep common variants (MAF >= 5%)
    # chr1 region wide enough to cover n=10 (needs ~30k SNPs at ~1 SNP/kb density)
    region       = "1:10000000-50000000"

    max_snps_needed = maximum(ngbits.(nqubits))

    outdir  = joinpath(@__DIR__, "data")
    mkpath(outdir)
    outfile = joinpath(outdir, "snp_goals.h5")

    @printf("Fetching SNP data from 1000 Genomes Phase 3 (chr1)\n")
    @printf("Region: %s  |  min MAF: %.2f  |  target SNPs: %d\n",
            region, min_maf, max_snps_needed)
    flush(stdout)

    sample_ids, geno_mat = fetch_genotypes(region, min_maf;
                                           max_snps=max_snps_needed,
                                           nsamples_want=nsamples)

    nsnps_got, nsamples_got = size(geno_mat)
    @printf("Fetched %d SNPs for %d samples\n\n", nsnps_got, nsamples_got)

    if nsnps_got < max_snps_needed
        @warn "Only got $nsnps_got SNPs; needed $max_snps_needed for n=$(maximum(nqubits)). " *
              "Consider widening the region."
    end

    println("Goal vector summary:")
    println("-"^60)

    h5open(outfile, "w") do h5f
        HDF5.attributes(h5f)["vcf_url"]    = VCF_URL
        HDF5.attributes(h5f)["region"]     = region
        HDF5.attributes(h5f)["min_maf"]    = min_maf
        HDF5.attributes(h5f)["nsamples"]   = nsamples_got
        HDF5.attributes(h5f)["sample_ids"] = collect(sample_ids)

        for nqubit in nqubits
            nb = ngbits(nqubit)
            nb > nsnps_got && (@printf("nqubit=%2d: insufficient SNPs, skipping\n", nqubit); continue)

            goals = geno_mat[1:nb, :]   # [ngbits × nsamples]

            entropies = [hamming_entropy(sum(goals[:, s]), nb) for s in 1:nsamples_got]

            @printf("nqubit=%2d  ngbits=%6d  median_H=%.3f  (range %.3f–%.3f)\n",
                    nqubit, nb, median(entropies), minimum(entropies), maximum(entropies))

            gp = create_group(h5f, "nqubit_$(nqubit)")
            HDF5.attributes(gp)["nqubit"] = nqubit
            HDF5.attributes(gp)["ngbits"] = nb
            gp["goals"]     = goals        # [ngbits × nsamples], Int8
            gp["entropies"] = entropies    # [nsamples]
        end
    end

    println("\nSaved to $outfile")
end

main()
