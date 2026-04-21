using Pkg
Pkg.activate("..")

using ArgParse
using HDF5

function parse_merge_args()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--outdir"
            arg_type = String
            required = true
            help = "Directory containing .h5 files to merge"
        "--output"
            arg_type = String
            required = true
            help = "Path for merged output .h5 file"
    end
    return parse_args(s)
end

function copy_group(src_group, dst_group)
    for key in keys(src_group)
        obj = src_group[key]
        if obj isa HDF5.Group
            sub = create_group(dst_group, key)
            # Copy attributes
            for attr_name in keys(attributes(obj))
                attributes(sub)[attr_name] = read(attributes(obj)[attr_name])
            end
            copy_group(obj, sub)
        else
            dst_group[key] = read(obj)
        end
    end
end

function merge_files()
    args = parse_merge_args()
    outdir = args["outdir"]
    output = args["output"]

    h5files = filter(f -> endswith(f, ".h5"), readdir(outdir, join=true))
    if isempty(h5files)
        error("No .h5 files found in $outdir")
    end

    h5open(output, "w") do out_h5
        for filepath in sort(h5files)
            groupname = replace(basename(filepath), ".h5" => "")
            h5open(filepath, "r") do in_h5
                gp = create_group(out_h5, groupname)
                for key in keys(in_h5)
                    sub = create_group(gp, key)
                    src = in_h5[key]
                    # Copy attributes
                    for attr_name in keys(attributes(src))
                        attributes(sub)[attr_name] = read(attributes(src)[attr_name])
                    end
                    copy_group(src, sub)
                end
            end
            println("Merged: $filepath -> $groupname")
        end
    end
    println("Done. Merged $(length(h5files)) files into $output")
end

if abspath(PROGRAM_FILE) == @__FILE__
    merge_files()
end
