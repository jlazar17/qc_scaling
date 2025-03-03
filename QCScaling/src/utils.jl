function po_coverage(states::Vector{PseudoGHZState})
   a = zeros(Int, 3^8)
   for cxt in contexts[1:10]
       idxs = map(x-> x.index, cxt.pos)
       a[idxs] .+= 1
   end
   return a .> 0, (a[1:2:end-2] .* a[2:2:end-1]) .> 0
end
