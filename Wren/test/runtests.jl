using Test
using Wren

@testset verbose=true "Wren.jl" begin
    include("nuts.jl")
    include("sampling.jl")
    include("tracked_quantities.jl")
    include("pathfinder.jl")
    include("posteriordb.jl")
end
