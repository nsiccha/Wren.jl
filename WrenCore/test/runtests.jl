module WrenCoreTests

using Test

@testset verbose=true "WrenCore.jl" begin
    include("expr.jl")
    include("dataflow.jl")
    include("syntax.jl")
    include("evaluate.jl")
    include("interface.jl")
    include("space.jl")
    include("batch/general.jl")
    include("batch/normal.jl")
    include("batch/bernoulli.jl")
    include("batch/binomial.jl")
    include("batch/poisson.jl")
    include("batch/exponential.jl")
    include("batch/lognormal.jl")
    include("batch/categorical.jl")
    include("batch/tdist.jl")
    include("batch/two_parameter.jl")
    include("batch/typestability.jl")
    include("ext/flexichains.jl")
    include("aliasing.jl")
    include("submodel.jl")
    include("tq.jl")
    include("enzyme_conditioning.jl")
    include("vecmodel.jl")
    include("ad.jl")
    include("reactant.jl")
    include("posteriordb.jl")
    include("issues/55.jl")
    include("issues/69.jl")
    include("issues/132.jl")
    include("issues/146.jl")
    include("issues/155.jl")
    include("issues/192.jl")
end

end # module WrenCoreTests
