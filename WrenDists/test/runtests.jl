using Test

@testset "WrenDists" begin
    include("precision.jl")
    include("sampling.jl")

    # Continuous unconstrained
    include("univariate/normal.jl")
    include("univariate/laplace.jl")
    include("univariate/cauchy.jl")
    include("univariate/logistic.jl")
    include("univariate/tdist.jl")
    include("univariate/flat.jl")
    include("univariate/truncatedflat.jl")

    # Continuous positive
    include("univariate/lognormal.jl")
    include("univariate/exponential.jl")
    include("univariate/weibull.jl")
    include("univariate/gamma.jl")
    include("univariate/inversegamma.jl")
    include("univariate/halfnormal.jl")
    include("univariate/halfcauchy.jl")
    include("univariate/halftdist.jl")

    # Continuous bounded
    include("univariate/beta.jl")
    include("univariate/uniform.jl")

    # Discrete univariate
    include("univariate/bernoulli.jl")
    include("univariate/categorical.jl")
    include("univariate/binomial.jl")
    include("univariate/poisson.jl")
    include("univariate/geometric.jl")
    include("univariate/negativebinomial.jl")

    # Multivariate
    include("multivariate/mvnormalscale.jl")
    include("multivariate/dirichlet.jl")
    include("multivariate/multinomial.jl")

    include("independent.jl")
end
