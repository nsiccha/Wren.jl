# Gaussian regressions with unconstrained flat coefficients and positive flat scale.
module FlatGaussianRegression

using WrenCore: @wren, @observe, @batch, @iid, @addloglikelihood
using WrenDists: WrenDists as WD

@wren function flat_gaussian(y, X)
    @iid beta ~ WD.Flat() size=size(X, 2)
    log_sigma ~ WD.Flat()
    sigma = exp(log_sigma)
    # A flat density in positive sigma induces this Jacobian in log sigma.
    # Temporary until the constrained Flat API in #192 is available here.
    @addloglikelihood log_sigma
    mu = X * beta
    @batch for i in eachindex(y)
        @observe y[i] ~ WD.Normal(mu[i], sigma)
    end
    return nothing
end

end
