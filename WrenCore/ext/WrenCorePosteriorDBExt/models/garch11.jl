# PosteriorDB: garch-garch11
module Garch11

using PosteriorDB
using WrenCore: @wren, @observe, @batch, @addloglikelihood
using WrenDists: WrenDists as WD

const data = let
    pdb = PosteriorDB.database()
    PosteriorDB.load(PosteriorDB.dataset(PosteriorDB.posterior(pdb, "garch-garch11")))
end

@wren function garch11(y, sigma1)
    mu ~ WD.Flat()
    log_alpha0 ~ WD.Flat()
    alpha0 = exp(log_alpha0)
    # Temporary positive-flat alpha0 transform pending #192.
    @addloglikelihood log_alpha0
    # Stan declares alpha1 on (0, 1) and beta1 on (0, 1 - alpha1) with no prior
    # statement: a flat prior on the triangle alpha1 + beta1 < 1. Beta(1, 2)
    # times Uniform(0, 1 - alpha1) has the constant density 2 on that triangle,
    # so the alpha1 and beta1 lines after this comment give Stan's prior up to
    # log(2). Uniform alone would not: its 1/(1 - alpha1) depends on alpha1.
    # Equivalent alternatives:
    #     alpha1 ~ WD.Uniform(0.0, 1.0)
    #     beta1 ~ WD.Uniform(0.0, 1 - alpha1)
    #     @addloglikelihood log1p(-alpha1)  # cancels Uniform's 1/(1 - alpha1)
    # or, the direct translation of Stan's bounds, once a truncated flat
    # with parameter-dependent bounds is available (#203, tracked in #207):
    #     alpha1 ~ WD.Uniform(0.0, 1.0)
    #     beta1 ~ WD.truncated(WD.Flat(); lower=0.0, upper=1 - alpha1)
    alpha1 ~ WD.Beta(1.0, 2.0)
    beta1 ~ WD.Uniform(0.0, 1 - alpha1)
    sigma = Vector{typeof(alpha0 + mu)}(undef, length(y))
    sigma[1] = sigma1
    for t in 2:length(y)
        sigma[t] = sqrt(alpha0 + alpha1 * (y[t - 1] - mu)^2 +
            beta1 * sigma[t - 1]^2)
    end
    @batch for t in eachindex(y)
        @observe y[t] ~ WD.Normal(mu, sigma[t])
    end
    return nothing
end

const wren_model = garch11(data["y"], data["sigma1"])

end
