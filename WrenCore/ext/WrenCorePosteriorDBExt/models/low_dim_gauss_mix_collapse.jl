# PosteriorDB: low_dim_gauss_mix_collapse-low_dim_gauss_mix_collapse
module LowDimGaussMixCollapse

using PosteriorDB
using WrenCore: @wren, @addloglikelihood, @iid, logdensityof
using WrenDists: WrenDists as WD
using ..LogAddExp: logaddexp

const data = let
    pdb = PosteriorDB.database()
    PosteriorDB.load(PosteriorDB.dataset(PosteriorDB.posterior(
        pdb, "low_dim_gauss_mix_collapse-low_dim_gauss_mix_collapse",
    )))
end

function low_dim_mixture_logp(mu, sigma, theta, y)
    first_weight, second_weight = log(theta), log1p(-theta)
    total = zero(theta + mu[1] + sigma[1])
    for yi in y
        first = first_weight + logdensityof(WD.Normal(mu[1], sigma[1]), yi)
        second = second_weight + logdensityof(WD.Normal(mu[2], sigma[2]), yi)
        total += logaddexp(first, second)
    end
    return total
end

@wren function low_dim_gauss_mix_collapse(y)
    @iid mu ~ WD.Normal(0.0, 2.0) size=2
    @iid sigma ~ WD.HalfNormal(0.0, 2.0) size=2
    theta ~ WD.Beta(5.0, 5.0)
    @addloglikelihood low_dim_mixture_logp(mu, sigma, theta, y)
    return nothing
end

const wren_model = low_dim_gauss_mix_collapse(data["y"])

end
