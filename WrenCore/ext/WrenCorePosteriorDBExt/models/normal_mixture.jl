# PosteriorDB: normal_2-normal_mixture
module NormalMixture

using PosteriorDB
using WrenCore: @wren, @addloglikelihood, @iid, logdensityof
using WrenDists: WrenDists as WD
using ..LogAddExp: logaddexp

const data = let
    pdb = PosteriorDB.database()
    PosteriorDB.load(PosteriorDB.dataset(
        PosteriorDB.posterior(pdb, "normal_2-normal_mixture"),
    ))
end

function mixture_logp(theta, mu, y)
    log_theta, log_other = log(theta), log1p(-theta)
    return sum(logaddexp(
        log_theta + logdensityof(WD.Normal(mu[1], 1.0), yi),
        log_other + logdensityof(WD.Normal(mu[2], 1.0), yi),
    ) for yi in y)
end

@wren function normal_mixture(y)
    theta ~ WD.Uniform(0.0, 1.0)
    @iid mu ~ WD.Normal(0.0, 10.0) size=2
    @addloglikelihood mixture_logp(theta, mu, y)
    return nothing
end

const wren_model = normal_mixture(data["y"])

end
