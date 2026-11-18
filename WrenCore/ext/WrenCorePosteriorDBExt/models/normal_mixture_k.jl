# PosteriorDB: normal_5-normal_mixture_k
module NormalMixtureK

using PosteriorDB
using WrenCore: @wren, @addloglikelihood, @iid, logdensityof
using WrenDists: WrenDists as WD
using ..LogAddExp: logaddexp

const data = let
    pdb = PosteriorDB.database()
    PosteriorDB.load(PosteriorDB.dataset(
        PosteriorDB.posterior(pdb, "normal_5-normal_mixture_k"),
    ))
end

function mixture_k_logp(theta, mu, sigma, y)
    total = zero(theta[1] + mu[1] + sigma[1])
    for yi in y
        part = log(theta[1]) + logdensityof(WD.Normal(mu[1], sigma[1]), yi)
        for k in 2:length(theta)
            next = log(theta[k]) + logdensityof(WD.Normal(mu[k], sigma[k]), yi)
            part = logaddexp(part, next)
        end
        total += part
    end
    return total
end

@wren function normal_mixture_k(y, K)
    # Stan's implicit flat simplex factor differs only by a constant.
    theta ~ WD.Dirichlet(ones(K))
    @iid mu ~ WD.Normal(0.0, 10.0) size=K
    @iid sigma ~ WD.Uniform(0.0, 10.0) size=K
    @addloglikelihood mixture_k_logp(theta, mu, sigma, y)
    return nothing
end

const wren_model = normal_mixture_k(data["y"], data["K"])

end
