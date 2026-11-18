# PosteriorDB: low_dim_gauss_mix-low_dim_gauss_mix
module LowDimGaussMix

using PosteriorDB
using ..LowDimGaussMixCollapse: low_dim_mixture_logp
using WrenCore: @wren, @addloglikelihood, @iid, logdensityof
using WrenDists: WrenDists as WD

const data = let
    pdb = PosteriorDB.database()
    PosteriorDB.load(PosteriorDB.dataset(PosteriorDB.posterior(
        pdb, "low_dim_gauss_mix-low_dim_gauss_mix",
    )))
end

@wren function low_dim_gauss_mix(y)
    mu1 ~ WD.Normal(0.0, 2.0)
    log_mu_gap ~ WD.Flat()
    mu2 = mu1 + exp(log_mu_gap)
    # Stan's ordered[2] mapping is (mu1, log(mu2-mu1)); include its Jacobian.
    # Native ordered prior support is tracked in #207.
    @addloglikelihood (logdensityof(WD.Normal(0.0, 2.0), mu2) + log_mu_gap)
    @iid sigma ~ WD.HalfNormal(0.0, 2.0) size=2
    theta ~ WD.Beta(5.0, 5.0)
    @addloglikelihood low_dim_mixture_logp([mu1, mu2], sigma, theta, y)
    return nothing
end

const wren_model = low_dim_gauss_mix(data["y"])

end
