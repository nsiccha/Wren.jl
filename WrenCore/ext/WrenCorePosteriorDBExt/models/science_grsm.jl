# PosteriorDB: science_irt-grsm_latent_reg_irt
module ScienceGRSM

using PosteriorDB
using WrenCore: @wren, @addloglikelihood, @iid, logdensityof
using WrenDists: WrenDists as WD
using ..IRTCommon: adjusted_covariates, beta_theta_logdensity, graded_logdensity

const data = let
    pdb = PosteriorDB.database()
    PosteriorDB.load(PosteriorDB.dataset(PosteriorDB.posterior(
        pdb, "science_irt-grsm_latent_reg_irt",
    )))
end

function science_logdensity(y, ii, jj, W_adj, alpha, beta_free,
        kappa_free, theta, lambda_adj)
    total = beta_theta_logdensity(W_adj, beta_free, theta, lambda_adj)
    kappa_last = -sum(kappa_free)
    for k in kappa_free
        total += logdensityof(WD.Normal(0.0, 3.0), k)
    end
    total += logdensityof(WD.Normal(0.0, 3.0), kappa_last)
    kappa = [kappa_free; kappa_last]
    I, m = length(alpha), length(kappa)
    beta_last = -sum(beta_free)
    for n in eachindex(y)
        i = ii[n]
        beta_i = i == I ? beta_last : beta_free[i]
        total += graded_logdensity(y[n], theta[jj[n]] * alpha[i],
            kappa, beta_i, 1, m)
    end
    return total
end

@wren function latent_reg_grsm(y, ii, jj, W_adj, I, J, K, m)
    @iid alpha ~ WD.LogNormal(1.0, 1.0) size=I
    @iid beta_free ~ WD.Flat() size=(I - 1)
    @iid kappa_free ~ WD.Flat() size=(m - 1)
    @iid theta ~ WD.Flat() size=J
    @iid lambda_adj ~ WD.TDist(3.0, 0.0, 1.0) size=K
    @addloglikelihood science_logdensity(y, ii, jj, W_adj,
        alpha, beta_free, kappa_free, theta, lambda_adj)
    return nothing
end

const wren_model = let
    data["I"] == 7 && data["J"] == 392 && data["N"] == 2744 &&
        data["K"] == 1 || error("unexpected public science IRT fixture")
    W_adj = adjusted_covariates(data["W"])
    m = maximum(data["y"])
    latent_reg_grsm(data["y"], data["ii"], data["jj"], W_adj,
        data["I"], data["J"], data["K"], m)
end

end
