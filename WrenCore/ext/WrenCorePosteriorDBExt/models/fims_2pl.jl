# PosteriorDB: fims_Aus_Jpn_irt-2pl_latent_reg_irt
module FIMS2PL

using PosteriorDB
using WrenCore: @wren, @addloglikelihood, @iid, logdensityof
using WrenDists: WrenDists as WD
using ..IRTCommon: adjusted_covariates, beta_theta_logdensity

const data = let
    pdb = PosteriorDB.database()
    PosteriorDB.load(PosteriorDB.dataset(PosteriorDB.posterior(
        pdb, "fims_Aus_Jpn_irt-2pl_latent_reg_irt",
    )))
end

function fims_logdensity(y, ii, jj, W_adj, alpha, beta_free, theta, lambda_adj)
    I = length(alpha)
    total = beta_theta_logdensity(W_adj, beta_free, theta, lambda_adj)
    beta_last = -sum(beta_free)
    for n in eachindex(y)
        i = ii[n]
        beta_i = i == I ? beta_last : beta_free[i]
        total += logdensityof(WD.BernoulliLogit(alpha[i] * theta[jj[n]] - beta_i), y[n])
    end
    return total
end

@wren function latent_reg_2pl(y, ii, jj, W_adj, I, J, K)
    @iid alpha ~ WD.LogNormal(1.0, 1.0) size=I
    @iid beta_free ~ WD.Flat() size=(I - 1)
    @iid theta ~ WD.Flat() size=J
    @iid lambda_adj ~ WD.TDist(3.0, 0.0, 1.0) size=K
    @addloglikelihood fims_logdensity(y, ii, jj, W_adj,
        alpha, beta_free, theta, lambda_adj)
    return nothing
end

const wren_model = let
    data["I"] == 14 && data["J"] == 500 && data["N"] == 7000 &&
        data["K"] == 4 || error("unexpected public FIMS fixture")
    W_adj = adjusted_covariates(data["W"])
    latent_reg_2pl(data["y"], data["ii"], data["jj"], W_adj,
        data["I"], data["J"], data["K"])
end

end
