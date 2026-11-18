# PosteriorDB: timssAusTwn_irt-gpcm_latent_reg_irt
module TIMSSGPCM

using PosteriorDB
using WrenCore: @wren, @addloglikelihood, @iid
using WrenDists: WrenDists as WD
using ..IRTCommon: adjusted_covariates, beta_theta_logdensity, graded_logdensity

const data = let
    pdb = PosteriorDB.database()
    PosteriorDB.load(PosteriorDB.dataset(PosteriorDB.posterior(
        pdb, "timssAusTwn_irt-gpcm_latent_reg_irt",
    )))
end

function item_steps(y, ii, I)
    m = zeros(Int, I)
    for n in eachindex(y)
        m[ii[n]] = max(m[ii[n]], y[n])
    end
    pos = cumsum([1; m[1:(I - 1)]])
    return m, pos
end

function timss_logdensity(y, ii, jj, W_adj, m, pos,
        alpha, beta_free, theta, lambda_adj)
    total = beta_theta_logdensity(W_adj, beta_free, theta, lambda_adj)
    beta = [beta_free; -sum(beta_free)]
    for n in eachindex(y)
        i = ii[n]
        total += graded_logdensity(y[n], theta[jj[n]] * alpha[i],
            beta, 0.0, pos[i], m[i])
    end
    return total
end

@wren function latent_reg_gpcm(y, ii, jj, W_adj, m, pos, I, J, K, nsteps)
    @iid alpha ~ WD.LogNormal(1.0, 1.0) size=I
    @iid beta_free ~ WD.Flat() size=(nsteps - 1)
    @iid theta ~ WD.Flat() size=J
    @iid lambda_adj ~ WD.TDist(3.0, 0.0, 1.0) size=K
    @addloglikelihood timss_logdensity(y, ii, jj, W_adj, m, pos,
        alpha, beta_free, theta, lambda_adj)
    return nothing
end

const wren_model = let
    data["I"] == 11 && data["J"] == 500 && data["N"] == 5500 &&
        data["K"] == 5 || error("unexpected public TIMSS fixture")
    W_adj = adjusted_covariates(data["W"])
    m, pos = item_steps(data["y"], data["ii"], data["I"])
    latent_reg_gpcm(data["y"], data["ii"], data["jj"], W_adj,
        m, pos, data["I"], data["J"], data["K"], sum(m))
end

end
