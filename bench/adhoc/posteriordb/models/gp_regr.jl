# PosteriorDB: gp_pois_regr-gp_regr
module GPRegr

using PosteriorDB
using DynamicPPL, Distributions
using LinearAlgebra: Symmetric, I

const data = let
    pdb = PosteriorDB.database()
    PosteriorDB.load(
        PosteriorDB.dataset(PosteriorDB.posterior(pdb, "gp_pois_regr-gp_regr")),
    )
end

@model function dppl_gp_regr(dist_sq, y, N)
    rho ~ Gamma(25, 1 / 4)
    alpha ~ truncated(Normal(0, 2); lower=0)
    sigma ~ truncated(Normal(0, 1); lower=0)
    covariance = alpha^2 .* exp.(-0.5 .* dist_sq ./ rho^2) + sigma * I
    return y ~ MvNormal(zeros(N), Symmetric(covariance))
end

const turing_model = let
    x = data["x"]
    dppl_gp_regr((x .- x') .^ 2, data["y"], data["N"])
end

end
