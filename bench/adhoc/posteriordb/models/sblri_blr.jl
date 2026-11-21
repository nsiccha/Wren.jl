# PosteriorDB: sblri-blr
module SblriBLR

using PosteriorDB
using DynamicPPL, Distributions
using FillArrays: Fill

const data = let
    pdb = PosteriorDB.database()
    PosteriorDB.load(PosteriorDB.dataset(PosteriorDB.posterior(pdb, "sblri-blr")))
end

@model function dppl_sblri_blr(X, y, D)
    beta ~ product_distribution(Fill(Normal(0, 10), D))
    sigma ~ truncated(Normal(0, 10); lower=0)
    mu = X * beta
    for i in eachindex(y)
        y[i] ~ Normal(mu[i], sigma)
    end
end

const turing_model = dppl_sblri_blr(data["X"], data["y"], data["D"])

end
