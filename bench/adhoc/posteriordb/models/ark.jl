# https://github.com/stan-dev/posteriordb/blob/master/posterior_database/models/stan/arK.stan

module ArK

using PosteriorDB
using DynamicPPL, Distributions
using FillArrays: Fill

const data = let
    pdb = PosteriorDB.database()
    PosteriorDB.load(PosteriorDB.dataset(PosteriorDB.posterior(pdb, "arK-arK")))
end

@model function arK_arK(y, lags, K)
    alpha ~ Normal(0, 10)
    beta ~ product_distribution(Fill(Normal(0, 10), K))
    sigma ~ truncated(Cauchy(0, 2.5); lower=0)
    mu = alpha .+ lags * beta
    for i in eachindex(y)
        y[i] ~ Normal(mu[i], sigma)
    end
end

const turing_model = let
    K = data["K"]
    T = data["T"]
    series = data["y"]
    lags = [series[t-k] for t in (K+1):T, k in 1:K]
    arK_arK(series[(K+1):T], lags, K)
end

end
