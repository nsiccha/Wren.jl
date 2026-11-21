# PosteriorDB: kidiq-kidscore_interaction
module KidscoreInteraction

using PosteriorDB
using DynamicPPL, Distributions
using FillArrays: Fill

const data = let
    pdb = PosteriorDB.database()
    PosteriorDB.load(
        PosteriorDB.dataset(PosteriorDB.posterior(pdb, "kidiq-kidscore_interaction")),
    )
end

@model function kidiq_kidscore_interaction(kid_score, X)
    beta ~ product_distribution(Fill(Flat(), size(X, 2)))
    sigma ~ truncated(Cauchy(0, 2.5); lower=0)
    mu = X * beta
    for i in eachindex(kid_score)
        kid_score[i] ~ Normal(mu[i], sigma)
    end
end

const turing_model = let
    mom_hs = data["mom_hs"]
    mom_iq = data["mom_iq"]
    X = hcat(ones(length(mom_hs)), mom_hs, mom_iq, mom_hs .* mom_iq)
    kidiq_kidscore_interaction(data["kid_score"], X)
end

end
