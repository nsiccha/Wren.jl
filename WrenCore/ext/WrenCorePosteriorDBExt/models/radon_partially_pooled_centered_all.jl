# PosteriorDB: radon_all-radon_partially_pooled_centered
module RadonPartiallyPooledCenteredAll

using PosteriorDB
using WrenCore: @wren, @observe, @batch, @iid
using WrenDists: WrenDists as WD

const data = let
    pdb = PosteriorDB.database()
    PosteriorDB.load(PosteriorDB.dataset(
        PosteriorDB.posterior(pdb, "radon_all-radon_partially_pooled_centered"),
    ))
end

@wren function radon_partially_pooled_centered(J, county_idx, log_radon)
    @iid alpha ~ WD.Flat() size=J
    mu_alpha ~ WD.Normal(0.0, 10.0)
    sigma_alpha ~ WD.HalfNormal(0.0, 1.0)
    sigma_y ~ WD.HalfNormal(0.0, 1.0)
    @observe @iid alpha ~ WD.Normal(mu_alpha, sigma_alpha) size=J
    mu = alpha[county_idx]
    @batch for i in eachindex(log_radon)
        @observe log_radon[i] ~ WD.Normal(mu[i], sigma_y)
    end
    return nothing
end

const wren_model = radon_partially_pooled_centered(
    data["J"], data["county_idx"], data["log_radon"],
)

end
