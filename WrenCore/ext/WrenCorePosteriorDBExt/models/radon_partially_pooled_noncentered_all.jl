# PosteriorDB: radon_all-radon_partially_pooled_noncentered
module RadonPartiallyPooledNoncenteredAll

using PosteriorDB
using WrenCore: @wren, @observe, @batch, @iid
using WrenDists: WrenDists as WD

const data = let
    pdb = PosteriorDB.database()
    PosteriorDB.load(PosteriorDB.dataset(
        PosteriorDB.posterior(pdb, "radon_all-radon_partially_pooled_noncentered"),
    ))
end

@wren function radon_partially_pooled_noncentered(J, county_idx, log_radon)
    @iid alpha_raw ~ WD.Normal(0.0, 1.0) size=J
    mu_alpha ~ WD.Normal(0.0, 10.0)
    sigma_alpha ~ WD.HalfNormal(0.0, 1.0)
    sigma_y ~ WD.HalfNormal(0.0, 1.0)
    alpha = mu_alpha .+ sigma_alpha .* alpha_raw
    mu = alpha[county_idx]
    @batch for i in eachindex(log_radon)
        @observe log_radon[i] ~ WD.Normal(mu[i], sigma_y)
    end
    return nothing
end

const wren_model = radon_partially_pooled_noncentered(
    data["J"], data["county_idx"], data["log_radon"],
)

end
