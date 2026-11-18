# PosteriorDB: radon_all-radon_variable_slope_noncentered
module RadonVariableSlopeNoncenteredAll

using PosteriorDB
using WrenCore: @wren, @observe, @batch, @iid
using WrenDists: WrenDists as WD

const data = let
    pdb = PosteriorDB.database()
    PosteriorDB.load(PosteriorDB.dataset(
        PosteriorDB.posterior(pdb, "radon_all-radon_variable_slope_noncentered"),
    ))
end

@wren function radon_variable_slope_noncentered(
    J, county_idx, floor_measure, log_radon,
)
    alpha ~ WD.Normal(0.0, 10.0)
    @iid beta_raw ~ WD.Normal(0.0, 1.0) size=J
    mu_beta ~ WD.Normal(0.0, 10.0)
    sigma_beta ~ WD.HalfNormal(0.0, 1.0)
    sigma_y ~ WD.HalfNormal(0.0, 1.0)
    beta = mu_beta .+ sigma_beta .* beta_raw
    mu = alpha .+ floor_measure .* beta[county_idx]
    @batch for i in eachindex(log_radon)
        @observe log_radon[i] ~ WD.Normal(mu[i], sigma_y)
    end
    return nothing
end

const wren_model = radon_variable_slope_noncentered(
    data["J"], data["county_idx"], data["floor_measure"], data["log_radon"],
)

end
