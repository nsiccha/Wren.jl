# PosteriorDB: radon_all-radon_variable_intercept_slope_centered
module RadonVariableInterceptSlopeCenteredAll

using PosteriorDB
using WrenCore: @wren, @observe, @batch, @iid
using WrenDists: WrenDists as WD

const data = let
    pdb = PosteriorDB.database()
    PosteriorDB.load(PosteriorDB.dataset(
        PosteriorDB.posterior(pdb, "radon_all-radon_variable_intercept_slope_centered"),
    ))
end

@wren function radon_variable_intercept_slope_centered(
    J, county_idx, floor_measure, log_radon,
)
    sigma_y ~ WD.HalfNormal(0.0, 1.0)
    sigma_alpha ~ WD.HalfNormal(0.0, 1.0)
    sigma_beta ~ WD.HalfNormal(0.0, 1.0)
    @iid alpha ~ WD.Flat() size=J
    @iid beta ~ WD.Flat() size=J
    mu_alpha ~ WD.Normal(0.0, 10.0)
    mu_beta ~ WD.Normal(0.0, 10.0)
    @observe @iid alpha ~ WD.Normal(mu_alpha, sigma_alpha) size=J
    @observe @iid beta ~ WD.Normal(mu_beta, sigma_beta) size=J
    mu = alpha[county_idx] .+ beta[county_idx] .* floor_measure
    @batch for i in eachindex(log_radon)
        @observe log_radon[i] ~ WD.Normal(mu[i], sigma_y)
    end
    return nothing
end

const wren_model = radon_variable_intercept_slope_centered(
    data["J"], data["county_idx"], data["floor_measure"], data["log_radon"],
)

end
