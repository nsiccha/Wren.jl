# PosteriorDB: radon_all-radon_county_intercept
module RadonCountyInterceptAll

using PosteriorDB
using WrenCore: @wren, @observe, @batch, @iid
using WrenDists: WrenDists as WD

const data = let
    pdb = PosteriorDB.database()
    PosteriorDB.load(PosteriorDB.dataset(
        PosteriorDB.posterior(pdb, "radon_all-radon_county_intercept"),
    ))
end

@wren function radon_county_intercept(J, county_idx, floor_measure, log_radon)
    @iid alpha ~ WD.Normal(0.0, 10.0) size=J
    beta ~ WD.Normal(0.0, 10.0)
    sigma_y ~ WD.HalfNormal(0.0, 1.0)
    mu = alpha[county_idx] .+ beta .* floor_measure
    @batch for i in eachindex(log_radon)
        @observe log_radon[i] ~ WD.Normal(mu[i], sigma_y)
    end
    return nothing
end

const wren_model = radon_county_intercept(
    data["J"], data["county_idx"], data["floor_measure"], data["log_radon"],
)

end
