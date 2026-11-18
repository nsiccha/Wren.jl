# PosteriorDB: radon_all-radon_hierarchical_intercept_noncentered
module RadonHierarchicalInterceptNoncenteredAll

using PosteriorDB
using WrenCore: @wren, @observe, @batch, @iid
using WrenDists: WrenDists as WD

const data = let
    pdb = PosteriorDB.database()
    PosteriorDB.load(PosteriorDB.dataset(
        PosteriorDB.posterior(pdb, "radon_all-radon_hierarchical_intercept_noncentered"),
    ))
end

@wren function radon_hierarchical_intercept_noncentered(
    J, county_idx, log_uppm, floor_measure, log_radon,
)
    @iid alpha_raw ~ WD.Normal(0.0, 1.0) size=J
    @iid beta ~ WD.Normal(0.0, 10.0) size=2
    mu_alpha ~ WD.Normal(0.0, 10.0)
    sigma_alpha ~ WD.HalfNormal(0.0, 1.0)
    sigma_y ~ WD.HalfNormal(0.0, 1.0)
    alpha = mu_alpha .+ sigma_alpha .* alpha_raw
    mu = alpha[county_idx] .+ beta[1] .* log_uppm .+ beta[2] .* floor_measure
    @batch for i in eachindex(log_radon)
        @observe log_radon[i] ~ WD.Normal(mu[i], sigma_y)
    end
    return nothing
end

const wren_model = radon_hierarchical_intercept_noncentered(
    data["J"], data["county_idx"], data["log_uppm"], data["floor_measure"],
    data["log_radon"],
)

end
