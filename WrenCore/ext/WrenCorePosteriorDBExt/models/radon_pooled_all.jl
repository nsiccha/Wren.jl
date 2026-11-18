# PosteriorDB: radon_all-radon_pooled
module RadonPooledAll

using PosteriorDB
using WrenCore: @wren, @observe, @batch
using WrenDists: WrenDists as WD

const data = let
    pdb = PosteriorDB.database()
    PosteriorDB.load(PosteriorDB.dataset(PosteriorDB.posterior(pdb, "radon_all-radon_pooled")))
end

@wren function radon_pooled_model(log_radon, floor_measure)
    alpha ~ WD.Normal(0.0, 10.0)
    beta ~ WD.Normal(0.0, 10.0)
    sigma_y ~ WD.HalfNormal(0.0, 1.0)
    mu = alpha .+ beta .* floor_measure
    @batch for i in eachindex(log_radon)
        @observe log_radon[i] ~ WD.Normal(mu[i], sigma_y)
    end
    return nothing
end

const wren_model = radon_pooled_model(data["log_radon"], data["floor_measure"])

end
