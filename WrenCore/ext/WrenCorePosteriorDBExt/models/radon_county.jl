# PosteriorDB: radon_mod-radon_county
module RadonCounty

using PosteriorDB
using WrenCore: @wren, @observe, @batch, @iid
using WrenDists: WrenDists as WD

const data = let
    pdb = PosteriorDB.database()
    PosteriorDB.load(PosteriorDB.dataset(PosteriorDB.posterior(
        pdb, "radon_mod-radon_county",
    )))
end

@wren function radon_county(county, y, J)
    mu_a ~ WD.Normal(0.0, 1.0)
    sigma_a ~ WD.Uniform(0.0, 100.0)
    sigma_y ~ WD.Uniform(0.0, 100.0)
    @iid a ~ WD.Normal(mu_a, sigma_a) size=J
    y_hat = a[county]
    @batch for i in eachindex(y)
        @observe y[i] ~ WD.Normal(y_hat[i], sigma_y)
    end
    return nothing
end

const wren_model = radon_county(data["county"], data["y"], data["J"])

end
