# PosteriorDB: dugongs_data-dugongs_model
module Dugongs

using PosteriorDB
using WrenCore: @wren, @observe, @batch
using WrenDists: WrenDists as WD

const data = let
    pdb = PosteriorDB.database()
    PosteriorDB.load(PosteriorDB.dataset(
        PosteriorDB.posterior(pdb, "dugongs_data-dugongs_model"),
    ))
end

@wren function dugongs_model(x, y)
    alpha ~ WD.Normal(0.0, 1000.0)
    beta ~ WD.Normal(0.0, 1000.0)
    lambda ~ WD.Uniform(0.5, 1.0)
    # Stan Gamma uses a rate; WrenDists Gamma uses a scale.
    tau ~ WD.Gamma(0.0001, 10000.0)
    sigma = inv(sqrt(tau))
    mu = alpha .- beta .* (lambda .^ x)
    @batch for i in eachindex(y)
        @observe y[i] ~ WD.Normal(mu[i], sigma)
    end
    return nothing
end

const wren_model = dugongs_model(data["x"], data["Y"])

end
