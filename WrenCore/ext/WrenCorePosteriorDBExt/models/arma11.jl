# PosteriorDB: arma-arma11
module ARMA11

using PosteriorDB
using WrenCore: @wren, @observe, @batch
using WrenDists: WrenDists as WD

const data = let
    pdb = PosteriorDB.database()
    PosteriorDB.load(PosteriorDB.dataset(PosteriorDB.posterior(pdb, "arma-arma11")))
end

@wren function arma11(y)
    mu ~ WD.Normal(0.0, 10.0)
    phi ~ WD.Normal(0.0, 2.0)
    theta ~ WD.Normal(0.0, 2.0)
    # Stan's positive Cauchy factor differs by the constant log(2).
    sigma ~ WD.HalfCauchy(0.0, 2.5)
    prediction = mu + phi * mu
    predictions = Vector{typeof(prediction)}(undef, length(y))
    predictions[1] = prediction
    previous_error = y[1] - prediction
    for t in 2:length(y)
        prediction = mu + phi * y[t - 1] + theta * previous_error
        predictions[t] = prediction
        previous_error = y[t] - prediction
    end
    @batch for t in eachindex(y)
        @observe y[t] ~ WD.Normal(predictions[t], sigma)
    end
    return nothing
end

const wren_model = arma11(data["y"])

end
