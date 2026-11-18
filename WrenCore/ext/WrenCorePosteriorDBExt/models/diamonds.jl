# PosteriorDB: diamonds-diamonds
module Diamonds

using PosteriorDB
using Statistics: mean
using WrenCore: @wren, @observe, @batch, @iid
using WrenDists: WrenDists as WD

const data = let
    pdb = PosteriorDB.database()
    PosteriorDB.load(PosteriorDB.dataset(PosteriorDB.posterior(pdb, "diamonds-diamonds")))
end

@wren function diamonds(y, Xc, prior_only)
    @iid b ~ WD.Normal(0.0, 1.0) size=size(Xc, 2)
    Intercept ~ WD.TDist(3.0, 8.0, 10.0)
    sigma ~ WD.HalfTDist(3.0, 0.0, 10.0)
    if prior_only == 0
        mu = Intercept .+ Xc * b
        @batch for i in eachindex(y)
            @observe y[i] ~ WD.Normal(mu[i], sigma)
        end
    end
    return nothing
end

const wren_model = let
    X = data["X"]
    Xc = X[:, 2:end] .- mean(X[:, 2:end]; dims=1)
    diamonds(data["Y"], Xc, data["prior_only"])
end

end
