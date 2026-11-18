# PosteriorDB: dogs-dogs
module Dogs

using PosteriorDB
using WrenCore: @wren, @observe, @batch, @iid
using WrenDists: WrenDists as WD

const data = let
    pdb = PosteriorDB.database()
    PosteriorDB.load(PosteriorDB.dataset(PosteriorDB.posterior(pdb, "dogs-dogs")))
end

function dog_history(y)
    dogs, trials = size(y)
    shocks = [sum((y[j, s] for s in 1:(t - 1)); init=0)
        for t in 1:trials for j in 1:dogs]
    avoids = [t - 1 - sum((y[j, s] for s in 1:(t - 1)); init=0)
        for t in 1:trials for j in 1:dogs]
    return vec(y), shocks, avoids
end

@wren function dogs_model(y, shocks, avoids)
    @iid beta ~ WD.Normal(0.0, 100.0) size=3
    eta = beta[1] .+ beta[2] .* avoids .+ beta[3] .* shocks
    @batch for i in eachindex(y)
        @observe y[i] ~ WD.BernoulliLogit(eta[i])
    end
    return nothing
end

const wren_model = dogs_model(dog_history(data["y"])...)

end
