# PosteriorDB: dogs-dogs_log
module DogsLog

using PosteriorDB
using WrenCore: @wren, @observe, @batch, @iid
using WrenDists: WrenDists as WD
using ..Dogs: dog_history

const data = let
    pdb = PosteriorDB.database()
    PosteriorDB.load(PosteriorDB.dataset(PosteriorDB.posterior(pdb, "dogs-dogs_log")))
end

@wren function dogs_log(y, shocks, avoids)
    # Stan declares unconstrained beta and imposes bounded uniform factors in the model block.
    @iid beta ~ WD.Flat() size=2
    @observe beta[1] ~ WD.Uniform(-100.0, 0.0)
    @observe beta[2] ~ WD.Uniform(0.0, 100.0)
    eta = beta[1] .* avoids .+ beta[2] .* shocks
    @batch for i in eachindex(y)
        @observe y[i] ~ WD.BernoulliLogit(eta[i])
    end
    return nothing
end

const wren_model = dogs_log(dog_history(data["y"])...)

end
