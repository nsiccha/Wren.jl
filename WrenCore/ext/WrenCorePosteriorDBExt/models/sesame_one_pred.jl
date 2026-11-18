# PosteriorDB: sesame_data-sesame_one_pred_a
module SesameOnePred

using PosteriorDB
using WrenCore: @wren, @observe, @batch, @iid, @addloglikelihood
using WrenDists: WrenDists as WD

const data = let
    pdb = PosteriorDB.database()
    PosteriorDB.load(PosteriorDB.dataset(PosteriorDB.posterior(
        pdb, "sesame_data-sesame_one_pred_a",
    )))
end

@wren function sesame_one_pred(encouraged, watched)
    @iid beta ~ WD.Flat() size=2
    log_sigma ~ WD.Flat()
    sigma = exp(log_sigma)
    # Temporary positive-flat scale transform pending #192.
    @addloglikelihood log_sigma
    mu = beta[1] .+ beta[2] .* encouraged
    @batch for i in eachindex(watched)
        @observe watched[i] ~ WD.Normal(mu[i], sigma)
    end
    return nothing
end

const wren_model = sesame_one_pred(data["encouraged"], data["watched"])

end
