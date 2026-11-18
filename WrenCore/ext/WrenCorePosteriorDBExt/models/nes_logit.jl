# PosteriorDB: nes_logit_data-nes_logit_model
module NESLogit

using PosteriorDB
using WrenCore: @wren, @observe, @batch, @iid
using WrenDists: WrenDists as WD

const data = let
    pdb = PosteriorDB.database()
    PosteriorDB.load(PosteriorDB.dataset(
        PosteriorDB.posterior(pdb, "nes_logit_data-nes_logit_model"),
    ))
end

@wren function nes_logit(income, vote)
    alpha ~ WD.Flat()
    @iid beta ~ WD.Flat() size=1
    eta = alpha .+ beta[1] .* income
    @batch for i in eachindex(vote)
        @observe vote[i] ~ WD.BernoulliLogit(eta[i])
    end
    return nothing
end

const wren_model = nes_logit(data["income"], data["vote"])

end
