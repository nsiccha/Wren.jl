# PosteriorDB: Rate_4_data-Rate_4_model
module Rate4

using PosteriorDB
using WrenCore: @wren, @observe
using WrenDists: WrenDists as WD

const data = let
    pdb = PosteriorDB.database()
    PosteriorDB.load(PosteriorDB.dataset(PosteriorDB.posterior(pdb, "Rate_4_data-Rate_4_model")))
end

@wren function rate_4_model(k, n)
    theta ~ WD.Beta(1.0, 1.0)
    thetaprior ~ WD.Beta(1.0, 1.0)
    @observe k ~ WD.Binomial(n, theta)
    return nothing
end

const wren_model = rate_4_model(data["k"], data["n"])

end
