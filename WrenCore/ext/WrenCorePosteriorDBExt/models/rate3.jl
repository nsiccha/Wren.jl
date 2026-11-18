# PosteriorDB: Rate_3_data-Rate_3_model
module Rate3

using PosteriorDB
using WrenCore: @wren, @observe
using WrenDists: WrenDists as WD

const data = let
    pdb = PosteriorDB.database()
    PosteriorDB.load(PosteriorDB.dataset(PosteriorDB.posterior(pdb, "Rate_3_data-Rate_3_model")))
end

@wren function rate_3_model(k1, n1, k2, n2)
    theta ~ WD.Beta(1.0, 1.0)
    @observe k1 ~ WD.Binomial(n1, theta)
    @observe k2 ~ WD.Binomial(n2, theta)
    return nothing
end

const wren_model = rate_3_model(data["k1"], data["n1"], data["k2"], data["n2"])

end
