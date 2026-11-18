# PosteriorDB: Rate_2_data-Rate_2_model
module Rate2

using PosteriorDB
using WrenCore: @wren, @observe
using WrenDists: WrenDists as WD

const data = let
    pdb = PosteriorDB.database()
    PosteriorDB.load(PosteriorDB.dataset(PosteriorDB.posterior(pdb, "Rate_2_data-Rate_2_model")))
end

@wren function rate_2_model(k1, n1, k2, n2)
    theta1 ~ WD.Beta(1.0, 1.0)
    theta2 ~ WD.Beta(1.0, 1.0)
    @observe k1 ~ WD.Binomial(n1, theta1)
    @observe k2 ~ WD.Binomial(n2, theta2)
    return nothing
end

const wren_model = rate_2_model(data["k1"], data["n1"], data["k2"], data["n2"])

end
