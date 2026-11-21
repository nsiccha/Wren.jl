# PosteriorDB: Rate_1_data-Rate_1_model
module Rate1

using PosteriorDB
using DynamicPPL, Distributions

const data = let
    pdb = PosteriorDB.database()
    PosteriorDB.load(
        PosteriorDB.dataset(PosteriorDB.posterior(pdb, "Rate_1_data-Rate_1_model")),
    )
end

@model function pdb_Rate_1_model(n, k)
    theta ~ Beta(1, 1)
    return k ~ Binomial(n, theta)
end

const turing_model = pdb_Rate_1_model(data["n"], data["k"])

end
