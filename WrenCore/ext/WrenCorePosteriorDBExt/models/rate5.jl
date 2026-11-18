# PosteriorDB: Rate_5_data-Rate_5_model
module Rate5

using PosteriorDB
using ..Rate3: rate_3_model

const data = let
    pdb = PosteriorDB.database()
    PosteriorDB.load(PosteriorDB.dataset(PosteriorDB.posterior(pdb, "Rate_5_data-Rate_5_model")))
end

# The generated predictive counts in the Stan source are outside the target density.
const wren_model = rate_3_model(data["k1"], data["n1"], data["k2"], data["n2"])

end
