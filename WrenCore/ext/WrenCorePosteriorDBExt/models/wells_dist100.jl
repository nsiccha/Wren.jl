# PosteriorDB: wells_data-wells_dist100_model
module WellsDist100

using PosteriorDB
using ..WellsDist: wells_glm_model

const data = let
    pdb = PosteriorDB.database()
    PosteriorDB.load(
        PosteriorDB.dataset(PosteriorDB.posterior(pdb, "wells_data-wells_dist100_model")),
    )
end

const wren_model = wells_glm_model(data["switched"], hcat(data["dist"] ./ 100.0))

end
