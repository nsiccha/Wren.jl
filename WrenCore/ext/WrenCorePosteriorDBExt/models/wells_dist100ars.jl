# PosteriorDB: wells_data-wells_dist100ars_model
module WellsDist100Ars

using PosteriorDB
using ..WellsDist: wells_glm_model

const data = let
    pdb = PosteriorDB.database()
    PosteriorDB.load(PosteriorDB.dataset(
        PosteriorDB.posterior(pdb, "wells_data-wells_dist100ars_model"),
    ))
end

const wren_model = wells_glm_model(data["switched"],
    hcat(data["dist"] ./ 100.0, data["arsenic"]))

end
