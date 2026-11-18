# PosteriorDB: wells_data-wells_dae_model
module WellsDAE

using PosteriorDB
using ..WellsDist: wells_glm_model

const data = let
    pdb = PosteriorDB.database()
    PosteriorDB.load(PosteriorDB.dataset(
        PosteriorDB.posterior(pdb, "wells_data-wells_dae_model"),
    ))
end

const wren_model = wells_glm_model(data["switched"],
    hcat(data["dist"] ./ 100.0, data["arsenic"], data["educ"] ./ 4.0))

end
