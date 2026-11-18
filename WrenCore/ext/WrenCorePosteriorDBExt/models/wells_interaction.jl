# PosteriorDB: wells_data-wells_interaction_model
module WellsInteraction

using PosteriorDB
using ..WellsDist: wells_glm_model

const data = let
    pdb = PosteriorDB.database()
    PosteriorDB.load(
        PosteriorDB.dataset(PosteriorDB.posterior(pdb, "wells_data-wells_interaction_model")),
    )
end

const wren_model = let
    dist100 = data["dist"] ./ 100.0
    arsenic = data["arsenic"]
    X = hcat(dist100, arsenic, dist100 .* arsenic)
    wells_glm_model(data["switched"], X)
end

end
