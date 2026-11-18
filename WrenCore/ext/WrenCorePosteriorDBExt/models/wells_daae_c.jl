# PosteriorDB: wells_data-wells_daae_c_model
module WellsDAAEC

using PosteriorDB
using Statistics: mean
using ..WellsDist: wells_glm_model

const data = let
    pdb = PosteriorDB.database()
    PosteriorDB.load(PosteriorDB.dataset(
        PosteriorDB.posterior(pdb, "wells_data-wells_daae_c_model"),
    ))
end

const wren_model = let
    dist = data["dist"]
    arsenic = data["arsenic"]
    c_dist100 = (dist .- mean(dist)) ./ 100.0
    c_arsenic = arsenic .- mean(arsenic)
    X = hcat(c_dist100, c_arsenic, c_dist100 .* c_arsenic,
        data["assoc"], data["educ"] ./ 4.0)
    wells_glm_model(data["switched"], X)
end

end
