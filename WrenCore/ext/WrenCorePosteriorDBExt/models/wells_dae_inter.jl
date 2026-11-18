# PosteriorDB: wells_data-wells_dae_inter_model
module WellsDAEInter

using PosteriorDB
using Statistics: mean
using ..WellsDist: wells_glm_model

const data = let
    pdb = PosteriorDB.database()
    PosteriorDB.load(PosteriorDB.dataset(
        PosteriorDB.posterior(pdb, "wells_data-wells_dae_inter_model"),
    ))
end

const wren_model = let
    dist, arsenic, educ = data["dist"], data["arsenic"], data["educ"]
    c_dist100 = (dist .- mean(dist)) ./ 100.0
    c_arsenic = arsenic .- mean(arsenic)
    c_educ4 = (educ .- mean(educ)) ./ 4.0
    X = hcat(c_dist100, c_arsenic, c_educ4,
        c_dist100 .* c_arsenic, c_dist100 .* c_educ4, c_arsenic .* c_educ4)
    wells_glm_model(data["switched"], X)
end

end
