# Four public Kidscore regressions with flat beta and positive flat sigma priors.
module KidscoreFlatCommon

using PosteriorDB
using ..FlatGaussianRegression: flat_gaussian

const data = let
    pdb = PosteriorDB.database()
    PosteriorDB.load(PosteriorDB.dataset(PosteriorDB.posterior(
        pdb, "kidiq_with_mom_work-kidscore_interaction_c",
    )))
end

end

module KidscoreInteractionC
using Statistics: mean
using ..KidscoreFlatCommon: data, flat_gaussian
const wren_model = let
    hs = data["mom_hs"] .- mean(data["mom_hs"])
    iq = data["mom_iq"] .- mean(data["mom_iq"])
    flat_gaussian(data["kid_score"], hcat(ones(length(hs)), hs, iq, hs .* iq))
end
end

module KidscoreInteractionC2
using ..KidscoreFlatCommon: data, flat_gaussian
const wren_model = let
    hs = data["mom_hs"] .- 0.5
    iq = data["mom_iq"] .- 100.0
    flat_gaussian(data["kid_score"], hcat(ones(length(hs)), hs, iq, hs .* iq))
end
end

module KidscoreInteractionZ
using Statistics: mean, std
using ..KidscoreFlatCommon: data, flat_gaussian
const wren_model = let
    hs = (data["mom_hs"] .- mean(data["mom_hs"])) ./ (2 * std(data["mom_hs"]))
    iq = (data["mom_iq"] .- mean(data["mom_iq"])) ./ (2 * std(data["mom_iq"]))
    flat_gaussian(data["kid_score"], hcat(ones(length(hs)), hs, iq, hs .* iq))
end
end

module KidscoreMomWork
using ..KidscoreFlatCommon: data, flat_gaussian
const wren_model = let
    work = data["mom_work"]
    flat_gaussian(data["kid_score"], hcat(
        ones(length(work)), work .== 2, work .== 3, work .== 4,
    ))
end
end
