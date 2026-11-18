# PosteriorDB: kidiq-kidscore_momhs
module KidscoreMomhs

using PosteriorDB
using ..KidscoreInteraction: kidscore_interaction

const data = let
    pdb = PosteriorDB.database()
    PosteriorDB.load(PosteriorDB.dataset(PosteriorDB.posterior(pdb, "kidiq-kidscore_momhs")))
end

const wren_model = kidscore_interaction(
    data["kid_score"], hcat(ones(length(data["mom_hs"])), data["mom_hs"]),
)

end
