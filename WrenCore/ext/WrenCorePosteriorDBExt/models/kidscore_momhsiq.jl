# PosteriorDB: kidiq-kidscore_momhsiq
module KidscoreMomhsiq

using PosteriorDB
using ..KidscoreInteraction: kidscore_interaction

const data = let
    pdb = PosteriorDB.database()
    PosteriorDB.load(PosteriorDB.dataset(PosteriorDB.posterior(pdb, "kidiq-kidscore_momhsiq")))
end

const wren_model = kidscore_interaction(
    data["kid_score"],
    hcat(ones(length(data["mom_hs"])), data["mom_hs"], data["mom_iq"]),
)

end
