# PosteriorDB: kidiq-kidscore_momiq
module KidscoreMomiq

using PosteriorDB
using ..KidscoreInteraction: kidscore_interaction

const data = let
    pdb = PosteriorDB.database()
    PosteriorDB.load(PosteriorDB.dataset(PosteriorDB.posterior(pdb, "kidiq-kidscore_momiq")))
end

const wren_model = kidscore_interaction(
    data["kid_score"], hcat(ones(length(data["mom_iq"])), data["mom_iq"]),
)

end
