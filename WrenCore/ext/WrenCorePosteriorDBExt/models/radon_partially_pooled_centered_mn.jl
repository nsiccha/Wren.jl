# PosteriorDB: radon_mn-radon_partially_pooled_centered
module RadonPartiallyPooledCenteredMn

using PosteriorDB
using ..RadonPartiallyPooledCenteredAll: radon_partially_pooled_centered

const data = let
    pdb = PosteriorDB.database()
    PosteriorDB.load(PosteriorDB.dataset(
        PosteriorDB.posterior(pdb, "radon_mn-radon_partially_pooled_centered"),
    ))
end

const wren_model = radon_partially_pooled_centered(
    data["J"], data["county_idx"], data["log_radon"],
)

end
