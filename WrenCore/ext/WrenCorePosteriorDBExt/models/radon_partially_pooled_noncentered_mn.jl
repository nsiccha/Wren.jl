# PosteriorDB: radon_mn-radon_partially_pooled_noncentered
module RadonPartiallyPooledNoncenteredMn

using PosteriorDB
using ..RadonPartiallyPooledNoncenteredAll: radon_partially_pooled_noncentered

const data = let
    pdb = PosteriorDB.database()
    PosteriorDB.load(PosteriorDB.dataset(
        PosteriorDB.posterior(pdb, "radon_mn-radon_partially_pooled_noncentered"),
    ))
end

const wren_model = radon_partially_pooled_noncentered(
    data["J"], data["county_idx"], data["log_radon"],
)

end
