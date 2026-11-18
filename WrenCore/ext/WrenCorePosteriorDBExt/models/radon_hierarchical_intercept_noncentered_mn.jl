# PosteriorDB: radon_mn-radon_hierarchical_intercept_noncentered
module RadonHierarchicalInterceptNoncenteredMn

using PosteriorDB
using ..RadonHierarchicalInterceptNoncenteredAll: radon_hierarchical_intercept_noncentered

const data = let
    pdb = PosteriorDB.database()
    PosteriorDB.load(PosteriorDB.dataset(
        PosteriorDB.posterior(pdb, "radon_mn-radon_hierarchical_intercept_noncentered"),
    ))
end

const wren_model = radon_hierarchical_intercept_noncentered(
    data["J"], data["county_idx"], data["log_uppm"], data["floor_measure"],
    data["log_radon"],
)

end
