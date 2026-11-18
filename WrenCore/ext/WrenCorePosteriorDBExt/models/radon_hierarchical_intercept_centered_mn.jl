# PosteriorDB: radon_mn-radon_hierarchical_intercept_centered
module RadonHierarchicalInterceptCenteredMn

using PosteriorDB
using ..RadonHierarchicalInterceptCenteredAll: radon_hierarchical_intercept_centered

const data = let
    pdb = PosteriorDB.database()
    PosteriorDB.load(PosteriorDB.dataset(
        PosteriorDB.posterior(pdb, "radon_mn-radon_hierarchical_intercept_centered"),
    ))
end

const wren_model = radon_hierarchical_intercept_centered(
    data["J"], data["county_idx"], data["log_uppm"], data["floor_measure"], data["log_radon"],
)

end
