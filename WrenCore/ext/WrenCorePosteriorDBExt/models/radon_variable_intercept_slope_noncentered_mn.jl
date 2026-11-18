# PosteriorDB: radon_mn-radon_variable_intercept_slope_noncentered
module RadonVariableInterceptSlopeNoncenteredMn

using PosteriorDB
using ..RadonVariableInterceptSlopeNoncenteredAll: radon_variable_intercept_slope_noncentered

const data = let
    pdb = PosteriorDB.database()
    PosteriorDB.load(PosteriorDB.dataset(
        PosteriorDB.posterior(pdb, "radon_mn-radon_variable_intercept_slope_noncentered"),
    ))
end

const wren_model = radon_variable_intercept_slope_noncentered(
    data["J"], data["county_idx"], data["floor_measure"], data["log_radon"],
)

end
