# PosteriorDB: radon_mn-radon_variable_intercept_slope_centered
module RadonVariableInterceptSlopeCenteredMn

using PosteriorDB
using ..RadonVariableInterceptSlopeCenteredAll: radon_variable_intercept_slope_centered

const data = let
    pdb = PosteriorDB.database()
    PosteriorDB.load(PosteriorDB.dataset(
        PosteriorDB.posterior(pdb, "radon_mn-radon_variable_intercept_slope_centered"),
    ))
end

const wren_model = radon_variable_intercept_slope_centered(
    data["J"], data["county_idx"], data["floor_measure"], data["log_radon"],
)

end
