# PosteriorDB: radon_mn-radon_variable_intercept_centered
module RadonVariableInterceptCenteredMn

using PosteriorDB
using ..RadonVariableInterceptCenteredAll: radon_variable_intercept_centered

const data = let
    pdb = PosteriorDB.database()
    PosteriorDB.load(PosteriorDB.dataset(
        PosteriorDB.posterior(pdb, "radon_mn-radon_variable_intercept_centered"),
    ))
end

const wren_model = radon_variable_intercept_centered(
    data["J"], data["county_idx"], data["floor_measure"], data["log_radon"],
)

end
