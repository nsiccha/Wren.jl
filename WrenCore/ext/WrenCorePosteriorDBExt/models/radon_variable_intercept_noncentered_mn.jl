# PosteriorDB: radon_mn-radon_variable_intercept_noncentered
module RadonVariableInterceptNoncenteredMn

using PosteriorDB
using ..RadonVariableInterceptNoncenteredAll: radon_variable_intercept_noncentered

const data = let
    pdb = PosteriorDB.database()
    PosteriorDB.load(PosteriorDB.dataset(
        PosteriorDB.posterior(pdb, "radon_mn-radon_variable_intercept_noncentered"),
    ))
end

const wren_model = radon_variable_intercept_noncentered(
    data["J"], data["county_idx"], data["floor_measure"], data["log_radon"],
)

end
