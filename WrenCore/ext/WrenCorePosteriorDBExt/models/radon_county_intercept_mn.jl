# PosteriorDB: radon_mn-radon_county_intercept
module RadonCountyInterceptMn

using PosteriorDB
using ..RadonCountyInterceptAll: radon_county_intercept

const data = let
    pdb = PosteriorDB.database()
    PosteriorDB.load(PosteriorDB.dataset(
        PosteriorDB.posterior(pdb, "radon_mn-radon_county_intercept"),
    ))
end

const wren_model = radon_county_intercept(
    data["J"], data["county_idx"], data["floor_measure"], data["log_radon"],
)

end
