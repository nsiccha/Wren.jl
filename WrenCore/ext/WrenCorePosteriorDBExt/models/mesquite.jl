# PosteriorDB: six Mesquite mass regressions.
module MesquiteCommon

using PosteriorDB
using ..FlatGaussianRegression: flat_gaussian

const data = let
    pdb = PosteriorDB.database()
    PosteriorDB.load(PosteriorDB.dataset(PosteriorDB.posterior(
        pdb, "mesquite-mesquite",
    )))
end

function fixture(kind::Symbol)
    d1, d2 = data["diam1"], data["diam2"]
    canopy, height = data["canopy_height"], data["total_height"]
    density, group = data["density"], data["group"]
    weight, N = data["weight"], data["N"]
    logvolume = log.(d1 .* d2 .* canopy)
    logarea = log.(d1 .* d2)
    logshape = log.(d1 ./ d2)
    if kind == :mesquite
        y = weight
        X = hcat(ones(N), d1, d2, canopy, height, density, group)
    elseif kind == :logmesquite
        y = log.(weight)
        X = hcat(ones(N), log.(d1), log.(d2), log.(canopy),
            log.(height), log.(density), group)
    elseif kind == :logmesquite_logva
        y = log.(weight)
        X = hcat(ones(N), logvolume, logarea, group)
    elseif kind == :logmesquite_logvas
        y = log.(weight)
        X = hcat(ones(N), logvolume, logarea, logshape,
            log.(height), log.(density), group)
    elseif kind == :logmesquite_logvash
        y = log.(weight)
        X = hcat(ones(N), logvolume, logarea, logshape, log.(height), group)
    elseif kind == :logmesquite_logvolume
        y = log.(weight)
        X = hcat(ones(N), logvolume)
    else
        throw(ArgumentError("unknown Mesquite model: $kind"))
    end
    return flat_gaussian(y, X)
end

end

module MesquiteRaw
using ..MesquiteCommon: fixture
const wren_model = fixture(:mesquite)
end
module LogMesquite
using ..MesquiteCommon: fixture
const wren_model = fixture(:logmesquite)
end
module LogMesquiteLogVA
using ..MesquiteCommon: fixture
const wren_model = fixture(:logmesquite_logva)
end
module LogMesquiteLogVAS
using ..MesquiteCommon: fixture
const wren_model = fixture(:logmesquite_logvas)
end
module LogMesquiteLogVASH
using ..MesquiteCommon: fixture
const wren_model = fixture(:logmesquite_logvash)
end
module LogMesquiteLogVolume
using ..MesquiteCommon: fixture
const wren_model = fixture(:logmesquite_logvolume)
end
