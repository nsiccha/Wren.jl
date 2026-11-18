# PosteriorDB: the seven earnings-height regression fixtures.
module EarningsCommon

using PosteriorDB
using Statistics: mean, std
using ..FlatGaussianRegression: flat_gaussian

const data = let
    pdb = PosteriorDB.database()
    PosteriorDB.load(PosteriorDB.dataset(PosteriorDB.posterior(
        pdb, "earnings-earn_height",
    )))
end

function fixture(kind::Symbol)
    earn, height, male = data["earn"], data["height"], data["male"]
    N = data["N"]
    if kind == :earn_height
        y = earn
        X = hcat(ones(N), height)
    elseif kind == :log10earn_height
        y = log10.(earn)
        X = hcat(ones(N), height)
    elseif kind == :logearn_height
        y = log.(earn)
        X = hcat(ones(N), height)
    elseif kind == :logearn_height_male
        y = log.(earn)
        X = hcat(ones(N), height, male)
    elseif kind == :logearn_interaction
        y = log.(earn)
        X = hcat(ones(N), height, male, height .* male)
    elseif kind == :logearn_interaction_z
        y = log.(earn)
        zheight = (height .- mean(height)) ./ std(height)
        X = hcat(ones(N), zheight, male, zheight .* male)
    elseif kind == :logearn_logheight_male
        y = log.(earn)
        X = hcat(ones(N), log.(height), male)
    else
        throw(ArgumentError("unknown earnings model: $kind"))
    end
    return flat_gaussian(y, X)
end

end

module EarnHeight
using ..EarningsCommon: fixture
const wren_model = fixture(:earn_height)
end
module Log10EarnHeight
using ..EarningsCommon: fixture
const wren_model = fixture(:log10earn_height)
end
module LogEarnHeight
using ..EarningsCommon: fixture
const wren_model = fixture(:logearn_height)
end
module LogEarnHeightMale
using ..EarningsCommon: fixture
const wren_model = fixture(:logearn_height_male)
end
module LogEarnInteraction
using ..EarningsCommon: fixture
const wren_model = fixture(:logearn_interaction)
end
module LogEarnInteractionZ
using ..EarningsCommon: fixture
const wren_model = fixture(:logearn_interaction_z)
end
module LogEarnLogheightMale
using ..EarningsCommon: fixture
const wren_model = fixture(:logearn_logheight_male)
end
