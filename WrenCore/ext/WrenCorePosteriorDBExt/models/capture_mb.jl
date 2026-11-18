# PosteriorDB: Mb_data-Mb_model
module CaptureMb

using PosteriorDB
using ..LogAddExp: logaddexp
using WrenCore: @wren, @addloglikelihood, logdensityof
using WrenDists: WrenDists as WD
using ..CaptureCommon: capture_counts

const data = let
    pdb = PosteriorDB.database()
    PosteriorDB.load(PosteriorDB.dataset(PosteriorDB.posterior(pdb, "Mb_data-Mb_model")))
end

function capture_mb_logp(omega, p, c, y, seen)
    present, absent = log(omega), log1p(-omega)
    total = zero(omega + p + c)
    for i in axes(y, 1)
        row = zero(total)
        for j in axes(y, 2)
            prob = j == 1 || y[i, j - 1] == 0 ? p : c
            row += logdensityof(WD.Bernoulli(prob), y[i, j])
        end
        row += present
        total += seen[i] ? row : logaddexp(row, absent)
    end
    return total
end

@wren function capture_mb(y, seen)
    omega ~ WD.Uniform(0.0, 1.0)
    p ~ WD.Uniform(0.0, 1.0)
    c ~ WD.Uniform(0.0, 1.0)
    @addloglikelihood capture_mb_logp(omega, p, c, y, seen)
    return nothing
end

const wren_model = let
    y = data["y"]
    capture_mb(y, capture_counts(y) .> 0)
end

end
