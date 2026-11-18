# PosteriorDB: Mt_data-Mt_model
module CaptureMt

using PosteriorDB
using ..LogAddExp: logaddexp
using WrenCore: @wren, @addloglikelihood, @iid, logdensityof
using WrenDists: WrenDists as WD
using ..CaptureCommon: capture_counts

const data = let
    pdb = PosteriorDB.database()
    PosteriorDB.load(PosteriorDB.dataset(PosteriorDB.posterior(pdb, "Mt_data-Mt_model")))
end

function capture_mt_logp(omega, p, y, seen)
    present, absent = log(omega), log1p(-omega)
    total = zero(omega + p[1])
    for i in axes(y, 1)
        row = zero(total)
        for j in axes(y, 2)
            row += logdensityof(WD.Bernoulli(p[j]), y[i, j])
        end
        row += present
        total += seen[i] ? row : logaddexp(row, absent)
    end
    return total
end

@wren function capture_mt(y, seen)
    omega ~ WD.Uniform(0.0, 1.0)
    @iid p ~ WD.Uniform(0.0, 1.0) size=size(y, 2)
    @addloglikelihood capture_mt_logp(omega, p, y, seen)
    return nothing
end

const wren_model = let
    y = data["y"]
    capture_mt(y, capture_counts(y) .> 0)
end

end
