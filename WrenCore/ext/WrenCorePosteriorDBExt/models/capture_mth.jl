# PosteriorDB: Mth_data-Mth_model
module CaptureMth

using PosteriorDB
using ..LogAddExp: logaddexp
using WrenCore: @wren, @addloglikelihood, @iid, logdensityof
using WrenDists: WrenDists as WD
using ..CaptureCommon: capture_counts

const data = let
    pdb = PosteriorDB.database()
    PosteriorDB.load(PosteriorDB.dataset(PosteriorDB.posterior(pdb, "Mth_data-Mth_model")))
end

function capture_mth_logp(omega, mean_p, sigma, eps_raw, y, seen)
    present, absent = log(omega), log1p(-omega)
    total = zero(omega + sigma + mean_p[1])
    for i in axes(y, 1)
        row = zero(total)
        for j in axes(y, 2)
            mean_logit = log(mean_p[j]) - log1p(-mean_p[j])
            eta = mean_logit + sigma * eps_raw[i]
            row += logdensityof(WD.BernoulliLogit(eta), y[i, j])
        end
        row += present
        total += seen[i] ? row : logaddexp(row, absent)
    end
    return total
end

@wren function capture_mth(y, seen)
    omega ~ WD.Uniform(0.0, 1.0)
    @iid mean_p ~ WD.Uniform(0.0, 1.0) size=size(y, 2)
    sigma ~ WD.Uniform(0.0, 5.0)
    @iid eps_raw ~ WD.Normal(0.0, 1.0) size=size(y, 1)
    @addloglikelihood capture_mth_logp(omega, mean_p, sigma, eps_raw, y, seen)
    return nothing
end

const wren_model = let
    y = data["y"]
    capture_mth(y, capture_counts(y) .> 0)
end

end
