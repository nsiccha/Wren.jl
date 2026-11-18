# PosteriorDB: M0_data-M0_model
module CaptureM0

using PosteriorDB
using ..LogAddExp: logaddexp
using WrenCore: @wren, @addloglikelihood, logdensityof
using WrenDists: WrenDists as WD
using ..CaptureCommon: capture_counts

const data = let
    pdb = PosteriorDB.database()
    PosteriorDB.load(PosteriorDB.dataset(PosteriorDB.posterior(pdb, "M0_data-M0_model")))
end

function capture_m0_logp(omega, p, s, T)
    present, absent = log(omega), log1p(-omega)
    total = zero(omega + p)
    for si in s
        row = present + logdensityof(WD.Binomial(T, p), si)
        total += si > 0 ? row : logaddexp(row, absent)
    end
    return total
end

@wren function capture_m0(s, T)
    omega ~ WD.Uniform(0.0, 1.0)
    p ~ WD.Uniform(0.0, 1.0)
    @addloglikelihood capture_m0_logp(omega, p, s, T)
    return nothing
end

const wren_model = capture_m0(capture_counts(data["y"]), data["T"])

end
