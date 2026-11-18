# PosteriorDB: Survey_data-Survey_model
module Survey

using PosteriorDB
using WrenCore: @wren, @addloglikelihood, logdensityof
using WrenDists: WrenDists as WD
using ..LogAddExp: logaddexp

const data = let
    pdb = PosteriorDB.database()
    PosteriorDB.load(PosteriorDB.dataset(
        PosteriorDB.posterior(pdb, "Survey_data-Survey_model"),
    ))
end

function survey_mixture_logp(theta, k, nmax)
    nmin = maximum(k)
    parts = [sum(logdensityof(WD.Binomial(n, theta), ki) for ki in k) - log(nmax)
        for n in nmin:nmax]
    return reduce(logaddexp, parts)
end

@wren function survey_model(k, nmax)
    theta ~ WD.Uniform(0.0, 1.0)
    @addloglikelihood survey_mixture_logp(theta, k, nmax)
    return nothing
end

const wren_model = survey_model(data["k"], data["nmax"])

end
