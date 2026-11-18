# PosteriorDB: hmm_example-hmm_example
module HMMExample

using PosteriorDB
using ..LogAddExp: logaddexp
using WrenCore: @wren, @addloglikelihood, logdensityof
using WrenDists: WrenDists as WD

const data = let
    pdb = PosteriorDB.database()
    PosteriorDB.load(PosteriorDB.dataset(PosteriorDB.posterior(
        pdb, "hmm_example-hmm_example",
    )))
end

@wren function hmm_example(y)
    p11 ~ WD.Uniform(0.0, 1.0)
    p21 ~ WD.Uniform(0.0, 1.0)
    raw_mu1 ~ WD.Flat()
    raw_gap ~ WD.Flat()
    mu1 = exp(raw_mu1)
    mu2 = mu1 + exp(raw_gap)
    # Stan's positive_ordered[2] transform still needs its Jacobian.
    # Native positive-ordered priors are tracked in #207.
    @addloglikelihood raw_mu1 + raw_gap +
        logdensityof(WD.Normal(3.0, 1.0), mu1) +
        logdensityof(WD.Normal(10.0, 1.0), mu2)
    g1 = logdensityof(WD.Normal(mu1, 1.0), y[1])
    g2 = logdensityof(WD.Normal(mu2, 1.0), y[1])
    for t in 2:length(y)
        next1 = logdensityof(WD.Normal(mu1, 1.0), y[t]) +
            logaddexp(g1 + log(p11), g2 + log(p21))
        next2 = logdensityof(WD.Normal(mu2, 1.0), y[t]) +
            logaddexp(g1 + log1p(-p11), g2 + log1p(-p21))
        g1, g2 = next1, next2
    end
    @addloglikelihood logaddexp(g1, g2)
    return nothing
end

const wren_model = let
    data["K"] == 2 || error("public HMM example fixture requires K=2")
    hmm_example(data["y"])
end

end
