# PosteriorDB: hmm_gaussian_simulated-hmm_gaussian
module HMMGaussian

using PosteriorDB
using ..LogAddExp: logaddexp
using WrenCore: @wren, @addloglikelihood, logdensityof
using WrenDists: WrenDists as WD

const data = let
    pdb = PosteriorDB.database()
    PosteriorDB.load(PosteriorDB.dataset(PosteriorDB.posterior(
        pdb, "hmm_gaussian_simulated-hmm_gaussian",
    )))
end

@wren function hmm_gaussian(y)
    pi ~ WD.Dirichlet(fill(1.0, 3))
    a1 ~ WD.Dirichlet(fill(1.0, 3))
    a2 ~ WD.Dirichlet(fill(1.0, 3))
    a3 ~ WD.Dirichlet(fill(1.0, 3))
    raw_mu1 ~ WD.Flat()
    raw_gap2 ~ WD.Flat()
    raw_gap3 ~ WD.Flat()
    raw_sigma1 ~ WD.Flat()
    raw_sigma2 ~ WD.Flat()
    raw_sigma3 ~ WD.Flat()

    mu1 = raw_mu1
    mu2 = mu1 + exp(raw_gap2)
    mu3 = mu2 + exp(raw_gap3)
    sigma1 = exp(raw_sigma1)
    sigma2 = exp(raw_sigma2)
    sigma3 = exp(raw_sigma3)
    # Ordered means and positive flat scales still need Jacobians.
    # Native ordered priors are tracked in #207; positive-flat scales in #192.
    @addloglikelihood raw_gap2 + raw_gap3 + raw_sigma1 + raw_sigma2 + raw_sigma3

    # The Stan source starts the forward pass with
    #     logalpha[1] = log(pi1) + normal_lpdf(y[1] | mu, sigma);
    # where the vectorised `normal_lpdf` is one number, the sum over all three
    # states. Every state therefore gets the same first emission term. This is the
    # PosteriorDB target, not the usual per-state emission.
    first_emission = logdensityof(WD.Normal(mu1, sigma1), y[1]) +
        logdensityof(WD.Normal(mu2, sigma2), y[1]) +
        logdensityof(WD.Normal(mu3, sigma3), y[1])
    g1 = log(pi[1]) + first_emission
    g2 = log(pi[2]) + first_emission
    g3 = log(pi[3]) + first_emission
    for t in 2:length(y)
        next1 = logdensityof(WD.Normal(mu1, sigma1), y[t]) +
            logaddexp(logaddexp(g1 + log(a1[1]), g2 + log(a2[1])),
                g3 + log(a3[1]))
        next2 = logdensityof(WD.Normal(mu2, sigma2), y[t]) +
            logaddexp(logaddexp(g1 + log(a1[2]), g2 + log(a2[2])),
                g3 + log(a3[2]))
        next3 = logdensityof(WD.Normal(mu3, sigma3), y[t]) +
            logaddexp(logaddexp(g1 + log(a1[3]), g2 + log(a2[3])),
                g3 + log(a3[3]))
        g1, g2, g3 = next1, next2, next3
    end
    @addloglikelihood logaddexp(logaddexp(g1, g2), g3)
    return nothing
end

const wren_model = let
    data["K"] == 3 || error("public Gaussian HMM fixture requires K=3")
    hmm_gaussian(data["y"])
end

end
