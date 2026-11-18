# PosteriorDB: bball_drive_event_0-hmm_drive_0
module BballDrive0

using PosteriorDB
using ..LogAddExp: logaddexp
using WrenCore: @wren, @addloglikelihood, logdensityof
using WrenDists: WrenDists as WD

const data = let
    pdb = PosteriorDB.database()
    PosteriorDB.load(PosteriorDB.dataset(PosteriorDB.posterior(
        pdb, "bball_drive_event_0-hmm_drive_0",
    )))
end

@wren function hmm_drive_0(u, v)
    p11 ~ WD.Beta(4.0, 2.0)
    p21 ~ WD.Beta(2.0, 4.0)
    raw_phi1 ~ WD.Flat()
    raw_phi_gap ~ WD.Flat()
    raw_lambda1 ~ WD.Flat()
    raw_lambda_gap ~ WD.Flat()

    phi1 = exp(raw_phi1)
    phi2 = phi1 + exp(raw_phi_gap)
    lambda1 = exp(raw_lambda1)
    lambda2 = lambda1 + exp(raw_lambda_gap)

    # The ordered positive rates still require their transform Jacobians.
    # Native positive-ordered priors are tracked in #207.
    @addloglikelihood raw_phi1 + raw_phi_gap + raw_lambda1 + raw_lambda_gap +
        logdensityof(WD.Normal(0.0, 1.0), phi1) +
        logdensityof(WD.Normal(3.0, 1.0), phi2) +
        logdensityof(WD.Normal(0.0, 1.0), lambda1) +
        logdensityof(WD.Normal(3.0, 1.0), lambda2)

    g1 = log(phi1) - phi1 * u[1] + log(lambda1) - lambda1 * v[1]
    g2 = log(phi2) - phi2 * u[1] + log(lambda2) - lambda2 * v[1]
    for t in 2:length(u)
        next1 = log(phi1) - phi1 * u[t] + log(lambda1) - lambda1 * v[t] +
            logaddexp(g1 + log(p11), g2 + log(p21))
        next2 = log(phi2) - phi2 * u[t] + log(lambda2) - lambda2 * v[t] +
            logaddexp(g1 + log1p(-p11), g2 + log1p(-p21))
        g1, g2 = next1, next2
    end
    @addloglikelihood logaddexp(g1, g2)
    return nothing
end

const wren_model = let
    data["K"] == 2 || error("public basketball HMM fixture requires K=2")
    data["alpha"] == [4 2; 2 4] || error("unexpected transition prior")
    all(>=(0), data["u"]) && all(>=(0), data["v"]) ||
        error("exponential observations must be nonnegative")
    hmm_drive_0(data["u"], data["v"])
end

end
