# PosteriorDB: bball_drive_event_1-hmm_drive_1
module BballDrive1

using PosteriorDB
using ..LogAddExp: logaddexp
using WrenCore: @wren, @addloglikelihood, logdensityof
using WrenDists: WrenDists as WD

const data = let
    pdb = PosteriorDB.database()
    PosteriorDB.load(PosteriorDB.dataset(PosteriorDB.posterior(
        pdb, "bball_drive_event_1-hmm_drive_1",
    )))
end

@wren function hmm_drive_1(u, v, tau, rho)
    p11 ~ WD.Beta(4.0, 2.0)
    p21 ~ WD.Beta(2.0, 4.0)
    phi1 ~ WD.Flat()
    raw_phi_gap ~ WD.Flat()
    lambda1 ~ WD.Flat()
    raw_lambda_gap ~ WD.Flat()

    phi2 = phi1 + exp(raw_phi_gap)
    lambda2 = lambda1 + exp(raw_lambda_gap)

    # The ordered means still require their transform Jacobians.
    # Native ordered priors are tracked in #207.
    @addloglikelihood raw_phi_gap + raw_lambda_gap +
        logdensityof(WD.Normal(0.0, 1.0), phi1) +
        logdensityof(WD.Normal(3.0, 1.0), phi2) +
        logdensityof(WD.Normal(0.0, 1.0), lambda1) +
        logdensityof(WD.Normal(3.0, 1.0), lambda2)

    g1 = logdensityof(WD.Normal(phi1, tau), u[1]) +
        logdensityof(WD.Normal(lambda1, rho), v[1])
    g2 = logdensityof(WD.Normal(phi2, tau), u[1]) +
        logdensityof(WD.Normal(lambda2, rho), v[1])
    for t in 2:length(u)
        next1 = logdensityof(WD.Normal(phi1, tau), u[t]) +
            logdensityof(WD.Normal(lambda1, rho), v[t]) +
            logaddexp(g1 + log(p11), g2 + log(p21))
        next2 = logdensityof(WD.Normal(phi2, tau), u[t]) +
            logdensityof(WD.Normal(lambda2, rho), v[t]) +
            logaddexp(g1 + log1p(-p11), g2 + log1p(-p21))
        g1, g2 = next1, next2
    end
    @addloglikelihood logaddexp(g1, g2)
    return nothing
end

const wren_model = let
    data["K"] == 2 || error("public basketball HMM fixture requires K=2")
    data["alpha"] == [4 2; 2 4] || error("unexpected transition prior")
    hmm_drive_1(data["u"], data["v"], data["tau"], data["rho"])
end

end
