# PosteriorDB: kilpisjarvi_mod-kilpisjarvi
module Kilpisjarvi

using PosteriorDB
using WrenCore: @wren, @observe, @batch, @addloglikelihood
using WrenDists: WrenDists as WD

const data = let
    pdb = PosteriorDB.database()
    PosteriorDB.load(PosteriorDB.dataset(PosteriorDB.posterior(
        pdb, "kilpisjarvi_mod-kilpisjarvi",
    )))
end

@wren function kilpisjarvi_model(x, y, pmualpha, psalpha, pmubeta, psbeta)
    alpha ~ WD.Normal(pmualpha, psalpha)
    beta ~ WD.Normal(pmubeta, psbeta)
    log_sigma ~ WD.Flat()
    sigma = exp(log_sigma)
    # Stan's positive, unprioritized sigma is flat in sigma, not log sigma.
    # Positive flat scale: the log-coordinate Jacobian is needed (#192).
    @addloglikelihood log_sigma
    mu = alpha .+ beta .* x
    @batch for i in eachindex(y)
        @observe y[i] ~ WD.Normal(mu[i], sigma)
    end
    return nothing
end

const wren_model = kilpisjarvi_model(data["x"], data["y"], data["pmualpha"],
    data["psalpha"], data["pmubeta"], data["psbeta"])

end
