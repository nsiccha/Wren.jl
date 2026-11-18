# PosteriorDB: rats_data-rats_model
module RatsModel

using PosteriorDB
using WrenCore: @wren, @observe, @batch, @iid, @addloglikelihood
using WrenDists: WrenDists as WD

const data = let
    pdb = PosteriorDB.database()
    PosteriorDB.load(PosteriorDB.dataset(PosteriorDB.posterior(
        pdb, "rats_data-rats_model",
    )))
end

@wren function rats_model(rat, x, y, xbar, N)
    mu_alpha ~ WD.Normal(0.0, 100.0)
    mu_beta ~ WD.Normal(0.0, 100.0)
    log_sigma_y ~ WD.Flat()
    log_sigma_alpha ~ WD.Flat()
    log_sigma_beta ~ WD.Flat()
    sigma_y = exp(log_sigma_y)
    sigma_alpha = exp(log_sigma_alpha)
    sigma_beta = exp(log_sigma_beta)
    # The three unprioritized positive scales are flat on their original scales.
    # Positive flat scales: each log-coordinate Jacobian is needed (#192).
    @addloglikelihood log_sigma_y + log_sigma_alpha + log_sigma_beta
    @iid alpha ~ WD.Normal(mu_alpha, sigma_alpha) size=N
    @iid beta ~ WD.Normal(mu_beta, sigma_beta) size=N
    mu = alpha[rat] .+ beta[rat] .* (x .- xbar)
    @batch for i in eachindex(y)
        @observe y[i] ~ WD.Normal(mu[i], sigma_y)
    end
    return nothing
end

const wren_model = rats_model(data["rat"], data["x"], data["y"], data["xbar"],
    data["N"])

end
