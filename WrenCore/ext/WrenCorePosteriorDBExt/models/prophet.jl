# PosteriorDB: rstan_downloads-prophet
module Prophet

using PosteriorDB
using WrenCore: @wren, @observe, @batch, @iid
using WrenDists: WrenDists as WD

const data = let
    pdb = PosteriorDB.database()
    PosteriorDB.load(PosteriorDB.dataset(PosteriorDB.posterior(
        pdb, "rstan_downloads-prophet",
    )))
end

@wren function prophet_linear(y, t, t_change, A, X, sigmas, tau, s_a, s_m)
    k ~ WD.Normal(0.0, 5.0)
    m ~ WD.Normal(0.0, 5.0)
    @iid delta ~ WD.Laplace(0.0, tau) size=length(t_change)
    sigma_obs ~ WD.HalfNormal(0.0, 0.5)
    @iid beta ~ WD.Normal(0.0, sigmas[1]) size=length(sigmas)
    trend = (k .+ A * delta) .* t .+ m .- A * (t_change .* delta)
    mu = trend .* (1 .+ X * (beta .* s_m)) .+ X * (beta .* s_a)
    @batch for i in eachindex(y)
        @observe y[i] ~ WD.Normal(mu[i], sigma_obs)
    end
    return nothing
end

const wren_model = let
    data["trend_indicator"] == 0 || error("public Prophet fixture requires linear trend")
    sigmas = data["sigmas"]
    all(==(sigmas[1]), sigmas) || error("public Prophet fixture has varying beta scales")
    t, t_change = data["t"], data["t_change"]
    A = Float64.(t .>= t_change')
    prophet_linear(data["y"], t, t_change, A, data["X"], sigmas,
        data["tau"], data["s_a"], data["s_m"])
end

end
