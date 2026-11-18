# PosteriorDB: mcycle_splines-accel_splines
module AccelSplines

using PosteriorDB
using WrenCore: @wren, @observe, @batch, @iid
using WrenDists: WrenDists as WD

const data = let
    pdb = PosteriorDB.database()
    PosteriorDB.load(PosteriorDB.dataset(PosteriorDB.posterior(
        pdb, "mcycle_splines-accel_splines",
    )))
end

@wren function accel_splines(y, Xs, Zs, Xs_sigma, Zs_sigma, prior_only)
    Intercept ~ WD.TDist(3.0, -13.0, 36.0)
    @iid bs ~ WD.Flat() size=size(Xs, 2)
    @iid zs ~ WD.Normal(0.0, 1.0) size=size(Zs, 2)
    sds ~ WD.HalfTDist(3.0, 0.0, 36.0)
    Intercept_sigma ~ WD.TDist(3.0, 0.0, 10.0)
    @iid bs_sigma ~ WD.Flat() size=size(Xs_sigma, 2)
    @iid zs_sigma ~ WD.Normal(0.0, 1.0) size=size(Zs_sigma, 2)
    sds_sigma ~ WD.HalfTDist(3.0, 0.0, 36.0)
    if prior_only == 0
        mu = Intercept .+ Xs * bs .+ Zs * (sds .* zs)
        log_sigma = Intercept_sigma .+ Xs_sigma * bs_sigma .+
            Zs_sigma * (sds_sigma .* zs_sigma)
        sigma = exp.(log_sigma)
        @batch for i in eachindex(y)
            @observe y[i] ~ WD.Normal(mu[i], sigma[i])
        end
    end
    return nothing
end

const wren_model = accel_splines(
    data["Y"], data["Xs"], data["Zs_1_1"], data["Xs_sigma"],
    data["Zs_sigma_1_1"], data["prior_only"],
)

end
