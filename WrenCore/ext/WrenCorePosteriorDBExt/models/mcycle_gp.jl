# PosteriorDB: mcycle_gp-accel_gp
module McycleGP

using PosteriorDB
using WrenCore: @wren, @observe, @batch, @iid
using WrenDists: WrenDists as WD

const data = let
    pdb = PosteriorDB.database()
    PosteriorDB.load(PosteriorDB.dataset(PosteriorDB.posterior(
        pdb, "mcycle_gp-accel_gp",
    )))
end

# brms' approximate GP: basis functions weighted by the square root of the
# exponentiated-quadratic spectral density at the square-root eigenvalues.
function approximate_gp(X, slambda, sdgp, lscale, z)
    spd = sdgp^2 * sqrt(2pi) * lscale .* exp.(-0.5 * lscale^2 .* slambda .^ 2)
    return X * (sqrt.(spd) .* z)
end

@wren function accel_gp(Y, Xgp, slambda, Xgp_sigma, slambda_sigma, prior_only)
    Intercept ~ WD.TDist(3.0, -13.0, 36.0)
    sdgp ~ WD.HalfTDist(3.0, 0.0, 36.0)
    lscale ~ WD.InverseGamma(1.124909, 0.0177)
    @iid zgp ~ WD.Normal(0.0, 1.0) size=size(Xgp, 2)
    Intercept_sigma ~ WD.TDist(3.0, 0.0, 10.0)
    sdgp_sigma ~ WD.HalfTDist(3.0, 0.0, 36.0)
    lscale_sigma ~ WD.InverseGamma(1.124909, 0.0177)
    @iid zgp_sigma ~ WD.Normal(0.0, 1.0) size=size(Xgp_sigma, 2)
    if prior_only == 0
        mu = Intercept .+ approximate_gp(Xgp, slambda, sdgp, lscale, zgp)
        sigma = exp.(Intercept_sigma .+ approximate_gp(Xgp_sigma, slambda_sigma,
            sdgp_sigma, lscale_sigma, zgp_sigma))
        @batch for i in eachindex(Y)
            @observe Y[i] ~ WD.Normal(mu[i], sigma[i])
        end
    end
    return nothing
end

const wren_model = accel_gp(
    data["Y"], data["Xgp_1"], vec(data["slambda_1"]),
    data["Xgp_sigma_1"], vec(data["slambda_sigma_1"]), data["prior_only"],
)

end
